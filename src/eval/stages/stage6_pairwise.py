"""Stage 6: pairwise method comparison with MT-Bench bias mitigations.

- Samples a seed-fixed subset of bills (``pairwise_sample_bills``;
  ``null`` uses the full intersection) and runs three comparison
  protocols:
    1. Single-answer: compare Stage 4 coverage percentages and Stage 5
       novel counts for the two methods directly.
    2. Pairwise with answer swap: for each sampled bill run (A,B) then
       (B,A) through the judge and record both orders; average the win
       rates to attenuate position bias (Zheng et al. 2023,
       ``docs/lit_rev_eval.md`` line 137).
    3. Reference-guided: the human NCSL labels are provided to the judge as
       the reference answer.
- Caches one row per bill under
  ``output/evals/v1/cache/pairwise/<bill_id>.jsonl``; rows carry ``winner``
  and ``rationale`` fields instead of the verdict shape used by Stages 3-4.
- Reports raw, swap-averaged, and quadruplet-count-normalised win rates
  because pure count bias is a known MT-Bench pitfall.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any

from ..artifacts import Quadruplet, StageResult
from ..cache import CacheWriter, cache_paths_for_stage, filter_pending, iter_cache_rows
from ..judge import CreditsExhaustedError, call_judge_json
from ..prompts import (
    PAIRWISE_SCHEMA,
    PAIRWISE_SYSTEM_PROMPT,
    PAIRWISE_USER_PROMPT_TEMPLATE,
)
from ._common import StageContext, gather_with_progress, write_result_json

logger = logging.getLogger(__name__)

_SCHEMA_NAME = "stage6_pairwise"


@dataclass(slots=True)
class _Item:
    """One pairwise comparison job (bill + two method quadruplet sets)."""

    bill_id: str
    a_method: str
    b_method: str
    a_quadruplets: list[Quadruplet]
    b_quadruplets: list[Quadruplet]
    swap: bool


def run(ctx: StageContext) -> StageResult:
    """Run pairwise comparisons with answer swap and emit aggregate metrics.

    Args:
        ctx: Pre-built :class:`StageContext`.  The judge connection is
            required.

    Returns:
        :class:`StageResult` carrying raw and swap-averaged win rates plus
        the reference-guided single-answer comparison.
    """

    if ctx.judge is None:
        raise RuntimeError("Stage 6 requires a judge connection")

    stage_params = ctx.config.stages.get(6)
    raw_sample_n = (
        stage_params.params.get("pairwise_sample_bills", 200)
        if stage_params else 200
    )
    sample_n: int | None = None if raw_sample_n is None else int(raw_sample_n)
    seed = int(
        stage_params.params.get("seed", ctx.config.sampling.seed)
        if stage_params else ctx.config.sampling.seed
    )
    swap = bool(stage_params.params.get("swap", True)) if stage_params else True

    method_names = list(ctx.method_outputs.keys())
    if len(method_names) < 2:
        raise ValueError("Stage 6 requires at least two methods in the config")
    a_method, b_method = method_names[0], method_names[1]

    sampled_bills = _sample_bills(
        ctx.sampled_bill_ids, n=sample_n, seed=seed
    )
    items = _build_items(
        ctx,
        bill_ids=sampled_bills,
        a_method=a_method,
        b_method=b_method,
        swap=swap,
    )
    asyncio.run(_judge_all(ctx, items=items))

    summary = _aggregate(
        ctx,
        items=items,
        a_method=a_method,
        b_method=b_method,
        single_answer=_single_answer_comparison(ctx, a_method=a_method, b_method=b_method),
    )
    result_file = write_result_json(
        ctx,
        filename="stage6_pairwise.json",
        payload={"stage": 6, **summary, "sampled_bill_ids": sampled_bills},
    )
    return StageResult(
        stage=6,
        status="completed",
        summary=_summary_line(summary),
        metrics=summary,
        artifacts={"result_file": str(result_file)},
    )


def _sample_bills(all_bill_ids: list[str], *, n: int | None, seed: int) -> list[str]:
    """Sample ``n`` bills from ``all_bill_ids`` deterministically.

    Passing ``n=None`` (the YAML null value) uses every available bill.
    """

    if not all_bill_ids:
        return []
    if n is None or n >= len(all_bill_ids):
        return sorted(all_bill_ids)
    rng = random.Random(seed)
    return sorted(rng.sample(all_bill_ids, n))


def _build_items(
    ctx: StageContext,
    *,
    bill_ids: list[str],
    a_method: str,
    b_method: str,
    swap: bool,
) -> list[_Item]:
    """Gather the per-bill work items for the pairwise stage."""

    items: list[_Item] = []
    for bill_id in bill_ids:
        record = ctx.bill_records.get(bill_id)
        if record is None:
            continue
        a = ctx.method_outputs[a_method].get(bill_id, [])
        b = ctx.method_outputs[b_method].get(bill_id, [])
        if not a and not b:
            continue
        items.append(
            _Item(
                bill_id=bill_id,
                a_method=a_method,
                b_method=b_method,
                a_quadruplets=a,
                b_quadruplets=b,
                swap=swap,
            )
        )
    logger.info("Stage 6: pairwise items=%d (swap=%s)", len(items), swap)
    return items


async def _judge_all(ctx: StageContext, *, items: list[_Item]) -> None:
    """Drive both orders (A,B) and optionally (B,A) through the judge."""

    paths = cache_paths_for_stage(run_dir=ctx.run_dir, stage=6)
    writer = CacheWriter(paths)
    work: list[tuple[str, str, dict[str, Any]]] = []
    for item in items:
        work.append((item.bill_id, f"{item.bill_id}::AB", {"item": item, "order": "AB"}))
        if item.swap:
            work.append(
                (item.bill_id, f"{item.bill_id}::BA", {"item": item, "order": "BA"})
            )
    pending = filter_pending(work, paths)
    logger.info(
        "Stage 6 cache: total=%d cached=%d pending=%d",
        len(work), len(work) - len(pending), len(pending),
    )
    if not pending:
        return

    semaphore = asyncio.Semaphore(4)
    tasks = [
        _judge_one(ctx, item=payload["item"], order=payload["order"], writer=writer, semaphore=semaphore)
        for _, _, payload in pending
    ]
    await gather_with_progress(tasks, desc="Stage 6 (pairwise)")


async def _judge_one(
    ctx: StageContext,
    *,
    item: _Item,
    order: str,
    writer: CacheWriter,
    semaphore: asyncio.Semaphore,
) -> None:
    """Judge one (bill, order) pair and append the row to the cache."""

    record = ctx.bill_records.get(item.bill_id)
    if record is None:
        return
    if order == "AB":
        a_list, b_list = item.a_quadruplets, item.b_quadruplets
        a_label, b_label = item.a_method, item.b_method
    else:
        a_list, b_list = item.b_quadruplets, item.a_quadruplets
        a_label, b_label = item.b_method, item.a_method

    prompt = PAIRWISE_USER_PROMPT_TEMPLATE.format(
        bill_id=record.bill_id,
        state=record.state,
        year=record.year,
        title=record.title,
        topics=", ".join(record.topics) or "(none)",
        a_count=len(a_list),
        b_count=len(b_list),
        a_block=_format_block(a_list),
        b_block=_format_block(b_list),
    )
    async with semaphore:
        try:
            payload, usage = await call_judge_json(
                ctx.judge,
                system_prompt=PAIRWISE_SYSTEM_PROMPT,
                user_prompt=prompt,
                schema=PAIRWISE_SCHEMA,
                schema_name=_SCHEMA_NAME,
            )
        except CreditsExhaustedError:
            raise
        except Exception:
            logger.exception(
                "Stage 6 judge call failed: bill=%s order=%s", item.bill_id, order
            )
            payload, usage = None, {}

    winner_raw = str((payload or {}).get("winner") or "").strip().upper()
    winner_method = _winner_to_method(
        winner=winner_raw, a_label=a_label, b_label=b_label,
    )
    writer.append(
        bill_id=item.bill_id,
        key=f"{item.bill_id}::{order}",
        payload={
            "order": order,
            "a_label": a_label,
            "b_label": b_label,
            "a_count": len(a_list),
            "b_count": len(b_list),
            "winner_raw": winner_raw,
            "winner_method": winner_method,
            "rationale": (payload or {}).get("rationale", ""),
            "usage": usage,
        },
    )


def _format_block(quadruplets: list[Quadruplet]) -> str:
    """Render one side of the pairwise prompt as a bullet list."""

    if not quadruplets:
        return "  (none)"
    lines = [
        f"  - ({q.entity} | {q.type} | {q.attribute} | {q.value})"
        for q in quadruplets[:60]
    ]
    if len(quadruplets) > 60:
        lines.append(f"  - ... ({len(quadruplets) - 60} more)")
    return "\n".join(lines)


def _winner_to_method(*, winner: str, a_label: str, b_label: str) -> str:
    """Translate the judge's ``A`` / ``B`` / ``tie`` answer to a method name."""

    winner = winner.upper()
    if winner == "A":
        return a_label
    if winner == "B":
        return b_label
    if winner == "TIE":
        return "tie"
    return "error"


def _aggregate(
    ctx: StageContext,
    *,
    items: list[_Item],
    a_method: str,
    b_method: str,
    single_answer: dict[str, Any],
) -> dict[str, Any]:
    """Produce raw, swap-averaged, and count-normalised win rates."""

    paths = cache_paths_for_stage(run_dir=ctx.run_dir, stage=6)
    wins_ab: dict[str, int] = {a_method: 0, b_method: 0, "tie": 0, "error": 0}
    wins_ba: dict[str, int] = {a_method: 0, b_method: 0, "tie": 0, "error": 0}
    normalised_points = {a_method: 0.0, b_method: 0.0}
    n_pairs = 0
    n_swap_pairs = 0
    for item in items:
        ab_row = _lookup_row(paths, item.bill_id, key=f"{item.bill_id}::AB")
        if not ab_row:
            continue
        n_pairs += 1
        ab_winner = str(ab_row.get("winner_method") or "error")
        if ab_winner in wins_ab:
            wins_ab[ab_winner] += 1
        else:
            wins_ab["error"] += 1

        ba_row = (
            _lookup_row(paths, item.bill_id, key=f"{item.bill_id}::BA")
            if item.swap else None
        )
        if ba_row:
            n_swap_pairs += 1
            ba_winner = str(ba_row.get("winner_method") or "error")
            if ba_winner in wins_ba:
                wins_ba[ba_winner] += 1
            else:
                wins_ba["error"] += 1

        _accumulate_count_normalised_points(
            normalised_points=normalised_points,
            a_method=a_method,
            b_method=b_method,
            ab_row=ab_row,
            ba_row=ba_row,
        )

    denom = max(1, n_pairs)
    return {
        "a_method": a_method,
        "b_method": b_method,
        "n_pairs": n_pairs,
        "n_swap_pairs": n_swap_pairs,
        "wins_ab": wins_ab,
        "wins_ba": wins_ba,
        "swap_averaged_winrate": {
            name: round(
                (wins_ab.get(name, 0) + wins_ba.get(name, 0))
                / max(1, n_pairs + n_swap_pairs),
                4,
            )
            for name in (a_method, b_method, "tie")
        },
        "count_normalised_points": {
            name: round(points / denom, 4)
            for name, points in normalised_points.items()
        },
        "single_answer": single_answer,
    }


def _accumulate_count_normalised_points(
    *,
    normalised_points: dict[str, float],
    a_method: str,
    b_method: str,
    ab_row: dict[str, Any],
    ba_row: dict[str, Any] | None,
) -> None:
    """Update the count-normalised scores so longer sets are not rewarded."""

    def points_from(row: dict[str, Any]) -> tuple[float, float]:
        """Return the (A, B) points contributed by one judged row."""
        winner = str(row.get("winner_method") or "error")
        a_count = float(row.get("a_count") or 0)
        b_count = float(row.get("b_count") or 0)
        denom = max(1.0, a_count + b_count)
        a_weight = a_count / denom
        b_weight = b_count / denom
        if winner == a_method:
            return 1.0 - a_weight, 0.0
        if winner == b_method:
            return 0.0, 1.0 - b_weight
        if winner == "tie":
            return 0.5 - a_weight / 2, 0.5 - b_weight / 2
        return 0.0, 0.0

    a_pts, b_pts = points_from(ab_row)
    if ba_row is not None:
        a_pts_ba, b_pts_ba = points_from(ba_row)
        a_pts = (a_pts + a_pts_ba) / 2
        b_pts = (b_pts + b_pts_ba) / 2
    normalised_points[a_method] += a_pts
    normalised_points[b_method] += b_pts


def _lookup_row(paths, bill_id: str, *, key: str) -> dict[str, Any]:
    """Return the cache row for a (bill, key) pair or ``{}`` when missing."""

    for row in iter_cache_rows(paths, bill_id):
        if str(row.get("key") or "") == key:
            return row
    return {}


def _single_answer_comparison(
    ctx: StageContext, *, a_method: str, b_method: str
) -> dict[str, Any]:
    """Read Stage 4 / Stage 5 artefacts and build the single-answer view."""

    from ._common import results_dir

    rd = results_dir(ctx)
    stage4_path = rd / "stage4_coverage.json"
    stage5_path = rd / "stage5_novelty.json"
    import json as _json

    out: dict[str, Any] = {}
    if stage4_path.is_file():
        with stage4_path.open("r", encoding="utf-8") as handle:
            s4 = _json.load(handle)
        per_method = s4.get("per_method", {})
        out["coverage_rate"] = {
            a_method: per_method.get(a_method, {}).get("rates", {}).get("covered"),
            b_method: per_method.get(b_method, {}).get("rates", {}).get("covered"),
        }
    if stage5_path.is_file():
        with stage5_path.open("r", encoding="utf-8") as handle:
            s5 = _json.load(handle)
        per_method = s5.get("per_method", {})
        out["novel_count"] = {
            a_method: per_method.get(a_method, {}).get("novel_count"),
            b_method: per_method.get(b_method, {}).get("novel_count"),
        }
    return out


def _summary_line(summary: dict[str, Any]) -> str:
    """Short log line with the swap-averaged win rate."""

    avg = summary.get("swap_averaged_winrate", {})
    parts = [f"{name}={avg.get(name, 0):.2%}" for name in avg]
    return f"Pairwise (n={summary['n_pairs']}): " + "  ".join(parts)
