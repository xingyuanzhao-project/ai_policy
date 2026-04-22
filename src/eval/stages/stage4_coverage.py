"""Stage 4: set-to-label coverage (inverted Atomic-SNLI).

- For each (bill, human-assigned label) pair, asks the judge whether any
  subset of the bill's quadruplets collectively covers the label.  This is
  the asymmetric inversion of Atomic-SNLI called out in
  ``docs/lit_rev_eval.md`` ("Why no single paper suffices").
- Only considers quadruplets that passed Stage 3 with verdict ``entailed``
  (or ``neutral`` when a permissive mode is requested); Stage 2 failures
  are excluded upstream.
- Caches one row per (bill_id, label) under
  ``output/evals/v1/cache/<method>/stage4/<bill_id>.jsonl`` so reruns skip
  already-judged labels.
- Writes ``results/stage4_coverage.json`` with per-method coverage rates,
  per-label breakdowns, and per-state breakdowns.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from ..artifacts import BillRecord, JudgeVerdict, Quadruplet, StageResult
from ..cache import CacheWriter, cache_paths_for_stage, filter_pending, iter_cache_rows
from ..judge import CreditsExhaustedError, call_judge
from ..prompts import (
    COVERAGE_SCHEMA,
    COVERAGE_SYSTEM_PROMPT,
    COVERAGE_USER_PROMPT_TEMPLATE,
)
from ._common import (
    StageContext,
    gather_with_progress,
    read_stage2_pass_set,
    write_result_json,
)

logger = logging.getLogger(__name__)

_SCHEMA_NAME = "stage4_coverage"


@dataclass(slots=True)
class _Item:
    """One (method, bill, label) work item routed through the judge."""

    method: str
    bill_id: str
    label: str
    quadruplets: list[Quadruplet]


def run(ctx: StageContext) -> StageResult:
    """Issue coverage verdicts for every (bill, label) pair and aggregate.

    Args:
        ctx: Pre-built :class:`StageContext` with the judge connection
            attached.

    Returns:
        :class:`StageResult` with coverage rates per method, per label,
        and per state.
    """

    if ctx.judge is None:
        raise RuntimeError("Stage 4 requires a judge connection")

    stage_params = ctx.config.stages.get(4)
    max_concurrency = int(
        stage_params.params.get("max_concurrency", 4) if stage_params else 4
    )

    pass_set = read_stage2_pass_set(ctx)
    stage3_survivors = _load_stage3_entailed_ids(ctx)

    items_by_method = _build_work_items(
        ctx, pass_set=pass_set, stage3_survivors=stage3_survivors
    )
    asyncio.run(
        _judge_all_methods(
            ctx,
            items_by_method=items_by_method,
            max_concurrency=max_concurrency,
        )
    )
    summary = _aggregate(ctx, items_by_method=items_by_method, stage3_survivors=stage3_survivors)
    result_file = write_result_json(
        ctx, filename="stage4_coverage.json", payload={"stage": 4, **summary}
    )
    return StageResult(
        stage=4,
        status="completed",
        summary=_summary_line(summary),
        metrics=summary,
        artifacts={"result_file": str(result_file)},
    )


def _build_work_items(
    ctx: StageContext,
    *,
    pass_set: dict[str, dict[str, set[str]]],
    stage3_survivors: dict[str, set[str]],
) -> dict[str, list[_Item]]:
    """Assemble the (method, bill, label) items that need a judge verdict."""

    items_by_method: dict[str, list[_Item]] = {}
    for method_name, bills in ctx.method_outputs.items():
        accepted_pass = pass_set.get(method_name, {})
        survivors = stage3_survivors.get(method_name, set())
        items: list[_Item] = []
        for bill_id, quadruplets in bills.items():
            record = ctx.bill_records.get(bill_id)
            if record is None or not record.topics:
                continue
            allowed_ids = set(accepted_pass.get(bill_id, set()))
            if survivors:
                allowed_ids &= survivors
            kept = [
                q for q in quadruplets if q.quadruplet_id in allowed_ids
            ]
            if not kept:
                continue
            for label in record.topics:
                items.append(
                    _Item(
                        method=method_name,
                        bill_id=bill_id,
                        label=label,
                        quadruplets=kept,
                    )
                )
        items_by_method[method_name] = items
        logger.info(
            "Stage 4 method=%s: bill-label pairs=%d  unique bills=%d",
            method_name,
            len(items),
            len({it.bill_id for it in items}),
        )
    return items_by_method


async def _judge_all_methods(
    ctx: StageContext,
    *,
    items_by_method: dict[str, list[_Item]],
    max_concurrency: int,
) -> None:
    """Judge every (bill, label) pair with a bounded fan-out.

    Tasks from both methods are combined into a single list before
    dispatch so one progress bar tracks the whole stage. Per-method
    ``total / cached / pending`` counts are still logged for
    resume-friendly debugging.
    """

    semaphore = asyncio.Semaphore(max_concurrency)
    all_tasks: list[Any] = []
    for method_name, items in items_by_method.items():
        if not items:
            continue
        paths = cache_paths_for_stage(
            run_dir=ctx.run_dir, stage=4, method=method_name
        )
        writer = CacheWriter(paths)
        pending = filter_pending(
            ((it.bill_id, _label_key(it.bill_id, it.label), it) for it in items),
            paths,
        )
        cached_count = len(items) - len(pending)
        logger.info(
            "Stage 4 method=%s: total=%d cached=%d pending=%d",
            method_name, len(items), cached_count, len(pending),
        )
        if not pending:
            continue
        all_tasks.extend(
            _judge_one(ctx, item=payload, writer=writer, semaphore=semaphore)
            for _, _, payload in pending
        )
    await gather_with_progress(all_tasks, desc="Stage 4 (coverage)")


async def _judge_one(
    ctx: StageContext,
    *,
    item: _Item,
    writer: CacheWriter,
    semaphore: asyncio.Semaphore,
) -> None:
    """Judge one (bill, label) pair and append the verdict to the cache."""

    record = ctx.bill_records.get(item.bill_id)
    key = _label_key(item.bill_id, item.label)
    if record is None:
        writer.append(
            bill_id=item.bill_id,
            key=key,
            payload={
                "method": item.method,
                "label": item.label,
                "verdict": "error",
                "rationale": "bill record missing",
                "supporting_ids": [],
            },
        )
        return

    prompt = _render_user_prompt(
        record=record,
        method=item.method,
        label=item.label,
        quadruplets=item.quadruplets,
    )
    async with semaphore:
        try:
            verdict = await call_judge(
                ctx.judge,
                system_prompt=COVERAGE_SYSTEM_PROMPT,
                user_prompt=prompt,
                schema=COVERAGE_SCHEMA,
                schema_name=_SCHEMA_NAME,
            )
        except CreditsExhaustedError:
            raise
        except Exception:
            logger.exception(
                "Stage 4 judge call failed: method=%s bill=%s label=%r",
                item.method, item.bill_id, item.label,
            )
            verdict = JudgeVerdict(verdict="error", rationale="judge exception")
    writer.append(
        bill_id=item.bill_id,
        key=key,
        payload={
            "method": item.method,
            "label": item.label,
            "verdict": verdict.verdict,
            "rationale": verdict.rationale,
            "supporting_ids": verdict.supporting_ids,
            "usage": verdict.usage,
        },
    )


def _render_user_prompt(
    *,
    record: BillRecord,
    method: str,
    label: str,
    quadruplets: list[Quadruplet],
) -> str:
    """Format the coverage user prompt for one (method, bill, label) triple."""

    lines: list[str] = []
    for q in quadruplets[:80]:
        lines.append(
            f"  - {q.quadruplet_id}: ({q.entity} | {q.type} | {q.attribute} | {q.value})"
        )
    block = "\n".join(lines) or "  (none)"
    return COVERAGE_USER_PROMPT_TEMPLATE.format(
        bill_id=record.bill_id,
        state=record.state,
        year=record.year,
        title=record.title,
        summary=(record.summary or "(none provided)")[:400],
        label=label,
        method=method,
        quadruplet_block=block,
    )


def _label_key(bill_id: str, label: str) -> str:
    """Compose the per-(bill, label) cache key."""

    return f"{bill_id}::{label}"


def _load_stage3_entailed_ids(ctx: StageContext) -> dict[str, set[str]]:
    """Return ``{method: {entailed_quadruplet_ids}}`` from Stage 3's cache.

    Missing cache is non-fatal: Stage 4 falls back to Stage 2's pass set in
    that case and the aggregator records ``"stage3_missing": true``.
    """

    survivors: dict[str, set[str]] = {}
    for method_name in ctx.method_outputs.keys():
        paths = cache_paths_for_stage(
            run_dir=ctx.run_dir, stage=3, method=method_name
        )
        method_ids: set[str] = set()
        for bill_id in ctx.method_outputs[method_name].keys():
            for row in iter_cache_rows(paths, bill_id):
                if str(row.get("verdict") or "") == "entailed":
                    qid = str(row.get("key") or "")
                    if qid:
                        method_ids.add(qid)
        if method_ids:
            survivors[method_name] = method_ids
        else:
            logger.info(
                "Stage 3 cache empty for method=%s; Stage 4 will use Stage 2 pass set only.",
                method_name,
            )
    return survivors


def _aggregate(
    ctx: StageContext,
    *,
    items_by_method: dict[str, list[_Item]],
    stage3_survivors: dict[str, set[str]],
) -> dict[str, Any]:
    """Walk each method's Stage 4 cache and aggregate coverage metrics."""

    per_method: dict[str, Any] = {}
    by_label: dict[str, dict[str, dict[str, int]]] = {}
    by_state: dict[str, dict[str, dict[str, int]]] = {}

    for method_name, items in items_by_method.items():
        paths = cache_paths_for_stage(
            run_dir=ctx.run_dir, stage=4, method=method_name
        )
        verdict_counts = {
            "covered": 0, "partially_covered": 0, "not_covered": 0, "error": 0,
        }
        label_counts: dict[str, dict[str, int]] = {}
        state_counts: dict[str, dict[str, int]] = {}
        survivors = stage3_survivors.get(method_name, set())
        strict_covered = 0
        for bill_id in {it.bill_id for it in items}:
            record = ctx.bill_records.get(bill_id)
            for row in iter_cache_rows(paths, bill_id):
                verdict = str(row.get("verdict") or "error")
                if verdict not in verdict_counts:
                    verdict = "error"
                verdict_counts[verdict] += 1

                label = str(row.get("label") or "UNKNOWN")
                label_map = label_counts.setdefault(
                    label,
                    {"covered": 0, "partially_covered": 0, "not_covered": 0, "error": 0},
                )
                label_map[verdict] += 1

                state = record.state if record else "UNKNOWN"
                state_map = state_counts.setdefault(
                    state,
                    {"covered": 0, "partially_covered": 0, "not_covered": 0, "error": 0},
                )
                state_map[verdict] += 1

                supporting = row.get("supporting_ids") or []
                if verdict == "covered":
                    if not supporting:
                        continue
                    if survivors and not all(str(x) in survivors for x in supporting):
                        continue
                    strict_covered += 1
        total = sum(verdict_counts.values()) or 1
        per_method[method_name] = {
            "counts": verdict_counts,
            "rates": {k: round(v / total, 4) for k, v in verdict_counts.items()},
            "total_judged": sum(verdict_counts.values()),
            "strict_covered_count": strict_covered,
            "strict_coverage_rate": round(strict_covered / total, 4),
        }
        by_label[method_name] = label_counts
        by_state[method_name] = state_counts
    return {
        "per_method": per_method,
        "by_label": by_label,
        "by_state": by_state,
    }


def _summary_line(summary: dict[str, Any]) -> str:
    """Return a short one-line summary the orchestrator logs."""

    parts = []
    for method_name, data in summary["per_method"].items():
        rates = data["rates"]
        parts.append(
            f"{method_name}: covered={rates.get('covered', 0):.2%} "
            f"partial={rates.get('partially_covered', 0):.2%} "
            f"strict={data['strict_coverage_rate']:.2%}"
        )
    return "  |  ".join(parts)
