"""Stage 3: per-quadruplet grounding verdict (entailed / neutral / contradicted).

- Reads the Stage 2 pass set so only plausibly-shaped quadruplets reach the
  judge (Mahbub et al. 2026 filter-before-ground pattern).
- Per quadruplet, asks the judge whether the bill text (a tight excerpt
  around the evidence spans) entails, is neutral about, or contradicts the
  extracted claim. This is the RefChecker three-way adaptation documented in
  ``docs/lit_rev_eval.md``.
- Caches one row per quadruplet id under
  ``output/evals/v1/cache/<method>/stage3/<bill_id>.jsonl`` so reruns skip
  already-judged items.
- Writes ``results/stage3_grounding.json`` aggregating verdict counts per
  method, per entity ``type``, and overall.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from ..artifacts import EvidenceSpan, JudgeVerdict, Quadruplet, StageResult
from ..cache import CacheWriter, cache_paths_for_stage, filter_pending, iter_cache_rows
from ..judge import CreditsExhaustedError, call_judge
from ..prompts import GROUNDING_SCHEMA, GROUNDING_SYSTEM_PROMPT, GROUNDING_USER_PROMPT_TEMPLATE
from ._common import (
    StageContext,
    gather_with_progress,
    read_stage2_pass_set,
    write_result_json,
)

logger = logging.getLogger(__name__)

_SCHEMA_NAME = "stage3_grounding"
_EXCERPT_RADIUS = 500


@dataclass(slots=True)
class _Item:
    """One judge work item (used for both cached and pending quadruplets)."""

    method: str
    bill_id: str
    quadruplet: Quadruplet


def run(ctx: StageContext) -> StageResult:
    """Issue grounding verdicts for every Stage-2 pass quadruplet.

    Args:
        ctx: Pre-built :class:`StageContext` that has the judge connection
            attached (raises :class:`RuntimeError` if missing).

    Returns:
        :class:`StageResult` with per-method grounding proportions and
        per-type breakdowns.
    """

    if ctx.judge is None:
        raise RuntimeError("Stage 3 requires a judge connection")

    stage_params = ctx.config.stages.get(3)
    max_concurrency = int(
        stage_params.params.get("max_concurrency", 8) if stage_params else 8
    )

    pass_set = read_stage2_pass_set(ctx)
    items_by_method: dict[str, list[_Item]] = {}
    for method_name, bills in ctx.method_outputs.items():
        accepted = pass_set.get(method_name, {})
        items: list[_Item] = []
        for bill_id, quadruplets in bills.items():
            keep_ids = accepted.get(bill_id)
            if not keep_ids:
                continue
            id_set = set(keep_ids)
            for quadruplet in quadruplets:
                if quadruplet.quadruplet_id in id_set:
                    items.append(
                        _Item(
                            method=method_name,
                            bill_id=bill_id,
                            quadruplet=quadruplet,
                        )
                    )
        items_by_method[method_name] = items

    asyncio.run(
        _judge_all_methods(
            ctx,
            items_by_method=items_by_method,
            max_concurrency=max_concurrency,
        )
    )

    summary = _aggregate_from_cache(ctx, items_by_method)
    result_file = write_result_json(
        ctx, filename="stage3_grounding.json", payload={"stage": 3, **summary}
    )
    return StageResult(
        stage=3,
        status="completed",
        summary=_summary_line(summary),
        metrics=summary,
        artifacts={"result_file": str(result_file)},
    )


async def _judge_all_methods(
    ctx: StageContext,
    *,
    items_by_method: dict[str, list[_Item]],
    max_concurrency: int,
) -> None:
    """Fan out judge calls across both methods with a bounded semaphore.

    Tasks from both methods are collected into a single list before
    being gathered so that a single tqdm progress bar covers the whole
    stage (satisfying the "no 0/2 method" progress rule). Per-method
    ``total / cached / pending`` counts are still logged so resume
    behaviour stays visible.
    """

    semaphore = asyncio.Semaphore(max_concurrency)
    all_tasks: list[Any] = []
    for method_name, items in items_by_method.items():
        if not items:
            logger.info("Stage 3: no Stage-2 survivors for method=%s", method_name)
            continue
        paths = cache_paths_for_stage(
            run_dir=ctx.run_dir, stage=3, method=method_name
        )
        writer = CacheWriter(paths)
        pending = filter_pending(
            ((it.bill_id, it.quadruplet.quadruplet_id, it) for it in items),
            paths,
        )
        cached_count = len(items) - len(pending)
        logger.info(
            "Stage 3 method=%s: total=%d cached=%d pending=%d",
            method_name, len(items), cached_count, len(pending),
        )
        if not pending:
            continue
        all_tasks.extend(
            _judge_one(ctx, item=payload, writer=writer, semaphore=semaphore)
            for _, _, payload in pending
        )
    await gather_with_progress(all_tasks, desc="Stage 3 (grounding)")


async def _judge_one(
    ctx: StageContext,
    *,
    item: _Item,
    writer: CacheWriter,
    semaphore: asyncio.Semaphore,
) -> None:
    """Judge one quadruplet and append the verdict to the per-bill JSONL."""

    record = ctx.bill_records.get(item.bill_id)
    if record is None:
        writer.append(
            bill_id=item.bill_id,
            key=item.quadruplet.quadruplet_id,
            payload={
                "method": item.method,
                "verdict": "error",
                "rationale": "bill text missing",
            },
        )
        return

    prompt = _render_user_prompt(record=record, quadruplet=item.quadruplet)
    async with semaphore:
        try:
            verdict = await call_judge(
                ctx.judge,
                system_prompt=GROUNDING_SYSTEM_PROMPT,
                user_prompt=prompt,
                schema=GROUNDING_SCHEMA,
                schema_name=_SCHEMA_NAME,
            )
        except CreditsExhaustedError:
            raise
        except Exception:
            logger.exception(
                "Stage 3 judge call failed: method=%s bill=%s id=%s",
                item.method,
                item.bill_id,
                item.quadruplet.quadruplet_id,
            )
            verdict = JudgeVerdict(verdict="error", rationale="judge exception")
    writer.append(
        bill_id=item.bill_id,
        key=item.quadruplet.quadruplet_id,
        payload={
            "method": item.method,
            "verdict": verdict.verdict,
            "rationale": verdict.rationale,
            "usage": verdict.usage,
        },
    )


def _render_user_prompt(*, record: Any, quadruplet: Quadruplet) -> str:
    """Format the grounding user prompt with evidence-aware bill excerpt."""

    evidence_lines: list[str] = []
    for span in quadruplet.all_spans():
        evidence_lines.append(
            f"  - [{span.start}:{span.end}] {_preview_span_text(span)}"
        )
    evidence_block = "\n".join(evidence_lines) or "  (none)"
    excerpt = _excerpt_around(
        record.text,
        [(span.start, span.end) for span in quadruplet.all_spans()],
    )
    return GROUNDING_USER_PROMPT_TEMPLATE.format(
        bill_id=record.bill_id,
        state=record.state,
        year=record.year,
        evidence_block=evidence_block,
        entity=quadruplet.entity,
        entity_type=quadruplet.type,
        attribute=quadruplet.attribute,
        value=quadruplet.value,
        bill_excerpt=excerpt,
    )


def _preview_span_text(span: EvidenceSpan, *, limit: int = 160) -> str:
    """Render the span text for the prompt, trimmed to ``limit`` characters."""

    text = (span.text or "").strip()
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text or "(empty)"


def _excerpt_around(
    text: str, spans: list[tuple[int, int]], *, radius: int = _EXCERPT_RADIUS
) -> str:
    """Return a bill text excerpt that covers every span plus ``radius`` padding."""

    if not text:
        return ""
    if not spans:
        return text[: radius * 2]
    start = max(0, min(s[0] for s in spans) - radius)
    end = min(len(text), max(s[1] for s in spans) + radius)
    return text[start:end]


def _aggregate_from_cache(
    ctx: StageContext, items_by_method: dict[str, list[_Item]]
) -> dict[str, Any]:
    """Walk each method's cache file set and tally verdicts for the result JSON."""

    per_method: dict[str, Any] = {}
    by_type: dict[str, dict[str, dict[str, int]]] = {}
    overall: dict[str, int] = {}

    for method_name, items in items_by_method.items():
        paths = cache_paths_for_stage(
            run_dir=ctx.run_dir, stage=3, method=method_name
        )
        verdict_counts = {
            "entailed": 0, "neutral": 0, "contradicted": 0, "error": 0,
        }
        type_counts: dict[str, dict[str, int]] = {}
        id_to_type: dict[str, str] = {
            it.quadruplet.quadruplet_id: it.quadruplet.type or "UNKNOWN"
            for it in items
        }
        bill_ids = {it.bill_id for it in items}
        for bill_id in bill_ids:
            for row in iter_cache_rows(paths, bill_id):
                verdict = str(row.get("verdict") or "error")
                if verdict not in verdict_counts:
                    verdict = "error"
                verdict_counts[verdict] += 1
                overall[verdict] = overall.get(verdict, 0) + 1
                qid = str(row.get("key") or "")
                entity_type = id_to_type.get(qid, "UNKNOWN")
                tc = type_counts.setdefault(
                    entity_type,
                    {"entailed": 0, "neutral": 0, "contradicted": 0, "error": 0},
                )
                tc[verdict] += 1
        total = sum(verdict_counts.values()) or 1
        per_method[method_name] = {
            "counts": verdict_counts,
            "rates": {k: round(v / total, 4) for k, v in verdict_counts.items()},
            "total_judged": sum(verdict_counts.values()),
        }
        by_type[method_name] = type_counts

    overall_total = sum(overall.values()) or 1
    overall_rates = {k: round(v / overall_total, 4) for k, v in overall.items()}
    return {
        "per_method": per_method,
        "by_type": by_type,
        "overall": {"counts": overall, "rates": overall_rates, "total": overall_total},
    }


def _summary_line(summary: dict[str, Any]) -> str:
    """Produce a short one-line summary the orchestrator can log."""

    parts = []
    for method_name, data in summary["per_method"].items():
        rates = data["rates"]
        parts.append(
            f"{method_name}: entailed={rates.get('entailed', 0):.2%} "
            f"neutral={rates.get('neutral', 0):.2%} "
            f"contradicted={rates.get('contradicted', 0):.2%}"
        )
    return "  |  ".join(parts)
