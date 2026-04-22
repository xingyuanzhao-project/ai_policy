"""Stage 1: prompt calibration against a 30-bill development sample.

- Draws a seeded development sample (default 30 bills) from the bill
  intersection of every configured method.
- Emits the frozen judge prompts for Stage 3 (grounding) and Stage 4
  (coverage) to ``output/evals/v1/prompts/``, with two rendered in-corpus
  examples appended so they are inspectable.
- Does not invoke the judge at scale: this is the "schema-locked prompts"
  step from Mahbub et al. 2026 as adapted in ``docs/lit_rev_eval.md``.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from ..artifacts import StageResult
from ..prompts import (
    COVERAGE_SCHEMA,
    COVERAGE_SYSTEM_PROMPT,
    COVERAGE_USER_PROMPT_TEMPLATE,
    GROUNDING_SCHEMA,
    GROUNDING_SYSTEM_PROMPT,
    GROUNDING_USER_PROMPT_TEMPLATE,
)
from ._common import StageContext, prompts_dir, write_result_json

logger = logging.getLogger(__name__)


def run(ctx: StageContext) -> StageResult:
    """Emit the calibrated prompt artefacts for Stages 3, 4, 6 and 8.

    Args:
        ctx: Pre-built :class:`StageContext` with the sampled bill set and
            method outputs already loaded.

    Returns:
        :class:`StageResult` describing the artefacts written and the size
        of the dev sample used to build the illustrative examples.
    """

    params = ctx.config.stages.get(1)
    dev_n = int(params.params.get("dev_sample_bills", 30)) if params else 30
    seed = int(params.params.get("seed", 20260417)) if params else 20260417

    dev_bill_ids = _sample_dev_bills(ctx.sampled_bill_ids, n=dev_n, seed=seed)
    logger.info(
        "Stage 1: emitting locked prompts using dev sample of %d bills", len(dev_bill_ids)
    )

    out_dir = prompts_dir(ctx)
    grounding_path = _write_grounding_prompt(ctx, dev_bill_ids, out_dir)
    coverage_path = _write_coverage_prompt(ctx, dev_bill_ids, out_dir)
    schemas_path = _write_schemas(out_dir)

    result_payload: dict[str, Any] = {
        "stage": 1,
        "dev_sample_bill_ids": dev_bill_ids,
        "artifacts": {
            "grounding_prompt": str(grounding_path),
            "coverage_prompt": str(coverage_path),
            "schemas": str(schemas_path),
        },
    }
    result_file = write_result_json(
        ctx, filename="stage1_calibration.json", payload=result_payload
    )
    return StageResult(
        stage=1,
        status="completed",
        summary=(
            f"Emitted grounding and coverage prompt artefacts using "
            f"{len(dev_bill_ids)}-bill dev sample."
        ),
        metrics={"dev_sample_size": len(dev_bill_ids)},
        artifacts={
            "grounding_prompt": str(grounding_path),
            "coverage_prompt": str(coverage_path),
            "schemas": str(schemas_path),
            "result_file": str(result_file),
        },
    )


def _sample_dev_bills(all_bill_ids: list[str], *, n: int, seed: int) -> list[str]:
    """Return a deterministic sample of dev-set bill ids capped at ``n``."""

    if not all_bill_ids:
        return []
    if n >= len(all_bill_ids):
        return sorted(all_bill_ids)
    rng = random.Random(seed)
    return sorted(rng.sample(all_bill_ids, n))


def _write_grounding_prompt(
    ctx: StageContext, dev_bill_ids: list[str], out_dir: Path
) -> Path:
    """Write ``stage3_grounding.txt`` with a rendered in-corpus example."""

    example_block = _render_grounding_example(ctx, dev_bill_ids)
    content = (
        "# Stage 3 grounding prompt (frozen for eval v1)\n"
        "# Source: src/eval/prompts.py (GROUNDING_SYSTEM_PROMPT + USER template)\n\n"
        "=== SYSTEM ===\n"
        f"{GROUNDING_SYSTEM_PROMPT}\n"
        "=== USER TEMPLATE ===\n"
        f"{GROUNDING_USER_PROMPT_TEMPLATE}\n"
        "=== JSON SCHEMA ===\n"
        f"{json.dumps(GROUNDING_SCHEMA, indent=2)}\n\n"
        "=== ILLUSTRATIVE EXAMPLE (rendered from dev sample) ===\n"
        f"{example_block}\n"
    )
    path = out_dir / "stage3_grounding.txt"
    path.write_text(content, encoding="utf-8")
    return path


def _write_coverage_prompt(
    ctx: StageContext, dev_bill_ids: list[str], out_dir: Path
) -> Path:
    """Write ``stage4_coverage.txt`` with a rendered in-corpus example."""

    example_block = _render_coverage_example(ctx, dev_bill_ids)
    content = (
        "# Stage 4 coverage prompt (frozen for eval v1)\n"
        "# Source: src/eval/prompts.py (COVERAGE_SYSTEM_PROMPT + USER template)\n\n"
        "=== SYSTEM ===\n"
        f"{COVERAGE_SYSTEM_PROMPT}\n"
        "=== USER TEMPLATE ===\n"
        f"{COVERAGE_USER_PROMPT_TEMPLATE}\n"
        "=== JSON SCHEMA ===\n"
        f"{json.dumps(COVERAGE_SCHEMA, indent=2)}\n\n"
        "=== ILLUSTRATIVE EXAMPLE (rendered from dev sample) ===\n"
        f"{example_block}\n"
    )
    path = out_dir / "stage4_coverage.txt"
    path.write_text(content, encoding="utf-8")
    return path


def _write_schemas(out_dir: Path) -> Path:
    """Dump both JSON schemas in one file for offline inspection."""

    payload = {
        "grounding": GROUNDING_SCHEMA,
        "coverage": COVERAGE_SCHEMA,
    }
    path = out_dir / "schemas.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def _render_grounding_example(
    ctx: StageContext, dev_bill_ids: list[str]
) -> str:
    """Render one grounding-prompt example from the dev sample if possible."""

    pick = _find_example_quadruplet(ctx, dev_bill_ids)
    if pick is None:
        return "(no dev bill had a quadruplet with a non-empty evidence span)"
    method, bill_id, quadruplet = pick
    record = ctx.bill_records.get(bill_id)
    if record is None:
        return f"(bill {bill_id!r} missing NCSL text; example skipped)"
    evidence_lines = []
    for span in quadruplet.all_spans():
        evidence_lines.append(
            f"  - [{span.start}:{span.end}] {span.text!r}"
        )
    evidence_block = "\n".join(evidence_lines) or "  (none)"
    bill_excerpt = _excerpt_around(
        record.text, [(s.start, s.end) for s in quadruplet.all_spans()]
    )
    rendered = GROUNDING_USER_PROMPT_TEMPLATE.format(
        bill_id=bill_id,
        state=record.state,
        year=record.year,
        evidence_block=evidence_block,
        entity=quadruplet.entity,
        entity_type=quadruplet.type,
        attribute=quadruplet.attribute,
        value=quadruplet.value,
        bill_excerpt=bill_excerpt,
    )
    return f"[example method={method}]\n{rendered}"


def _render_coverage_example(
    ctx: StageContext, dev_bill_ids: list[str]
) -> str:
    """Render one coverage-prompt example from the dev sample if possible."""

    pick = _find_example_for_coverage(ctx, dev_bill_ids)
    if pick is None:
        return "(no dev bill had both NCSL labels and quadruplets)"
    method, bill_id, label = pick
    record = ctx.bill_records[bill_id]
    quadruplets = ctx.method_outputs[method].get(bill_id, [])
    block = "\n".join(
        f"  - {q.quadruplet_id} : ({q.entity} | {q.type} | {q.attribute} | {q.value})"
        for q in quadruplets[:10]
    ) or "  (none)"
    rendered = COVERAGE_USER_PROMPT_TEMPLATE.format(
        bill_id=bill_id,
        state=record.state,
        year=record.year,
        title=record.title,
        summary=record.summary[:400],
        label=label,
        method=method,
        quadruplet_block=block,
    )
    return f"[example method={method}  label={label!r}]\n{rendered}"


def _find_example_quadruplet(
    ctx: StageContext, dev_bill_ids: list[str]
):
    """Pick one method/bill/quadruplet triple with a non-empty evidence span."""

    for method_name, bills in ctx.method_outputs.items():
        for bill_id in dev_bill_ids:
            for quadruplet in bills.get(bill_id, []):
                spans = quadruplet.all_spans()
                if spans and any(span.text for span in spans):
                    return method_name, bill_id, quadruplet
    return None


def _find_example_for_coverage(ctx: StageContext, dev_bill_ids: list[str]):
    """Pick one (method, bill, label) with quadruplets and at least one NCSL label."""

    for method_name, bills in ctx.method_outputs.items():
        for bill_id in dev_bill_ids:
            if not bills.get(bill_id):
                continue
            record = ctx.bill_records.get(bill_id)
            if record is None or not record.topics:
                continue
            return method_name, bill_id, record.topics[0]
    return None


def _excerpt_around(
    text: str, spans: list[tuple[int, int]], *, radius: int = 400
) -> str:
    """Return a compact excerpt covering every span with ``radius`` chars padding."""

    if not text:
        return ""
    if not spans:
        return text[:radius]
    start = max(0, min(s[0] for s in spans) - radius)
    end = min(len(text), max(s[1] for s in spans) + radius)
    return text[start:end]
