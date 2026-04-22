"""Stage 5: novel-entity bookkeeping derived from Stages 3 and 4.

- A quadruplet is "novel" when it survived Stage 2 and Stage 3 (entailed
  grounding) but was never cited as a ``supporting_id`` in any Stage 4
  coverage verdict.  These are the bits of structure the extractor added
  on top of NCSL's coarse labels.
- Emits per-method novel counts, a per-entity-type distribution (top-10
  types appear in the plot Stage 9 builds), per-state counts, and a
  stratified audit sample of 50 novel quadruplets with their Stage 3
  verdict for manual inspection.
- Does not call the judge; reads Stage 3 and Stage 4 caches only.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from typing import Any

from ..artifacts import Quadruplet, StageResult
from ..cache import cache_paths_for_stage, iter_cache_rows
from ._common import StageContext, write_result_json

logger = logging.getLogger(__name__)

_AUDIT_SAMPLE_SIZE = 50


def run(ctx: StageContext) -> StageResult:
    """Classify each Stage 3 survivor as novel or covered and aggregate.

    Args:
        ctx: Pre-built :class:`StageContext`. The judge connection is not
            used here.

    Returns:
        :class:`StageResult` with novel counts, per-type and per-state
        breakdowns, and the path to the 50-row audit sample.
    """

    stage_params = ctx.config.stages.get(5)
    seed = int(ctx.config.sampling.seed)

    per_method: dict[str, Any] = {}
    audit_sample: list[dict[str, Any]] = []

    for method_name, bills in ctx.method_outputs.items():
        entailed = _entailed_ids(ctx, method_name=method_name)
        covered_support = _coverage_support_ids(ctx, method_name=method_name)
        novel_ids = entailed - covered_support

        id_to_quadruplet: dict[str, Quadruplet] = {
            q.quadruplet_id: q
            for quadruplets in bills.values()
            for q in quadruplets
        }
        type_counter: Counter[str] = Counter()
        state_counter: Counter[str] = Counter()
        novel_quadruplets: list[Quadruplet] = []
        for qid in novel_ids:
            quadruplet = id_to_quadruplet.get(qid)
            if quadruplet is None:
                continue
            novel_quadruplets.append(quadruplet)
            type_counter[quadruplet.type or "UNKNOWN"] += 1
            record = ctx.bill_records.get(quadruplet.bill_id)
            state_counter[record.state if record else "UNKNOWN"] += 1

        per_method[method_name] = {
            "stage3_entailed_count": len(entailed),
            "stage4_supporting_count": len(covered_support),
            "novel_count": len(novel_quadruplets),
            "by_type_top10": type_counter.most_common(10),
            "by_state_top10": state_counter.most_common(10),
        }
        audit_sample.extend(
            _stratified_audit_sample(
                method_name, novel_quadruplets, seed=seed
            )
        )

    result_payload = {
        "stage": 5,
        "per_method": per_method,
        "audit_sample": audit_sample,
    }
    result_file = write_result_json(
        ctx, filename="stage5_novelty.json", payload=result_payload
    )
    summary_line = "  |  ".join(
        f"{m}: novel={data['novel_count']}"
        for m, data in per_method.items()
    )
    return StageResult(
        stage=5,
        status="completed",
        summary=f"Novel quadruplets -- {summary_line}",
        metrics={"per_method": per_method},
        artifacts={"result_file": str(result_file)},
    )


def _entailed_ids(ctx: StageContext, *, method_name: str) -> set[str]:
    """Collect the set of quadruplet ids with Stage 3 verdict ``entailed``."""

    ids: set[str] = set()
    paths = cache_paths_for_stage(
        run_dir=ctx.run_dir, stage=3, method=method_name
    )
    for bill_id in ctx.method_outputs[method_name].keys():
        for row in iter_cache_rows(paths, bill_id):
            if str(row.get("verdict") or "") == "entailed":
                qid = row.get("key")
                if qid:
                    ids.add(str(qid))
    return ids


def _coverage_support_ids(ctx: StageContext, *, method_name: str) -> set[str]:
    """Collect every quadruplet id the judge cited as supporting coverage."""

    ids: set[str] = set()
    paths = cache_paths_for_stage(
        run_dir=ctx.run_dir, stage=4, method=method_name
    )
    for bill_id in ctx.method_outputs[method_name].keys():
        for row in iter_cache_rows(paths, bill_id):
            verdict = str(row.get("verdict") or "")
            if verdict not in {"covered", "partially_covered"}:
                continue
            for qid in row.get("supporting_ids") or []:
                if qid:
                    ids.add(str(qid))
    return ids


def _stratified_audit_sample(
    method_name: str,
    novel_quadruplets: list[Quadruplet],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Return a type-stratified sample of novel quadruplets for manual audit."""

    if not novel_quadruplets:
        return []
    rng = random.Random(seed + hash(method_name) % 100_000)
    by_type: dict[str, list[Quadruplet]] = {}
    for quadruplet in novel_quadruplets:
        by_type.setdefault(quadruplet.type or "UNKNOWN", []).append(quadruplet)
    per_bucket = max(1, _AUDIT_SAMPLE_SIZE // max(1, len(by_type)))
    picked: list[Quadruplet] = []
    for bucket in by_type.values():
        rng.shuffle(bucket)
        picked.extend(bucket[:per_bucket])
    rng.shuffle(picked)
    picked = picked[:_AUDIT_SAMPLE_SIZE]
    return [
        {
            "method": method_name,
            "quadruplet_id": q.quadruplet_id,
            "bill_id": q.bill_id,
            "entity": q.entity,
            "type": q.type,
            "attribute": q.attribute,
            "value": q.value,
        }
        for q in picked
    ]
