"""Stage 7: expert agreement (optional; skipped if the expert file is absent).

- Reads ``data/eval/expert_coded.json`` (or whichever path the stage
  parameter points at).  If the file is missing, emits a ``skipped`` stage
  result with a clear reason and does nothing else; the rest of the
  pipeline continues uninterrupted.
- When the file exists, it must have the schema documented in
  ``docs/lit_rev_eval.md`` line 283: a list of ``{bill_id, method,
  quadruplet_id, expert_verdict}`` rows whose ``expert_verdict`` mirrors
  the Stage 3 three-way label (``entailed`` / ``neutral`` /
  ``contradicted``).
- Produces Gwet's AC1, Cohen's kappa, and the Han et al. 2025
  human-likeness z-score against whichever human-human kappa distribution
  is supplied in the expert file's ``human_kappa_distribution`` block.
- No judge calls; it consumes the Stage 3 cache and the expert file.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import StageResult
from ..cache import cache_paths_for_stage, iter_cache_rows
from ._common import StageContext, skipped_result, write_result_json

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _PairedRow:
    """One paired judge / expert verdict used by the agreement metrics."""

    method: str
    bill_id: str
    quadruplet_id: str
    judge_verdict: str
    expert_verdict: str


def run(ctx: StageContext) -> StageResult:
    """Compute judge / expert agreement metrics if an expert file is supplied.

    Args:
        ctx: Pre-built :class:`StageContext`; Stage 3 cache must already
            contain verdicts for the quadruplets the expert labelled.

    Returns:
        Either a ``completed`` :class:`StageResult` with agreement metrics
        or a ``skipped`` result when the expert file is missing.
    """

    params = ctx.config.stages.get(7)
    expert_file = _resolve_expert_file(ctx, params)
    if expert_file is None or not expert_file.is_file():
        logger.warning(
            "Stage 7 skipped: expert file not found at %s", expert_file
        )
        result_payload = {
            "stage": 7,
            "status": "skipped",
            "reason": "expert file not found",
            "expert_file": str(expert_file) if expert_file else None,
        }
        result_file = write_result_json(
            ctx, filename="stage7_expert.json", payload=result_payload
        )
        return skipped_result(
            7,
            summary="Stage 7 skipped (expert coded file not supplied).",
            reason="expert file not found",
            filename=str(result_file),
        )

    with expert_file.open("r", encoding="utf-8") as handle:
        expert_payload: dict[str, Any] = json.load(handle)
    expert_rows = expert_payload.get("rows") or []
    human_kappa_dist = expert_payload.get("human_kappa_distribution") or []
    if not expert_rows:
        raise ValueError(f"Stage 7 expert file has no 'rows': {expert_file}")

    paired = _pair_with_stage3(ctx, expert_rows=expert_rows)
    per_method = {
        method: _agreement_metrics(rows)
        for method, rows in _group_by_method(paired).items()
    }
    overall = _agreement_metrics(paired)
    overall["human_likeness_z"] = _human_likeness_z(
        overall["cohen_kappa"], human_kappa_dist
    )

    payload = {
        "stage": 7,
        "paired_count": len(paired),
        "per_method": per_method,
        "overall": overall,
        "expert_file": str(expert_file),
    }
    result_file = write_result_json(
        ctx, filename="stage7_expert.json", payload=payload
    )
    return StageResult(
        stage=7,
        status="completed",
        summary=(
            f"Expert agreement: kappa={overall['cohen_kappa']:.3f} "
            f"AC1={overall['gwet_ac1']:.3f} human_likeness_z="
            f"{overall['human_likeness_z']:.3f}"
            if overall["human_likeness_z"] is not None
            else f"Expert agreement: kappa={overall['cohen_kappa']:.3f} AC1={overall['gwet_ac1']:.3f}"
        ),
        metrics={"per_method": per_method, "overall": overall},
        artifacts={"result_file": str(result_file)},
    )


def _resolve_expert_file(ctx: StageContext, params: Any) -> Path | None:
    """Resolve the expert-file path, preferring the stage override."""

    configured = params.params.get("expert_file") if params else None
    if not configured:
        return None
    path = Path(configured)
    if not path.is_absolute():
        path = (ctx.config.project_root / path).resolve()
    return path


def _pair_with_stage3(
    ctx: StageContext, *, expert_rows: list[dict[str, Any]]
) -> list[_PairedRow]:
    """Pair each expert-labelled quadruplet with the Stage 3 cached verdict."""

    judge_index: dict[tuple[str, str], str] = {}
    for method_name in ctx.method_outputs.keys():
        paths = cache_paths_for_stage(
            run_dir=ctx.run_dir, stage=3, method=method_name
        )
        for bill_id in ctx.method_outputs[method_name].keys():
            for row in iter_cache_rows(paths, bill_id):
                qid = str(row.get("key") or "")
                if not qid:
                    continue
                verdict = str(row.get("verdict") or "error")
                judge_index[(method_name, qid)] = verdict

    paired: list[_PairedRow] = []
    for row in expert_rows:
        method = str(row.get("method") or "")
        qid = str(row.get("quadruplet_id") or "")
        bill_id = str(row.get("bill_id") or "")
        expert = str(row.get("expert_verdict") or "").strip().lower()
        judge_verdict = judge_index.get((method, qid))
        if judge_verdict is None:
            continue
        if expert not in {"entailed", "neutral", "contradicted"}:
            continue
        paired.append(
            _PairedRow(
                method=method,
                bill_id=bill_id,
                quadruplet_id=qid,
                judge_verdict=judge_verdict,
                expert_verdict=expert,
            )
        )
    return paired


def _group_by_method(paired: list[_PairedRow]) -> dict[str, list[_PairedRow]]:
    """Bucket the paired rows by method name for per-method metrics."""

    out: dict[str, list[_PairedRow]] = {}
    for row in paired:
        out.setdefault(row.method, []).append(row)
    return out


def _agreement_metrics(rows: list[_PairedRow]) -> dict[str, Any]:
    """Return observed agreement, Cohen's kappa, and Gwet's AC1 for ``rows``."""

    if not rows:
        return {
            "n": 0,
            "observed_agreement": None,
            "cohen_kappa": None,
            "gwet_ac1": None,
            "confusion": {},
        }
    labels = ("entailed", "neutral", "contradicted")
    n = len(rows)
    matrix: dict[tuple[str, str], int] = {
        (a, b): 0 for a in labels for b in labels
    }
    for row in rows:
        if (
            row.judge_verdict not in labels
            or row.expert_verdict not in labels
        ):
            continue
        matrix[(row.judge_verdict, row.expert_verdict)] += 1
    matched = sum(matrix[(a, a)] for a in labels)
    observed = matched / n
    p_marg_judge = {a: sum(matrix[(a, b)] for b in labels) / n for a in labels}
    p_marg_expert = {b: sum(matrix[(a, b)] for a in labels) / n for b in labels}
    chance_cohen = sum(
        p_marg_judge[a] * p_marg_expert[a] for a in labels
    )
    cohen_kappa = (
        (observed - chance_cohen) / (1 - chance_cohen)
        if chance_cohen < 1
        else 1.0
    )
    q = len(labels)
    chance_gwet = (1 - sum(
        ((p_marg_judge[a] + p_marg_expert[a]) / 2) ** 2 for a in labels
    )) / (q - 1)
    gwet_ac1 = (
        (observed - chance_gwet) / (1 - chance_gwet)
        if chance_gwet < 1
        else 1.0
    )
    confusion = {f"{a}->{b}": matrix[(a, b)] for a in labels for b in labels}
    return {
        "n": n,
        "observed_agreement": round(observed, 4),
        "cohen_kappa": round(cohen_kappa, 4),
        "gwet_ac1": round(gwet_ac1, 4),
        "confusion": confusion,
    }


def _human_likeness_z(
    cohen_kappa: float | None, dist: list[float]
) -> float | None:
    """Return the Han et al. 2025 ``(kappa - mu) / sigma`` against ``dist``."""

    if cohen_kappa is None or not dist:
        return None
    samples = [float(v) for v in dist if v is not None]
    if len(samples) < 2:
        return None
    mu = sum(samples) / len(samples)
    variance = sum((x - mu) ** 2 for x in samples) / (len(samples) - 1)
    sigma = math.sqrt(variance)
    if sigma == 0:
        return None
    return round((cohen_kappa - mu) / sigma, 4)
