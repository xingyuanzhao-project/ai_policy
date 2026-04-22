"""Stage 9: extrinsic-validity hooks for the downstream bill analysis.

- Emits ``results/stage9_extrinsic.csv`` joining each bill-method pair to
  its quadruplet count, coverage rate, novel count, and per-type counts.
  This is the handoff artefact that the outline's downstream analyses
  (``docs/mpsa_draft_outline.md`` lines 264-272) can join against without
  ever re-calling the judge.
- Additionally writes ``results/stage9_extrinsic.md`` explaining what each
  column means and which downstream tests the dataset supports.
- Does not run the downstream regressions itself; that is out of scope per
  the eval plan.
"""

from __future__ import annotations

import csv
import logging
from collections import Counter
from typing import Any

from ..artifacts import StageResult
from ..cache import cache_paths_for_stage, iter_cache_rows
from ._common import StageContext, results_dir

logger = logging.getLogger(__name__)

_CSV_COLUMNS = [
    "bill_id",
    "source_bill_id",
    "year",
    "state",
    "method",
    "quadruplet_count",
    "stage2_pass_count",
    "stage3_entailed_count",
    "stage4_labels_total",
    "stage4_labels_covered",
    "stage4_coverage_rate",
    "stage5_novel_count",
    "top_types",
]


def run(ctx: StageContext) -> StageResult:
    """Build the extrinsic-validity CSV and README.

    Args:
        ctx: Pre-built :class:`StageContext`.

    Returns:
        :class:`StageResult` pointing to the CSV and markdown note.
    """

    out_dir = results_dir(ctx)
    csv_path = out_dir / "stage9_extrinsic.csv"
    md_path = out_dir / "stage9_extrinsic.md"

    rows_written = 0
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for method_name, bills in ctx.method_outputs.items():
            stage3_paths = cache_paths_for_stage(
                run_dir=ctx.run_dir, stage=3, method=method_name
            )
            stage4_paths = cache_paths_for_stage(
                run_dir=ctx.run_dir, stage=4, method=method_name
            )
            entailed_map = _entailed_counts(ctx, method_name=method_name, paths=stage3_paths)
            coverage_map = _coverage_counts(ctx, method_name=method_name, paths=stage4_paths)
            novel_map = _novel_counts(ctx, method_name=method_name)
            for bill_id, quadruplets in bills.items():
                record = ctx.bill_records.get(bill_id)
                entailed_ids = entailed_map.get(bill_id, set())
                label_total, label_covered = coverage_map.get(
                    bill_id, (0, 0)
                )
                novel = novel_map.get(bill_id, 0)
                type_counter: Counter[str] = Counter(
                    (q.type or "UNKNOWN") for q in quadruplets
                )
                top_types = ";".join(
                    f"{t}:{c}" for t, c in type_counter.most_common(5)
                )
                row = {
                    "bill_id": bill_id,
                    "source_bill_id": record.source_bill_id if record else "",
                    "year": record.year if record else "",
                    "state": record.state if record else "",
                    "method": method_name,
                    "quadruplet_count": len(quadruplets),
                    "stage2_pass_count": _stage2_pass_count(ctx, method_name, bill_id),
                    "stage3_entailed_count": len(entailed_ids),
                    "stage4_labels_total": label_total,
                    "stage4_labels_covered": label_covered,
                    "stage4_coverage_rate": (
                        round(label_covered / label_total, 4) if label_total else 0.0
                    ),
                    "stage5_novel_count": novel,
                    "top_types": top_types,
                }
                writer.writerow(row)
                rows_written += 1

    md_path.write_text(_readme_text(rows_written), encoding="utf-8")
    logger.info(
        "Stage 9 wrote CSV rows=%d  csv=%s  md=%s", rows_written, csv_path, md_path
    )
    return StageResult(
        stage=9,
        status="completed",
        summary=f"Wrote extrinsic-validity dataset ({rows_written} rows).",
        metrics={"rows_written": rows_written},
        artifacts={
            "extrinsic_csv": str(csv_path),
            "extrinsic_md": str(md_path),
        },
    )


def _stage2_pass_count(
    ctx: StageContext, method_name: str, bill_id: str
) -> int:
    """Count Stage-2 survivors for one (method, bill) from the result file."""

    from ._common import read_stage2_pass_set

    try:
        pass_set = read_stage2_pass_set(ctx)
    except FileNotFoundError:
        return 0
    return len(pass_set.get(method_name, {}).get(bill_id, set()))


def _entailed_counts(
    ctx: StageContext, *, method_name: str, paths
) -> dict[str, set[str]]:
    """Bill -> set of ``entailed`` quadruplet ids from the Stage 3 cache."""

    out: dict[str, set[str]] = {}
    for bill_id in ctx.method_outputs[method_name].keys():
        ids: set[str] = set()
        for row in iter_cache_rows(paths, bill_id):
            if str(row.get("verdict") or "") == "entailed":
                qid = row.get("key")
                if qid:
                    ids.add(str(qid))
        if ids:
            out[bill_id] = ids
    return out


def _coverage_counts(
    ctx: StageContext, *, method_name: str, paths
) -> dict[str, tuple[int, int]]:
    """Bill -> (total labels judged, labels covered) from Stage 4 cache."""

    out: dict[str, tuple[int, int]] = {}
    for bill_id in ctx.method_outputs[method_name].keys():
        total = 0
        covered = 0
        for row in iter_cache_rows(paths, bill_id):
            verdict = str(row.get("verdict") or "")
            if verdict in {"covered", "partially_covered", "not_covered"}:
                total += 1
                if verdict == "covered":
                    covered += 1
        if total:
            out[bill_id] = (total, covered)
    return out


def _novel_counts(ctx: StageContext, *, method_name: str) -> dict[str, int]:
    """Count novel quadruplets per bill by differencing Stage 3 and Stage 4.

    A quadruplet is counted here only when its Stage 3 verdict is
    ``entailed`` and it never appears as a ``supporting_id`` in any
    ``covered`` or ``partially_covered`` Stage 4 row.
    """

    stage3_paths = cache_paths_for_stage(
        run_dir=ctx.run_dir, stage=3, method=method_name
    )
    stage4_paths = cache_paths_for_stage(
        run_dir=ctx.run_dir, stage=4, method=method_name
    )
    counts: dict[str, int] = {}
    for bill_id in ctx.method_outputs[method_name].keys():
        entailed: set[str] = set()
        for row in iter_cache_rows(stage3_paths, bill_id):
            if str(row.get("verdict") or "") == "entailed":
                qid = row.get("key")
                if qid:
                    entailed.add(str(qid))
        supporting: set[str] = set()
        for row in iter_cache_rows(stage4_paths, bill_id):
            verdict = str(row.get("verdict") or "")
            if verdict not in {"covered", "partially_covered"}:
                continue
            for qid in row.get("supporting_ids") or []:
                if qid:
                    supporting.add(str(qid))
        novel = entailed - supporting
        if novel:
            counts[bill_id] = len(novel)
    return counts


def _readme_text(rows: int) -> str:
    """Return the README markdown describing the CSV schema and hooks."""

    return (
        "# Stage 9 -- Extrinsic-validity dataset\n\n"
        f"Rows: {rows}\n\n"
        "This CSV is the handoff between the eval pipeline and the downstream "
        "analyses outlined in `docs/mpsa_draft_outline.md` (lines 264-272).  "
        "Each row is one (bill, method) pair and carries the quadruplet count, "
        "plausibility-pass count, grounded count, coverage rate over the NCSL "
        "topic labels, novel-quadruplet count, and the top 5 entity types.\n\n"
        "Downstream consumers join on `bill_id` (composite `{year}__{bill_id}`) "
        "for:\n\n"
        "1. Predictive-validity regressions of bill pass / governor signature "
        "probability on coverage rate and novel count.\n"
        "2. Variance-inflation / topic-leakage tests using the per-type columns.\n"
        "3. Cost-adjusted quality (combine with the usage-summary JSON to compute "
        "USD-per-covered-label).\n\n"
        "No downstream analyses are run here; this file is the input.\n"
    )
