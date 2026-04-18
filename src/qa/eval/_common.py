"""Shared scoring and reporting helpers for the QA eval harness.

- Holds the version-agnostic scoring rubric (keyword, citation, count metrics)
  and the per-question + aggregate report writers used by V1, V2, and V3.
- Keeps ``runner.py`` (V3), ``runner_v1.py``, and ``runner_v2.py`` in sync on
  the exact same grading rules so their ``results.json`` files are directly
  comparable.
- Does not know anything about which retrieval / synthesis stack produced the
  ``AnswerResult`` -- it only consumes the dataclass and the ground-truth entry.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.qa.artifacts import AnswerResult


@dataclass(slots=True)
class QuestionScore:
    """Per-question scoring breakdown used in the aggregate report."""

    question_id: str
    difficulty: str
    pattern: str
    passed: bool
    keyword_score: float
    forbidden_hit: bool
    citation_precision: float | None
    citation_recall: float | None
    citation_f1: float | None
    count_score: float | None
    citation_count: int
    expected_min_citations: int
    latency_seconds: float
    notes: list[str] = field(default_factory=list)


def normalize_bill_id(bill_id: str) -> str:
    """Collapse internal whitespace so padded and unpadded bill_ids match."""

    return re.sub(r"\s+", " ", bill_id.strip()).lower()


def answer_contains(answer: str, needle: str) -> bool:
    """Case-insensitive substring check."""

    return needle.lower() in answer.lower()


def score_keywords(answer: str, required: list[str]) -> float:
    """Fraction of required keywords present in the answer."""

    if not required:
        return 1.0
    hits = sum(1 for keyword in required if answer_contains(answer, keyword))
    return hits / len(required)


def forbidden_hit(answer: str, forbidden: list[str]) -> bool:
    """Return True if any forbidden keyword appears in the answer."""

    return any(answer_contains(answer, keyword) for keyword in forbidden)


def score_citations(
    cited_bill_ids: list[str], expected_bill_ids: list[str]
) -> tuple[float | None, float | None, float | None]:
    """Precision / recall / F1 for cited bill ids vs the expected ground-truth set."""

    if not expected_bill_ids:
        return None, None, None
    cited_norm = {normalize_bill_id(b) for b in cited_bill_ids}
    expected_norm = {normalize_bill_id(b) for b in expected_bill_ids}
    if not cited_norm:
        return 0.0, 0.0, 0.0
    overlap = cited_norm & expected_norm
    precision = len(overlap) / len(cited_norm)
    recall = len(overlap) / len(expected_norm)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


_COUNT_RE = re.compile(r"\b(\d{1,4})\b")


def score_count(answer: str, expected: int, tolerance_abs: int | None) -> float | None:
    """Score numeric-count questions by the best-matching integer in the answer."""

    if expected is None:
        return None
    candidates = [int(match) for match in _COUNT_RE.findall(answer)]
    if not candidates:
        return 0.0
    best = min(candidates, key=lambda value: abs(value - expected))
    delta = abs(best - expected)
    if tolerance_abs is not None and delta <= tolerance_abs:
        return 1.0
    return max(0.0, 1.0 - (delta / max(expected, 1)))


def score_question(
    question: dict[str, Any], result: AnswerResult, latency_seconds: float
) -> QuestionScore:
    """Apply the ground-truth scoring rules to one QA service result."""

    gt = question["ground_truth"]
    required = list(gt.get("required_keywords", []))
    forbidden = list(gt.get("forbidden_keywords", []))
    expected_bill_ids = list(gt.get("expected_bill_ids", []))
    expected_count = gt.get("expected_count")
    tolerance_abs = gt.get("count_tolerance_abs")
    min_citations = int(gt.get("min_citations", 0))
    keyword_threshold = float(gt.get("keyword_threshold", 1.0))

    answer_text = result.answer or ""
    cited_bill_ids = [citation.bill_id for citation in result.citations]

    keyword_value = score_keywords(answer_text, required)
    forbidden_flag = forbidden_hit(answer_text, forbidden)
    precision, recall, f1 = score_citations(cited_bill_ids, expected_bill_ids)
    count_value = score_count(answer_text, expected_count, tolerance_abs) if expected_count else None

    notes: list[str] = []
    if forbidden_flag:
        notes.append(f"forbidden keyword appeared: {forbidden!r}")
    if min_citations and len(cited_bill_ids) < min_citations:
        notes.append(f"citations {len(cited_bill_ids)} < min {min_citations}")
    if expected_bill_ids and (recall is None or recall < 0.25):
        notes.append(f"low citation recall ({recall})")

    passed = (
        keyword_value >= keyword_threshold
        and not forbidden_flag
        and len(cited_bill_ids) >= min_citations
        and (expected_count is None or (count_value or 0.0) >= 0.5)
    )

    return QuestionScore(
        question_id=question["id"],
        difficulty=question["difficulty"],
        pattern=question["pattern"],
        passed=passed,
        keyword_score=round(keyword_value, 4),
        forbidden_hit=forbidden_flag,
        citation_precision=None if precision is None else round(precision, 4),
        citation_recall=None if recall is None else round(recall, 4),
        citation_f1=None if f1 is None else round(f1, 4),
        count_score=None if count_value is None else round(count_value, 4),
        citation_count=len(cited_bill_ids),
        expected_min_citations=min_citations,
        latency_seconds=round(latency_seconds, 3),
        notes=notes,
    )


def aggregate_summary(scores: list[QuestionScore]) -> dict[str, Any]:
    """Group pass-rates and key metrics by difficulty for the report."""

    def _mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    overall_pass = _mean([1.0 if score.passed else 0.0 for score in scores])
    by_diff: dict[str, dict[str, Any]] = {}
    for difficulty in sorted({score.difficulty for score in scores}):
        subset = [score for score in scores if score.difficulty == difficulty]
        by_diff[difficulty] = {
            "count": len(subset),
            "pass_rate": _mean([1.0 if score.passed else 0.0 for score in subset]),
            "mean_keyword_score": _mean([score.keyword_score for score in subset]),
            "mean_citation_f1": _mean(
                [score.citation_f1 for score in subset if score.citation_f1 is not None]
            ),
            "mean_count_score": _mean(
                [score.count_score for score in subset if score.count_score is not None]
            ),
            "mean_latency": _mean([score.latency_seconds for score in subset]),
        }
    return {
        "n_questions": len(scores),
        "overall_pass_rate": overall_pass,
        "by_difficulty": by_diff,
    }


def load_ground_truth(path: Path) -> dict[str, Any]:
    """Load and minimally validate the ground-truth file."""

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "questions" not in payload or not isinstance(payload["questions"], list):
        raise ValueError(f"Ground truth file {path} missing 'questions' list")
    return payload


def build_raw_result_entry(
    question: dict[str, Any],
    result: AnswerResult,
    latency_seconds: float,
) -> dict[str, Any]:
    """Build the raw per-question entry written into ``results.json``."""

    return {
        "id": question["id"],
        "difficulty": question["difficulty"],
        "pattern": question["pattern"],
        "question": question["question"],
        "answer": result.answer,
        "answer_model": result.answer_model,
        "applied_filters": dict(result.applied_filters or {}),
        "cited_bill_ids": [citation.bill_id for citation in result.citations],
        "latency_seconds": round(latency_seconds, 3),
    }


def build_failed_result_entry(
    question: dict[str, Any],
    error: BaseException,
    latency_seconds: float,
) -> dict[str, Any]:
    """Build the raw per-question entry for a question that raised."""

    return {
        "id": question["id"],
        "difficulty": question["difficulty"],
        "pattern": question["pattern"],
        "question": question["question"],
        "answer": "",
        "answer_model": "",
        "applied_filters": {},
        "cited_bill_ids": [],
        "latency_seconds": round(latency_seconds, 3),
        "error": f"{type(error).__name__}: {error}",
    }


def build_failed_score(
    question: dict[str, Any],
    latency_seconds: float,
    error: BaseException,
) -> QuestionScore:
    """Build a zeroed ``QuestionScore`` for a question that raised."""

    return QuestionScore(
        question_id=question["id"],
        difficulty=question["difficulty"],
        pattern=question["pattern"],
        passed=False,
        keyword_score=0.0,
        forbidden_hit=False,
        citation_precision=None,
        citation_recall=None,
        citation_f1=None,
        count_score=None,
        citation_count=0,
        expected_min_citations=int(question["ground_truth"].get("min_citations", 0)),
        latency_seconds=round(latency_seconds, 3),
        notes=[f"exception: {type(error).__name__}: {error}"],
    )


def write_markdown_report(
    *,
    gt_path: Path,
    summary: dict[str, Any],
    scores: list[QuestionScore],
    raw_results: list[dict[str, Any]],
    report_path: Path,
    answer_model: str,
    corpus_path: str,
    retrieval_backend: str,
    chunk_count: int,
    version_label: str,
) -> None:
    """Write a Markdown summary alongside the JSON results."""

    lines: list[str] = []
    lines.append(f"# QA Eval Report -- {version_label}")
    lines.append("")
    lines.append(f"- Ground truth: `{gt_path.as_posix()}`")
    lines.append(f"- Corpus: `{corpus_path}`")
    lines.append(f"- Retrieval backend: `{retrieval_backend}`, chunks indexed: {chunk_count}")
    lines.append(f"- Answer model: `{answer_model}`")
    lines.append(f"- N questions: **{summary['n_questions']}**")
    lines.append(f"- Overall pass rate: **{summary['overall_pass_rate']:.2%}**")
    lines.append("")
    lines.append("## Pass rate by difficulty")
    lines.append("")
    lines.append("| Difficulty | N | Pass rate | Mean keyword | Mean citation F1 | Mean count score | Mean latency (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for difficulty, stats in summary["by_difficulty"].items():
        lines.append(
            "| {difficulty} | {n} | {pass_rate:.0%} | {kw:.2f} | {cf1:.2f} | {cs:.2f} | {lat:.2f} |".format(
                difficulty=difficulty,
                n=stats["count"],
                pass_rate=stats["pass_rate"],
                kw=stats["mean_keyword_score"],
                cf1=stats["mean_citation_f1"],
                cs=stats["mean_count_score"],
                lat=stats["mean_latency"],
            )
        )
    lines.append("")
    lines.append("## Per-question detail")
    lines.append("")
    lines.append("| ID | Difficulty | Pattern | Pass | KW | Cit P | Cit R | Cit F1 | Count | #Cit | Notes |")
    lines.append("|---|---|---|:---:|---:|---:|---:|---:|---:|---:|---|")
    for score in scores:
        lines.append(
            "| {qid} | {diff} | {pattern} | {pass_flag} | {kw:.2f} | {cp} | {cr} | {cf1} | {cs} | {nc} | {notes} |".format(
                qid=score.question_id,
                diff=score.difficulty,
                pattern=score.pattern,
                pass_flag="YES" if score.passed else "no",
                kw=score.keyword_score,
                cp="-" if score.citation_precision is None else f"{score.citation_precision:.2f}",
                cr="-" if score.citation_recall is None else f"{score.citation_recall:.2f}",
                cf1="-" if score.citation_f1 is None else f"{score.citation_f1:.2f}",
                cs="-" if score.count_score is None else f"{score.count_score:.2f}",
                nc=score.citation_count,
                notes="; ".join(score.notes) if score.notes else "",
            )
        )
    lines.append("")
    lines.append("## Per-question answers (first 1000 chars each)")
    for entry in raw_results:
        lines.append("")
        lines.append(f"### {entry['id']} -- {entry['difficulty']} / {entry['pattern']}")
        lines.append("")
        lines.append(f"**Q:** {entry['question']}")
        lines.append("")
        cited = ", ".join(entry["cited_bill_ids"]) or "(none)"
        lines.append(f"**Citations ({len(entry['cited_bill_ids'])}):** `{cited}`")
        lines.append("")
        applied = entry.get("applied_filters") or {}
        if applied:
            lines.append(f"**Applied filters:** `{json.dumps(applied, sort_keys=True)}`")
            lines.append("")
        answer = entry.get("answer") or ""
        snippet = answer[:1000] + ("..." if len(answer) > 1000 else "")
        lines.append("**A:**")
        lines.append("")
        lines.append("> " + snippet.replace("\n", "\n> "))
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_eval_artifacts(
    *,
    results_dir: Path,
    gt_path: Path,
    scores: list[QuestionScore],
    raw_results: list[dict[str, Any]],
    answer_model: str,
    retrieval_backend: str,
    chunk_count: int,
    corpus_path: str,
    version_label: str,
) -> dict[str, Any]:
    """Write ``results.json`` and ``report.md`` into ``results_dir``.

    Returns the aggregate ``summary`` dict so callers can print a console line
    without re-reading the JSON.
    """

    results_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate_summary(scores)
    results_payload = {
        "version_label": version_label,
        "ground_truth_file": gt_path.as_posix(),
        "answer_model": answer_model,
        "retrieval_backend": retrieval_backend,
        "chunk_count": chunk_count,
        "summary": summary,
        "scores": [asdict(score) for score in scores],
        "raw_results": raw_results,
    }
    results_path = results_dir / "results.json"
    report_path = results_dir / "report.md"
    results_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    write_markdown_report(
        gt_path=gt_path,
        summary=summary,
        scores=scores,
        raw_results=raw_results,
        report_path=report_path,
        answer_model=answer_model,
        corpus_path=corpus_path,
        retrieval_backend=retrieval_backend,
        chunk_count=chunk_count,
        version_label=version_label,
    )
    return summary


def print_summary(version_label: str, summary: dict[str, Any], scores: list[QuestionScore]) -> None:
    """Print the short aggregate line used by every runner's CLI."""

    print()
    print(
        f"[{version_label}] Overall pass rate: "
        f"{summary['overall_pass_rate']:.2%} "
        f"({sum(1 for s in scores if s.passed)}/{len(scores)})"
    )
    for difficulty, stats in summary["by_difficulty"].items():
        print(
            f"  {difficulty:10s} n={stats['count']:2d} "
            f"pass={stats['pass_rate']:.0%} "
            f"kw={stats['mean_keyword_score']:.2f} "
            f"cit_f1={stats['mean_citation_f1']:.2f} "
            f"count={stats['mean_count_score']:.2f}"
        )


__all__ = [
    "QuestionScore",
    "aggregate_summary",
    "build_failed_result_entry",
    "build_failed_score",
    "build_raw_result_entry",
    "load_ground_truth",
    "print_summary",
    "score_question",
    "write_eval_artifacts",
    "write_markdown_report",
]
