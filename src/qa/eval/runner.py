"""V3 eval runner: agentic RAG with planner (current production stack).

- Loads the hand-curated ground truth set at ``ground_truth.json``.
- Spins up the same QA runtime that ``scripts/run_qa_app.py`` uses so the
  evaluation sees the real filter extractor, retriever, planner agent, and
  answer model.
- Feeds each question to ``QAService.answer_question`` and scores the answer
  with the shared rubric in ``_common.py`` so V1 / V2 / V3 results stay
  directly comparable.
- Writes ``results.json`` and ``report.md`` to the caller-provided paths
  (defaults to the ``src/qa/eval/`` directory for backward compatibility with
  existing ad-hoc callers).
- Does NOT silently swap models, thresholds, or the corpus path.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.qa import build_qa_browser_runtime  # noqa: E402
from src.qa.eval._common import (  # noqa: E402
    QuestionScore,
    build_failed_result_entry,
    build_failed_score,
    build_raw_result_entry,
    load_ground_truth,
    print_summary,
    score_question,
    write_eval_artifacts,
)

DEFAULT_GROUND_TRUTH = _THIS_DIR / "ground_truth.json"
DEFAULT_RESULTS_JSON = _THIS_DIR / "results.json"
DEFAULT_REPORT_MD = _THIS_DIR / "report.md"

_VERSION_LABEL = "V3 -- agentic RAG with planner"


def run_eval(
    *,
    ground_truth_path: Path,
    results_path: Path,
    report_path: Path,
    max_questions: int | None = None,
    answer_model: str | None = None,
) -> None:
    """Load ground truth, hit the live QA service, score, and emit reports.

    ``results_path`` and ``report_path`` may point to any directory; the pair
    is written side-by-side. When invoked by the V1/V2/V3 orchestrator the
    caller passes paths under ``output/evals_app/v3/``; older ad-hoc callers
    that used the defaults continue to land next to ``ground_truth.json``.
    """

    ground_truth = load_ground_truth(ground_truth_path)
    questions = ground_truth["questions"]
    if max_questions is not None:
        questions = questions[:max_questions]

    runtime = build_qa_browser_runtime(_PROJECT_ROOT)
    try:
        answer_model_used = answer_model or runtime.qa_service.default_answer_model
        print(
            f"[V3] backend={runtime.retrieval_backend} "
            f"chunks={runtime.chunk_count} "
            f"answer_model={answer_model_used} "
            f"n_questions={len(questions)}"
        )
        corpus_path_used = "(n/a)"
        try:
            from src.qa import load_qa_config

            corpus_path_used = load_qa_config(_PROJECT_ROOT).corpus_path
        except Exception:
            pass

        scores: list[QuestionScore] = []
        raw_results: list[dict] = []
        for idx, question in enumerate(questions, start=1):
            print(f"[V3 {idx}/{len(questions)}] {question['id']}: {question['question'][:80]}")
            start = time.perf_counter()
            try:
                result = runtime.qa_service.answer_question(
                    question=question["question"],
                    answer_model=answer_model,
                )
                latency = time.perf_counter() - start
            except Exception as err:  # noqa: BLE001
                latency = time.perf_counter() - start
                scores.append(build_failed_score(question, latency, err))
                raw_results.append(build_failed_result_entry(question, err, latency))
                continue

            scores.append(score_question(question, result, latency))
            raw_results.append(build_raw_result_entry(question, result, latency))

        results_dir = results_path.parent
        results_dir.mkdir(parents=True, exist_ok=True)
        summary = write_eval_artifacts(
            results_dir=results_dir,
            gt_path=ground_truth_path,
            scores=scores,
            raw_results=raw_results,
            answer_model=answer_model_used,
            retrieval_backend=runtime.retrieval_backend,
            chunk_count=runtime.chunk_count,
            corpus_path=corpus_path_used,
            version_label=_VERSION_LABEL,
        )

        _reconcile_output_paths(
            results_dir=results_dir,
            results_path=results_path,
            report_path=report_path,
        )

        print_summary(_VERSION_LABEL, summary, scores)
        print(f"[V3] Wrote {results_path}")
        print(f"[V3] Wrote {report_path}")
    finally:
        runtime.close()


def _reconcile_output_paths(
    *,
    results_dir: Path,
    results_path: Path,
    report_path: Path,
) -> None:
    """Align ``write_eval_artifacts`` outputs with caller-requested file paths.

    ``write_eval_artifacts`` always writes ``results.json`` and ``report.md``
    into its ``results_dir``. When callers request different filenames we
    rename the written files; identical names are left alone.
    """

    default_results = results_dir / "results.json"
    default_report = results_dir / "report.md"
    if results_path != default_results and default_results.exists():
        results_path.parent.mkdir(parents=True, exist_ok=True)
        default_results.replace(results_path)
    if report_path != default_report and default_report.exists():
        report_path.parent.mkdir(parents=True, exist_ok=True)
        default_report.replace(report_path)


def _parse_args() -> argparse.Namespace:
    """Command-line interface for ad-hoc eval runs."""

    parser = argparse.ArgumentParser(description="Run the V3 QA eval harness")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH,
        help="Path to the ground-truth JSON file",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_JSON,
        help="Path to write the per-question JSON results",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_MD,
        help="Path to write the Markdown summary report",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optional cap for smoke-testing the runner",
    )
    parser.add_argument(
        "--answer-model",
        default=None,
        help="Override the default answer model for this eval run",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for ``python -m src.qa.eval.runner``."""

    args = _parse_args()
    run_eval(
        ground_truth_path=args.ground_truth,
        results_path=args.results,
        report_path=args.report,
        max_questions=args.max_questions,
        answer_model=args.answer_model,
    )


if __name__ == "__main__":
    main()


__all__ = ["run_eval"]
