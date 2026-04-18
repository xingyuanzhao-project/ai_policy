"""V2 eval runner: self-query RAG (filter extractor + single-pass synthesis).

- Runs one ``FilterExtractor`` call per question to pull structured filters out
  of the natural-language question, then runs a single embed -> retrieve ->
  generate_answer pass against the filtered subset.
- No planner / tool loop; this is the "V2" middle ground between V1 (no
  filters at all) and V3 (agentic RAG with planner + workers).
- Scores with the same rubric as V1 and V3 (``_common.score_question``).
- Writes ``results.json`` and ``report.md`` to the caller-provided directory
  (typically ``output/evals_app/v2/``).
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

from src.qa.artifacts import STATUS_BUCKETS, AnswerResult  # noqa: E402
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
from src.qa.eval.pipeline import build_retrieval_only_runtime  # noqa: E402
from src.qa.filter_extractor import FilterExtractor  # noqa: E402
from src.qa.retriever import Retriever, _coerce_int_values, _coerce_str_values  # noqa: E402

DEFAULT_GROUND_TRUTH = _THIS_DIR / "ground_truth.json"
DEFAULT_RESULTS_DIR = _PROJECT_ROOT / "output" / "evals_app" / "v2"

_VERSION_LABEL = "V2 -- self-query RAG (filter extractor, no planner)"


def run_eval_v2(
    *,
    project_root: Path,
    ground_truth_path: Path,
    results_dir: Path,
    max_questions: int | None = None,
    answer_model: str | None = None,
) -> None:
    """Run the V2 eval and write ``results.json`` + ``report.md`` to ``results_dir``."""

    ground_truth = load_ground_truth(ground_truth_path)
    questions = ground_truth["questions"]
    if max_questions is not None:
        questions = questions[:max_questions]

    runtime = build_retrieval_only_runtime(project_root)
    try:
        answer_model_used = answer_model or runtime.default_answer_model
        filter_extractor = FilterExtractor(
            client=runtime.provider_client.openai_client,
            model=runtime.filter_extractor_model,
        )
        available_filter_values = _compute_available_filter_values(runtime.retriever)

        print(
            f"[V2] backend={runtime.retrieval_backend} "
            f"chunks={runtime.chunk_count} "
            f"answer_model={answer_model_used} "
            f"filter_extractor_model={runtime.filter_extractor_model} "
            f"n_questions={len(questions)}"
        )

        scores: list[QuestionScore] = []
        raw_results: list[dict] = []
        for idx, question in enumerate(questions, start=1):
            print(f"[V2 {idx}/{len(questions)}] {question['id']}: {question['question'][:80]}")
            start = time.perf_counter()
            try:
                extracted = filter_extractor.extract(
                    question["question"], available_filter_values
                )
                semantic_query = extracted.semantic_query or question["question"]
                normalized_filters = _normalize_filters(extracted.filters) or {}

                query_vector = runtime.provider_client.embed_query(semantic_query)
                retrieved = runtime.retriever.retrieve(
                    query_vector,
                    top_k=runtime.retrieval_top_k,
                    filters=normalized_filters or None,
                )
                answer_text = runtime.provider_client.generate_answer(
                    question["question"],
                    retrieved,
                    answer_model=answer_model_used,
                )
                latency = time.perf_counter() - start
                result = AnswerResult(
                    question=question["question"],
                    answer=answer_text,
                    answer_model=answer_model_used,
                    citations=list(retrieved),
                    applied_filters=dict(normalized_filters),
                )
            except Exception as err:  # noqa: BLE001
                latency = time.perf_counter() - start
                scores.append(build_failed_score(question, latency, err))
                raw_results.append(build_failed_result_entry(question, err, latency))
                continue

            scores.append(score_question(question, result, latency))
            raw_results.append(build_raw_result_entry(question, result, latency))

        summary = write_eval_artifacts(
            results_dir=results_dir,
            gt_path=ground_truth_path,
            scores=scores,
            raw_results=raw_results,
            answer_model=answer_model_used,
            retrieval_backend=runtime.retrieval_backend,
            chunk_count=runtime.chunk_count,
            corpus_path=runtime.corpus_path,
            version_label=_VERSION_LABEL,
        )
        print_summary(_VERSION_LABEL, summary, scores)
        print(f"[V2] Wrote {results_dir / 'results.json'}")
        print(f"[V2] Wrote {results_dir / 'report.md'}")
    finally:
        runtime.close()


def _compute_available_filter_values(retriever: Retriever) -> dict[str, list]:
    """Derive unique filter values from the retriever metadata.

    Mirrors ``QAService._compute_available_filter_values`` for the vector
    backend; V1/V2 eval runners use the vector-only runtime so the lexical
    branch is not needed here.
    """

    metadata = retriever.chunk_metadata
    years: set[int] = set()
    states: set[str] = set()
    topics: set[str] = set()

    years.update(int(year) for year in metadata.years.tolist() if int(year) > 0)
    states.update(str(state) for state in metadata.states.tolist() if str(state).strip())
    for topic_set in metadata.topic_sets:
        topics.update(topic_set)

    return {
        "year": sorted(years, reverse=True),
        "state": sorted(states),
        "status_bucket": list(STATUS_BUCKETS),
        "topics": sorted(topics),
    }


def _normalize_filters(filters: dict | None) -> dict | None:
    """Scalar/list normalization matching ``QAService._normalize_filters``.

    Keeps V2 ``applied_filters`` output identical in shape to V3 so the
    comparison markdown can line them up directly.
    """

    if not filters:
        return None
    cleaned: dict = {}

    years = _coerce_int_values(filters.get("year"))
    if years:
        cleaned["year"] = years[0] if len(years) == 1 else years

    for key in ("state", "status_bucket"):
        values = _coerce_str_values(filters.get(key))
        if values:
            cleaned[key] = values[0] if len(values) == 1 else values

    topics = _coerce_str_values(filters.get("topics"))
    if topics:
        cleaned["topics"] = topics

    return cleaned or None


def _parse_args() -> argparse.Namespace:
    """Command-line interface for ad-hoc V2 eval runs."""

    parser = argparse.ArgumentParser(description="Run the V2 self-query RAG eval harness")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH,
        help="Path to the ground-truth JSON file",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory that will receive results.json and report.md",
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
    """Entry point for ``python -m src.qa.eval.runner_v2``."""

    args = _parse_args()
    run_eval_v2(
        project_root=_PROJECT_ROOT,
        ground_truth_path=args.ground_truth,
        results_dir=args.results_dir,
        max_questions=args.max_questions,
        answer_model=args.answer_model,
    )


if __name__ == "__main__":
    main()


__all__ = ["run_eval_v2"]
