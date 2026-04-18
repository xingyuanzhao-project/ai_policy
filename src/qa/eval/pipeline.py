"""Orchestrator for running V1, V2, and V3 QA evals in one invocation.

- Provides the ``run_all_versions_and_compare`` entry point that the
  ``scripts/qa_eval.py`` script calls. One invocation executes the V1, V2,
  and V3 runners in order, writes per-version ``results.json`` + ``report.md``
  under ``output/evals_app/<version>/``, then generates comparison plots and
  an N-way comparison markdown under ``output/evals_app/_comparison/``.
- Hosts the ``build_retrieval_only_runtime`` helper used by V1 and V2 runners
  so they can reuse ``provider_client`` + ``Retriever`` without wiring the
  ``FilterExtractor`` / ``PlannerAgent`` / ``QAService`` stack.
- Does not own scoring rules, plot rendering, or HTTP server logic; those live
  in ``_common.py``, ``plots.py``, and ``runtime.py`` respectively.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src.qa.artifacts import IndexedChunk
from src.qa.config import load_provider_api_key, load_qa_config
from src.qa.indexer import QAIndexer
from src.qa.provider_client import OpenAICompatibleClient
from src.qa.retriever import Retriever


_VECTOR_RETRIEVAL_BACKEND = "vector"


@dataclass(slots=True)
class RetrievalOnlyRuntime:
    """Retrieval-only runtime shared by V1 and V2 eval runners."""

    provider_client: OpenAICompatibleClient
    retriever: Retriever
    chunks: Sequence[IndexedChunk]
    retrieval_backend: str
    chunk_count: int
    retrieval_top_k: int
    default_answer_model: str
    filter_extractor_model: str
    corpus_path: str

    def close(self) -> None:
        """Release HTTP resources held by the provider client."""

        self.provider_client.close()


def build_retrieval_only_runtime(project_root: Path) -> RetrievalOnlyRuntime:
    """Build a retrieval-only runtime (no planner, no filter extractor, no service).

    Mirrors the vector-backend branch of ``src/qa/runtime.py`` but skips every
    component V1 / V2 do not need. V1 and V2 runners then assemble whatever
    extra pieces they require (e.g. V2 constructs its own ``FilterExtractor``).
    """

    config = load_qa_config(project_root)
    config.validate()

    provider_client = OpenAICompatibleClient(
        api_key=load_provider_api_key(config.provider),
        api_base_url=config.provider.api_base_url,
        embedding_model=config.models.embedding_model,
        answer_model=config.models.answer_model,
    )
    indexer = QAIndexer(
        project_root=project_root,
        config=config,
        provider_client=provider_client,
    )
    loaded_index = indexer.load_ready_index()
    retriever = Retriever(loaded_index.chunks, loaded_index.embeddings)

    return RetrievalOnlyRuntime(
        provider_client=provider_client,
        retriever=retriever,
        chunks=loaded_index.chunks,
        retrieval_backend=_VECTOR_RETRIEVAL_BACKEND,
        chunk_count=loaded_index.manifest.total_chunks,
        retrieval_top_k=config.index.retrieval_top_k,
        default_answer_model=config.models.answer_model,
        filter_extractor_model=config.models.filter_extractor_model,
        corpus_path=config.corpus_path,
    )


def run_all_versions_and_compare(
    *,
    project_root: Path,
    ground_truth_path: Path,
    output_root: Path,
    versions: tuple[str, ...] = ("v1", "v2", "v3"),
    max_questions: int | None = None,
    answer_model: str | None = None,
) -> None:
    """Run V1/V2/V3 evals in sequence, then render comparison artifacts.

    Args:
        project_root: Repository root; passed through to runtime builders.
        ground_truth_path: Path to ``ground_truth.json``.
        output_root: Parent directory for ``v1/``, ``v2/``, ``v3/`` and the
            ``_comparison/`` sibling.
        versions: Ordered tuple of version keys to execute.
        max_questions: Optional cap for smoke-testing the harness.
        answer_model: Optional override of the default answer model.
    """

    resolved_ground_truth = _resolve_ground_truth_path(project_root, ground_truth_path)
    resolved_output_root = _resolve_output_root(project_root, output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    for version in versions:
        version_dir = resolved_output_root / version
        version_dir.mkdir(parents=True, exist_ok=True)
        _dispatch_version(
            version=version,
            project_root=project_root,
            ground_truth_path=resolved_ground_truth,
            results_dir=version_dir,
            max_questions=max_questions,
            answer_model=answer_model,
        )

    comparison_dir = resolved_output_root / "_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    version_results_paths: dict[str, Path] = {}
    for version in versions:
        results_path = resolved_output_root / version / "results.json"
        if results_path.exists():
            version_results_paths[version] = results_path
        else:
            print(f"[compare] Skipping {version}: {results_path} not found")

    if len(version_results_paths) >= 2:
        from src.qa.eval.plots import render_comparison_plots

        render_comparison_plots(
            version_results=version_results_paths,
            output_dir=comparison_dir,
            per_version_output_root=resolved_output_root,
        )
        _write_comparison_markdown(
            version_results=version_results_paths,
            output_path=comparison_dir / "comparison.md",
        )
        print(f"[compare] Wrote comparison artifacts to {comparison_dir}")
    else:
        print(
            f"[compare] Need >=2 version results to compare; got {len(version_results_paths)}"
        )


def _dispatch_version(
    *,
    version: str,
    project_root: Path,
    ground_truth_path: Path,
    results_dir: Path,
    max_questions: int | None,
    answer_model: str | None,
) -> None:
    """Dispatch one version to the correct runner."""

    if version == "v1":
        from src.qa.eval.runner_v1 import run_eval_v1

        run_eval_v1(
            project_root=project_root,
            ground_truth_path=ground_truth_path,
            results_dir=results_dir,
            max_questions=max_questions,
            answer_model=answer_model,
        )
    elif version == "v2":
        from src.qa.eval.runner_v2 import run_eval_v2

        run_eval_v2(
            project_root=project_root,
            ground_truth_path=ground_truth_path,
            results_dir=results_dir,
            max_questions=max_questions,
            answer_model=answer_model,
        )
    elif version == "v3":
        from src.qa.eval.runner import run_eval

        run_eval(
            ground_truth_path=ground_truth_path,
            results_path=results_dir / "results.json",
            report_path=results_dir / "report.md",
            max_questions=max_questions,
            answer_model=answer_model,
        )
    else:
        raise ValueError(f"Unknown eval version {version!r}; expected one of v1/v2/v3")


def _resolve_ground_truth_path(project_root: Path, ground_truth_path: Path) -> Path:
    """Resolve ``ground_truth_path`` relative to ``project_root`` if relative."""

    if ground_truth_path.is_absolute():
        return ground_truth_path
    return (project_root / ground_truth_path).resolve()


def _resolve_output_root(project_root: Path, output_root: Path) -> Path:
    """Resolve ``output_root`` relative to ``project_root`` if relative."""

    if output_root.is_absolute():
        return output_root
    return (project_root / output_root).resolve()


def _load_results(path: Path) -> dict[str, Any]:
    """Load one version's ``results.json`` payload."""

    return json.loads(path.read_text(encoding="utf-8"))


def _write_comparison_markdown(
    *,
    version_results: dict[str, Path],
    output_path: Path,
) -> None:
    """Write an N-way comparison markdown across all version results."""

    loaded = {version: _load_results(path) for version, path in version_results.items()}
    version_order = list(loaded.keys())

    lines: list[str] = []
    lines.append("# QA Eval Comparison")
    lines.append("")
    lines.append("Compares pass rate, keyword score, citation F1, and latency across:")
    for version in version_order:
        label = loaded[version].get("version_label", version)
        lines.append(f"- **{version}** -- {label}")
    lines.append("")

    lines.append("## Overall pass rate")
    lines.append("")
    lines.append("| Version | N | Pass rate | Mean keyword | Mean citation F1 | Mean latency (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for version in version_order:
        payload = loaded[version]
        summary = payload.get("summary", {})
        n = summary.get("n_questions", 0)
        pass_rate = summary.get("overall_pass_rate", 0.0)
        by_diff = summary.get("by_difficulty", {})
        kw = _mean_over_difficulties(by_diff, "mean_keyword_score")
        cf1 = _mean_over_difficulties(by_diff, "mean_citation_f1")
        lat = _mean_over_difficulties(by_diff, "mean_latency")
        lines.append(
            f"| {version} | {n} | {pass_rate:.0%} | {kw:.2f} | {cf1:.2f} | {lat:.2f} |"
        )
    lines.append("")

    lines.append("## Pass rate by difficulty")
    lines.append("")
    difficulties = _collect_difficulties(loaded)
    header = "| Difficulty | " + " | ".join(version_order) + " |"
    sep = "|---|" + "|".join([":---:"] * len(version_order)) + "|"
    lines.append(header)
    lines.append(sep)
    for difficulty in difficulties:
        row = [difficulty]
        for version in version_order:
            stats = loaded[version].get("summary", {}).get("by_difficulty", {}).get(difficulty)
            if stats is None:
                row.append("-")
            else:
                n = stats.get("count", 0)
                pr = stats.get("pass_rate", 0.0)
                row.append(f"{pr:.0%} (n={n})")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Per-question pass/fail")
    lines.append("")
    question_order = _collect_question_order(loaded)
    header = "| ID | Difficulty | " + " | ".join(version_order) + " | Delta |"
    sep = "|---|---|" + "|".join([":---:"] * len(version_order)) + "|---|"
    lines.append(header)
    lines.append(sep)
    for qid in question_order:
        per_version = _question_pass_flags(loaded, qid)
        difficulty = _question_difficulty(loaded, qid)
        flags = []
        for version in version_order:
            passed = per_version.get(version)
            if passed is None:
                flags.append("-")
            else:
                flags.append("YES" if passed else "no")
        delta = _describe_delta(per_version, version_order)
        lines.append(
            f"| {qid} | {difficulty} | " + " | ".join(flags) + f" | {delta} |"
        )
    lines.append("")

    lines.append("## Regressions and improvements (vs first listed version)")
    lines.append("")
    baseline = version_order[0]
    for target in version_order[1:]:
        regressions: list[str] = []
        improvements: list[str] = []
        for qid in question_order:
            per_version = _question_pass_flags(loaded, qid)
            base_pass = per_version.get(baseline)
            tgt_pass = per_version.get(target)
            if base_pass is None or tgt_pass is None:
                continue
            if base_pass and not tgt_pass:
                regressions.append(qid)
            elif not base_pass and tgt_pass:
                improvements.append(qid)
        lines.append(f"### {target} vs {baseline}")
        lines.append("")
        lines.append(f"- Improvements ({len(improvements)}): {', '.join(improvements) or '(none)'}")
        lines.append(f"- Regressions ({len(regressions)}): {', '.join(regressions) or '(none)'}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _mean_over_difficulties(by_diff: dict[str, dict[str, Any]], key: str) -> float:
    """Average a per-difficulty metric across difficulties present."""

    values = [
        float(stats.get(key, 0.0))
        for stats in by_diff.values()
        if key in stats
    ]
    return sum(values) / len(values) if values else 0.0


def _collect_difficulties(loaded: dict[str, dict[str, Any]]) -> list[str]:
    """Union of difficulties seen across versions, preserved in stable order."""

    preferred = ["easy", "medium", "hard", "very_hard"]
    seen: list[str] = []
    for version_payload in loaded.values():
        by_diff = version_payload.get("summary", {}).get("by_difficulty", {})
        for difficulty in by_diff.keys():
            if difficulty not in seen:
                seen.append(difficulty)
    return [d for d in preferred if d in seen] + [d for d in seen if d not in preferred]


def _collect_question_order(loaded: dict[str, dict[str, Any]]) -> list[str]:
    """Union of question ids across versions, preserving first-seen order."""

    seen: list[str] = []
    for version_payload in loaded.values():
        for score in version_payload.get("scores", []):
            qid = score.get("question_id")
            if qid and qid not in seen:
                seen.append(qid)
    return seen


def _question_pass_flags(
    loaded: dict[str, dict[str, Any]], qid: str
) -> dict[str, bool]:
    """Map version -> passed flag for one question id."""

    out: dict[str, bool] = {}
    for version, payload in loaded.items():
        for score in payload.get("scores", []):
            if score.get("question_id") == qid:
                out[version] = bool(score.get("passed", False))
                break
    return out


def _question_difficulty(loaded: dict[str, dict[str, Any]], qid: str) -> str:
    """Return the difficulty label for ``qid`` from whichever version has it."""

    for payload in loaded.values():
        for score in payload.get("scores", []):
            if score.get("question_id") == qid:
                return str(score.get("difficulty", ""))
    return ""


def _describe_delta(per_version: dict[str, bool], version_order: list[str]) -> str:
    """Summarize the pass pattern across versions for one question."""

    if not per_version:
        return ""
    pattern = "".join(
        "P" if per_version.get(version) else "F"
        for version in version_order
    )
    return pattern


__all__ = [
    "RetrievalOnlyRuntime",
    "build_retrieval_only_runtime",
    "run_all_versions_and_compare",
]
