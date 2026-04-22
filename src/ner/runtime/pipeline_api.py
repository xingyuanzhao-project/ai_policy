"""Public API for the NER pipeline.

- Exposes full-corpus, single-bill, chunk-rerun, and group-rerun functions.
- Ensures every call shares the same bootstrap and inference wiring.
- Persists a ``usage_summary.json`` to the run directory on completion.
- Does not implement stage logic directly; it delegates to runtime/bootstrap
  and orchestration modules.

Scripts in ``scripts/`` import from this module to run the pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path

from ..schemas.artifacts import CandidateQuadruplet, RefinedQuadruplet
from .bootstrap import PipelineContext, bootstrap
from .inference_loop import InferenceLoop

logger = logging.getLogger(__name__)


def run_full_corpus(
    project_root: Path,
    config_path: str | Path,
    ner_config_path: str | Path,
    prompt_config_path: str | Path,
    run_id: str | None = None,
    resume: bool = True,
) -> dict[str, list[RefinedQuadruplet]]:
    """Run the full NER pipeline on the configured corpus.

    Args:
        project_root (Path): Project root used for resolving relative config
            paths.
        config_path (str | Path): Project-level config file (relative to
            *project_root* or absolute).
        ner_config_path (str | Path): NER runtime config file.
        prompt_config_path (str | Path): NER prompt / schema config file.
        run_id (str | None): Optional stable run identifier. A random id is
            created when omitted.
        resume (bool): Whether persisted stage outputs may be reused.

    Returns:
        dict[str, list[RefinedQuadruplet]]: Mapping from ``bill_id`` to final
            refined quadruplets.
    """

    context, inference_loop = _build_runtime(
        project_root=project_root,
        run_id=run_id,
        config_path=config_path,
        ner_config_path=ner_config_path,
        prompt_config_path=prompt_config_path,
    )
    try:
        bills = context.corpus_store.load()
        return asyncio.run(inference_loop.run_corpus(bills, resume=resume))
    finally:
        _persist_usage_summary(context)
        context.llm_client.close()
        context.detach_run_log()


def run_single_bill(
    project_root: Path,
    bill_id: str,
    config_path: str | Path,
    ner_config_path: str | Path,
    prompt_config_path: str | Path,
    run_id: str | None = None,
    resume: bool = True,
) -> list[RefinedQuadruplet]:
    """Run the NER pipeline on a single bill for debugging.

    Args:
        project_root (Path): Project root used for resolving relative config
            paths.
        bill_id (str): Bill identifier to process.
        config_path (str | Path): Project-level config file (relative to
            *project_root* or absolute).
        ner_config_path (str | Path): NER runtime config file.
        prompt_config_path (str | Path): NER prompt / schema config file.
        run_id (str | None): Optional stable run identifier. A random id is
            created when omitted.
        resume (bool): Whether persisted stage outputs may be reused.

    Returns:
        list[RefinedQuadruplet]: Final refined quadruplets for the selected
            bill.
    """

    context, inference_loop = _build_runtime(
        project_root=project_root,
        run_id=run_id,
        config_path=config_path,
        ner_config_path=ner_config_path,
        prompt_config_path=prompt_config_path,
    )
    try:
        context.corpus_store.load()
        bill = context.corpus_store.get_bill(bill_id)
        return asyncio.run(inference_loop.run_bill(bill, resume=resume))
    finally:
        _persist_usage_summary(context)
        context.llm_client.close()
        context.detach_run_log()


def rerun_chunk(
    project_root: Path,
    bill_id: str,
    chunk_id: int,
    run_id: str,
    config_path: str | Path,
    ner_config_path: str | Path,
    prompt_config_path: str | Path,
) -> list[CandidateQuadruplet]:
    """Rerun zero-shot annotation for one persisted chunk id.

    Args:
        project_root (Path): Project root used for resolving relative config
            paths.
        bill_id (str): Bill identifier owning the chunk.
        chunk_id (int): Stable chunk identifier to rerun.
        run_id (str): Stable run identifier containing the persisted artifacts.
        config_path (str | Path): Project-level config file (relative to
            *project_root* or absolute).
        ner_config_path (str | Path): NER runtime config file.
        prompt_config_path (str | Path): NER prompt / schema config file.

    Returns:
        list[CandidateQuadruplet]: Candidate quadruplets regenerated for the
            requested chunk.
    """

    context, _ = _build_runtime(
        project_root=project_root,
        run_id=run_id,
        config_path=config_path,
        ner_config_path=ner_config_path,
        prompt_config_path=prompt_config_path,
    )
    try:
        context.corpus_store.load()
        bill = context.corpus_store.get_bill(bill_id)
        chunks = context.inference_unit_builder.build(bill)
        return context.orchestrator.rerun_chunk(
            bill_id=bill_id,
            chunks=chunks,
            chunk_id=chunk_id,
        )
    finally:
        context.llm_client.close()
        context.detach_run_log()


def rerun_group(
    project_root: Path,
    bill_id: str,
    group_id: int,
    run_id: str,
    config_path: str | Path,
    ner_config_path: str | Path,
    prompt_config_path: str | Path,
) -> list[RefinedQuadruplet]:
    """Rerun refinement for one grouped candidate set.

    Args:
        project_root (Path): Project root used for resolving relative config
            paths.
        bill_id (str): Bill identifier owning the grouped set.
        group_id (int): Stable grouped-set identifier to rerun.
        run_id (str): Stable run identifier containing the persisted artifacts.
        config_path (str | Path): Project-level config file (relative to
            *project_root* or absolute).
        ner_config_path (str | Path): NER runtime config file.
        prompt_config_path (str | Path): NER prompt / schema config file.

    Returns:
        list[RefinedQuadruplet]: Refined quadruplets regenerated for the
            requested group.
    """

    context, _ = _build_runtime(
        project_root=project_root,
        run_id=run_id,
        config_path=config_path,
        ner_config_path=ner_config_path,
        prompt_config_path=prompt_config_path,
    )
    try:
        return context.orchestrator.rerun_group(
            bill_id=bill_id,
            group_id=group_id,
        )
    finally:
        context.llm_client.close()
        context.detach_run_log()


_USAGE_SUMMARY_NUMERIC_KEYS = (
    "total_calls",
    "total_prompt_tokens",
    "total_completion_tokens",
    "total_tokens",
    "total_reasoning_tokens",
    "total_cached_tokens",
    "total_cost_usd",
    "total_elapsed_ms",
)


def _persist_usage_summary(context: PipelineContext) -> None:
    """Write cumulative LLM usage stats to the run directory.

    If a ``usage_summary.json`` already exists from a previous session (e.g.
    a partial run that was interrupted), the current session's values are added
    on top so the file always reflects the **cumulative** resource consumption
    across all sessions for this run_id.
    """

    session_summary = context.llm_client.usage_stats.summary_dict()
    run_dir = context.artifact_store.run_dir(context.run_id)
    summary_path = run_dir / "usage_summary.json"

    if summary_path.exists():
        try:
            with open(summary_path, encoding="utf-8") as handle:
                previous = json.load(handle)
            for key in _USAGE_SUMMARY_NUMERIC_KEYS:
                session_summary[key] = (
                    previous.get(key, 0) + session_summary[key]
                )
            session_summary["total_cost_usd"] = round(
                session_summary["total_cost_usd"], 8
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Could not merge previous usage_summary.json; overwriting")

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(session_summary, handle, indent=2, ensure_ascii=False)
    logger.info(
        "Usage summary persisted: calls=%d  tokens=%d  cost=$%.6f  file=%s",
        session_summary["total_calls"],
        session_summary["total_tokens"],
        session_summary["total_cost_usd"],
        summary_path,
    )


def _build_runtime(
    project_root: Path,
    run_id: str | None,
    config_path: str | Path,
    ner_config_path: str | Path,
    prompt_config_path: str | Path,
) -> tuple[PipelineContext, InferenceLoop]:
    """Build one shared runtime context plus the main inference loop.

    Args:
        project_root (Path): Project root used for resolving relative config
            paths.
        run_id (str | None): Optional stable run identifier.
        config_path (str | Path): Project-level config file.
        ner_config_path (str | Path): NER runtime config file.
        prompt_config_path (str | Path): NER prompt / schema config file.

    Returns:
        tuple[PipelineContext, InferenceLoop]: Tuple of fully initialized
            pipeline context and inference loop.
    """

    effective_run_id = run_id or uuid.uuid4().hex
    context = bootstrap(
        project_root=project_root,
        run_id=effective_run_id,
        config_path=config_path,
        ner_config_path=ner_config_path,
        prompt_config_path=prompt_config_path,
    )
    inference_loop = InferenceLoop(
        inference_unit_builder=context.inference_unit_builder,
        orchestrator=context.orchestrator,
        final_output_store=context.final_output_store,
        max_bill_text_chars=context.max_bill_text_chars,
    )
    return context, inference_loop

