"""Shared runtime entry points for the NER pipeline.

- Exposes full-corpus, single-bill, chunk-rerun, and group-rerun entry points.
- Ensures every entry point shares the same bootstrap and inference wiring.
- Does not implement stage logic directly; it delegates to runtime/bootstrap
  and orchestration modules.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from ..schemas.artifacts import CandidateQuadruplet, RefinedQuadruplet
from .bootstrap import PipelineContext, bootstrap
from .inference_loop import InferenceLoop


def run_full_corpus(
    project_root: Path,
    run_id: str | None = None,
    resume: bool = True,
    config_path: Path | None = None,
    ner_config_path: Path | None = None,
    prompt_config_path: Path | None = None,
) -> dict[str, list[RefinedQuadruplet]]:
    """Run the full NER pipeline on the configured corpus.

    Args:
        project_root (Path): Project root used for resolving relative config
            paths.
        run_id (str | None): Optional stable run identifier. A random id is
            created when omitted.
        resume (bool): Whether persisted stage outputs may be reused.
        config_path (Path | None): Optional override for the project-level
            config file.
        ner_config_path (Path | None): Optional override for the NER runtime
            config file.
        prompt_config_path (Path | None): Optional override for the NER prompt
            config file.

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
        context.llm_client.close()


def run_single_bill(
    project_root: Path,
    bill_id: str,
    run_id: str | None = None,
    resume: bool = True,
    config_path: Path | None = None,
    ner_config_path: Path | None = None,
    prompt_config_path: Path | None = None,
) -> list[RefinedQuadruplet]:
    """Run the NER pipeline on a single bill for debugging.

    Args:
        project_root (Path): Project root used for resolving relative config
            paths.
        bill_id (str): Bill identifier to process.
        run_id (str | None): Optional stable run identifier. A random id is
            created when omitted.
        resume (bool): Whether persisted stage outputs may be reused.
        config_path (Path | None): Optional override for the project-level
            config file.
        ner_config_path (Path | None): Optional override for the NER runtime
            config file.
        prompt_config_path (Path | None): Optional override for the NER prompt
            config file.

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
        context.llm_client.close()


def rerun_chunk(
    project_root: Path,
    bill_id: str,
    chunk_id: int,
    run_id: str,
    config_path: Path | None = None,
    ner_config_path: Path | None = None,
    prompt_config_path: Path | None = None,
) -> list[CandidateQuadruplet]:
    """Rerun zero-shot annotation for one persisted chunk id.

    Args:
        project_root (Path): Project root used for resolving relative config
            paths.
        bill_id (str): Bill identifier owning the chunk.
        chunk_id (int): Stable chunk identifier to rerun.
        run_id (str): Stable run identifier containing the persisted artifacts.
        config_path (Path | None): Optional override for the project-level
            config file.
        ner_config_path (Path | None): Optional override for the NER runtime
            config file.
        prompt_config_path (Path | None): Optional override for the NER prompt
            config file.

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


def rerun_group(
    project_root: Path,
    bill_id: str,
    group_id: int,
    run_id: str,
    config_path: Path | None = None,
    ner_config_path: Path | None = None,
    prompt_config_path: Path | None = None,
) -> list[RefinedQuadruplet]:
    """Rerun refinement for one grouped candidate set.

    Args:
        project_root (Path): Project root used for resolving relative config
            paths.
        bill_id (str): Bill identifier owning the grouped set.
        group_id (int): Stable grouped-set identifier to rerun.
        run_id (str): Stable run identifier containing the persisted artifacts.
        config_path (Path | None): Optional override for the project-level
            config file.
        ner_config_path (Path | None): Optional override for the NER runtime
            config file.
        prompt_config_path (Path | None): Optional override for the NER prompt
            config file.

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


def _build_runtime(
    project_root: Path,
    run_id: str | None,
    config_path: Path | None,
    ner_config_path: Path | None,
    prompt_config_path: Path | None,
) -> tuple[PipelineContext, InferenceLoop]:
    """Build one shared runtime context plus the main inference loop.

    Args:
        project_root (Path): Project root used for resolving relative config
            paths.
        run_id (str | None): Optional stable run identifier.
        config_path (Path | None): Optional override for the project-level
            config file.
        ner_config_path (Path | None): Optional override for the NER runtime
            config file.
        prompt_config_path (Path | None): Optional override for the NER prompt
            config file.

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
        max_bill_text_chars=context.max_bill_text_chars,
    )
    return context, inference_loop

