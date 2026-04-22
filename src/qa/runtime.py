"""Shared runtime wiring for the QA browser app.

- Centralizes QA service, planner agent, and Flask app construction for both
  the local CLI entry point and Render's Gunicorn entry point.
- Preserves the existing vector-first startup behavior while keeping the
  lexical fallback available when no ready index is present.
- Keeps provider client, local answer support, and answer-model option assembly
  in one place so deployment-specific entry points stay thin.
- Does not own command-line parsing, corpus indexing, or HTTP server binding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from flask import Flask

from .artifacts import IndexedChunk
from .config import load_provider_api_key, load_qa_config
from .diagnostics import emit_runtime_diagnostic
from .filter_extractor import FilterExtractor
from .indexer import IndexStateError, QAIndexer, build_indexed_chunks
from .lexical_retriever import LexicalRetriever
from .local_answer_support import AnswerModelOption, LocalAnswerSupport
from .planner_agent import PlannerAgent
from .provider_client import OpenAICompatibleClient
from .qa_tools import (
    LexicalSearchBackend,
    SearchBackend,
    VectorSearchBackend,
    build_bill_index,
)
from .quadruplet_store import QuadrupletStore, load_quadruplet_store
from .retriever import Retriever
from .service import QAService
from .web_app import create_app

_VECTOR_RETRIEVAL_BACKEND = "vector"
_LEXICAL_RETRIEVAL_BACKEND = "lexical"
_QUADRUPLET_SIDECAR_FILENAME = "quadruplets.jsonl"


@dataclass(slots=True)
class QABrowserRuntime:
    """Loaded QA runtime shared by local and hosted entry points."""

    app: Flask
    qa_service: QAService
    provider_client: OpenAICompatibleClient
    local_answer_support: LocalAnswerSupport
    retrieval_backend: str
    chunk_count: int

    def close(self) -> None:
        """Close network clients owned by the runtime."""

        self.provider_client.close()
        self.local_answer_support.close()


def build_qa_browser_runtime(
    project_root: Path,
    *,
    corpus_path: str | None = None,
    cache_dir: str | None = None,
    max_bills: int | None = None,
) -> QABrowserRuntime:
    """Build the QA browser runtime for the current project root."""

    config = load_qa_config(project_root)
    emit_runtime_diagnostic("QA runtime config loaded")
    if corpus_path:
        config.corpus_path = corpus_path
    if cache_dir:
        config.index.cache_dir = cache_dir
    config.validate()

    provider_client = OpenAICompatibleClient(
        api_key=load_provider_api_key(config.provider),
        api_base_url=config.provider.api_base_url,
        embedding_model=config.models.embedding_model,
        answer_model=config.models.answer_model,
    )
    emit_runtime_diagnostic("QA provider client ready")
    filter_extractor = FilterExtractor(
        client=provider_client.openai_client,
        model=config.models.filter_extractor_model,
    )
    emit_runtime_diagnostic("QA filter extractor ready")
    local_answer_support = LocalAnswerSupport.from_project_root(project_root)
    emit_runtime_diagnostic("QA local answer support ready")
    answer_model_options = _build_answer_model_options(
        config.models.available_answer_models, local_answer_support
    )
    available_answer_models = tuple(
        answer_model_option.option_id for answer_model_option in answer_model_options
    )

    quadruplet_store = _load_quadruplet_store_from_config(project_root, config)
    emit_runtime_diagnostic(
        f"QA quadruplet store loaded: {quadruplet_store.total_quadruplets} records"
    )

    indexer = QAIndexer(
        project_root=project_root,
        config=config,
        provider_client=provider_client,
    )
    try:
        loaded_index = indexer.load_ready_index(bill_limit=max_bills)
        retriever = Retriever(loaded_index.chunks, loaded_index.embeddings)
        vector_search_backend: SearchBackend = VectorSearchBackend(
            retriever=retriever,
            embed_query=provider_client.embed_query,
        )
        planner_agent = _build_planner_agent(
            provider_client=provider_client,
            chunks=loaded_index.chunks,
            search_backend=vector_search_backend,
            worker_model=config.models.worker_model,
            agent_config=config.agent,
            quadruplet_store=quadruplet_store,
        )
        emit_runtime_diagnostic("QA planner agent ready (vector backend)")
        qa_service = QAService(
            retriever=retriever,
            planner_agent=planner_agent,
            filter_extractor=filter_extractor,
            retrieval_top_k=config.index.retrieval_top_k,
            default_answer_model=config.models.answer_model,
            available_answer_models=available_answer_models,
            answer_model_options=answer_model_options,
            local_answer_support=local_answer_support,
            capture_trace=config.app.show_trace,
        )
        return QABrowserRuntime(
            app=create_app(qa_service, show_trace=config.app.show_trace),
            qa_service=qa_service,
            provider_client=provider_client,
            local_answer_support=local_answer_support,
            retrieval_backend=_VECTOR_RETRIEVAL_BACKEND,
            chunk_count=loaded_index.manifest.total_chunks,
        )
    except IndexStateError:
        emit_runtime_diagnostic("QA vector cache unavailable; falling back to lexical retrieval")
        chunks = build_indexed_chunks(
            config.resolve_corpus_path(project_root),
            chunk_size=config.chunking.chunk_size,
            overlap=config.chunking.overlap,
            max_bills=max_bills,
        )
        lexical_retriever = LexicalRetriever(chunks)
        lexical_search_backend: SearchBackend = LexicalSearchBackend(retriever=lexical_retriever)
        planner_agent = _build_planner_agent(
            provider_client=provider_client,
            chunks=chunks,
            search_backend=lexical_search_backend,
            worker_model=config.models.worker_model,
            agent_config=config.agent,
            quadruplet_store=quadruplet_store,
        )
        emit_runtime_diagnostic("QA planner agent ready (lexical backend)")
        qa_service = QAService(
            retriever=None,
            lexical_retriever=lexical_retriever,
            planner_agent=planner_agent,
            filter_extractor=filter_extractor,
            retrieval_top_k=config.index.retrieval_top_k,
            default_answer_model=config.models.answer_model,
            available_answer_models=available_answer_models,
            answer_model_options=answer_model_options,
            local_answer_support=local_answer_support,
            capture_trace=config.app.show_trace,
        )
        return QABrowserRuntime(
            app=create_app(qa_service, show_trace=config.app.show_trace),
            qa_service=qa_service,
            provider_client=provider_client,
            local_answer_support=local_answer_support,
            retrieval_backend=_LEXICAL_RETRIEVAL_BACKEND,
            chunk_count=len(chunks),
        )


def _build_planner_agent(
    *,
    provider_client: OpenAICompatibleClient,
    chunks: Sequence[IndexedChunk],
    search_backend: SearchBackend,
    worker_model: str,
    agent_config,
    quadruplet_store: QuadrupletStore,
) -> PlannerAgent:
    """Build a ``PlannerAgent`` backed by the shared OpenAI-compatible client.

    The bill-level metadata index is prebuilt here so the one-shot JSONL
    sweep for a production ``ChunkStore`` happens at startup rather than
    on the first question. ``quadruplet_store`` is forwarded unchanged; an
    empty store disables the ``query_quadruplets`` tool without any other
    runtime impact.
    """

    bill_index = build_bill_index(chunks)
    return PlannerAgent(
        planner_client=provider_client.openai_client,
        worker_client=provider_client.openai_client,
        chunks=chunks,
        search_backend=search_backend,
        worker_model=worker_model,
        agent_config=agent_config,
        bill_index=bill_index,
        quadruplet_store=quadruplet_store,
    )


def _load_quadruplet_store_from_config(
    project_root: Path,
    config,
) -> QuadrupletStore:
    """Resolve the sidecar path from ``config`` and load the store.

    The sidecar is always placed next to the persisted QA cache so the
    same ``QA_CACHE_DIR`` environment variable that moves ``chunks.jsonl``
    to Render's persistent disk also moves ``quadruplets.jsonl``.
    """

    cache_dir = config.resolve_cache_dir(project_root)
    sidecar_path = cache_dir / _QUADRUPLET_SIDECAR_FILENAME
    return load_quadruplet_store(sidecar_path)


def _build_answer_model_options(
    available_answer_models: tuple[str, ...],
    local_answer_support: LocalAnswerSupport,
) -> tuple[AnswerModelOption, ...]:
    """Build the full answer-model option list for the QA runtime."""

    return tuple(
        AnswerModelOption(option_id=model_name, label=model_name)
        for model_name in available_answer_models
    ) + local_answer_support.answer_model_options


__all__ = ["QABrowserRuntime", "build_qa_browser_runtime"]
