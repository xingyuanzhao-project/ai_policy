"""Shared runtime wiring for the QA browser app.

- Centralizes QA service and Flask app construction for both the local CLI entry
  point and Render's Gunicorn entry point.
- Preserves the existing vector-first startup behavior while keeping the
  lexical fallback available when no ready index is present.
- Keeps provider client, local answer support, and answer-model option assembly
  in one place so deployment-specific entry points stay thin.
- Does not own command-line parsing, corpus indexing, or HTTP server binding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from flask import Flask

from .config import QAConfig, load_provider_api_key, load_qa_config
from .gemini_client import OpenAICompatibleClient
from .indexer import IndexStateError, QAIndexer, build_indexed_chunks
from .lexical_retriever import LexicalRetriever
from .local_answer_support import AnswerModelOption, LocalAnswerSupport
from .retriever import Retriever
from .service import QAService
from .web_app import create_app

_VECTOR_RETRIEVAL_BACKEND = "vector"
_LEXICAL_RETRIEVAL_BACKEND = "lexical"


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
    local_answer_support = LocalAnswerSupport.from_project_root(project_root)
    answer_model_options = _build_answer_model_options(config, local_answer_support)
    available_answer_models = tuple(
        answer_model_option.option_id for answer_model_option in answer_model_options
    )

    indexer = QAIndexer(
        project_root=project_root,
        config=config,
        provider_client=provider_client,
    )
    try:
        loaded_index = indexer.load_ready_index(bill_limit=max_bills)
        qa_service = QAService(
            retriever=Retriever(loaded_index.chunks, loaded_index.embeddings),
            provider_client=provider_client,
            retrieval_top_k=config.index.retrieval_top_k,
            default_answer_model=config.models.answer_model,
            available_answer_models=available_answer_models,
            answer_model_options=answer_model_options,
            local_answer_support=local_answer_support,
        )
        return QABrowserRuntime(
            app=create_app(qa_service),
            qa_service=qa_service,
            provider_client=provider_client,
            local_answer_support=local_answer_support,
            retrieval_backend=_VECTOR_RETRIEVAL_BACKEND,
            chunk_count=loaded_index.manifest.total_chunks,
        )
    except IndexStateError:
        chunks = build_indexed_chunks(
            config.resolve_corpus_path(project_root),
            chunk_size=config.chunking.chunk_size,
            overlap=config.chunking.overlap,
            max_bills=max_bills,
        )
        qa_service = QAService(
            retriever=None,
            lexical_retriever=LexicalRetriever(chunks),
            provider_client=provider_client,
            retrieval_top_k=config.index.retrieval_top_k,
            default_answer_model=config.models.answer_model,
            available_answer_models=available_answer_models,
            answer_model_options=answer_model_options,
            local_answer_support=local_answer_support,
        )
        return QABrowserRuntime(
            app=create_app(qa_service),
            qa_service=qa_service,
            provider_client=provider_client,
            local_answer_support=local_answer_support,
            retrieval_backend=_LEXICAL_RETRIEVAL_BACKEND,
            chunk_count=len(chunks),
        )


def _build_answer_model_options(
    config: QAConfig,
    local_answer_support: LocalAnswerSupport,
) -> tuple[AnswerModelOption, ...]:
    """Build the full answer-model option list for the QA runtime."""

    return tuple(
        AnswerModelOption(option_id=model_name, label=model_name)
        for model_name in config.models.available_answer_models
    ) + local_answer_support.answer_model_options


__all__ = ["QABrowserRuntime", "build_qa_browser_runtime"]
