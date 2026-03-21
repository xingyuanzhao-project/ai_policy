"""Typed QA package for the local bills RAG app."""

from .artifacts import (
    AnswerResult,
    INDEX_STATUS_BUILDING,
    INDEX_STATUS_READY,
    IndexManifest,
    IndexedChunk,
    QAArtifactValidationError,
    RetrievedChunk,
    validate_answer_result,
    validate_index_manifest,
    validate_indexed_chunk,
    validate_retrieved_chunk,
)
from .config import (
    ModelConfig,
    ProviderConfig,
    QAAppConfig,
    QAChunkingConfig,
    QAConfig,
    QAConfigValidationError,
    QAIndexConfig,
    load_provider_api_key,
    load_qa_config,
)
from .gemini_client import GeminiClient, OpenAICompatibleClient
from .indexer import IndexStateError, LoadedIndex, QAIndexer, build_indexed_chunks
from .lexical_retriever import LexicalRetriever
from .local_answer_support import AnswerModelOption, LocalAnswerSupport, LocalAnswerTarget
from .retriever import Retriever
from .runtime import QABrowserRuntime, build_qa_browser_runtime
from .secrets import SecretStoreError, load_secret, save_secret
from .service import QAService
from .web_app import create_app

__all__ = [
    "AnswerResult",
    "AnswerModelOption",
    "GeminiClient",
    "INDEX_STATUS_BUILDING",
    "INDEX_STATUS_READY",
    "IndexManifest",
    "IndexStateError",
    "IndexedChunk",
    "LexicalRetriever",
    "LocalAnswerSupport",
    "LocalAnswerTarget",
    "LoadedIndex",
    "ModelConfig",
    "OpenAICompatibleClient",
    "ProviderConfig",
    "QAAppConfig",
    "QAArtifactValidationError",
    "QABrowserRuntime",
    "QAChunkingConfig",
    "QAConfig",
    "QAConfigValidationError",
    "QAIndexConfig",
    "QAIndexer",
    "QAService",
    "RetrievedChunk",
    "Retriever",
    "SecretStoreError",
    "build_qa_browser_runtime",
    "build_indexed_chunks",
    "create_app",
    "load_provider_api_key",
    "load_qa_config",
    "load_secret",
    "save_secret",
    "validate_answer_result",
    "validate_index_manifest",
    "validate_indexed_chunk",
    "validate_retrieved_chunk",
]
