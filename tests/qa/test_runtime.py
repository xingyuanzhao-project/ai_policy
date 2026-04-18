"""Offline tests for shared QA runtime construction."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.qa.artifacts import INDEX_STATUS_READY, IndexManifest, IndexedChunk
from src.qa.config import (
    AgentConfig,
    ModelConfig,
    ProviderConfig,
    QAAppConfig,
    QAChunkingConfig,
    QAConfig,
    QAIndexConfig,
)
from src.qa.indexer import IndexStateError, LoadedIndex
from src.qa.local_answer_support import LocalAnswerSupport
from src.qa.runtime import build_qa_browser_runtime

_PROVIDER_API_BASE_URL = "https://openrouter.ai/api/v1"
_EMBEDDING_MODEL = "openai/text-embedding-3-small"
_DEFAULT_ANSWER_MODEL = "google/gemini-2.5-flash"
_ALTERNATE_ANSWER_MODEL = "anthropic/claude-haiku-4.5"


class QARuntimeTests(unittest.TestCase):
    """Verify the shared runtime preserves the intended startup behavior."""

    def test_build_runtime_prefers_ready_vector_index(self) -> None:
        """Verify a ready cache yields vector retrieval instead of lexical fallback."""

        provider_client = MagicMock()
        with (
            patch("src.qa.runtime.load_qa_config", return_value=_make_config()),
            patch("src.qa.runtime.load_provider_api_key", return_value="secret"),
            patch(
                "src.qa.runtime.OpenAICompatibleClient",
                return_value=provider_client,
            ),
            patch(
                "src.qa.runtime.LocalAnswerSupport.from_project_root",
                return_value=LocalAnswerSupport(),
            ),
            patch(
                "src.qa.runtime.QAIndexer.load_ready_index",
                return_value=LoadedIndex(
                    manifest=_make_manifest(total_chunks=1),
                    chunks=[_make_chunk()],
                    embeddings=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
                ),
            ),
            patch("src.qa.runtime.build_indexed_chunks") as mock_build_indexed_chunks,
        ):
            runtime = build_qa_browser_runtime(Path("C:/project"))

        self.assertEqual(runtime.retrieval_backend, "vector")
        self.assertEqual(runtime.chunk_count, 1)
        mock_build_indexed_chunks.assert_not_called()
        runtime.close()
        provider_client.close.assert_called_once()

    def test_build_runtime_falls_back_to_lexical_retrieval(self) -> None:
        """Verify startup degrades to lexical retrieval when no ready cache exists."""

        provider_client = MagicMock()
        chunk = _make_chunk()
        with (
            patch("src.qa.runtime.load_qa_config", return_value=_make_config()),
            patch("src.qa.runtime.load_provider_api_key", return_value="secret"),
            patch(
                "src.qa.runtime.OpenAICompatibleClient",
                return_value=provider_client,
            ),
            patch(
                "src.qa.runtime.LocalAnswerSupport.from_project_root",
                return_value=LocalAnswerSupport(),
            ),
            patch(
                "src.qa.runtime.QAIndexer.load_ready_index",
                side_effect=IndexStateError("missing cache"),
            ),
            patch(
                "src.qa.runtime.build_indexed_chunks",
                return_value=[chunk],
            ) as mock_build_indexed_chunks,
        ):
            runtime = build_qa_browser_runtime(
                Path("C:/project"),
                corpus_path="custom-corpus.jsonl",
                cache_dir="custom-cache",
                max_bills=3,
            )

        self.assertEqual(runtime.retrieval_backend, "lexical")
        self.assertEqual(runtime.chunk_count, 1)
        mock_build_indexed_chunks.assert_called_once()
        self.assertIsNone(runtime.qa_service._retriever)
        self.assertIsNotNone(runtime.qa_service._lexical_retriever)
        runtime.close()
        provider_client.close.assert_called_once()


def _make_chunk() -> IndexedChunk:
    """Build one representative indexed chunk fixture."""

    return IndexedChunk(
        chunk_id=1,
        bill_id="BILL-001",
        text="Impact assessments are required for covered systems.",
        start_offset=0,
        end_offset=52,
        state="CA",
        title="AI Accountability Act",
        status="Introduced",
    )


def _make_config() -> QAConfig:
    """Build a minimal in-memory QA config for runtime tests."""

    return QAConfig(
        corpus_path="data/ncsl/us_ai_legislation_ncsl_text.jsonl",
        chunking=QAChunkingConfig(chunk_size=64, overlap=12),
        index=QAIndexConfig(
            cache_dir="data/qa_cache",
            batch_size=2,
            retrieval_top_k=3,
        ),
        provider=ProviderConfig(
            api_base_url=_PROVIDER_API_BASE_URL,
            api_key_env_var="OPENROUTER_API_KEY",
            keyring_service="ai_policy.qa",
            keyring_username="openrouter",
        ),
        models=ModelConfig(
            embedding_model=_EMBEDDING_MODEL,
            answer_model=_DEFAULT_ANSWER_MODEL,
            available_answer_models=(_DEFAULT_ANSWER_MODEL, _ALTERNATE_ANSWER_MODEL),
            filter_extractor_model=_DEFAULT_ANSWER_MODEL,
            worker_model=_DEFAULT_ANSWER_MODEL,
        ),
        app=QAAppConfig(host="127.0.0.1", port=5050),
        agent=AgentConfig(
            max_planner_turns=4,
            max_planner_tokens=512,
            planner_temperature=0.0,
            max_worker_tokens=256,
            worker_temperature=0.0,
            max_tool_calls=8,
            max_worker_calls=3,
            max_bills_per_list=10,
            max_chunks_per_bill=3,
            max_citations_per_bill=2,
        ),
    )


def _make_manifest(total_chunks: int) -> IndexManifest:
    """Build a ready manifest fixture for runtime tests."""

    return IndexManifest(
        index_format_version=2,
        status=INDEX_STATUS_READY,
        corpus_path="C:/tmp/us_ai_legislation_ncsl_text.jsonl",
        corpus_fingerprint="abc123",
        chunk_size=64,
        overlap=12,
        provider_api_base_url=_PROVIDER_API_BASE_URL,
        embedding_model=_EMBEDDING_MODEL,
        batch_size=2,
        total_chunks=total_chunks,
        completed_batch_count=1,
        built_at_utc="2026-03-21T00:00:00+00:00",
        bill_limit=None,
    )


if __name__ == "__main__":
    unittest.main()
