"""Offline tests for QA indexing, manifest validation, and resume behavior."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.ner.schemas.artifacts import BillRecord
from src.qa.artifacts import INDEX_STATUS_READY, IndexManifest
from src.qa.config import (
    ModelConfig,
    ProviderConfig,
    QAAppConfig,
    QAChunkingConfig,
    QAConfig,
    QAIndexConfig,
)
from src.qa.indexer import IndexStateError, QAIndexer


class FakeGeminiIndexerClient:
    """Deterministic fake embedding client used by offline QA index tests."""

    def __init__(
        self,
        fail_on_call: int | None = None,
        retryable_failures: int = 0,
        retryable_parse_failures: int = 0,
        retryable_body_failures: int = 0,
    ) -> None:
        self.fail_on_call = fail_on_call
        self.retryable_failures = retryable_failures
        self.retryable_parse_failures = retryable_parse_failures
        self.retryable_body_failures = retryable_body_failures
        self.document_calls = 0

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        self.document_calls += 1
        if self.retryable_failures > 0:
            self.retryable_failures -= 1
            raise RetryableQuotaError()
        if self.retryable_body_failures > 0:
            self.retryable_body_failures -= 1
            raise RetryableBodyQuotaError()
        if self.retryable_parse_failures > 0:
            self.retryable_parse_failures -= 1
            raise TypeError("'NoneType' object is not iterable")
        if self.fail_on_call is not None and self.document_calls == self.fail_on_call:
            raise RuntimeError("simulated Gemini interruption")
        return [self._vector_for_text(text) for text in texts]

    def _vector_for_text(self, text: str) -> np.ndarray:
        lowered = text.lower()
        vector = np.array(
            [
                lowered.count("risk") + lowered.count("impact"),
                lowered.count("sandbox") + lowered.count("innovation"),
                lowered.count("penalty") + lowered.count("fine"),
            ],
            dtype=np.float32,
        )
        if float(np.linalg.norm(vector)) == 0.0:
            vector = np.array([1.0, 0.5, 0.25], dtype=np.float32)
        return vector / np.linalg.norm(vector)


class QAIndexerTests(unittest.TestCase):
    """Verify the local QA index cache and manifest behavior."""

    def test_build_or_resume_persists_ready_manifest_and_embeddings(self) -> None:
        """Verify a fresh build writes a ready manifest, chunks, and embeddings."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            corpus_path = _write_bill_jsonl(
                project_root,
                [
                    _make_bill(
                        bill_id="BILL-001",
                        text=(
                            "Risk assessments are required for covered AI systems. "
                            "The documentation must explain the impact review."
                        ),
                    ),
                    _make_bill(
                        bill_id="BILL-002",
                        text=(
                            "Innovation sandboxes may allow a temporary safe harbor. "
                            "The sandbox program encourages experimentation."
                        ),
                    ),
                ],
            )
            config = _make_config()

            loaded_index = QAIndexer(
                project_root=project_root,
                config=config,
                provider_client=FakeGeminiIndexerClient(),
            ).build_or_resume()

            self.assertEqual(loaded_index.manifest.status, INDEX_STATUS_READY)
            self.assertEqual(Path(loaded_index.manifest.corpus_path), corpus_path.resolve())
            self.assertGreater(loaded_index.manifest.total_chunks, 0)
            self.assertEqual(len(loaded_index.chunks), loaded_index.embeddings.shape[0])
            self.assertTrue((project_root / "data" / "qa_cache" / "manifest.json").exists())
            self.assertTrue((project_root / "data" / "qa_cache" / "chunks.json").exists())

    def test_load_ready_index_detects_stale_manifest_when_corpus_changes(self) -> None:
        """Verify manifest validation rejects a stale index after corpus drift."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            corpus_path = _write_bill_jsonl(
                project_root,
                [_make_bill(text="Risk documentation is required for public agency AI systems.")],
            )
            config = _make_config()
            indexer = QAIndexer(
                project_root=project_root,
                config=config,
                provider_client=FakeGeminiIndexerClient(),
            )
            indexer.build_or_resume()

            with open(corpus_path, "a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        _make_bill(
                            bill_id="BILL-999",
                            text="Penalty notices apply when required disclosures are omitted.",
                        ).to_dict(),
                        ensure_ascii=False,
                    )
                )
                handle.write("\n")

            with self.assertRaises(IndexStateError):
                indexer.load_ready_index()

    def test_load_ready_index_accepts_portable_manifest_on_new_project_root(self) -> None:
        """Verify a ready QA cache can move to a new absolute corpus path."""

        with (
            tempfile.TemporaryDirectory() as source_directory,
            tempfile.TemporaryDirectory() as destination_directory,
        ):
            source_root = Path(source_directory)
            destination_root = Path(destination_directory)
            source_corpus_path = _write_bill_jsonl(
                source_root,
                [
                    _make_bill(
                        text=(
                            "Impact assessments are required for covered AI systems "
                            "and public disclosures must be retained."
                        )
                    )
                ],
            )
            config = _make_config()
            QAIndexer(
                project_root=source_root,
                config=config,
                provider_client=FakeGeminiIndexerClient(),
            ).build_or_resume()

            shutil.copytree(source_root / "data", destination_root / "data")
            loaded_index = QAIndexer(
                project_root=destination_root,
                config=config,
            ).load_ready_index()

            self.assertEqual(loaded_index.manifest.status, INDEX_STATUS_READY)
            self.assertEqual(
                Path(loaded_index.manifest.corpus_path),
                source_corpus_path.resolve(),
            )
            self.assertEqual(
                len(loaded_index.chunks),
                loaded_index.embeddings.shape[0],
            )

    def test_build_or_resume_skips_finished_batches_after_interruption(self) -> None:
        """Verify resumable builds do not repeat already-persisted embedding batches."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            _write_bill_jsonl(
                project_root,
                [
                    _make_bill(
                        bill_id="BILL-001",
                        text=(
                            "Risk documentation is required. "
                            "Impact reports must be published. "
                            "Penalty notices apply for missing reports."
                        ),
                    )
                ],
            )
            config = _make_config(batch_size=1, chunk_size=48, overlap=8)
            interrupted_client = FakeGeminiIndexerClient(fail_on_call=2)
            interrupted_indexer = QAIndexer(
                project_root=project_root,
                config=config,
                provider_client=interrupted_client,
            )

            with self.assertRaises(RuntimeError):
                interrupted_indexer.build_or_resume()

            manifest_path = project_root / "data" / "qa_cache" / "manifest.json"
            manifest = IndexManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
            self.assertEqual(manifest.completed_batch_count, 1)

            resumed_client = FakeGeminiIndexerClient()
            resumed_index = QAIndexer(
                project_root=project_root,
                config=config,
                provider_client=resumed_client,
            ).build_or_resume()

            expected_remaining_batches = resumed_index.manifest.completed_batch_count - 1
            self.assertEqual(resumed_client.document_calls, expected_remaining_batches)

    def test_build_retries_retryable_quota_errors(self) -> None:
        """Verify transient Gemini quota errors are retried during index builds."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            _write_bill_jsonl(
                project_root,
                [
                    _make_bill(
                        text=(
                            "Risk documentation is required for covered systems, "
                            "and the impact analysis must be filed."
                        ),
                    )
                ],
            )
            retrying_client = FakeGeminiIndexerClient(retryable_failures=1)

            loaded_index = QAIndexer(
                project_root=project_root,
                config=_make_config(batch_size=8, chunk_size=256, overlap=32),
                provider_client=retrying_client,
            ).build_or_resume()

            self.assertEqual(retrying_client.document_calls, 2)
            self.assertEqual(loaded_index.manifest.status, INDEX_STATUS_READY)

    def test_build_retries_transient_embedding_parse_errors(self) -> None:
        """Verify transient provider parse errors are retried during index builds."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            _write_bill_jsonl(
                project_root,
                [
                    _make_bill(
                        text=(
                            "Impact reports are required and enforcement notices "
                            "apply when agencies omit them."
                        ),
                    )
                ],
            )
            retrying_client = FakeGeminiIndexerClient(retryable_parse_failures=1)

            loaded_index = QAIndexer(
                project_root=project_root,
                config=_make_config(batch_size=8, chunk_size=256, overlap=32),
                provider_client=retrying_client,
            ).build_or_resume()

            self.assertEqual(retrying_client.document_calls, 2)
            self.assertEqual(loaded_index.manifest.status, INDEX_STATUS_READY)

    def test_build_honors_retry_delay_from_openai_error_body(self) -> None:
        """Verify retry delay parsing works for OpenAI-compatible error bodies."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            _write_bill_jsonl(
                project_root,
                [
                    _make_bill(
                        text=(
                            "Covered deployers must keep impact assessments and "
                            "disclosure records."
                        )
                    )
                ],
            )
            retrying_client = FakeGeminiIndexerClient(retryable_body_failures=1)
            with unittest.mock.patch("src.qa.indexer.time.sleep") as sleep_mock:
                loaded_index = QAIndexer(
                    project_root=project_root,
                    config=_make_config(batch_size=8, chunk_size=256, overlap=32),
                    provider_client=retrying_client,
                ).build_or_resume()

            sleep_mock.assert_called_once_with(5.0)
            self.assertEqual(retrying_client.document_calls, 2)
            self.assertEqual(loaded_index.manifest.status, INDEX_STATUS_READY)


def _make_bill(
    bill_id: str = "BILL-001",
    text: str = "Artificial intelligence impact assessments are required.",
) -> BillRecord:
    """Build a representative bill record fixture for QA index tests."""

    return BillRecord(
        bill_id=bill_id,
        state="CA",
        bill_url="https://example.com/bill",
        title=f"{bill_id} Title",
        status="Introduced",
        summary="Summary",
        text=text,
    )


def _make_config(
    *,
    batch_size: int = 2,
    chunk_size: int = 64,
    overlap: int = 12,
) -> QAConfig:
    """Build an in-memory QA config fixture."""

    return QAConfig(
        corpus_path="data/ncsl/us_ai_legislation_ncsl_text.jsonl",
        chunking=QAChunkingConfig(chunk_size=chunk_size, overlap=overlap),
        index=QAIndexConfig(
            cache_dir="data/qa_cache",
            batch_size=batch_size,
            retrieval_top_k=3,
        ),
        provider=ProviderConfig(
            api_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key_env_var="GEMINI_API_KEY",
            keyring_service="ai_policy.qa",
            keyring_username="gemini",
        ),
        models=ModelConfig(
            embedding_model="fake-embedding-model",
            answer_model="fake-answer-model",
            available_answer_models=("fake-answer-model",),
        ),
        app=QAAppConfig(host="127.0.0.1", port=5050),
    )


def _write_bill_jsonl(project_root: Path, bills: list[BillRecord]) -> Path:
    """Write bill fixtures into the expected repo-style JSONL location."""

    corpus_path = project_root / "data" / "ncsl" / "us_ai_legislation_ncsl_text.jsonl"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with open(corpus_path, "w", encoding="utf-8") as handle:
        for bill in bills:
            handle.write(json.dumps(bill.to_dict(), ensure_ascii=False))
            handle.write("\n")
    return corpus_path


class RetryableQuotaError(Exception):
    """Small fake Gemini error that exercises retry behavior."""

    code = 429
    status = "RESOURCE_EXHAUSTED"
    details = {"error": {"details": [{"retryDelay": "0s"}]}}


class RetryableBodyQuotaError(Exception):
    """Small fake OpenAI-style quota error that exposes retry info via body."""

    code = 429
    status = None
    body = {"error": {"details": [{"retryDelay": "5s"}]}}


if __name__ == "__main__":
    unittest.main()
