"""Offline tests for typed QA results and retrieval ranking."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.qa.artifacts import AnswerResult, IndexedChunk
from src.qa.embedding_store import EmbeddingBatchSpec, EmbeddingStore
from src.qa.lexical_retriever import LexicalRetriever
from src.qa.local_answer_support import (
    AnswerModelOption,
    LocalAnswerSupport,
    LocalAnswerTarget,
)
from src.qa.retriever import Retriever
from src.qa.service import QAService

_DEFAULT_ANSWER_MODEL = "google/gemini-2.5-flash"
_ALTERNATE_ANSWER_MODEL = "anthropic/claude-haiku-4.5"


class FakeProviderClient:
    """Deterministic fake provider client used by QA service tests."""

    def __init__(self) -> None:
        self.embed_query_call_count = 0
        self.answer_call_count = 0
        self.last_answer_model: str | None = None

    def embed_query(self, question: str) -> np.ndarray:
        self.embed_query_call_count += 1
        lowered = question.lower()
        vector = np.array(
            [
                1.0 if "risk" in lowered or "impact" in lowered else 0.0,
                1.0 if "sandbox" in lowered or "innovation" in lowered else 0.0,
                1.0 if "penalty" in lowered or "fine" in lowered else 0.0,
            ],
            dtype=np.float32,
        )
        if float(np.linalg.norm(vector)) == 0.0:
            vector = np.array([1.0, 0.5, 0.25], dtype=np.float32)
        return vector / np.linalg.norm(vector)

    def generate_answer(
        self,
        question: str,
        retrieved_chunks,
        answer_model: str | None = None,
    ) -> str:
        self.answer_call_count += 1
        self.last_answer_model = answer_model
        return (
            f"The retrieved bill text indicates that {retrieved_chunks[0].text} "
            f"[{retrieved_chunks[0].rank}]"
        )


class FakeLocalAnswerClient:
    """Small fake local client used to verify answer-client routing."""

    def __init__(self) -> None:
        self.answer_call_count = 0
        self.last_answer_model: str | None = None

    def generate_answer(
        self,
        question: str,
        retrieved_chunks,
        answer_model: str | None = None,
    ) -> str:
        self.answer_call_count += 1
        self.last_answer_model = answer_model
        return f"Local answer for {retrieved_chunks[0].bill_id}. [{retrieved_chunks[0].rank}]"

    def close(self) -> None:
        """Mirror the real client close surface used by the helper."""


class QAServiceTests(unittest.TestCase):
    """Verify retrieval ranking and typed answer payloads."""

    def test_retriever_ranks_expected_chunk_first(self) -> None:
        """Verify the similarity search returns the most relevant chunk first."""

        retriever = Retriever(
            chunks=_make_chunks(),
            embeddings=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )

        results = retriever.retrieve(
            query_embedding=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            top_k=2,
        )

        self.assertEqual(results[0].bill_id, "BILL-002")
        self.assertGreaterEqual(results[0].score, results[1].score)

    def test_retriever_supports_streamed_embedding_batches(self) -> None:
        """Verify retrieval can rank results from persisted embedding batches."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            first_batch_path = temporary_root / "batch_00000.npy"
            second_batch_path = temporary_root / "batch_00001.npy"
            np.save(
                first_batch_path,
                np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
            )
            np.save(
                second_batch_path,
                np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            )
            retriever = Retriever(
                chunks=_make_chunks(),
                embeddings=EmbeddingStore(
                    batch_specs=(
                        EmbeddingBatchSpec(path=first_batch_path),
                        EmbeddingBatchSpec(path=second_batch_path),
                    ),
                    total_rows=3,
                    embedding_dimension=3,
                ),
            )

            results = retriever.retrieve(
                query_embedding=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                top_k=2,
            )

        self.assertEqual(results[0].bill_id, "BILL-002")
        self.assertGreaterEqual(results[0].score, results[1].score)

    def test_service_returns_typed_answer_result_with_citations(self) -> None:
        """Verify the QA service returns an AnswerResult rather than loose dicts."""

        service = QAService(
            retriever=Retriever(
                chunks=_make_chunks(),
                embeddings=np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            ),
            provider_client=FakeProviderClient(),
            retrieval_top_k=2,
            default_answer_model=_DEFAULT_ANSWER_MODEL,
            available_answer_models=(
                _DEFAULT_ANSWER_MODEL,
                _ALTERNATE_ANSWER_MODEL,
            ),
        )

        result = service.answer_question("What risk documentation is required?")

        self.assertIsInstance(result, AnswerResult)
        self.assertEqual(result.question, "What risk documentation is required?")
        self.assertEqual(result.answer_model, _DEFAULT_ANSWER_MODEL)
        self.assertEqual(len(result.citations), 2)
        self.assertEqual(result.citations[0].bill_id, "BILL-001")
        self.assertIn("[1]", result.answer)
        round_trip = AnswerResult.from_dict(result.to_dict())
        self.assertEqual(round_trip.citations[0].chunk_id, result.citations[0].chunk_id)
        self.assertEqual(round_trip.answer_model, result.answer_model)

    def test_service_rejects_unknown_answer_model(self) -> None:
        """Verify answer-model validation happens before provider inference."""

        service = QAService(
            retriever=Retriever(
                chunks=_make_chunks(),
                embeddings=np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            ),
            provider_client=FakeProviderClient(),
            retrieval_top_k=2,
            default_answer_model=_DEFAULT_ANSWER_MODEL,
            available_answer_models=(
                _DEFAULT_ANSWER_MODEL,
                _ALTERNATE_ANSWER_MODEL,
            ),
        )

        with self.assertRaises(ValueError):
            service.answer_question(
                "What risk documentation is required?",
                answer_model="not-a-real-model",
            )

    def test_service_supports_lexical_retrieval_backend(self) -> None:
        """Verify the QA service can answer over the full local lexical retriever."""

        service = QAService(
            retriever=None,
            lexical_retriever=LexicalRetriever(_make_chunks()),
            provider_client=FakeProviderClient(),
            retrieval_top_k=2,
            default_answer_model=_DEFAULT_ANSWER_MODEL,
            available_answer_models=(
                _DEFAULT_ANSWER_MODEL,
                _ALTERNATE_ANSWER_MODEL,
            ),
        )

        result = service.answer_question("Which bill creates an innovation sandbox?")

        self.assertEqual(result.answer_model, _DEFAULT_ANSWER_MODEL)
        self.assertEqual(result.citations[0].bill_id, "BILL-002")

    def test_service_routes_selected_local_answer_model_through_local_support(self) -> None:
        """Verify local answer-model ids are routed through the local helper only."""

        remote_client = FakeProviderClient()
        local_client = FakeLocalAnswerClient()
        local_option_id = "local::hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
        local_label = "Local / hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
        local_answer_support = LocalAnswerSupport(
            [
                LocalAnswerTarget(
                    option=AnswerModelOption(
                        option_id=local_option_id,
                        label=local_label,
                    ),
                    raw_model_name="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
                    client=local_client,
                )
            ]
        )
        service = QAService(
            retriever=Retriever(
                chunks=_make_chunks(),
                embeddings=np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            ),
            provider_client=remote_client,
            retrieval_top_k=2,
            default_answer_model=_DEFAULT_ANSWER_MODEL,
            available_answer_models=(
                _DEFAULT_ANSWER_MODEL,
                local_option_id,
            ),
            answer_model_options=(
                AnswerModelOption(
                    option_id=_DEFAULT_ANSWER_MODEL,
                    label=_DEFAULT_ANSWER_MODEL,
                ),
                AnswerModelOption(
                    option_id=local_option_id,
                    label=local_label,
                ),
            ),
            local_answer_support=local_answer_support,
        )

        result = service.answer_question(
            "What risk documentation is required?",
            answer_model=local_option_id,
        )

        self.assertEqual(remote_client.embed_query_call_count, 1)
        self.assertEqual(remote_client.answer_call_count, 0)
        self.assertEqual(local_client.answer_call_count, 1)
        self.assertEqual(
            local_client.last_answer_model,
            "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        )
        self.assertEqual(result.answer_model, local_label)


def _make_chunks() -> list[IndexedChunk]:
    """Build indexed chunk fixtures aligned to the retrieval matrix."""

    return [
        IndexedChunk(
            chunk_id=1,
            bill_id="BILL-001",
            text="Risk documentation must be published for covered systems.",
            start_offset=0,
            end_offset=58,
            state="CA",
            title="Risk Disclosure Act",
            status="Introduced",
        ),
        IndexedChunk(
            chunk_id=2,
            bill_id="BILL-002",
            text="Innovation sandboxes may allow a temporary pilot program.",
            start_offset=0,
            end_offset=58,
            state="TX",
            title="Sandbox Pilot Act",
            status="Engrossed",
        ),
        IndexedChunk(
            chunk_id=3,
            bill_id="BILL-003",
            text="Civil penalties apply after repeated disclosure failures.",
            start_offset=0,
            end_offset=56,
            state="NY",
            title="Penalty Enforcement Act",
            status="Passed Senate",
        ),
    ]


if __name__ == "__main__":
    unittest.main()
