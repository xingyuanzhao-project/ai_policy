"""Offline tests for the planner-driven QA service."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.qa.artifacts import AnswerResult, IndexedChunk, RetrievedChunk
from src.qa.embedding_store import EmbeddingBatchSpec, EmbeddingStore
from src.qa.filter_extractor import ExtractedQuery
from src.qa.lexical_retriever import LexicalRetriever
from src.qa.local_answer_support import (
    AnswerModelOption,
    LocalAnswerSupport,
    LocalAnswerTarget,
)
from src.qa.planner_agent import PlannerAnswer
from src.qa.retriever import Retriever
from src.qa.service import QAService

_DEFAULT_ANSWER_MODEL = "google/gemini-2.5-flash"
_ALTERNATE_ANSWER_MODEL = "anthropic/claude-haiku-4.5"


@dataclass
class _RecordedPlannerCall:
    """Captured arguments from one fake planner call."""

    question: str
    semantic_query: str
    initial_filters: dict[str, Any]
    available_filter_values: dict[str, list]
    planner_model: str
    citation_cap: int
    worker_model: str | None = None


class FakePlannerAgent:
    """Record ``PlannerAgent.answer`` invocations and return a preset payload."""

    def __init__(
        self,
        *,
        answer_text: str = "The planner answer text.",
        citations: list[RetrievedChunk] | None = None,
    ) -> None:
        self._answer_text = answer_text
        self._citations = citations or []
        self.call_count = 0
        self.calls: list[_RecordedPlannerCall] = []

    def answer(
        self,
        question: str,
        *,
        semantic_query: str,
        initial_filters: dict[str, Any],
        available_filter_values: dict[str, list],
        planner_model: str,
        citation_cap: int,
        capture_trace: bool = False,
        worker_model: str | None = None,
    ) -> PlannerAnswer:
        """Record the invocation and return the canned answer payload."""

        self.call_count += 1
        self.calls.append(
            _RecordedPlannerCall(
                question=question,
                semantic_query=semantic_query,
                initial_filters=dict(initial_filters),
                available_filter_values=dict(available_filter_values),
                planner_model=planner_model,
                citation_cap=citation_cap,
                worker_model=worker_model,
            )
        )
        return PlannerAnswer(
            answer_text=self._answer_text,
            citations=list(self._citations),
            planner_turns=1,
            tool_calls=2,
            worker_calls=0,
            routing_path="agent",
        )


@dataclass
class FakeFilterExtractor:
    """Return a preset ``ExtractedQuery`` and track invocations."""

    extracted: ExtractedQuery
    raise_exc: Exception | None = None
    call_count: int = 0
    last_question: str | None = None
    last_available: dict | None = field(default=None)

    def extract(
        self,
        question: str,
        available_filter_values: dict,
    ) -> ExtractedQuery:
        """Record the invocation and return the canned extraction payload."""

        self.call_count += 1
        self.last_question = question
        self.last_available = available_filter_values
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.extracted


class FakeLocalAnswerClient:
    """Marker-only local answer-client stub used in rejection tests."""

    def close(self) -> None:
        """Mirror the real client close surface used by the helper."""


class RetrieverRankingTests(unittest.TestCase):
    """Verify retriever ranking remains correct under the new service wiring."""

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


class QAServicePlannerTests(unittest.TestCase):
    """Verify QAService always delegates to the planner and surfaces its payload."""

    def test_service_returns_typed_answer_result_with_planner_citations(self) -> None:
        """Verify the service returns an ``AnswerResult`` sourced from the planner."""

        planner_citations = [_make_retrieved_chunk("BILL-001")]
        planner = FakePlannerAgent(
            answer_text="Planner-generated answer. [1]",
            citations=planner_citations,
        )
        extractor = FakeFilterExtractor(
            extracted=ExtractedQuery(semantic_query="risk documentation", filters={}),
        )
        service = _build_vector_service(
            planner=planner,
            extractor=extractor,
        )

        result = service.answer_question("What risk documentation is required?")

        self.assertIsInstance(result, AnswerResult)
        self.assertEqual(result.question, "What risk documentation is required?")
        self.assertEqual(result.answer, "Planner-generated answer. [1]")
        self.assertEqual(result.answer_model, _DEFAULT_ANSWER_MODEL)
        self.assertEqual(len(result.citations), 1)
        self.assertEqual(result.citations[0].bill_id, "BILL-001")
        self.assertEqual(result.applied_filters.get("routing_path"), "agent")
        self.assertEqual(extractor.call_count, 1)
        self.assertEqual(planner.call_count, 1)
        round_trip = AnswerResult.from_dict(result.to_dict())
        self.assertEqual(round_trip.citations[0].chunk_id, result.citations[0].chunk_id)

    def test_service_rejects_unknown_answer_model(self) -> None:
        """Verify answer-model validation happens before planner invocation."""

        planner = FakePlannerAgent()
        extractor = FakeFilterExtractor(
            extracted=ExtractedQuery(semantic_query="anything", filters={}),
        )
        service = _build_vector_service(planner=planner, extractor=extractor)

        with self.assertRaises(ValueError):
            service.answer_question(
                "What risk documentation is required?",
                answer_model="not-a-real-model",
            )
        self.assertEqual(planner.call_count, 0)

    def test_service_supports_lexical_retrieval_backend(self) -> None:
        """Verify the service runs the lexical backend when no vector index is present."""

        planner = FakePlannerAgent(
            answer_text="Lexical path answer.",
            citations=[_make_retrieved_chunk("BILL-002")],
        )
        extractor = FakeFilterExtractor(
            extracted=ExtractedQuery(
                semantic_query="innovation sandbox",
                filters={},
            ),
        )
        service = QAService(
            retriever=None,
            lexical_retriever=LexicalRetriever(_make_chunks()),
            planner_agent=planner,
            filter_extractor=extractor,
            retrieval_top_k=2,
            default_answer_model=_DEFAULT_ANSWER_MODEL,
            available_answer_models=(_DEFAULT_ANSWER_MODEL, _ALTERNATE_ANSWER_MODEL),
        )

        result = service.answer_question("Which bill creates an innovation sandbox?")

        self.assertEqual(result.answer, "Lexical path answer.")
        self.assertEqual(result.answer_model, _DEFAULT_ANSWER_MODEL)
        self.assertEqual(result.citations[0].bill_id, "BILL-002")
        self.assertEqual(planner.call_count, 1)

    def test_service_rejects_local_answer_model_selection(self) -> None:
        """Verify local answer models are rejected because they cannot drive tool calls."""

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
                    client=FakeLocalAnswerClient(),
                )
            ]
        )
        planner = FakePlannerAgent()
        extractor = FakeFilterExtractor(
            extracted=ExtractedQuery(semantic_query="anything", filters={}),
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
            planner_agent=planner,
            filter_extractor=extractor,
            retrieval_top_k=2,
            default_answer_model=_DEFAULT_ANSWER_MODEL,
            available_answer_models=(_DEFAULT_ANSWER_MODEL, local_option_id),
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

        with self.assertRaises(ValueError):
            service.answer_question(
                "What risk documentation is required?",
                answer_model=local_option_id,
            )
        self.assertEqual(planner.call_count, 0)

    def test_service_invokes_extractor_when_filters_unset(self) -> None:
        """Verify the extractor runs when callers omit filters and seeds the planner."""

        extractor = FakeFilterExtractor(
            extracted=ExtractedQuery(
                semantic_query="impact disclosure",
                filters={"state": "CA"},
            )
        )
        planner = FakePlannerAgent(
            citations=[_make_retrieved_chunk("BILL-CA")],
        )
        service = _build_vector_service(planner=planner, extractor=extractor)

        result = service.answer_question("tell me about California impact disclosure")

        self.assertEqual(extractor.call_count, 1)
        self.assertEqual(extractor.last_question, "tell me about California impact disclosure")
        self.assertEqual(planner.call_count, 1)
        call = planner.calls[0]
        self.assertEqual(call.semantic_query, "impact disclosure")
        self.assertEqual(call.initial_filters, {"state": "CA"})
        self.assertEqual(
            result.applied_filters,
            {"state": "CA", "routing_path": "agent"},
        )

    def test_service_bypasses_extractor_when_filters_passed_explicitly(self) -> None:
        """Verify explicit filter callers skip the extractor and reuse the question verbatim."""

        extractor = FakeFilterExtractor(
            extracted=ExtractedQuery(semantic_query="should-not-run", filters={"state": "NY"})
        )
        planner = FakePlannerAgent()
        service = _build_vector_service(planner=planner, extractor=extractor)

        result = service.answer_question(
            "what risk documentation is required?",
            filters={"state": "TX"},
        )

        self.assertEqual(extractor.call_count, 0)
        self.assertEqual(planner.call_count, 1)
        call = planner.calls[0]
        self.assertEqual(call.semantic_query, "what risk documentation is required?")
        self.assertEqual(call.initial_filters, {"state": "TX"})
        self.assertEqual(
            result.applied_filters,
            {"state": "TX", "routing_path": "agent"},
        )

    def test_service_falls_back_when_extractor_returns_empty(self) -> None:
        """Verify an extractor with no filters still produces a planner answer."""

        extractor = FakeFilterExtractor(
            extracted=ExtractedQuery(
                semantic_query="what risk documentation is required?",
                filters={},
            )
        )
        planner = FakePlannerAgent(
            citations=[_make_retrieved_chunk("BILL-X"), _make_retrieved_chunk("BILL-Y")],
        )
        service = _build_vector_service(planner=planner, extractor=extractor)

        result = service.answer_question("what risk documentation is required?")

        self.assertEqual(extractor.call_count, 1)
        self.assertEqual(planner.call_count, 1)
        self.assertEqual(result.applied_filters, {"routing_path": "agent"})
        self.assertEqual(len(result.citations), 2)

    def test_service_forwards_selected_answer_model_to_planner(self) -> None:
        """Verify the service uses the caller's answer model as the planner model."""

        extractor = FakeFilterExtractor(
            extracted=ExtractedQuery(semantic_query="q", filters={}),
        )
        planner = FakePlannerAgent()
        service = _build_vector_service(planner=planner, extractor=extractor)

        service.answer_question(
            "What risk documentation is required?",
            answer_model=_ALTERNATE_ANSWER_MODEL,
        )

        self.assertEqual(planner.call_count, 1)
        self.assertEqual(planner.calls[0].planner_model, _ALTERNATE_ANSWER_MODEL)

    def test_service_forwards_selected_answer_model_as_worker_model(self) -> None:
        """Selected dropdown answer model must also drive the worker model.

        The single UI model dropdown controls both the planner and the
        summarize/compare workers so one choice swaps both halves of the
        agent. This keeps the YAML ``worker_model`` as a default that users
        override per-request by picking a different model.
        """

        extractor = FakeFilterExtractor(
            extracted=ExtractedQuery(semantic_query="q", filters={}),
        )
        planner = FakePlannerAgent()
        service = _build_vector_service(planner=planner, extractor=extractor)

        service.answer_question(
            "What risk documentation is required?",
            answer_model=_ALTERNATE_ANSWER_MODEL,
        )

        self.assertEqual(planner.call_count, 1)
        self.assertEqual(planner.calls[0].planner_model, _ALTERNATE_ANSWER_MODEL)
        self.assertEqual(planner.calls[0].worker_model, _ALTERNATE_ANSWER_MODEL)

    def test_service_defaults_worker_model_when_answer_model_unset(self) -> None:
        """Worker model still follows the planner selection when no override is passed."""

        extractor = FakeFilterExtractor(
            extracted=ExtractedQuery(semantic_query="q", filters={}),
        )
        planner = FakePlannerAgent()
        service = _build_vector_service(planner=planner, extractor=extractor)

        service.answer_question("What risk documentation is required?")

        self.assertEqual(planner.call_count, 1)
        self.assertEqual(planner.calls[0].planner_model, _DEFAULT_ANSWER_MODEL)
        self.assertEqual(planner.calls[0].worker_model, _DEFAULT_ANSWER_MODEL)


def _build_vector_service(
    *,
    planner: FakePlannerAgent,
    extractor: FakeFilterExtractor,
) -> QAService:
    """Return a vector-backed QAService wired to the supplied fakes."""

    return QAService(
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
        planner_agent=planner,
        filter_extractor=extractor,
        retrieval_top_k=2,
        default_answer_model=_DEFAULT_ANSWER_MODEL,
        available_answer_models=(
            _DEFAULT_ANSWER_MODEL,
            _ALTERNATE_ANSWER_MODEL,
        ),
    )


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


def _make_retrieved_chunk(bill_id: str) -> RetrievedChunk:
    """Build a minimal ``RetrievedChunk`` the planner can "cite"."""

    return RetrievedChunk(
        rank=1,
        score=0.9,
        chunk_id=hash(bill_id) & 0xFFFF,
        bill_id=bill_id,
        text=f"Text of {bill_id}",
        start_offset=0,
        end_offset=20,
        state="CA",
        title=f"Title of {bill_id}",
        status="Introduced",
    )


if __name__ == "__main__":
    unittest.main()
