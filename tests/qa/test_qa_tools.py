"""Offline tests for the planner tools and citation plumbing."""

from __future__ import annotations

import json
import unittest
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.qa.artifacts import IndexedChunk, RetrievedChunk
from src.qa.config import AgentConfig
from src.qa.lexical_retriever import LexicalRetriever
from src.qa.qa_tools import (
    CitationAccumulator,
    LexicalSearchBackend,
    VectorSearchBackend,
    WorkerBudgetExceededError,
    WorkerCallBudget,
    build_bill_index,
    build_qa_tool_registry,
)
from src.qa.retriever import Retriever


_DEFAULT_WORKER_MODEL = "google/gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_config(
    *,
    max_planner_turns: int = 8,
    max_planner_tokens: int = 4096,
    planner_temperature: float = 0.0,
    max_worker_tokens: int = 1024,
    worker_temperature: float = 0.0,
    max_tool_calls: int = 16,
    max_worker_calls: int = 6,
    max_bills_per_list: int = 50,
    max_chunks_per_bill: int = 6,
    max_citations_per_bill: int = 2,
) -> AgentConfig:
    """Return an ``AgentConfig`` with defaults that individual tests can override."""

    return AgentConfig(
        max_planner_turns=max_planner_turns,
        max_planner_tokens=max_planner_tokens,
        planner_temperature=planner_temperature,
        max_worker_tokens=max_worker_tokens,
        worker_temperature=worker_temperature,
        max_tool_calls=max_tool_calls,
        max_worker_calls=max_worker_calls,
        max_bills_per_list=max_bills_per_list,
        max_chunks_per_bill=max_chunks_per_bill,
        max_citations_per_bill=max_citations_per_bill,
    )


def _make_chunks() -> list[IndexedChunk]:
    """Build a small multi-bill corpus used across the tool tests."""

    return [
        IndexedChunk(
            chunk_id=101,
            bill_id="CA-2024-AI",
            text="California impact assessments must be disclosed.",
            start_offset=0,
            end_offset=50,
            state="CA",
            title="California AI Accountability Act",
            status="Enacted",
            status_bucket="Enacted",
            year=2024,
            topics_list=["Private Sector Use"],
            bill_url="https://example.test/ca-ai",
        ),
        IndexedChunk(
            chunk_id=102,
            bill_id="CA-2024-AI",
            text="Covered entities must publish bias audits annually.",
            start_offset=50,
            end_offset=110,
            state="CA",
            title="California AI Accountability Act",
            status="Enacted",
            status_bucket="Enacted",
            year=2024,
            topics_list=["Private Sector Use"],
            bill_url="https://example.test/ca-ai",
        ),
        IndexedChunk(
            chunk_id=201,
            bill_id="TX-2023-SB",
            text="Texas sandbox program allows pilots without full licensure.",
            start_offset=0,
            end_offset=60,
            state="TX",
            title="Texas AI Sandbox Act",
            status="Enacted",
            status_bucket="Enacted",
            year=2023,
            topics_list=["Government Use"],
            bill_url="https://example.test/tx-sandbox",
        ),
        IndexedChunk(
            chunk_id=301,
            bill_id="NY-2025-HB",
            text="New York restricts automated decisioning in hiring.",
            start_offset=0,
            end_offset=55,
            state="NY",
            title="New York Hiring AI Act",
            status="Pending",
            status_bucket="Pending",
            year=2025,
            topics_list=["Private Sector Use", "Employment"],
            bill_url="https://example.test/ny-hiring",
        ),
    ]


def _make_retrieved_chunk(
    bill_id: str,
    *,
    rank: int,
    chunk_id: int,
    state: str = "CA",
    score: float = 0.5,
) -> RetrievedChunk:
    """Build a ``RetrievedChunk`` fixture used by the citation tests."""

    return RetrievedChunk(
        rank=rank,
        score=score,
        chunk_id=chunk_id,
        bill_id=bill_id,
        text=f"text of chunk {chunk_id}",
        start_offset=0,
        end_offset=10,
        state=state,
        title=f"Title {bill_id}",
        status="Introduced",
    )


@dataclass
class _FakeCompletionMessage:
    """Stand-in for the SDK's ``response.choices[0].message``."""

    content: str


@dataclass
class _FakeCompletionChoice:
    """Stand-in for one element of ``response.choices``."""

    message: _FakeCompletionMessage


@dataclass
class _FakeCompletionResponse:
    """Stand-in for an ``openai`` chat-completion response object."""

    choices: list[_FakeCompletionChoice]


class _FakeChatCompletions:
    """Capture planner worker calls and return a canned completion."""

    def __init__(self, completion_text: str = "worker-generated output") -> None:
        self._completion_text = completion_text
        self.calls: list[dict[str, Any]] = []

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> _FakeCompletionResponse:
        """Record the completion arguments and return a canned response."""

        self.calls.append(
            {
                "model": model,
                "messages": list(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return _FakeCompletionResponse(
            choices=[_FakeCompletionChoice(message=_FakeCompletionMessage(content=self._completion_text))]
        )


@dataclass
class _FakeOpenAIClient:
    """Minimal ``openai``-style client exposing ``chat.completions.create``."""

    chat: Any = field(default=None)

    @classmethod
    def with_completion(cls, completion_text: str = "worker output") -> "_FakeOpenAIClient":
        """Build a fake client whose chat.completions returns ``completion_text``."""

        completions = _FakeChatCompletions(completion_text=completion_text)
        wrapper = type("_ChatWrapper", (), {"completions": completions})
        return cls(chat=wrapper)

    @property
    def completion_calls(self) -> list[dict[str, Any]]:
        """Return the raw calls list recorded by the underlying completions stub."""

        return self.chat.completions.calls  # type: ignore[no-any-return]


class _RecordingSearchBackend:
    """Search backend double that records arguments and returns preset chunks."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.calls: list[dict[str, Any]] = []

    def search(
        self,
        query_text: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        """Record the search invocation and return the canned chunk list."""

        self.calls.append(
            {"query_text": query_text, "top_k": top_k, "filters": filters}
        )
        return list(self._chunks)


# ---------------------------------------------------------------------------
# build_bill_index
# ---------------------------------------------------------------------------


class BuildBillIndexTests(unittest.TestCase):
    """Verify the in-memory bill metadata index from a chunk sequence."""

    def test_aggregates_multiple_chunks_into_one_summary(self) -> None:
        """Verify multiple chunks for the same bill produce one merged summary."""

        bill_index = build_bill_index(_make_chunks())

        self.assertEqual(len(bill_index), 3)
        self.assertIn("CA-2024-AI", bill_index)
        summary = bill_index["CA-2024-AI"]
        self.assertEqual(summary.state, "CA")
        self.assertEqual(summary.year, 2024)
        self.assertEqual(summary.status_bucket, "Enacted")
        self.assertEqual(summary.topics, ("Private Sector Use",))
        self.assertEqual(summary.chunk_row_indices, (0, 1))

    def test_missing_bill_id_is_skipped(self) -> None:
        """Verify chunks without a bill_id do not corrupt the index."""

        chunks = _make_chunks() + [
            IndexedChunk(
                chunk_id=999,
                bill_id="",
                text="orphan chunk",
                start_offset=0,
                end_offset=5,
                state="ZZ",
            )
        ]

        bill_index = build_bill_index(chunks)

        self.assertEqual(set(bill_index.keys()), {"CA-2024-AI", "TX-2023-SB", "NY-2025-HB"})


# ---------------------------------------------------------------------------
# CitationAccumulator
# ---------------------------------------------------------------------------


class CitationAccumulatorTests(unittest.TestCase):
    """Verify dedup, per-bill caps, and export renumbering."""

    def test_deduplicates_by_chunk_id(self) -> None:
        """Verify duplicate ``chunk_id`` values are ignored on add."""

        accumulator = CitationAccumulator(max_per_bill=3)
        accumulator.add(_make_retrieved_chunk("B1", rank=1, chunk_id=1))
        accumulator.add(_make_retrieved_chunk("B1", rank=2, chunk_id=1))

        exported = accumulator.export()

        self.assertEqual(len(exported), 1)
        self.assertEqual(exported[0].chunk_id, 1)
        self.assertEqual(exported[0].rank, 1)

    def test_enforces_per_bill_cap(self) -> None:
        """Verify extra chunks from the same bill are dropped at the cap."""

        accumulator = CitationAccumulator(max_per_bill=2)
        accumulator.add(_make_retrieved_chunk("B1", rank=1, chunk_id=1))
        accumulator.add(_make_retrieved_chunk("B1", rank=2, chunk_id=2))
        accumulator.add(_make_retrieved_chunk("B1", rank=3, chunk_id=3))
        accumulator.add(_make_retrieved_chunk("B2", rank=4, chunk_id=10))

        exported = accumulator.export()
        chunk_ids = [chunk.chunk_id for chunk in exported]

        self.assertEqual(chunk_ids, [1, 2, 10])

    def test_export_truncates_and_renumbers(self) -> None:
        """Verify exporting with ``max_total`` renumbers ranks from 1."""

        accumulator = CitationAccumulator(max_per_bill=5)
        accumulator.add(_make_retrieved_chunk("B1", rank=4, chunk_id=1, score=0.8))
        accumulator.add(_make_retrieved_chunk("B2", rank=6, chunk_id=2, score=0.5))
        accumulator.add(_make_retrieved_chunk("B3", rank=9, chunk_id=3, score=0.3))

        exported = accumulator.export(max_total=2)

        self.assertEqual(len(exported), 2)
        self.assertEqual([chunk.rank for chunk in exported], [1, 2])
        self.assertEqual([chunk.chunk_id for chunk in exported], [1, 2])


# ---------------------------------------------------------------------------
# WorkerCallBudget
# ---------------------------------------------------------------------------


class WorkerCallBudgetTests(unittest.TestCase):
    """Verify the worker-call cap enforcement."""

    def test_consume_raises_when_cap_exceeded(self) -> None:
        """Verify ``consume`` refuses calls past ``max_calls``."""

        budget = WorkerCallBudget(max_calls=2)
        budget.consume()
        budget.consume()

        with self.assertRaises(WorkerBudgetExceededError):
            budget.consume()
        self.assertEqual(budget.used, 2)


# ---------------------------------------------------------------------------
# Search backend adapters
# ---------------------------------------------------------------------------


class SearchBackendAdapterTests(unittest.TestCase):
    """Verify the vector and lexical adapters delegate correctly."""

    def test_vector_backend_calls_embed_then_retrieve(self) -> None:
        """Verify the vector adapter embeds the query then delegates to the retriever."""

        retriever = Retriever(
            chunks=_make_chunks(),
            embeddings=np.eye(4, dtype=np.float32),
        )
        embed_calls: list[str] = []

        def _fake_embed(text: str) -> np.ndarray:
            embed_calls.append(text)
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        backend = VectorSearchBackend(retriever=retriever, embed_query=_fake_embed)

        results = backend.search(query_text="California", top_k=2, filters={"state": "CA"})

        self.assertEqual(embed_calls, ["California"])
        self.assertTrue(all(result.state == "CA" for result in results))

    def test_vector_backend_returns_empty_on_blank_query(self) -> None:
        """Verify the vector adapter short-circuits on a blank query."""

        retriever = Retriever(
            chunks=_make_chunks(),
            embeddings=np.eye(4, dtype=np.float32),
        )
        backend = VectorSearchBackend(
            retriever=retriever,
            embed_query=lambda _: (_ for _ in ()).throw(AssertionError("must not be called")),
        )

        self.assertEqual(backend.search("   ", top_k=3, filters=None), [])

    def test_lexical_backend_delegates_to_retrieve_question(self) -> None:
        """Verify the lexical adapter forwards the query to BM25 retrieval."""

        retriever = LexicalRetriever(_make_chunks())
        backend = LexicalSearchBackend(retriever=retriever)

        results = backend.search("sandbox pilots", top_k=2, filters=None)

        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(any(result.bill_id == "TX-2023-SB" for result in results))


# ---------------------------------------------------------------------------
# Tool handlers (via ``build_qa_tool_registry``)
# ---------------------------------------------------------------------------


class ListBillsToolTests(unittest.TestCase):
    """Verify ``list_bills`` in both metadata and semantic modes."""

    def test_metadata_only_listing_respects_limit_and_filters(self) -> None:
        """Verify metadata mode filters by state and respects the limit."""

        chunks = _make_chunks()
        bill_index = build_bill_index(chunks)
        search_backend = _RecordingSearchBackend(chunks=[])
        accumulator = CitationAccumulator(max_per_bill=2)
        registry = build_qa_tool_registry(
            chunks=chunks,
            bill_index=bill_index,
            search_backend=search_backend,
            worker_client=_FakeOpenAIClient.with_completion(),
            worker_model=_DEFAULT_WORKER_MODEL,
            agent_config=_make_agent_config(),
            accumulator=accumulator,
            worker_budget=WorkerCallBudget(max_calls=3),
        )

        raw_result = registry.execute(
            "list_bills",
            {"filters": {"state": "CA"}, "limit": 5},
        )
        payload = json.loads(raw_result)

        self.assertEqual(payload["bill_count"], 1)
        self.assertEqual(payload["bills"][0]["bill_id"], "CA-2024-AI")
        self.assertEqual(payload["applied_filters"], {"state": "CA"})
        self.assertEqual(payload["semantic_query"], "")
        self.assertEqual(search_backend.calls, [])

    def test_semantic_listing_uses_search_backend_and_populates_citations(self) -> None:
        """Verify semantic mode calls the backend and records citations."""

        chunks = _make_chunks()
        bill_index = build_bill_index(chunks)
        search_hits = [
            _make_retrieved_chunk("CA-2024-AI", rank=1, chunk_id=101, state="CA"),
            _make_retrieved_chunk("NY-2025-HB", rank=2, chunk_id=301, state="NY"),
        ]
        search_backend = _RecordingSearchBackend(chunks=search_hits)
        accumulator = CitationAccumulator(max_per_bill=2)
        registry = build_qa_tool_registry(
            chunks=chunks,
            bill_index=bill_index,
            search_backend=search_backend,
            worker_client=_FakeOpenAIClient.with_completion(),
            worker_model=_DEFAULT_WORKER_MODEL,
            agent_config=_make_agent_config(),
            accumulator=accumulator,
            worker_budget=WorkerCallBudget(max_calls=3),
        )

        raw_result = registry.execute(
            "list_bills",
            {"semantic_query": "bias audits", "limit": 2},
        )
        payload = json.loads(raw_result)

        self.assertEqual(len(search_backend.calls), 1)
        self.assertEqual(search_backend.calls[0]["query_text"], "bias audits")
        self.assertEqual(payload["semantic_query"], "bias audits")
        self.assertEqual(
            [bill["bill_id"] for bill in payload["bills"]],
            ["CA-2024-AI", "NY-2025-HB"],
        )
        cited_chunk_ids = {chunk.chunk_id for chunk in accumulator.export()}
        self.assertEqual(cited_chunk_ids, {101, 301})

    def test_limit_is_capped_by_agent_config(self) -> None:
        """Verify the returned bill count never exceeds ``max_bills_per_list``."""

        chunks = _make_chunks()
        bill_index = build_bill_index(chunks)
        search_backend = _RecordingSearchBackend(chunks=[])
        registry = build_qa_tool_registry(
            chunks=chunks,
            bill_index=bill_index,
            search_backend=search_backend,
            worker_client=_FakeOpenAIClient.with_completion(),
            worker_model=_DEFAULT_WORKER_MODEL,
            agent_config=_make_agent_config(max_bills_per_list=1),
            accumulator=CitationAccumulator(max_per_bill=2),
            worker_budget=WorkerCallBudget(max_calls=3),
        )

        raw_result = registry.execute("list_bills", {"limit": 50})
        payload = json.loads(raw_result)

        self.assertEqual(payload["limit"], 1)
        self.assertEqual(payload["bill_count"], 1)

    def test_list_bills_accepts_list_valued_state_filter(self) -> None:
        """Verify ``filters={"state": ["CA","TX"]}`` returns both states via OR."""

        chunks = _make_chunks()
        bill_index = build_bill_index(chunks)
        search_backend = _RecordingSearchBackend(chunks=[])
        registry = build_qa_tool_registry(
            chunks=chunks,
            bill_index=bill_index,
            search_backend=search_backend,
            worker_client=_FakeOpenAIClient.with_completion(),
            worker_model=_DEFAULT_WORKER_MODEL,
            agent_config=_make_agent_config(),
            accumulator=CitationAccumulator(max_per_bill=2),
            worker_budget=WorkerCallBudget(max_calls=3),
        )

        raw_result = registry.execute(
            "list_bills",
            {"filters": {"state": ["CA", "TX"]}, "limit": 10},
        )
        payload = json.loads(raw_result)

        returned_ids = {bill["bill_id"] for bill in payload["bills"]}
        self.assertEqual(returned_ids, {"CA-2024-AI", "TX-2023-SB"})
        self.assertEqual(payload["applied_filters"], {"state": ["CA", "TX"]})

    def test_list_bills_accepts_list_valued_status_bucket_filter(self) -> None:
        """Verify ``status_bucket=["Enacted","Pending"]`` matches either bucket."""

        chunks = _make_chunks()
        bill_index = build_bill_index(chunks)
        search_backend = _RecordingSearchBackend(chunks=[])
        registry = build_qa_tool_registry(
            chunks=chunks,
            bill_index=bill_index,
            search_backend=search_backend,
            worker_client=_FakeOpenAIClient.with_completion(),
            worker_model=_DEFAULT_WORKER_MODEL,
            agent_config=_make_agent_config(),
            accumulator=CitationAccumulator(max_per_bill=2),
            worker_budget=WorkerCallBudget(max_calls=3),
        )

        raw_result = registry.execute(
            "list_bills",
            {
                "filters": {"status_bucket": ["Enacted", "Pending"]},
                "limit": 10,
            },
        )
        payload = json.loads(raw_result)

        returned_ids = {bill["bill_id"] for bill in payload["bills"]}
        self.assertEqual(
            returned_ids, {"CA-2024-AI", "TX-2023-SB", "NY-2025-HB"}
        )

    def test_list_bills_collapses_single_element_list_to_scalar_echo(self) -> None:
        """Verify single-element list still filters correctly and echoes as scalar."""

        chunks = _make_chunks()
        bill_index = build_bill_index(chunks)
        search_backend = _RecordingSearchBackend(chunks=[])
        registry = build_qa_tool_registry(
            chunks=chunks,
            bill_index=bill_index,
            search_backend=search_backend,
            worker_client=_FakeOpenAIClient.with_completion(),
            worker_model=_DEFAULT_WORKER_MODEL,
            agent_config=_make_agent_config(),
            accumulator=CitationAccumulator(max_per_bill=2),
            worker_budget=WorkerCallBudget(max_calls=3),
        )

        raw_result = registry.execute(
            "list_bills",
            {"filters": {"state": ["CA"]}, "limit": 10},
        )
        payload = json.loads(raw_result)

        returned_ids = {bill["bill_id"] for bill in payload["bills"]}
        self.assertEqual(returned_ids, {"CA-2024-AI"})
        self.assertEqual(payload["applied_filters"], {"state": "CA"})


class GetBillContentToolTests(unittest.TestCase):
    """Verify ``get_bill_content`` assembly and error handling."""

    def test_returns_concatenated_bill_text(self) -> None:
        """Verify the handler concatenates chunks in offset order."""

        chunks = _make_chunks()
        bill_index = build_bill_index(chunks)
        accumulator = CitationAccumulator(max_per_bill=3)
        registry = build_qa_tool_registry(
            chunks=chunks,
            bill_index=bill_index,
            search_backend=_RecordingSearchBackend(chunks=[]),
            worker_client=_FakeOpenAIClient.with_completion(),
            worker_model=_DEFAULT_WORKER_MODEL,
            agent_config=_make_agent_config(),
            accumulator=accumulator,
            worker_budget=WorkerCallBudget(max_calls=3),
        )

        raw_result = registry.execute(
            "get_bill_content",
            {"bill_id": "CA-2024-AI"},
        )
        payload = json.loads(raw_result)

        self.assertEqual(payload["bill"]["bill_id"], "CA-2024-AI")
        self.assertIn("California impact assessments", payload["text"])
        self.assertIn("bias audits annually", payload["text"])
        self.assertEqual(payload["chunks_returned"], 2)
        self.assertEqual(payload["total_chunks_in_bill"], 2)
        cited_chunk_ids = {chunk.chunk_id for chunk in accumulator.export()}
        self.assertEqual(cited_chunk_ids, {101, 102})

    def test_unknown_bill_id_returns_error(self) -> None:
        """Verify a bogus bill_id surfaces a structured error payload."""

        chunks = _make_chunks()
        bill_index = build_bill_index(chunks)
        registry = build_qa_tool_registry(
            chunks=chunks,
            bill_index=bill_index,
            search_backend=_RecordingSearchBackend(chunks=[]),
            worker_client=_FakeOpenAIClient.with_completion(),
            worker_model=_DEFAULT_WORKER_MODEL,
            agent_config=_make_agent_config(),
            accumulator=CitationAccumulator(max_per_bill=3),
            worker_budget=WorkerCallBudget(max_calls=3),
        )

        raw_result = registry.execute(
            "get_bill_content",
            {"bill_id": "does-not-exist"},
        )
        payload = json.loads(raw_result)

        self.assertIn("error", payload)
        self.assertIn("does-not-exist", payload["error"])

    def test_max_chars_truncates_returned_text(self) -> None:
        """Verify ``max_chars`` caps the concatenated text length."""

        chunks = _make_chunks()
        bill_index = build_bill_index(chunks)
        registry = build_qa_tool_registry(
            chunks=chunks,
            bill_index=bill_index,
            search_backend=_RecordingSearchBackend(chunks=[]),
            worker_client=_FakeOpenAIClient.with_completion(),
            worker_model=_DEFAULT_WORKER_MODEL,
            agent_config=_make_agent_config(),
            accumulator=CitationAccumulator(max_per_bill=3),
            worker_budget=WorkerCallBudget(max_calls=3),
        )

        raw_result = registry.execute(
            "get_bill_content",
            {"bill_id": "CA-2024-AI", "max_chars": 200},
        )
        payload = json.loads(raw_result)

        self.assertLessEqual(len(payload["text"]), 200)


class SummarizeBillToolTests(unittest.TestCase):
    """Verify the summarize_bill worker path and budget integration."""

    def test_invokes_worker_client_with_configured_params(self) -> None:
        """Verify the summarize handler dispatches one completion call."""

        chunks = _make_chunks()
        bill_index = build_bill_index(chunks)
        worker_client = _FakeOpenAIClient.with_completion("- Summary bullet")
        registry = build_qa_tool_registry(
            chunks=chunks,
            bill_index=bill_index,
            search_backend=_RecordingSearchBackend(chunks=[]),
            worker_client=worker_client,
            worker_model=_DEFAULT_WORKER_MODEL,
            agent_config=_make_agent_config(
                max_worker_tokens=512, worker_temperature=0.1
            ),
            accumulator=CitationAccumulator(max_per_bill=3),
            worker_budget=WorkerCallBudget(max_calls=3),
        )

        raw_result = registry.execute(
            "summarize_bill",
            {"bill_id": "CA-2024-AI", "focus": "bias audits"},
        )
        payload = json.loads(raw_result)

        self.assertEqual(payload["summary"], "- Summary bullet")
        self.assertEqual(payload["bill_id"], "CA-2024-AI")
        self.assertEqual(payload["worker_calls_used"], 1)
        self.assertEqual(len(worker_client.completion_calls), 1)
        invocation = worker_client.completion_calls[0]
        self.assertEqual(invocation["model"], _DEFAULT_WORKER_MODEL)
        self.assertEqual(invocation["max_tokens"], 512)
        self.assertEqual(invocation["temperature"], 0.1)
        self.assertEqual(invocation["messages"][0]["role"], "system")
        self.assertEqual(invocation["messages"][1]["role"], "user")

    def test_worker_budget_exceeded_returns_error_payload(self) -> None:
        """Verify exhausted budget surfaces an error without raising."""

        chunks = _make_chunks()
        bill_index = build_bill_index(chunks)
        worker_client = _FakeOpenAIClient.with_completion()
        registry = build_qa_tool_registry(
            chunks=chunks,
            bill_index=bill_index,
            search_backend=_RecordingSearchBackend(chunks=[]),
            worker_client=worker_client,
            worker_model=_DEFAULT_WORKER_MODEL,
            agent_config=_make_agent_config(),
            accumulator=CitationAccumulator(max_per_bill=3),
            worker_budget=WorkerCallBudget(max_calls=0),
        )

        raw_result = registry.execute(
            "summarize_bill",
            {"bill_id": "CA-2024-AI"},
        )
        payload = json.loads(raw_result)

        self.assertIn("error", payload)
        self.assertIn("budget", payload["error"].lower())
        self.assertEqual(len(worker_client.completion_calls), 0)


class CompareBillsToolTests(unittest.TestCase):
    """Verify the compare_bills worker path and validation."""

    def test_requires_at_least_two_bill_ids(self) -> None:
        """Verify single-bill inputs are rejected with an error payload."""

        chunks = _make_chunks()
        bill_index = build_bill_index(chunks)
        registry = build_qa_tool_registry(
            chunks=chunks,
            bill_index=bill_index,
            search_backend=_RecordingSearchBackend(chunks=[]),
            worker_client=_FakeOpenAIClient.with_completion(),
            worker_model=_DEFAULT_WORKER_MODEL,
            agent_config=_make_agent_config(),
            accumulator=CitationAccumulator(max_per_bill=3),
            worker_budget=WorkerCallBudget(max_calls=3),
        )

        raw_result = registry.execute(
            "compare_bills",
            {"bill_ids": ["CA-2024-AI"], "question": "What overlaps?"},
        )
        payload = json.loads(raw_result)

        self.assertIn("error", payload)
        self.assertIn("at least 2", payload["error"])

    def test_dispatches_one_worker_call_for_valid_inputs(self) -> None:
        """Verify the compare handler runs one completion call for 2+ bills."""

        chunks = _make_chunks()
        bill_index = build_bill_index(chunks)
        worker_client = _FakeOpenAIClient.with_completion("- Common theme")
        registry = build_qa_tool_registry(
            chunks=chunks,
            bill_index=bill_index,
            search_backend=_RecordingSearchBackend(chunks=[]),
            worker_client=worker_client,
            worker_model=_DEFAULT_WORKER_MODEL,
            agent_config=_make_agent_config(),
            accumulator=CitationAccumulator(max_per_bill=3),
            worker_budget=WorkerCallBudget(max_calls=3),
        )

        raw_result = registry.execute(
            "compare_bills",
            {
                "bill_ids": ["CA-2024-AI", "NY-2025-HB"],
                "question": "How do they treat automated decisioning?",
            },
        )
        payload = json.loads(raw_result)

        self.assertEqual(payload["comparison"], "- Common theme")
        self.assertEqual(payload["bill_ids"], ["CA-2024-AI", "NY-2025-HB"])
        self.assertEqual(len(worker_client.completion_calls), 1)


if __name__ == "__main__":
    unittest.main()
