"""Offline tests for metadata filtering across the QA retrieval stack."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.qa.artifacts import STATUS_BUCKETS, IndexedChunk, RetrievedChunk
from src.qa.filter_extractor import ExtractedQuery
from src.qa.indexer import _normalize_status, _split_topics
from src.qa.lexical_retriever import LexicalRetriever
from src.qa.planner_agent import PlannerAnswer
from src.qa.retriever import Retriever
from src.qa.service import QAService

_DEFAULT_ANSWER_MODEL = "google/gemini-2.5-flash"


@dataclass
class _RecordedFilterPlannerCall:
    """Captured arguments from one fake planner call (filter tests only)."""

    question: str
    semantic_query: str
    initial_filters: dict[str, Any]
    planner_model: str


class _FakeFilterPlanner:
    """Planner stub that records its ``initial_filters`` to let tests assert plumbing."""

    def __init__(self) -> None:
        self.calls: list[_RecordedFilterPlannerCall] = []

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
        """Record the invocation and return a stable canned payload."""

        self.calls.append(
            _RecordedFilterPlannerCall(
                question=question,
                semantic_query=semantic_query,
                initial_filters=dict(initial_filters),
                planner_model=planner_model,
            )
        )
        return PlannerAnswer(
            answer_text="filter-plumbing planner answer",
            citations=[
                RetrievedChunk(
                    rank=1,
                    score=0.5,
                    chunk_id=999,
                    bill_id=initial_filters.get("bill_id_marker", "BILL-MARKER"),
                    text="marker text",
                    start_offset=0,
                    end_offset=10,
                    state=str(initial_filters.get("state", "")),
                    title="marker title",
                    status="Introduced",
                )
            ],
        )


@dataclass
class _FakeFilterExtractor:
    """Extractor stub that always returns an empty ``ExtractedQuery``."""

    extracted: ExtractedQuery = field(
        default_factory=lambda: ExtractedQuery(semantic_query="", filters={})
    )
    call_count: int = 0

    def extract(
        self,
        question: str,
        available_filter_values: dict,
    ) -> ExtractedQuery:
        """Return a fixed extraction while counting calls."""

        self.call_count += 1
        return self.extracted


class NormalizeStatusTests(unittest.TestCase):
    """Verify the canonical status-bucket mapping."""

    def test_prefix_in_whitelist_returns_bucket(self) -> None:
        self.assertEqual(_normalize_status("Failed - Adjourned"), "Failed")
        self.assertEqual(
            _normalize_status("Vetoed - Vetoed by Governor"),
            "Vetoed",
        )
        self.assertEqual(_normalize_status("Pending"), "Pending")
        self.assertEqual(_normalize_status("Enacted"), "Enacted")

    def test_prefix_outside_whitelist_collapses_to_other(self) -> None:
        self.assertEqual(_normalize_status("Adopted - Adopted"), "Other")
        self.assertEqual(_normalize_status("To governor"), "Other")
        self.assertEqual(_normalize_status(""), "Other")

    def test_non_string_input_returns_other(self) -> None:
        self.assertEqual(_normalize_status(None), "Other")  # type: ignore[arg-type]
        self.assertEqual(_normalize_status(42), "Other")  # type: ignore[arg-type]


class SplitTopicsTests(unittest.TestCase):
    """Verify topic string parsing across comma and semicolon separators."""

    def test_empty_string_returns_empty_list(self) -> None:
        self.assertEqual(_split_topics(""), [])
        self.assertEqual(_split_topics("   "), [])

    def test_comma_separated_topics_produce_stripped_tokens(self) -> None:
        self.assertEqual(
            _split_topics("Health Use, Private Sector Use"),
            ["Health Use", "Private Sector Use"],
        )

    def test_semicolon_separated_topics_produce_stripped_tokens(self) -> None:
        self.assertEqual(
            _split_topics("Child Pornography; Criminal Use"),
            ["Child Pornography", "Criminal Use"],
        )

    def test_mixed_separators_are_both_respected(self) -> None:
        self.assertEqual(
            _split_topics("A, B; C, D"),
            ["A", "B", "C", "D"],
        )


class RetrieverMaskTests(unittest.TestCase):
    """Verify the in-memory retriever's boolean filter mask."""

    def _build_retriever(self) -> Retriever:
        chunks = [
            IndexedChunk(
                chunk_id=1,
                bill_id="TX2023-1",
                text="Texas sandbox provisions apply.",
                start_offset=0,
                end_offset=30,
                state="Texas",
                year=2023,
                status_bucket="Enacted",
                topics_list=["Government Use"],
            ),
            IndexedChunk(
                chunk_id=2,
                bill_id="TX2025-1",
                text="Texas new risk framework introduced.",
                start_offset=0,
                end_offset=36,
                state="Texas",
                year=2025,
                status_bucket="Failed",
                topics_list=["Private Sector Use", "Health Use"],
            ),
            IndexedChunk(
                chunk_id=3,
                bill_id="CA2023-1",
                text="California innovation pilot bill.",
                start_offset=0,
                end_offset=32,
                state="California",
                year=2023,
                status_bucket="Enacted",
                topics_list=["Private Sector Use"],
            ),
            IndexedChunk(
                chunk_id=4,
                bill_id="NY2024-1",
                text="New York consumer protection act.",
                start_offset=0,
                end_offset=32,
                state="New York",
                year=2024,
                status_bucket="Pending",
                topics_list=["Criminal Use"],
            ),
            IndexedChunk(
                chunk_id=5,
                bill_id="NY2025-1",
                text="New York AI disclosure requirements.",
                start_offset=0,
                end_offset=35,
                state="New York",
                year=2025,
                status_bucket="Other",
                topics_list=[],
            ),
        ]
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms == 0, 1, norms)
        return Retriever(chunks=chunks, embeddings=embeddings)

    def test_year_filter_keeps_only_matching_year(self) -> None:
        retriever = self._build_retriever()
        query = np.ones(3, dtype=np.float32) / np.sqrt(3.0)

        results = retriever.retrieve(
            query_embedding=query,
            top_k=5,
            filters={"year": 2023},
        )

        returned_years = {result.year for result in results}
        self.assertEqual(returned_years, {2023})
        self.assertEqual(len(results), 2)

    def test_state_and_status_combined_filter(self) -> None:
        retriever = self._build_retriever()
        query = np.ones(3, dtype=np.float32) / np.sqrt(3.0)

        results = retriever.retrieve(
            query_embedding=query,
            top_k=5,
            filters={"state": "Texas", "status_bucket": "Enacted"},
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].bill_id, "TX2023-1")

    def test_topics_filter_uses_or_semantics(self) -> None:
        retriever = self._build_retriever()
        query = np.ones(3, dtype=np.float32) / np.sqrt(3.0)

        results = retriever.retrieve(
            query_embedding=query,
            top_k=5,
            filters={"topics": ["Health Use", "Criminal Use"]},
        )

        bill_ids = {result.bill_id for result in results}
        self.assertEqual(bill_ids, {"TX2025-1", "NY2024-1"})

    def test_state_filter_accepts_list_for_or_within_field(self) -> None:
        """Verify a list of states matches any chunk whose state is in the list."""

        retriever = self._build_retriever()
        query = np.ones(3, dtype=np.float32) / np.sqrt(3.0)

        results = retriever.retrieve(
            query_embedding=query,
            top_k=5,
            filters={"state": ["California", "New York"]},
        )

        bill_ids = {result.bill_id for result in results}
        self.assertEqual(bill_ids, {"CA2023-1", "NY2024-1", "NY2025-1"})

    def test_year_filter_accepts_list_for_or_within_field(self) -> None:
        """Verify ``year=[2023, 2025]`` matches chunks from either year."""

        retriever = self._build_retriever()
        query = np.ones(3, dtype=np.float32) / np.sqrt(3.0)

        results = retriever.retrieve(
            query_embedding=query,
            top_k=5,
            filters={"year": [2023, 2025]},
        )

        returned_years = {result.year for result in results}
        self.assertEqual(returned_years, {2023, 2025})
        self.assertEqual(len(results), 4)

    def test_status_bucket_filter_accepts_list(self) -> None:
        """Verify ``status_bucket=["Enacted","Pending"]`` honors OR-within-field."""

        retriever = self._build_retriever()
        query = np.ones(3, dtype=np.float32) / np.sqrt(3.0)

        results = retriever.retrieve(
            query_embedding=query,
            top_k=5,
            filters={"status_bucket": ["Enacted", "Pending"]},
        )

        bill_ids = {result.bill_id for result in results}
        self.assertEqual(bill_ids, {"TX2023-1", "CA2023-1", "NY2024-1"})

    def test_multi_field_with_lists_combines_with_and(self) -> None:
        """Verify list-valued fields still AND across fields (state OR x year OR)."""

        retriever = self._build_retriever()
        query = np.ones(3, dtype=np.float32) / np.sqrt(3.0)

        results = retriever.retrieve(
            query_embedding=query,
            top_k=5,
            filters={
                "state": ["New York", "California"],
                "year": [2023, 2024],
            },
        )

        bill_ids = {result.bill_id for result in results}
        self.assertEqual(bill_ids, {"CA2023-1", "NY2024-1"})

    def test_filter_with_no_matches_returns_empty(self) -> None:
        retriever = self._build_retriever()
        query = np.ones(3, dtype=np.float32) / np.sqrt(3.0)

        results = retriever.retrieve(
            query_embedding=query,
            top_k=5,
            filters={"year": 1999},
        )

        self.assertEqual(results, [])

    def test_no_filter_returns_all_top_k_in_score_order(self) -> None:
        retriever = self._build_retriever()
        query = np.ones(3, dtype=np.float32) / np.sqrt(3.0)

        results = retriever.retrieve(query_embedding=query, top_k=5)

        self.assertEqual(len(results), 5)
        scores = [result.score for result in results]
        self.assertEqual(scores, sorted(scores, reverse=True))


class LexicalRetrieverFilterTests(unittest.TestCase):
    """Verify the lexical retriever honors filters when provided."""

    def _chunks(self) -> list[IndexedChunk]:
        return [
            IndexedChunk(
                chunk_id=1,
                bill_id="TX2023-1",
                text="Texas innovation sandbox pilot program.",
                start_offset=0,
                end_offset=40,
                state="Texas",
                year=2023,
                status_bucket="Enacted",
                topics_list=["Government Use"],
            ),
            IndexedChunk(
                chunk_id=2,
                bill_id="CA2025-1",
                text="California innovation sandbox rules.",
                start_offset=0,
                end_offset=37,
                state="California",
                year=2025,
                status_bucket="Failed",
                topics_list=["Private Sector Use"],
            ),
        ]

    def test_state_filter_restricts_matches(self) -> None:
        retriever = LexicalRetriever(self._chunks())

        results = retriever.retrieve_question(
            "innovation sandbox",
            top_k=5,
            filters={"state": "Texas"},
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].bill_id, "TX2023-1")

    def test_filter_with_no_matches_returns_empty(self) -> None:
        retriever = LexicalRetriever(self._chunks())

        results = retriever.retrieve_question(
            "innovation sandbox",
            top_k=5,
            filters={"year": 1999},
        )

        self.assertEqual(results, [])

    def test_state_filter_accepts_list_for_or_within_field(self) -> None:
        """Verify the lexical retriever honors list-valued state OR filters."""

        retriever = LexicalRetriever(self._chunks())

        results = retriever.retrieve_question(
            "innovation sandbox",
            top_k=5,
            filters={"state": ["Texas", "California"]},
        )

        bill_ids = {result.bill_id for result in results}
        self.assertEqual(bill_ids, {"TX2023-1", "CA2025-1"})


class ServiceFilterPlumbingTests(unittest.TestCase):
    """Verify filters flow from QAService into the planner invocation."""

    def _build_service(self) -> tuple[QAService, _FakeFilterPlanner, _FakeFilterExtractor]:
        chunks = [
            IndexedChunk(
                chunk_id=1,
                bill_id="TX2023-1",
                text="Texas sandbox provisions apply.",
                start_offset=0,
                end_offset=30,
                state="Texas",
                year=2023,
                status_bucket="Enacted",
                topics_list=["Government Use"],
            ),
            IndexedChunk(
                chunk_id=2,
                bill_id="CA2025-1",
                text="California disclosure requirements.",
                start_offset=0,
                end_offset=36,
                state="California",
                year=2025,
                status_bucket="Failed",
                topics_list=["Private Sector Use"],
            ),
        ]
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms == 0, 1, norms)
        planner = _FakeFilterPlanner()
        extractor = _FakeFilterExtractor()
        service = QAService(
            retriever=Retriever(chunks=chunks, embeddings=embeddings),
            planner_agent=planner,
            filter_extractor=extractor,
            retrieval_top_k=5,
            default_answer_model=_DEFAULT_ANSWER_MODEL,
            available_answer_models=(_DEFAULT_ANSWER_MODEL,),
        )
        return service, planner, extractor

    def test_normalize_filters_drops_empty_fields(self) -> None:
        cleaned = QAService._normalize_filters(
            {"year": "", "state": "  ", "status_bucket": "", "topics": []}
        )
        self.assertIsNone(cleaned)

    def test_normalize_filters_keeps_valid_fields(self) -> None:
        cleaned = QAService._normalize_filters(
            {
                "year": "2023",
                "state": "Texas",
                "status_bucket": "Enacted",
                "topics": ["Government Use", ""],
            }
        )
        self.assertEqual(
            cleaned,
            {
                "year": 2023,
                "state": "Texas",
                "status_bucket": "Enacted",
                "topics": ["Government Use"],
            },
        )

    def test_normalize_filters_collapses_single_element_list_to_scalar(self) -> None:
        """Verify ``state=["CA"]`` normalizes to scalar ``state="CA"``."""

        cleaned = QAService._normalize_filters(
            {
                "year": [2024],
                "state": ["CA"],
                "status_bucket": ["Enacted"],
            }
        )
        self.assertEqual(
            cleaned,
            {"year": 2024, "state": "CA", "status_bucket": "Enacted"},
        )

    def test_normalize_filters_preserves_multi_value_lists(self) -> None:
        """Verify multi-value fields stay as lists so OR-within-field survives."""

        cleaned = QAService._normalize_filters(
            {
                "year": [2024, 2025],
                "state": ["CA", "TX"],
                "status_bucket": ["Enacted", "Pending"],
                "topics": ["Private Sector Use"],
            }
        )
        self.assertEqual(
            cleaned,
            {
                "year": [2024, 2025],
                "state": ["CA", "TX"],
                "status_bucket": ["Enacted", "Pending"],
                "topics": ["Private Sector Use"],
            },
        )

    def test_normalize_filters_dedupes_and_drops_bad_values(self) -> None:
        """Verify duplicates and non-numeric year entries are dropped."""

        cleaned = QAService._normalize_filters(
            {
                "year": [2024, "2024", "bad"],
                "state": ["CA", "CA", "  "],
            }
        )
        self.assertEqual(cleaned, {"year": 2024, "state": "CA"})

    def test_explicit_filters_reach_planner_and_skip_extractor(self) -> None:
        """Verify explicit filters bypass the extractor and land in the planner call."""

        service, planner, extractor = self._build_service()

        result = service.answer_question(
            "what provisions apply?",
            filters={"state": "Texas"},
        )

        self.assertEqual(extractor.call_count, 0)
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(planner.calls[0].initial_filters, {"state": "Texas"})
        self.assertEqual(
            result.applied_filters,
            {"state": "Texas", "routing_path": "agent"},
        )

    def test_available_filter_values_exposes_unique_facets(self) -> None:
        service, _planner, _extractor = self._build_service()

        facets = service.available_filter_values

        self.assertEqual(facets["year"], [2025, 2023])
        self.assertEqual(facets["state"], ["California", "Texas"])
        self.assertEqual(facets["status_bucket"], list(STATUS_BUCKETS))
        self.assertIn("Government Use", facets["topics"])
        self.assertIn("Private Sector Use", facets["topics"])


if __name__ == "__main__":
    unittest.main()
