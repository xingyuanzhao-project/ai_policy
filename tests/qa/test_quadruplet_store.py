"""Offline tests for the quadruplet sidecar store and its planner tool."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.qa.qa_tools import _query_quadruplets_handler, build_qa_tool_registry
from src.qa.quadruplet_store import (
    QuadrupletRecord,
    QuadrupletSpan,
    QuadrupletStore,
    load_quadruplet_store,
)


def _record(
    *,
    bill_id: str = "2024__CA A   1",
    state: str = "California",
    year: int = 2024,
    regulated_entity: str = "automated decision system",
    entity_type: str = "AI application",
    regulatory_mechanism: str = "disclosure requirement",
    provision_text: str = "must notify individuals of AI-driven decisions",
    entity_span: QuadrupletSpan | None = None,
    provision_span: QuadrupletSpan | None = None,
) -> QuadrupletRecord:
    """Return a populated ``QuadrupletRecord`` with sensible defaults."""

    return QuadrupletRecord(
        bill_id=bill_id,
        state=state,
        year=year,
        regulated_entity=regulated_entity,
        entity_type=entity_type,
        regulatory_mechanism=regulatory_mechanism,
        provision_text=provision_text,
        entity_span=entity_span,
        provision_span=provision_span,
    )


class QuadrupletStoreSearchTests(unittest.TestCase):
    """Filter semantics and limit handling for :meth:`QuadrupletStore.search`."""

    def setUp(self) -> None:
        """Populate a small three-bill store used across every search test."""

        self.store = QuadrupletStore(
            [
                _record(
                    bill_id="2024__CA A   1",
                    state="California",
                    year=2024,
                    regulated_entity="automated decision system",
                    entity_type="AI application",
                    regulatory_mechanism="disclosure requirement",
                    provision_text="must notify individuals of AI-driven decisions",
                ),
                _record(
                    bill_id="2024__CA A   2",
                    state="California",
                    year=2024,
                    regulated_entity="facial recognition technology",
                    entity_type="technology",
                    regulatory_mechanism="prohibition",
                    provision_text="prohibits use of facial recognition by police",
                ),
                _record(
                    bill_id="2023__TX S   9",
                    state="Texas",
                    year=2023,
                    regulated_entity="generative AI",
                    entity_type="AI application",
                    regulatory_mechanism="disclosure requirement",
                    provision_text="disclosure required when generative AI authored content",
                ),
            ]
        )

    def test_regulated_entity_substring_is_case_insensitive(self) -> None:
        """``regulated_entity`` filter matches as a case-insensitive substring."""

        hits = self.store.search(regulated_entity="FACIAL")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].bill_id, "2024__CA A   2")

    def test_state_and_year_combine_with_AND(self) -> None:
        """Categorical filters intersect (AND across fields, OR within list)."""

        hits = self.store.search(state=["California"], year=[2024])
        self.assertEqual(len(hits), 2)
        self.assertEqual(
            {hit.bill_id for hit in hits},
            {"2024__CA A   1", "2024__CA A   2"},
        )

    def test_regulatory_mechanism_substring_matches_partial(self) -> None:
        """``regulatory_mechanism`` substring catches 'disclosure' prefix matches."""

        hits = self.store.search(regulatory_mechanism="disclosure")
        self.assertEqual(len(hits), 2)
        self.assertTrue(all("disclosure" in hit.regulatory_mechanism for hit in hits))

    def test_provision_contains_filters_on_body(self) -> None:
        """``provision_contains`` matches against the provision_text body."""

        hits = self.store.search(provision_contains="police")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].bill_id, "2024__CA A   2")

    def test_entity_type_list_is_OR(self) -> None:
        """Multi-valued ``entity_type`` list returns the union of matches."""

        hits = self.store.search(entity_type=["AI application", "technology"])
        self.assertEqual(len(hits), 3)

    def test_bill_id_exact_match_restricts_to_one_bill(self) -> None:
        """``bill_id`` filter enumerates only the rows from that one bill."""

        hits = self.store.search(bill_id=["2023__TX S   9"])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].state, "Texas")

    def test_limit_truncates_the_result_list(self) -> None:
        """``limit=1`` returns only the first matching record."""

        hits = self.store.search(entity_type=["AI application"], limit=1)
        self.assertEqual(len(hits), 1)

    def test_zero_limit_returns_empty_list(self) -> None:
        """Non-positive ``limit`` short-circuits to an empty list."""

        hits = self.store.search(regulated_entity="AI", limit=0)
        self.assertEqual(hits, [])

    def test_missing_filters_return_all_records(self) -> None:
        """No filters returns every record up to the default limit."""

        hits = self.store.search(limit=10)
        self.assertEqual(len(hits), 3)


class QuadrupletStoreLoadingTests(unittest.TestCase):
    """JSONL loading and missing-file fallbacks."""

    def test_missing_file_returns_empty_store(self) -> None:
        """An absent sidecar is a non-fatal signal (empty store)."""

        store = load_quadruplet_store(Path("does/not/exist/quadruplets.jsonl"))
        self.assertEqual(len(store), 0)
        self.assertEqual(store.total_quadruplets, 0)

    def test_jsonl_round_trip_preserves_fields(self) -> None:
        """Records round-trip through JSONL with their spans intact."""

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "quadruplets.jsonl"
            payload = {
                "bill_id": "2024__CA A   1",
                "state": "California",
                "year": 2024,
                "regulated_entity": "automated decision system",
                "entity_type": "AI application",
                "regulatory_mechanism": "disclosure requirement",
                "provision_text": "must notify individuals",
                "entity_span": {"start": 100, "end": 125, "text": "span-text"},
                "provision_span": {"start": 200, "end": 240, "text": "prov-text"},
            }
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            store = load_quadruplet_store(path)
            self.assertEqual(len(store), 1)
            record = store.search(limit=1)[0]
            self.assertEqual(record.bill_id, "2024__CA A   1")
            self.assertEqual(record.year, 2024)
            self.assertIsNotNone(record.entity_span)
            self.assertEqual(record.entity_span.start, 100)
            self.assertEqual(record.provision_span.end, 240)

    def test_malformed_line_is_skipped(self) -> None:
        """Corrupt JSON lines are logged and skipped without raising."""

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "quadruplets.jsonl"
            good_payload = json.dumps(
                {
                    "bill_id": "2024__CA A   1",
                    "state": "California",
                    "year": 2024,
                    "regulated_entity": "e",
                    "entity_type": "t",
                    "regulatory_mechanism": "r",
                    "provision_text": "p",
                }
            )
            path.write_text(f"{good_payload}\nnot json\n", encoding="utf-8")
            store = load_quadruplet_store(path)
            self.assertEqual(len(store), 1)

    def test_row_missing_required_fields_is_skipped(self) -> None:
        """Rows missing a mandatory string field never enter the store."""

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "quadruplets.jsonl"
            incomplete = json.dumps({"bill_id": "x", "state": "CA"})
            path.write_text(incomplete + "\n", encoding="utf-8")
            store = load_quadruplet_store(path)
            self.assertEqual(len(store), 0)


class QuadrupletStoreVocabularyTests(unittest.TestCase):
    """Vocabulary counts surface top categories in descending frequency."""

    def test_top_entity_types_sorted_by_count(self) -> None:
        """``entity_types`` lists the most common categories first."""

        store = QuadrupletStore(
            [
                _record(entity_type="AI application"),
                _record(entity_type="AI application"),
                _record(entity_type="technology"),
            ]
        )
        vocab = store.vocabulary(top_entity_types=2, top_mechanisms=2)
        self.assertEqual(vocab.entity_types[0], ("AI application", 2))
        self.assertEqual(vocab.entity_types[1], ("technology", 1))


class QueryQuadrupletsToolTests(unittest.TestCase):
    """End-to-end behavior of the ``query_quadruplets`` tool handler."""

    def setUp(self) -> None:
        """Build a small store reused across all tool-handler tests."""

        self.store = QuadrupletStore(
            [
                _record(
                    bill_id="2024__CA A   1",
                    regulated_entity="automated decision system",
                    regulatory_mechanism="disclosure requirement",
                    provision_span=QuadrupletSpan(
                        start=0, end=50, text="sample-provision"
                    ),
                ),
                _record(
                    bill_id="2024__CA A   2",
                    regulated_entity="facial recognition",
                    regulatory_mechanism="prohibition",
                ),
            ]
        )

    def test_handler_returns_matches_and_applied_filters(self) -> None:
        """Successful call echoes the applied filters and full match list."""

        payload = _query_quadruplets_handler(
            {"regulatory_mechanism": "prohibition"},
            quadruplet_store=self.store,
        )
        self.assertEqual(payload["match_count"], 1)
        self.assertEqual(payload["applied_filters"], {"regulatory_mechanism": "prohibition"})
        self.assertFalse(payload["truncated"])
        self.assertEqual(payload["total_quadruplets_available"], 2)

    def test_handler_truncates_and_flags_when_limit_exceeded(self) -> None:
        """``truncated`` is ``True`` when the raw hit count exceeds ``limit``."""

        payload = _query_quadruplets_handler(
            {"limit": 1},
            quadruplet_store=self.store,
        )
        self.assertEqual(payload["match_count"], 1)
        self.assertTrue(payload["truncated"])
        self.assertEqual(payload["limit"], 1)

    def test_handler_truncates_long_provision_text(self) -> None:
        """Provision text is truncated in the payload to keep it prompt-safe."""

        long_text = "a" * 1000
        store = QuadrupletStore(
            [_record(provision_text=long_text, bill_id="2024__CA A   1")]
        )
        payload = _query_quadruplets_handler(
            {"bill_id": "2024__CA A   1"}, quadruplet_store=store
        )
        provision = payload["matches"][0]["provision_text"]
        self.assertTrue(provision.endswith("..."))
        self.assertLess(len(provision), 1000)
        self.assertTrue(payload["matches"][0]["provision_truncated"])

    def test_empty_store_returns_note_and_zero_matches(self) -> None:
        """An empty store short-circuits with an explanatory note."""

        payload = _query_quadruplets_handler(
            {"regulated_entity": "anything"},
            quadruplet_store=QuadrupletStore.empty(),
        )
        self.assertEqual(payload["match_count"], 0)
        self.assertEqual(payload["total_quadruplets_available"], 0)
        self.assertIn("note", payload)

    def test_handler_resolves_usps_state_abbreviation(self) -> None:
        """Verify ``state="TX"`` resolves against a store that indexes ``"Texas"``.

        The store's records use full state names; the LLM frequently emits
        USPS codes. The handler is expected to fold the code through the
        canonical state vocabulary and echo the canonical form.
        """

        store = QuadrupletStore(
            [
                _record(bill_id="2024__CA A   1", state="California"),
                _record(bill_id="2023__TX S   9", state="Texas"),
            ]
        )
        payload = _query_quadruplets_handler(
            {"state": "TX"}, quadruplet_store=store
        )
        self.assertEqual(payload["match_count"], 1)
        self.assertEqual(payload["applied_filters"]["state"], "Texas")
        self.assertEqual(payload["matches"][0]["bill_id"], "2023__TX S   9")

    def test_handler_resolves_lowercase_full_state_name(self) -> None:
        """Verify ``state="california"`` folds to the canonical ``"California"``."""

        store = QuadrupletStore(
            [
                _record(bill_id="2024__CA A   1", state="California"),
                _record(bill_id="2023__TX S   9", state="Texas"),
            ]
        )
        payload = _query_quadruplets_handler(
            {"state": "california"}, quadruplet_store=store
        )
        self.assertEqual(payload["match_count"], 1)
        self.assertEqual(payload["applied_filters"]["state"], "California")

    def test_handler_resolves_case_insensitive_entity_type(self) -> None:
        """Verify ``entity_type="ai application"`` folds to canonical casing.

        Quadruplet ``entity_type`` values come from free-text NER output and
        are stored in whatever casing the extractor produced. The handler
        must let the planner query with any casing and still hit.
        """

        store = QuadrupletStore(
            [
                _record(bill_id="b1", entity_type="AI application"),
                _record(bill_id="b2", entity_type="technology"),
            ]
        )
        payload = _query_quadruplets_handler(
            {"entity_type": ["ai application"]}, quadruplet_store=store
        )
        self.assertEqual(payload["match_count"], 1)
        self.assertEqual(
            payload["applied_filters"]["entity_type"], "AI application"
        )


class ToolRegistryIntegrationTests(unittest.TestCase):
    """``build_qa_tool_registry`` only registers ``query_quadruplets`` with data."""

    def test_tool_is_registered_when_store_has_records(self) -> None:
        """A non-empty store causes the registry to expose the new tool."""

        from src.qa.config import AgentConfig
        from src.qa.qa_tools import CitationAccumulator, WorkerCallBudget

        registry = build_qa_tool_registry(
            chunks=[],
            bill_index={},
            search_backend=_NullSearchBackend(),
            worker_client=None,
            worker_model="ignored",
            agent_config=AgentConfig(
                max_planner_turns=1,
                max_planner_tokens=1,
                planner_temperature=0.0,
                max_worker_tokens=1,
                worker_temperature=0.0,
                max_tool_calls=1,
                max_worker_calls=1,
                max_bills_per_list=1,
                max_chunks_per_bill=1,
                max_citations_per_bill=1,
            ),
            accumulator=CitationAccumulator(max_per_bill=1),
            worker_budget=WorkerCallBudget(max_calls=1),
            quadruplet_store=QuadrupletStore([_record()]),
        )
        tool_names = {tool["function"]["name"] for tool in registry.definitions()}
        self.assertIn("query_quadruplets", tool_names)

    def test_tool_is_absent_when_store_is_empty(self) -> None:
        """An empty store disables the tool at registration time."""

        from src.qa.config import AgentConfig
        from src.qa.qa_tools import CitationAccumulator, WorkerCallBudget

        registry = build_qa_tool_registry(
            chunks=[],
            bill_index={},
            search_backend=_NullSearchBackend(),
            worker_client=None,
            worker_model="ignored",
            agent_config=AgentConfig(
                max_planner_turns=1,
                max_planner_tokens=1,
                planner_temperature=0.0,
                max_worker_tokens=1,
                worker_temperature=0.0,
                max_tool_calls=1,
                max_worker_calls=1,
                max_bills_per_list=1,
                max_chunks_per_bill=1,
                max_citations_per_bill=1,
            ),
            accumulator=CitationAccumulator(max_per_bill=1),
            worker_budget=WorkerCallBudget(max_calls=1),
            quadruplet_store=QuadrupletStore.empty(),
        )
        tool_names = {tool["function"]["name"] for tool in registry.definitions()}
        self.assertNotIn("query_quadruplets", tool_names)


class _NullSearchBackend:
    """Minimal ``SearchBackend`` double that returns no chunks."""

    def search(self, query_text, top_k, filters):
        """Return an empty list for any search request."""

        return []


if __name__ == "__main__":
    unittest.main()
