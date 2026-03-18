"""Round-trip and contract tests for Step 1 of the NER plan.

- Verifies artifact serialization, storage round-trips, and schema boundaries.
- Verifies corpus loading for both CSV and JSONL input surfaces.
- Does not execute online LLM calls.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from src.ner.agents.shared import AgentExecutionConfig
from src.ner.agents.zero_shot_annotator import ZeroShotAnnotator
from src.ner.schemas.artifacts import (
    BillRecord,
    CandidateQuadruplet,
    ContextChunk,
    GroupedCandidateSet,
    RefinedQuadruplet,
    RefinementArtifact,
    SpanRef,
)
from src.ner.schemas.constants import CANONICAL_FIELD_ORDER
from src.ner.schemas.inference_unit_builder import ChunkingConfig, InferenceUnitBuilder
from src.ner.schemas.validation import (
    SchemaValidationError,
    validate_context_chunk,
    validate_refinement_artifact,
)
from src.ner.storage.artifact_store import ArtifactStore
from src.ner.storage.corpus_store import CorpusStore


class RoundTripTests(unittest.TestCase):
    """Verify round-trip, shape, and schema guarantees from Step 1."""

    def test_bill_record_round_trips_through_storage(self) -> None:
        """Verify bill records round-trip through JSON storage intact."""

        bill = _make_bill_record()
        with tempfile.TemporaryDirectory() as temporary_directory:
            json_path = Path(temporary_directory) / "bill.json"
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump(bill.to_dict(), handle, indent=2)
            with open(json_path, encoding="utf-8") as handle:
                restored = BillRecord.from_dict(json.load(handle))

        self.assertEqual(restored.bill_id, bill.bill_id)
        self.assertEqual(restored.text, bill.text)
        self.assertEqual(restored.state, bill.state)

    def test_context_chunk_round_trips_through_storage(self) -> None:
        """Verify context chunks preserve ids and offsets when serialized."""

        chunk = _make_context_chunk()
        with tempfile.TemporaryDirectory() as temporary_directory:
            json_path = Path(temporary_directory) / "chunk.json"
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump(chunk.to_dict(), handle, indent=2)
            with open(json_path, encoding="utf-8") as handle:
                restored = ContextChunk.from_dict(json.load(handle))

        self.assertEqual(restored.chunk_id, chunk.chunk_id)
        self.assertEqual(restored.bill_id, chunk.bill_id)
        self.assertEqual(restored.start_offset, chunk.start_offset)
        self.assertEqual(restored.end_offset, chunk.end_offset)

    def test_artifacts_round_trip_through_artifact_store(self) -> None:
        """Verify persisted intermediate artifacts can be reloaded without drift."""

        candidate = _make_candidate()
        grouped_set = _make_grouped_set(candidate.candidate_id)
        refined_output = _make_refined_output(grouped_set.group_id, candidate.candidate_id)
        refinement_artifact = _make_refinement_artifact(grouped_set.group_id, candidate.candidate_id)

        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_store = ArtifactStore(Path(temporary_directory))
            artifact_store.save_candidates("run-1", "bill-1", 1, [candidate], '{"candidates": []}')
            artifact_store.save_grouped("run-1", "bill-1", [grouped_set], '{"groups": []}')
            artifact_store.save_refined_group_outputs(
                "run-1",
                "bill-1",
                grouped_set.group_id,
                [refined_output],
                refinement_artifact,
                '{"refined_quadruplets": []}',
            )

            loaded_candidates = artifact_store.load_candidates("run-1", "bill-1", 1)
            loaded_grouped = artifact_store.load_grouped("run-1", "bill-1")
            loaded_refined, loaded_artifact = artifact_store.load_refined_group_outputs(
                "run-1",
                "bill-1",
                grouped_set.group_id,
            )

        self.assertEqual(loaded_candidates[0].candidate_id, candidate.candidate_id)
        self.assertEqual(
            loaded_grouped[0].field_score_matrix,
            grouped_set.field_score_matrix,
        )
        self.assertEqual(
            loaded_grouped[0].field_order,
            CANONICAL_FIELD_ORDER,
        )
        self.assertEqual(loaded_refined[0].source_group_id, grouped_set.group_id)
        self.assertIsNotNone(loaded_artifact)
        self.assertEqual(loaded_artifact.group_id, grouped_set.group_id)

    def test_field_score_matrix_is_persisted_as_declared_schema_field(self) -> None:
        """Verify grouped-set score matrices are stored as explicit schema fields."""

        grouped_set = _make_grouped_set(101)
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_store = ArtifactStore(Path(temporary_directory))
            artifact_store.save_grouped("run-1", "bill-1", [grouped_set], '{"groups": []}')
            raw_payload = json.loads(
                Path(
                    temporary_directory,
                    "runs",
                    "run-1",
                    "grouped",
                    "bill-1.json",
                ).read_text(encoding="utf-8")
            )

        self.assertIn("field_score_matrix", raw_payload[0])
        self.assertEqual(raw_payload[0]["field_order"], list(CANONICAL_FIELD_ORDER))

    def test_refinement_artifact_rejects_non_canonical_relation_labels(self) -> None:
        """Verify invalid refinement relation labels are rejected."""

        invalid_artifact = RefinementArtifact(
            group_id=10,
            candidate_ids=[101],
            entity_relations=[["bad_label"]],
            type_relations=[[None]],
            attribute_relations=[[None]],
            value_relations=[[None]],
        )
        with self.assertRaises(SchemaValidationError):
            validate_refinement_artifact(invalid_artifact)

    def test_corpus_records_stay_distinct_from_context_chunks(self) -> None:
        """Verify raw bill records remain distinct from derived context chunks."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "subset.csv"
            _write_bill_csv(csv_path, [_make_bill_record()])
            corpus_store = CorpusStore(csv_path)
            [loaded_bill] = corpus_store.load()

        builder = InferenceUnitBuilder(ChunkingConfig(chunk_size=64, overlap=8))
        [loaded_chunk, *_] = builder.build(loaded_bill)
        self.assertIsInstance(loaded_bill, BillRecord)
        self.assertIsInstance(loaded_chunk, ContextChunk)
        self.assertNotEqual(type(loaded_bill), type(loaded_chunk))

    def test_corpus_store_reads_jsonl_realistically(self) -> None:
        """Verify the corpus store reads realistic JSONL bill input."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            jsonl_path = Path(temporary_directory) / "subset.jsonl"
            _write_bill_jsonl(jsonl_path, [_make_bill_record()])
            corpus_store = CorpusStore(jsonl_path)
            [loaded_bill] = corpus_store.load()

        self.assertEqual(loaded_bill.bill_id, "BILL-001")
        self.assertIn("Artificial intelligence systems", loaded_bill.text)
        self.assertEqual(loaded_bill.state, "CA")

    def test_builder_emits_schema_valid_context_chunks(self) -> None:
        """Verify the chunk builder emits schema-valid context chunks."""

        builder = InferenceUnitBuilder(ChunkingConfig(chunk_size=64, overlap=8))
        chunks = builder.build(_make_bill_record())
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            validate_context_chunk(chunk)
            self.assertEqual(chunk.bill_id, "BILL-001")

    def test_builder_prefers_paragraph_boundaries_before_hard_splitting(self) -> None:
        """Verify recursive chunking keeps paragraph boundaries when they fit."""

        builder = InferenceUnitBuilder(ChunkingConfig(chunk_size=30, overlap=5))
        bill = BillRecord(
            bill_id="BILL-PARA",
            state="CA",
            text=("A" * 20) + "\n\n" + ("B" * 20),
        )

        chunks = builder.build(bill)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].end_offset, 22)
        self.assertTrue(chunks[0].text.endswith("\n\n"))
        self.assertEqual(chunks[1].start_offset, 17)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 30)

    def test_builder_falls_back_to_word_boundaries_within_long_paragraphs(self) -> None:
        """Verify recursive chunking uses word separators before hard cuts."""

        builder = InferenceUnitBuilder(ChunkingConfig(chunk_size=15, overlap=2))
        bill = BillRecord(
            bill_id="BILL-WORD",
            state="CA",
            text="alpha beta gamma delta epsilon zeta",
        )

        chunks = builder.build(bill)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 15)
        for chunk in chunks[:-1]:
            self.assertTrue(chunk.text.endswith(" "))

    def test_zero_shot_annotator_rejects_ad_hoc_dict_input(self) -> None:
        """Verify the annotator rejects ad-hoc dictionary input payloads."""

        prompt_executor = Mock()
        prompt_executor.execute.return_value = '{"candidates": []}'
        annotator = ZeroShotAnnotator(
            prompt_template="{text}",
            output_schema={"type": "object"},
            execution_config=AgentExecutionConfig(temperature=0.0, max_tokens=64),
            prompt_executor=prompt_executor,
        )
        with self.assertRaises(SchemaValidationError):
            annotator.run({"text": "not a context chunk"})  # type: ignore[arg-type]


def _make_bill_record() -> BillRecord:
    """Build a representative bill record for round-trip tests.

    Returns:
        Canonical bill record fixture used by test helpers.
    """

    return BillRecord(
        bill_id="BILL-001",
        state="CA",
        bill_url="https://example.com/bill",
        title="AI Bill",
        status="Introduced",
        text=(
            "Artificial intelligence systems used by state agencies must publish "
            "risk documentation and impact disclosures."
        ),
    )


def _make_context_chunk() -> ContextChunk:
    """Build a representative context chunk for round-trip tests.

    Returns:
        Canonical context chunk fixture used by test helpers.
    """

    return ContextChunk(
        chunk_id=1,
        bill_id="BILL-001",
        text="Artificial intelligence systems must publish risk documentation.",
        start_offset=100,
        end_offset=166,
    )


def _make_span() -> SpanRef:
    """Build a representative evidence span for round-trip tests.

    Returns:
        Canonical evidence span fixture used by test helpers.
    """

    return SpanRef(
        span_id=11,
        start=100,
        end=122,
        text="Artificial intelligence",
        chunk_id=1,
    )


def _make_candidate() -> CandidateQuadruplet:
    """Build a representative candidate quadruplet for round-trip tests.

    Returns:
        Canonical candidate fixture used by test helpers.
    """

    return CandidateQuadruplet(
        candidate_id=101,
        entity="artificial intelligence systems",
        type="technology",
        attribute="risk documentation",
        value="must publish",
        entity_evidence=[_make_span()],
    )


def _make_grouped_set(candidate_id: int) -> GroupedCandidateSet:
    """Build a representative grouped candidate set for round-trip tests.

    Args:
        candidate_id: Candidate id to place in the grouped-set fixture.

    Returns:
        Grouped candidate set aligned to the supplied candidate id.
    """

    return GroupedCandidateSet(
        group_id=202,
        candidate_ids=[candidate_id],
        field_score_matrix=[[0.9, 0.8, 0.7, 0.6]],
    )


def _make_refined_output(group_id: int, candidate_id: int) -> RefinedQuadruplet:
    """Build a representative refined quadruplet for round-trip tests.

    Args:
        group_id: Source group id for the refined fixture.
        candidate_id: Source candidate id for the refined fixture.

    Returns:
        Refined quadruplet fixture aligned to the supplied ids.
    """

    return RefinedQuadruplet(
        refined_id=303,
        source_group_id=group_id,
        source_candidate_ids=[candidate_id],
        entity="artificial intelligence systems",
        type="technology",
        attribute="risk documentation",
        value="must publish",
        entity_evidence=[_make_span()],
    )


def _make_refinement_artifact(group_id: int, candidate_id: int) -> RefinementArtifact:
    """Build a representative refinement artifact for round-trip tests.

    Args:
        group_id: Source group id for the artifact fixture.
        candidate_id: Candidate id referenced by the artifact fixture.

    Returns:
        Refinement artifact fixture aligned to the supplied ids.
    """

    return RefinementArtifact(
        group_id=group_id,
        candidate_ids=[candidate_id],
        entity_relations=[[None]],
        type_relations=[[None]],
        attribute_relations=[[None]],
        value_relations=[[None]],
    )


def _write_bill_csv(csv_path: Path, bills: list[BillRecord]) -> None:
    """Write bill fixtures to a temporary CSV file.

    Args:
        csv_path: Destination CSV path.
        bills: Bill fixtures to serialize.
    """

    fieldnames = [
        "state",
        "bill_id",
        "bill_url",
        "title",
        "status",
        "date_of_last_action",
        "author",
        "topics",
        "summary",
        "history",
        "text",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for bill in bills:
            writer.writerow(
                {
                    "state": bill.state,
                    "bill_id": bill.bill_id,
                    "bill_url": bill.bill_url,
                    "title": bill.title,
                    "status": bill.status,
                    "date_of_last_action": bill.date_of_last_action,
                    "author": bill.author,
                    "topics": bill.topics,
                    "summary": bill.summary,
                    "history": bill.history,
                    "text": bill.text,
                }
            )


def _write_bill_jsonl(jsonl_path: Path, bills: list[BillRecord]) -> None:
    """Write bill fixtures to a temporary JSONL file.

    Args:
        jsonl_path: Destination JSONL path.
        bills: Bill fixtures to serialize.
    """

    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for bill in bills:
            handle.write(json.dumps(bill.to_dict(), ensure_ascii=False))
            handle.write("\n")


if __name__ == "__main__":
    unittest.main()

