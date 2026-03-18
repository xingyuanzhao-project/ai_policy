"""Orchestration, resume, and rerun tests for the NER runtime.

- Verifies explicit stage order, durable resume, and rerun invalidation rules.
- Uses deterministic test doubles to isolate orchestration logic from the LLM.
- Does not depend on live online model execution.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from src.ner.agents.base import AgentResult
from src.ner.agents.granularity_refiner import RefinementRequest
from src.ner.orchestration.orchestrator import Orchestrator, StageFailure
from src.ner.schemas.artifacts import (
    CandidateQuadruplet,
    ContextChunk,
    GroupedCandidateSet,
    RefinedQuadruplet,
    RefinementArtifact,
    SpanRef,
)
from src.ner.schemas.validation import SchemaValidationError
from src.ner.storage.artifact_store import ArtifactStore
from src.ner.storage.final_output_store import FinalOutputStore


class OrchestrationTests(unittest.TestCase):
    """Verify explicit stage order, durable resume, and rerun semantics."""

    def test_run_bill_uses_explicit_stage_order_and_marks_completion(self) -> None:
        """Verify the orchestrator executes stages in the planned order."""

        chunks = _make_chunks()
        call_log: list[tuple[str, int]] = []
        with tempfile.TemporaryDirectory() as temporary_directory:
            orchestrator, artifact_store, output_store = _build_orchestrator(
                base_dir=Path(temporary_directory),
                call_log=call_log,
            )

            refined_outputs = asyncio.run(
                orchestrator.run_bill("BILL-001", chunks, resume=True)
            )

            self.assertEqual(
                call_log,
                [
                    ("annotation", chunks[0].chunk_id),
                    ("annotation", chunks[1].chunk_id),
                    ("grouping", 2),
                    ("refinement", 1001),
                    ("refinement", 1002),
                ],
            )
            self.assertEqual(len(refined_outputs), 2)
            self.assertTrue(artifact_store.is_stage_complete("run-1", "BILL-001", "annotation"))
            self.assertTrue(artifact_store.is_stage_complete("run-1", "BILL-001", "grouping"))
            self.assertTrue(artifact_store.is_stage_complete("run-1", "BILL-001", "refinement"))
            self.assertTrue(output_store.exists("run-1", "BILL-001"))

    def test_annotation_resume_uses_saved_chunk_outputs(self) -> None:
        """Verify annotation resume skips chunks that were already persisted."""

        chunks = _make_chunks()
        failing_call_log: list[tuple[str, int]] = []
        stable_call_log: list[tuple[str, int]] = []

        with tempfile.TemporaryDirectory() as temporary_directory:
            base_dir = Path(temporary_directory)
            failing_orchestrator, artifact_store, output_store = _build_orchestrator(
                base_dir=base_dir,
                call_log=failing_call_log,
                fail_chunk_id=chunks[1].chunk_id,
            )
            with self.assertRaises(StageFailure):
                asyncio.run(
                    failing_orchestrator.run_bill("BILL-001", chunks, resume=True)
                )

            self.assertTrue(
                artifact_store.candidate_chunk_exists("run-1", "BILL-001", chunks[0].chunk_id)
            )
            self.assertFalse(
                artifact_store.candidate_chunk_exists("run-1", "BILL-001", chunks[1].chunk_id)
            )
            self.assertFalse(
                artifact_store.is_stage_complete("run-1", "BILL-001", "annotation")
            )

            resumed_orchestrator, _, _ = _build_orchestrator(
                base_dir=base_dir,
                call_log=stable_call_log,
            )
            refined_outputs = asyncio.run(
                resumed_orchestrator.run_bill("BILL-001", chunks, resume=True)
            )

            self.assertEqual(
                stable_call_log,
                [
                    ("annotation", chunks[1].chunk_id),
                    ("grouping", 2),
                    ("refinement", 1001),
                    ("refinement", 1002),
                ],
            )
            self.assertEqual(len(refined_outputs), 2)
            self.assertTrue(
                artifact_store.is_stage_complete("run-1", "BILL-001", "annotation")
            )
            self.assertTrue(output_store.exists("run-1", "BILL-001"))

    def test_refinement_resume_skips_completed_annotation_and_grouping(self) -> None:
        """Verify refinement resume reuses completed upstream stage outputs."""

        chunks = _make_chunks()
        failing_refiner_log: list[tuple[str, int]] = []
        resumed_refiner_log: list[tuple[str, int]] = []

        with tempfile.TemporaryDirectory() as temporary_directory:
            base_dir = Path(temporary_directory)
            failing_orchestrator, artifact_store, output_store = _build_orchestrator(
                base_dir=base_dir,
                call_log=failing_refiner_log,
                fail_group_id=1002,
            )
            with self.assertRaises(StageFailure):
                asyncio.run(
                    failing_orchestrator.run_bill("BILL-001", chunks, resume=True)
                )

            self.assertTrue(
                artifact_store.refined_group_exists("run-1", "BILL-001", 1001)
            )
            self.assertFalse(
                artifact_store.refined_group_exists("run-1", "BILL-001", 1002)
            )
            self.assertFalse(
                artifact_store.is_stage_complete("run-1", "BILL-001", "refinement")
            )

            resumed_orchestrator = Orchestrator(
                run_id="run-1",
                annotator=_RaiseIfCalledAnnotator(),
                assembler=_RaiseIfCalledAssembler(),
                refiner=_RecordingRefiner(resumed_refiner_log),
                artifact_store=artifact_store,
                final_output_store=output_store,
                concurrency=1,
            )
            refined_outputs = asyncio.run(
                resumed_orchestrator.run_bill("BILL-001", chunks, resume=True)
            )

            self.assertEqual(resumed_refiner_log, [("refinement", 1002)])
            self.assertEqual(len(refined_outputs), 2)
            self.assertTrue(
                artifact_store.is_stage_complete("run-1", "BILL-001", "refinement")
            )
            self.assertTrue(output_store.exists("run-1", "BILL-001"))

    def test_rerun_chunk_invalidates_grouping_and_refinement(self) -> None:
        """Verify rerunning one chunk invalidates all downstream persisted stages."""

        chunks = _make_chunks()
        with tempfile.TemporaryDirectory() as temporary_directory:
            orchestrator, artifact_store, output_store = _build_orchestrator(
                base_dir=Path(temporary_directory),
                call_log=[],
            )
            asyncio.run(orchestrator.run_bill("BILL-001", chunks, resume=True))
            self.assertTrue(output_store.exists("run-1", "BILL-001"))

            rerun_candidates = orchestrator.rerun_chunk(
                bill_id="BILL-001",
                chunks=chunks,
                chunk_id=chunks[0].chunk_id,
            )

            self.assertEqual(len(rerun_candidates), 1)
            self.assertTrue(
                artifact_store.is_stage_complete("run-1", "BILL-001", "annotation")
            )
            self.assertFalse(
                artifact_store.is_stage_complete("run-1", "BILL-001", "grouping")
            )
            self.assertFalse(
                artifact_store.is_stage_complete("run-1", "BILL-001", "refinement")
            )
            self.assertFalse(output_store.exists("run-1", "BILL-001"))

    def test_rerun_group_reassembles_final_outputs(self) -> None:
        """Verify rerunning one group rebuilds final bill-level outputs."""

        chunks = _make_chunks()
        with tempfile.TemporaryDirectory() as temporary_directory:
            base_dir = Path(temporary_directory)
            initial_orchestrator, artifact_store, output_store = _build_orchestrator(
                base_dir=base_dir,
                call_log=[],
            )
            asyncio.run(
                initial_orchestrator.run_bill("BILL-001", chunks, resume=True)
            )

            rerun_log: list[tuple[str, int]] = []
            rerun_orchestrator = Orchestrator(
                run_id="run-1",
                annotator=_RaiseIfCalledAnnotator(),
                assembler=_RaiseIfCalledAssembler(),
                refiner=_RecordingRefiner(rerun_log, replacement_value="rerun-value"),
                artifact_store=artifact_store,
                final_output_store=output_store,
                concurrency=1,
            )
            rerun_outputs = rerun_orchestrator.rerun_group("BILL-001", 1001)
            all_outputs = output_store.load("run-1", "BILL-001")

            self.assertEqual(rerun_log, [("refinement", 1001)])
            self.assertEqual(len(rerun_outputs), 1)
            self.assertEqual(len(all_outputs), 2)
            self.assertTrue(
                artifact_store.is_stage_complete("run-1", "BILL-001", "refinement")
            )


class _RecordingAnnotator:
    """Deterministic annotation-stage test double used by orchestration tests."""

    def __init__(self, call_log: list[tuple[str, int]], fail_chunk_id: int | None = None) -> None:
        """Initialize the recording annotator test double.

        Args:
            call_log: Shared list used to record call order and identifiers.
            fail_chunk_id: Optional chunk id that should raise once when seen.
        """

        self._call_log = call_log
        self._fail_chunk_id = fail_chunk_id
        self._failed = False

    def run(self, chunk: ContextChunk) -> AgentResult[list[CandidateQuadruplet]]:
        """Return deterministic candidate output for one chunk.

        Args:
            chunk: Context chunk supplied by the orchestrator.

        Returns:
            Deterministic agent result with one candidate per chunk.

        Raises:
            SchemaValidationError: If the configured failing chunk id is seen for
                the first time.
        """

        if self._fail_chunk_id == chunk.chunk_id and not self._failed:
            self._failed = True
            raise SchemaValidationError("synthetic annotation failure")
        self._call_log.append(("annotation", chunk.chunk_id))
        candidate = CandidateQuadruplet(
            candidate_id=chunk.chunk_id,
            entity=f"entity-{chunk.chunk_id}",
            type="technology",
            attribute="definition",
            value=f"value-{chunk.chunk_id}",
            entity_evidence=[
                SpanRef(
                    span_id=chunk.chunk_id + 5000,
                    start=chunk.start_offset,
                    end=min(chunk.start_offset + 10, chunk.end_offset),
                    text=chunk.text[:10],
                    chunk_id=chunk.chunk_id,
                )
            ],
        )
        return AgentResult(
            input_schema_name="ContextChunk",
            output_schema_name="list[CandidateQuadruplet]",
            raw_response='{"candidates": []}',
            parsed_response=[candidate],
        )


class _RecordingAssembler:
    """Deterministic grouping-stage test double used by orchestration tests."""

    def __init__(self, call_log: list[tuple[str, int]]) -> None:
        """Initialize the recording assembler test double.

        Args:
            call_log: Shared list used to record call order and identifiers.
        """

        self._call_log = call_log

    def run(self, candidates: list[CandidateQuadruplet]) -> AgentResult[list[GroupedCandidateSet]]:
        """Return deterministic grouped-set output for the supplied candidates.

        Args:
            candidates: Candidate pool supplied by the orchestrator.

        Returns:
            Deterministic grouped-set result with one group per candidate.
        """

        self._call_log.append(("grouping", len(candidates)))
        groups = [
            GroupedCandidateSet(
                group_id=1000 + index + 1,
                candidate_ids=[candidate.candidate_id],
                field_score_matrix=[[1.0, 1.0, 1.0, 1.0]],
            )
            for index, candidate in enumerate(candidates)
        ]
        return AgentResult(
            input_schema_name="list[CandidateQuadruplet]",
            output_schema_name="list[GroupedCandidateSet]",
            raw_response='{"groups": []}',
            parsed_response=groups,
        )


class _RecordingRefiner:
    """Deterministic refinement-stage test double used by orchestration tests."""

    def __init__(
        self,
        call_log: list[tuple[str, int]],
        fail_group_id: int | None = None,
        replacement_value: str | None = None,
    ) -> None:
        """Initialize the recording refiner test double.

        Args:
            call_log: Shared list used to record call order and identifiers.
            fail_group_id: Optional group id that should raise once when seen.
            replacement_value: Optional value override injected into refined
                outputs for rerun assertions.
        """

        self._call_log = call_log
        self._fail_group_id = fail_group_id
        self._failed = False
        self._replacement_value = replacement_value

    def run(
        self,
        refinement_request: RefinementRequest,
    ) -> AgentResult[tuple[list[RefinedQuadruplet], RefinementArtifact | None]]:
        """Return deterministic refinement output for one grouped candidate set.

        Args:
            refinement_request: Grouped-set refinement request supplied by the
                orchestrator.

        Returns:
            Deterministic refined output plus optional refinement artifact.

        Raises:
            SchemaValidationError: If the configured failing group id is seen
                for the first time.
        """

        grouped_set = refinement_request.grouped_candidate_set
        if self._fail_group_id == grouped_set.group_id and not self._failed:
            self._failed = True
            raise SchemaValidationError("synthetic refinement failure")

        self._call_log.append(("refinement", grouped_set.group_id))
        candidate_id = grouped_set.candidate_ids[0]
        refined_output = RefinedQuadruplet(
            refined_id=2000 + grouped_set.group_id,
            source_group_id=grouped_set.group_id,
            source_candidate_ids=[candidate_id],
            entity=f"entity-{candidate_id}",
            type="technology",
            attribute="definition",
            value=self._replacement_value or f"value-{candidate_id}",
            entity_evidence=[
                SpanRef(
                    span_id=3000 + grouped_set.group_id,
                    start=0,
                    end=5,
                    text="entity",
                    chunk_id=candidate_id,
                )
            ],
        )
        refinement_artifact = RefinementArtifact(
            group_id=grouped_set.group_id,
            candidate_ids=grouped_set.candidate_ids,
            entity_relations=[[None]],
            type_relations=[[None]],
            attribute_relations=[[None]],
            value_relations=[[None]],
        )
        return AgentResult(
            input_schema_name="RefinementRequest",
            output_schema_name="tuple[list[RefinedQuadruplet], RefinementArtifact | None]",
            raw_response='{"refined_quadruplets": []}',
            parsed_response=([refined_output], refinement_artifact),
        )


class _RaiseIfCalledAnnotator:
    """Sentinel annotator that fails when resume logic calls annotation unexpectedly."""

    def run(self, _: ContextChunk) -> AgentResult[list[CandidateQuadruplet]]:
        """Fail the test if annotation is invoked unexpectedly.

        Args:
            _: Ignored chunk argument.

        Returns:
            This method never returns.

        Raises:
            AssertionError: Always, because annotation should have resumed from
                persisted outputs.
        """

        raise AssertionError("Annotation should have resumed from persisted outputs")


class _RaiseIfCalledAssembler:
    """Sentinel assembler that fails when resume logic calls grouping unexpectedly."""

    def run(self, _: list[CandidateQuadruplet]) -> AgentResult[list[GroupedCandidateSet]]:
        """Fail the test if grouping is invoked unexpectedly.

        Args:
            _: Ignored candidate-pool argument.

        Returns:
            This method never returns.

        Raises:
            AssertionError: Always, because grouping should have resumed from
                persisted outputs.
        """

        raise AssertionError("Grouping should have resumed from persisted outputs")


def _build_orchestrator(
    base_dir: Path,
    call_log: list[tuple[str, int]],
    fail_chunk_id: int | None = None,
    fail_group_id: int | None = None,
) -> tuple[Orchestrator, ArtifactStore, FinalOutputStore]:
    """Build a test orchestrator wired to deterministic test doubles.

    Args:
        base_dir: Temporary artifact directory for the test run.
        call_log: Shared list used to record call order and identifiers.
        fail_chunk_id: Optional chunk id that should fail once during annotation.
        fail_group_id: Optional group id that should fail once during
            refinement.

    Returns:
        Tuple of test orchestrator, artifact store, and final-output store.
    """

    artifact_store = ArtifactStore(base_dir)
    output_store = FinalOutputStore(base_dir)
    orchestrator = Orchestrator(
        run_id="run-1",
        annotator=_RecordingAnnotator(call_log, fail_chunk_id=fail_chunk_id),
        assembler=_RecordingAssembler(call_log),
        refiner=_RecordingRefiner(call_log, fail_group_id=fail_group_id),
        artifact_store=artifact_store,
        final_output_store=output_store,
        concurrency=1,
    )
    return orchestrator, artifact_store, output_store


def _make_chunks() -> list[ContextChunk]:
    """Build a representative chunk list for orchestration tests.

    Returns:
        Ordered chunk fixtures used across orchestration tests.
    """

    return [
        ContextChunk(
            chunk_id=1,
            bill_id="BILL-001",
            text="Artificial intelligence systems must publish documentation.",
            start_offset=0,
            end_offset=59,
        ),
        ContextChunk(
            chunk_id=2,
            bill_id="BILL-001",
            text="State agencies must disclose high-risk AI use cases.",
            start_offset=59,
            end_offset=113,
        ),
    ]


if __name__ == "__main__":
    unittest.main()

