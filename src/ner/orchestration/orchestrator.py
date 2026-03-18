"""Explicit stage orchestration for the NER multi-agent pipeline."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ..agents.eval_assembler import EvalAssembler
from ..agents.granularity_refiner import GranularityRefiner, RefinementRequest
from ..agents.zero_shot_annotator import ZeroShotAnnotator
from ..schemas.artifacts import CandidateQuadruplet, ContextChunk, GroupedCandidateSet, RefinedQuadruplet
from ..schemas.validation import SchemaValidationError
from ..storage.artifact_store import ArtifactStore
from ..storage.final_output_store import FinalOutputStore


@dataclass(slots=True)
class StageFailure(RuntimeError):
    """Raised when one stage fails for a specific bill.

    Attributes:
        stage: Name of the failed pipeline stage.
        bill_id: Bill identifier whose processing failed.
        detail: Human-readable validation or runtime failure detail.
    """

    stage: str
    bill_id: str
    detail: str

    def __str__(self) -> str:
        """Return the formatted failure message.

        Returns:
            Combined stage and bill-level error message.
        """

        return f"Stage '{self.stage}' failed for bill '{self.bill_id}': {self.detail}"


class Orchestrator:
    """Run the three NER stages in explicit order with durable resume state."""

    def __init__(
        self,
        run_id: str,
        annotator: ZeroShotAnnotator,
        assembler: EvalAssembler,
        refiner: GranularityRefiner,
        artifact_store: ArtifactStore,
        final_output_store: FinalOutputStore,
        concurrency: int = 4,
    ) -> None:
        """Initialize the pipeline orchestrator.

        Args:
            run_id: Stable run identifier used for persisted artifacts.
            annotator: Chunk-level zero-shot annotation stage.
            assembler: Bill-level candidate grouping stage.
            refiner: Group-level refinement stage.
            artifact_store: Store used for intermediate artifacts and stage state.
            final_output_store: Store used for bill-level final outputs.
            concurrency: Maximum number of concurrent LLM calls within a single
                fan-out stage (annotation chunks or refinement groups).
        """

        self._run_id = run_id
        self._annotator = annotator
        self._assembler = assembler
        self._refiner = refiner
        self._artifact_store = artifact_store
        self._final_output_store = final_output_store
        self._concurrency = concurrency

    async def run_bill(
        self,
        bill_id: str,
        chunks: list[ContextChunk],
        resume: bool = True,
    ) -> list[RefinedQuadruplet]:
        """Run one bill through annotation, grouping, and refinement.

        Annotation and refinement fan out their independent work units
        (chunks and groups respectively) with bounded concurrency. Grouping
        remains a synchronous single-call barrier between the two stages.

        Args:
            bill_id: Bill identifier being processed.
            chunks: Ordered chunk list derived from the bill text.
            resume: Whether previously completed stage outputs may be reused.

        Returns:
            Final refined quadruplets for the bill.
        """

        if not chunks:
            self._artifact_store.mark_stage_complete(
                self._run_id,
                bill_id,
                "annotation",
                {"chunk_ids_in_order": [], "candidate_counts_by_chunk": {}},
            )
            self._artifact_store.mark_stage_complete(
                self._run_id,
                bill_id,
                "grouping",
                {"group_ids_in_order": []},
            )
            self._final_output_store.save(self._run_id, bill_id, [])
            self._artifact_store.mark_stage_complete(
                self._run_id,
                bill_id,
                "refinement",
                {"group_ids_in_order": [], "refined_output_count": 0},
            )
            return []

        candidates = await self._run_annotation(bill_id=bill_id, chunks=chunks, resume=resume)
        grouped_sets = self._run_grouping(
            bill_id=bill_id,
            candidates=candidates,
            resume=resume,
        )
        refined_outputs = await self._run_refinement(
            bill_id=bill_id,
            grouped_sets=grouped_sets,
            candidates=candidates,
            resume=resume,
        )
        return refined_outputs

    def rerun_chunk(
        self,
        bill_id: str,
        chunks: list[ContextChunk],
        chunk_id: int,
    ) -> list[CandidateQuadruplet]:
        """Rerun one chunk's annotation output and invalidate downstream stages.

        Args:
            bill_id: Bill identifier owning the chunk.
            chunks: Full ordered chunk list for the bill.
            chunk_id: Stable chunk identifier to rerun.

        Returns:
            Candidate quadruplets regenerated for the target chunk.

        Raises:
            KeyError: If the requested chunk id does not belong to the bill.
            StageFailure: If annotation fails validation or runtime execution.
        """

        target_chunk = next((chunk for chunk in chunks if chunk.chunk_id == chunk_id), None)
        if target_chunk is None:
            raise KeyError(f"Chunk '{chunk_id}' is not part of bill '{bill_id}'")

        try:
            result = self._annotator.run(target_chunk)
        except SchemaValidationError as exc:
            raise StageFailure("annotation", bill_id, str(exc)) from exc

        self._artifact_store.save_candidates(
            self._run_id,
            bill_id,
            target_chunk.chunk_id,
            result.parsed_response,
            result.raw_response,
        )
        self._artifact_store.invalidate_from_grouping(self._run_id, bill_id)
        self._final_output_store.delete(self._run_id, bill_id)
        self._artifact_store.mark_stage_complete(
            self._run_id,
            bill_id,
            "annotation",
            self._annotation_stage_payload(bill_id, chunks),
        )
        return result.parsed_response

    def rerun_group(
        self,
        bill_id: str,
        group_id: int,
    ) -> list[RefinedQuadruplet]:
        """Rerun refinement for one grouped candidate set and rebuild final outputs.

        Args:
            bill_id: Bill identifier owning the grouped candidate set.
            group_id: Stable grouped-set identifier to rerun.

        Returns:
            Refined quadruplets regenerated for the target grouped set.

        Raises:
            KeyError: If the requested group id does not belong to the bill.
            StageFailure: If refinement fails validation or runtime execution.
        """

        grouped_sets = self._load_grouped_sets_for_resume(bill_id)
        target_group = next(
            (grouped_set for grouped_set in grouped_sets if grouped_set.group_id == group_id),
            None,
        )
        if target_group is None:
            raise KeyError(f"Group '{group_id}' is not part of bill '{bill_id}'")

        annotation_state = self._artifact_store.load_stage_state(self._run_id, bill_id, "annotation")
        chunk_ids_in_order = annotation_state.get("chunk_ids_in_order", [])
        candidate_pool = {
            candidate.candidate_id: candidate
            for candidate in self._artifact_store.load_all_candidates_for_bill(
                self._run_id,
                bill_id,
                chunk_ids_in_order=[int(chunk_id) for chunk_id in chunk_ids_in_order],
            )
        }

        self._artifact_store.invalidate_refinement(self._run_id, bill_id, group_id=group_id)
        self._final_output_store.delete(self._run_id, bill_id)
        group_outputs = self._run_refinement_for_group(
            bill_id=bill_id,
            grouped_set=target_group,
            candidate_pool_by_id=candidate_pool,
        )

        all_refined_outputs, _ = self._artifact_store.load_all_refined_for_bill(
            self._run_id,
            bill_id,
            group_ids_in_order=[grouped_set.group_id for grouped_set in grouped_sets],
        )
        self._final_output_store.save(self._run_id, bill_id, all_refined_outputs)
        self._artifact_store.mark_stage_complete(
            self._run_id,
            bill_id,
            "refinement",
            self._refinement_stage_payload(bill_id, grouped_sets),
        )
        return group_outputs

    async def _run_annotation(
        self,
        bill_id: str,
        chunks: list[ContextChunk],
        resume: bool,
    ) -> list[CandidateQuadruplet]:
        """Run or resume chunk-level zero-shot annotation for one bill.

        Independent chunks are dispatched with bounded concurrency via
        ``asyncio.to_thread``. Cached chunks bypass the semaphore entirely so
        they do not consume concurrency slots.

        Args:
            bill_id: Bill identifier being processed.
            chunks: Ordered chunk list derived from the bill text.
            resume: Whether persisted chunk outputs may be reused.

        Returns:
            Bill-level candidate pool aggregated in chunk order.

        Raises:
            StageFailure: If annotation fails validation or runtime execution.
        """

        semaphore = asyncio.Semaphore(self._concurrency)

        async def _annotate_or_resume(chunk: ContextChunk) -> list[CandidateQuadruplet]:
            if resume and self._artifact_store.candidate_chunk_exists(
                self._run_id, bill_id, chunk.chunk_id,
            ):
                return self._artifact_store.load_candidates(
                    self._run_id, bill_id, chunk.chunk_id,
                )
            async with semaphore:
                return await asyncio.to_thread(
                    self._annotate_single_chunk, bill_id, chunk,
                )

        chunk_results = await asyncio.gather(
            *[_annotate_or_resume(chunk) for chunk in chunks]
        )

        candidates: list[CandidateQuadruplet] = []
        for chunk_candidates in chunk_results:
            candidates.extend(chunk_candidates)

        self._artifact_store.mark_stage_complete(
            self._run_id,
            bill_id,
            "annotation",
            self._annotation_stage_payload(bill_id, chunks),
        )
        return candidates

    def _annotate_single_chunk(
        self,
        bill_id: str,
        chunk: ContextChunk,
    ) -> list[CandidateQuadruplet]:
        """Run the annotator on one chunk and persist its output.

        This method is the unit of work dispatched to a thread by the bounded
        annotation fan-out. It is also usable from sync callers directly.

        Args:
            bill_id: Bill identifier owning the chunk.
            chunk: Context chunk to annotate.

        Returns:
            Candidate quadruplets extracted from the chunk.

        Raises:
            StageFailure: If annotation fails validation or runtime execution.
        """

        try:
            result = self._annotator.run(chunk)
        except SchemaValidationError as exc:
            raise StageFailure("annotation", bill_id, str(exc)) from exc

        self._artifact_store.save_candidates(
            self._run_id,
            bill_id,
            chunk.chunk_id,
            result.parsed_response,
            result.raw_response,
        )
        return result.parsed_response

    def _run_grouping(
        self,
        bill_id: str,
        candidates: list[CandidateQuadruplet],
        resume: bool,
    ) -> list[GroupedCandidateSet]:
        """Run or resume bill-level candidate grouping.

        Args:
            bill_id: Bill identifier being processed.
            candidates: Bill-level candidate pool aggregated from all chunks.
            resume: Whether persisted grouped outputs may be reused.

        Returns:
            Grouped candidate sets for the bill.

        Raises:
            StageFailure: If grouping fails validation or runtime execution.
        """

        if resume and self._artifact_store.is_stage_complete(self._run_id, bill_id, "grouping"):
            return self._load_grouped_sets_for_resume(bill_id)

        try:
            result = self._assembler.run(candidates)
        except SchemaValidationError as exc:
            raise StageFailure("grouping", bill_id, str(exc)) from exc

        self._artifact_store.save_grouped(
            self._run_id,
            bill_id,
            result.parsed_response,
            result.raw_response,
        )
        self._artifact_store.mark_stage_complete(
            self._run_id,
            bill_id,
            "grouping",
            {
                "group_ids_in_order": [
                    grouped_set.group_id for grouped_set in result.parsed_response
                ]
            },
        )
        return result.parsed_response

    async def _run_refinement(
        self,
        bill_id: str,
        grouped_sets: list[GroupedCandidateSet],
        candidates: list[CandidateQuadruplet],
        resume: bool,
    ) -> list[RefinedQuadruplet]:
        """Run or resume group-level refinement and assemble bill-level outputs.

        Independent groups are dispatched with bounded concurrency via
        ``asyncio.to_thread``. Cached groups bypass the semaphore entirely so
        they do not consume concurrency slots.

        Args:
            bill_id: Bill identifier being processed.
            grouped_sets: Ordered grouped candidate sets for the bill.
            candidates: Bill-level candidate pool referenced by group ids.
            resume: Whether persisted refinement outputs may be reused.

        Returns:
            Bill-level refined quadruplets assembled in grouped-set order.

        Raises:
            StageFailure: If refinement fails validation or runtime execution.
        """

        if resume and self._artifact_store.is_stage_complete(
            self._run_id,
            bill_id,
            "refinement",
        ):
            return self._final_output_store.load(self._run_id, bill_id)

        candidate_pool_by_id = {
            candidate.candidate_id: candidate for candidate in candidates
        }
        semaphore = asyncio.Semaphore(self._concurrency)

        async def _refine_or_resume(
            grouped_set: GroupedCandidateSet,
        ) -> list[RefinedQuadruplet]:
            if resume and self._artifact_store.refined_group_exists(
                self._run_id, bill_id, grouped_set.group_id,
            ):
                persisted_group_outputs, _ = self._artifact_store.load_refined_group_outputs(
                    self._run_id, bill_id, grouped_set.group_id,
                )
                return persisted_group_outputs
            async with semaphore:
                return await asyncio.to_thread(
                    self._run_refinement_for_group,
                    bill_id,
                    grouped_set,
                    candidate_pool_by_id,
                )

        group_results = await asyncio.gather(
            *[_refine_or_resume(gs) for gs in grouped_sets]
        )

        refined_outputs: list[RefinedQuadruplet] = []
        for group_outputs in group_results:
            refined_outputs.extend(group_outputs)

        self._final_output_store.save(self._run_id, bill_id, refined_outputs)
        self._artifact_store.mark_stage_complete(
            self._run_id,
            bill_id,
            "refinement",
            self._refinement_stage_payload(bill_id, grouped_sets),
        )
        return refined_outputs

    def _run_refinement_for_group(
        self,
        bill_id: str,
        grouped_set: GroupedCandidateSet,
        candidate_pool_by_id: dict[int, CandidateQuadruplet],
    ) -> list[RefinedQuadruplet]:
        """Run refinement for one grouped candidate set and persist group outputs.

        Args:
            bill_id: Bill identifier being processed.
            grouped_set: Grouped candidate set to refine.
            candidate_pool_by_id: Candidate lookup table referenced by the group.

        Returns:
            Refined quadruplets emitted for the grouped set.

        Raises:
            StageFailure: If the refiner fails validation or runtime execution.
        """

        try:
            result = self._refiner.run(
                RefinementRequest(
                    grouped_candidate_set=grouped_set,
                    candidate_pool_by_id=candidate_pool_by_id,
                )
            )
        except SchemaValidationError as exc:
            raise StageFailure("refinement", bill_id, str(exc)) from exc

        refined_outputs, refinement_artifact = result.parsed_response
        self._artifact_store.save_refined_group_outputs(
            self._run_id,
            bill_id,
            grouped_set.group_id,
            refined_outputs,
            refinement_artifact,
            result.raw_response,
        )
        return refined_outputs

    def _annotation_stage_payload(
        self,
        bill_id: str,
        chunks: list[ContextChunk],
    ) -> dict[str, object]:
        """Build durable annotation-stage completion metadata.

        Args:
            bill_id: Bill identifier whose annotation stage completed.
            chunks: Ordered chunk list processed for the bill.

        Returns:
            Serializable stage-state payload used for durable resume.
        """

        candidate_counts_by_chunk: dict[str, int] = {}
        for chunk in chunks:
            persisted_candidates = self._artifact_store.load_candidates(
                self._run_id,
                bill_id,
                chunk.chunk_id,
            )
            candidate_counts_by_chunk[str(chunk.chunk_id)] = len(persisted_candidates)

        return {
            "chunk_ids_in_order": [chunk.chunk_id for chunk in chunks],
            "candidate_counts_by_chunk": candidate_counts_by_chunk,
        }

    def _refinement_stage_payload(
        self,
        bill_id: str,
        grouped_sets: list[GroupedCandidateSet],
    ) -> dict[str, object]:
        """Build durable refinement-stage completion metadata.

        Args:
            bill_id: Bill identifier whose refinement stage completed.
            grouped_sets: Ordered grouped candidate sets processed for the bill.

        Returns:
            Serializable stage-state payload used for durable resume.
        """

        refined_outputs, refinement_artifacts = self._artifact_store.load_all_refined_for_bill(
            self._run_id,
            bill_id,
            group_ids_in_order=[grouped_set.group_id for grouped_set in grouped_sets],
        )
        return {
            "group_ids_in_order": [grouped_set.group_id for grouped_set in grouped_sets],
            "refined_output_count": len(refined_outputs),
            "refinement_artifact_group_ids": [
                refinement_artifact.group_id for refinement_artifact in refinement_artifacts
            ],
        }

    def _load_grouped_sets_for_resume(self, bill_id: str) -> list[GroupedCandidateSet]:
        """Load grouped sets for a bill using durable grouping-stage evidence.

        Args:
            bill_id: Bill identifier whose grouped outputs should be reloaded.

        Returns:
            Persisted grouped candidate sets for the bill.

        Raises:
            StageFailure: If grouping stage markers exist but grouped outputs are
                missing on disk.
        """

        if not self._artifact_store.grouped_exists(self._run_id, bill_id):
            raise StageFailure(
                "grouping",
                bill_id,
                "Grouping stage marker exists but grouped output files are missing",
            )
        return self._artifact_store.load_grouped(self._run_id, bill_id)

