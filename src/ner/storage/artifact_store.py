"""Persistence for intermediate NER artifacts and stage completion state.

- Owns run-scoped persistence for intermediate parsed artifacts and raw LLM
  responses.
- Owns durable stage-state markers used for resume and rerun behavior.
- Does not validate artifacts, call the LLM, or assemble bill-level outputs.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ..schemas.artifacts import (
    CandidateQuadruplet,
    GroupedCandidateSet,
    RefinedQuadruplet,
    RefinementArtifact,
)


class ArtifactStore:
    """Persist parsed artifacts, raw responses, and durable stage metadata."""

    def __init__(self, base_dir: Path) -> None:
        """Initialize the artifact store.

        Args:
            base_dir (Path): Base directory under which run-scoped artifacts are
                stored.
        """

        self._base_dir = base_dir

    def run_dir(self, run_id: str) -> Path:
        """Return the root directory for one run id.

        Args:
            run_id (str): Stable run identifier.

        Returns:
            Path: Filesystem path to the run directory.
        """

        return self._base_dir / "runs" / run_id

    def _write_json(self, path: Path, payload: Any) -> None:
        """Write a JSON payload to disk.

        Args:
            path (Path): Destination file path.
            payload (Any): JSON-serializable payload to persist.

        Returns:
            None: This method writes the payload to disk in place.
        """

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def _read_json(self, path: Path) -> Any:
        """Read a JSON payload from disk.

        Args:
            path (Path): Source file path.

        Returns:
            Any: Decoded JSON payload.
        """

        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    def _delete_path(self, path: Path) -> None:
        """Delete a file or directory if it exists.

        Args:
            path (Path): Filesystem path to remove.

        Returns:
            None: This method deletes the target when it exists.
        """

        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()

    def _candidate_path(self, run_id: str, bill_id: str, chunk_id: int) -> Path:
        """Return the parsed candidate path for one chunk.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.
            chunk_id (int): Stable chunk identifier.

        Returns:
            Path: Filesystem path for parsed chunk candidates.
        """

        return self.run_dir(run_id) / "candidates" / bill_id / f"{chunk_id}.json"

    def _candidate_raw_path(self, run_id: str, bill_id: str, chunk_id: int) -> Path:
        """Return the raw candidate-response path for one chunk.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.
            chunk_id (int): Stable chunk identifier.

        Returns:
            Path: Filesystem path for the raw chunk response.
        """

        return self.run_dir(run_id) / "candidates" / bill_id / f"{chunk_id}.raw.json"

    def _grouped_path(self, run_id: str, bill_id: str) -> Path:
        """Return the parsed grouped-set path for one bill.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.

        Returns:
            Path: Filesystem path for grouped candidate sets.
        """

        return self.run_dir(run_id) / "grouped" / f"{bill_id}.json"

    def _grouped_raw_path(self, run_id: str, bill_id: str) -> Path:
        """Return the raw grouping-response path for one bill.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.

        Returns:
            Path: Filesystem path for the raw grouping response.
        """

        return self.run_dir(run_id) / "grouped" / f"{bill_id}.raw.json"

    def _refined_group_path(self, run_id: str, bill_id: str, group_id: int) -> Path:
        """Return the parsed refinement-output path for one group.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.
            group_id (int): Stable grouped-set identifier.

        Returns:
            Path: Filesystem path for parsed group refinement output.
        """

        return self.run_dir(run_id) / "refined_outputs" / bill_id / f"{group_id}.json"

    def _refined_group_raw_path(self, run_id: str, bill_id: str, group_id: int) -> Path:
        """Return the raw refinement-response path for one group.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.
            group_id (int): Stable grouped-set identifier.

        Returns:
            Path: Filesystem path for the raw group refinement response.
        """

        return self.run_dir(run_id) / "refined_outputs" / bill_id / f"{group_id}.raw.json"

    def _refinement_artifact_path(
        self,
        run_id: str,
        bill_id: str,
        group_id: int,
    ) -> Path:
        """Return the optional refinement-artifact path for one group.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.
            group_id (int): Stable grouped-set identifier.

        Returns:
            Path: Filesystem path for the refinement artifact.
        """

        return self.run_dir(run_id) / "refinement_artifacts" / bill_id / f"{group_id}.json"

    def _stage_state_path(self, run_id: str, bill_id: str, stage: str) -> Path:
        """Return the stage-state marker path for one bill stage.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.
            stage (str): Stage name whose state marker is requested.

        Returns:
            Path: Filesystem path for the stage-state marker.
        """

        return self.run_dir(run_id) / "stage_state" / bill_id / f"{stage}.json"

    def save_candidates(
        self,
        run_id: str,
        bill_id: str,
        chunk_id: int,
        candidates: list[CandidateQuadruplet],
        raw_response: str,
    ) -> None:
        """Persist one chunk's zero-shot candidate output plus raw response.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.
            chunk_id (int): Stable chunk identifier.
            candidates (list[CandidateQuadruplet]): Parsed candidates emitted
                for the chunk.
            raw_response (str): Raw model response text for the chunk.

        Returns:
            None: This method persists chunk-level parsed and raw outputs.
        """

        self._write_json(
            self._candidate_path(run_id, bill_id, chunk_id),
            [candidate.to_dict() for candidate in candidates],
        )
        self._write_json(
            self._candidate_raw_path(run_id, bill_id, chunk_id),
            {"raw_response": raw_response},
        )

    def candidate_chunk_exists(self, run_id: str, bill_id: str, chunk_id: int) -> bool:
        """Return whether one chunk's annotation output is already persisted.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.
            chunk_id (int): Stable chunk identifier.

        Returns:
            bool: ``True`` when both parsed and raw chunk outputs already
                exist.
        """

        return (
            self._candidate_path(run_id, bill_id, chunk_id).exists()
            and self._candidate_raw_path(run_id, bill_id, chunk_id).exists()
        )

    def load_candidates(
        self,
        run_id: str,
        bill_id: str,
        chunk_id: int,
    ) -> list[CandidateQuadruplet]:
        """Load the persisted candidate output for one chunk.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.
            chunk_id (int): Stable chunk identifier.

        Returns:
            list[CandidateQuadruplet]: Parsed candidate quadruplets for the
                chunk.
        """

        payload = self._read_json(self._candidate_path(run_id, bill_id, chunk_id))
        return [CandidateQuadruplet.from_dict(item) for item in payload]

    def load_all_candidates_for_bill(
        self,
        run_id: str,
        bill_id: str,
        chunk_ids_in_order: list[int] | None = None,
    ) -> list[CandidateQuadruplet]:
        """Load the full bill-level candidate pool in chunk order.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.
            chunk_ids_in_order (list[int] | None): Optional explicit chunk
                order to use when reconstructing the bill-level pool.

        Returns:
            list[CandidateQuadruplet]: Bill-level candidate pool aggregated in
                chunk order.
        """

        if chunk_ids_in_order is None:
            annotation_state = self.load_stage_state(run_id, bill_id, "annotation")
            chunk_ids_in_order = annotation_state.get("chunk_ids_in_order", [])

        candidates: list[CandidateQuadruplet] = []
        for chunk_id in chunk_ids_in_order:
            candidates.extend(self.load_candidates(run_id, bill_id, int(chunk_id)))
        return candidates

    def save_grouped(
        self,
        run_id: str,
        bill_id: str,
        grouped_sets: list[GroupedCandidateSet],
        raw_response: str,
    ) -> None:
        """Persist the bill-level grouped candidate sets plus raw response.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.
            grouped_sets (list[GroupedCandidateSet]): Parsed grouped candidate
                sets for the bill.
            raw_response (str): Raw model response text for the grouping stage.

        Returns:
            None: This method persists bill-level grouped outputs.
        """

        self._write_json(
            self._grouped_path(run_id, bill_id),
            [grouped_set.to_dict() for grouped_set in grouped_sets],
        )
        self._write_json(
            self._grouped_raw_path(run_id, bill_id),
            {"raw_response": raw_response},
        )

    def grouped_exists(self, run_id: str, bill_id: str) -> bool:
        """Return whether grouped output is already persisted for a bill.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.

        Returns:
            bool: ``True`` when both parsed and raw grouping outputs already
                exist.
        """

        return (
            self._grouped_path(run_id, bill_id).exists()
            and self._grouped_raw_path(run_id, bill_id).exists()
        )

    def load_grouped(self, run_id: str, bill_id: str) -> list[GroupedCandidateSet]:
        """Load the persisted grouped candidate sets for a bill.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.

        Returns:
            list[GroupedCandidateSet]: Parsed grouped candidate sets for the
                bill.
        """

        payload = self._read_json(self._grouped_path(run_id, bill_id))
        return [GroupedCandidateSet.from_dict(item) for item in payload]

    def save_refined_group_outputs(
        self,
        run_id: str,
        bill_id: str,
        group_id: int,
        refined_outputs: list[RefinedQuadruplet],
        refinement_artifact: RefinementArtifact | None,
        raw_response: str,
    ) -> None:
        """Persist refinement-stage outputs for one grouped candidate set.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.
            group_id (int): Stable grouped-set identifier.
            refined_outputs (list[RefinedQuadruplet]): Parsed refined outputs
                for the grouped set.
            refinement_artifact (RefinementArtifact | None): Optional
                refinement-side artifact for the grouped set.
            raw_response (str): Raw model response text for the refinement
                stage.

        Returns:
            None: This method persists group-level refinement outputs.
        """

        self._write_json(
            self._refined_group_path(run_id, bill_id, group_id),
            [refined_output.to_dict() for refined_output in refined_outputs],
        )
        self._write_json(
            self._refined_group_raw_path(run_id, bill_id, group_id),
            {"raw_response": raw_response},
        )

        artifact_path = self._refinement_artifact_path(run_id, bill_id, group_id)
        if refinement_artifact is None:
            self._delete_path(artifact_path)
        else:
            self._write_json(artifact_path, refinement_artifact.to_dict())

    def refined_group_exists(self, run_id: str, bill_id: str, group_id: int) -> bool:
        """Return whether one group's refinement outputs are already persisted.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.
            group_id (int): Stable grouped-set identifier.

        Returns:
            bool: ``True`` when both parsed and raw refinement outputs already
                exist.
        """

        return (
            self._refined_group_path(run_id, bill_id, group_id).exists()
            and self._refined_group_raw_path(run_id, bill_id, group_id).exists()
        )

    def load_refined_group_outputs(
        self,
        run_id: str,
        bill_id: str,
        group_id: int,
    ) -> tuple[list[RefinedQuadruplet], RefinementArtifact | None]:
        """Load one group's refinement-stage outputs.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.
            group_id (int): Stable grouped-set identifier.

        Returns:
            tuple[list[RefinedQuadruplet], RefinementArtifact | None]: Tuple of
                parsed refined outputs and optional refinement artifact.
        """

        refined_payload = self._read_json(self._refined_group_path(run_id, bill_id, group_id))
        refined_outputs = [
            RefinedQuadruplet.from_dict(item) for item in refined_payload
        ]
        artifact_path = self._refinement_artifact_path(run_id, bill_id, group_id)
        refinement_artifact = None
        if artifact_path.exists():
            refinement_artifact = RefinementArtifact.from_dict(self._read_json(artifact_path))
        return refined_outputs, refinement_artifact

    def load_all_refined_for_bill(
        self,
        run_id: str,
        bill_id: str,
        group_ids_in_order: list[int],
    ) -> tuple[list[RefinedQuadruplet], list[RefinementArtifact]]:
        """Load all refinement-stage outputs for a bill in grouped-set order.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.
            group_ids_in_order (list[int]): Ordered grouped-set ids used to
                reconstruct the bill-level refinement outputs.

        Returns:
            tuple[list[RefinedQuadruplet], list[RefinementArtifact]]: Tuple of
                bill-level refined outputs and available refinement artifacts in
                group order.
        """

        refined_outputs: list[RefinedQuadruplet] = []
        refinement_artifacts: list[RefinementArtifact] = []
        for group_id in group_ids_in_order:
            group_outputs, artifact = self.load_refined_group_outputs(run_id, bill_id, group_id)
            refined_outputs.extend(group_outputs)
            if artifact is not None:
                refinement_artifacts.append(artifact)
        return refined_outputs, refinement_artifacts

    def mark_stage_complete(
        self,
        run_id: str,
        bill_id: str,
        stage: str,
        payload: dict[str, Any],
    ) -> None:
        """Persist the durable completion evidence for one stage and bill.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.
            stage (str): Stage name being marked complete.
            payload (dict[str, Any]): Serializable stage metadata written to
                disk.

        Returns:
            None: This method persists a durable stage-state marker.
        """

        state_payload = {"completed": True, "stage": stage, **payload}
        self._write_json(self._stage_state_path(run_id, bill_id, stage), state_payload)

    def load_stage_state(self, run_id: str, bill_id: str, stage: str) -> dict[str, Any]:
        """Load one stage completion marker.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.
            stage (str): Stage name whose state marker should be loaded.

        Returns:
            dict[str, Any]: Parsed stage-state payload, or an empty dictionary
                when no marker exists.
        """

        path = self._stage_state_path(run_id, bill_id, stage)
        if not path.exists():
            return {}
        return self._read_json(path)

    def is_stage_complete(self, run_id: str, bill_id: str, stage: str) -> bool:
        """Return whether a durable completion marker exists for a bill stage.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.
            stage (str): Stage name to inspect.

        Returns:
            bool: ``True`` when the stage-state marker exists.
        """

        return self._stage_state_path(run_id, bill_id, stage).exists()

    def clear_stage_completion(self, run_id: str, bill_id: str, stage: str) -> None:
        """Delete one stage's completion marker.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.
            stage (str): Stage name whose marker should be removed.

        Returns:
            None: This method removes the stage-state marker if it exists.
        """

        self._delete_path(self._stage_state_path(run_id, bill_id, stage))

    def invalidate_from_grouping(self, run_id: str, bill_id: str) -> None:
        """Delete grouped and refinement-stage outputs after annotation changes.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier whose downstream artifacts should be
                reset.

        Returns:
            None: This method deletes grouped and refinement-stage outputs.
        """

        self._delete_path(self._grouped_path(run_id, bill_id))
        self._delete_path(self._grouped_raw_path(run_id, bill_id))
        self._delete_path(self.run_dir(run_id) / "refined_outputs" / bill_id)
        self._delete_path(self.run_dir(run_id) / "refinement_artifacts" / bill_id)
        self.clear_stage_completion(run_id, bill_id, "grouping")
        self.clear_stage_completion(run_id, bill_id, "refinement")

    def invalidate_refinement(
        self,
        run_id: str,
        bill_id: str,
        group_id: int | None = None,
    ) -> None:
        """Delete refinement-stage outputs for one group or one full bill.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier whose refinement outputs should be
                reset.
            group_id (int | None): Optional grouped-set identifier. When
                omitted, all bill-level refinement outputs are removed.

        Returns:
            None: This method deletes refinement outputs for the requested
                scope.
        """

        if group_id is None:
            self._delete_path(self.run_dir(run_id) / "refined_outputs" / bill_id)
            self._delete_path(self.run_dir(run_id) / "refinement_artifacts" / bill_id)
        else:
            self._delete_path(self._refined_group_path(run_id, bill_id, group_id))
            self._delete_path(self._refined_group_raw_path(run_id, bill_id, group_id))
            self._delete_path(self._refinement_artifact_path(run_id, bill_id, group_id))
        self.clear_stage_completion(run_id, bill_id, "refinement")

