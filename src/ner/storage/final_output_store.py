"""Persistence for normalized bill-level refined quadruplets.

- Owns run-scoped storage for final bill-level refined outputs.
- Keeps final outputs separate from intermediate candidates and grouping state.
- Does not validate artifacts, call the LLM, or orchestrate pipeline stages.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..schemas.artifacts import RefinedQuadruplet


class FinalOutputStore:
    """Store final bill-level refined quadruplets separately from intermediates."""

    def __init__(self, base_dir: Path) -> None:
        """Initialize the final-output store.

        Args:
            base_dir (Path): Base directory under which run-scoped outputs are
                stored.
        """

        self._base_dir = base_dir

    def _output_path(self, run_id: str, bill_id: str) -> Path:
        """Return the output path for one bill's final results.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.

        Returns:
            Path: Filesystem path for the bill-level final output JSON.
        """

        return self._base_dir / "runs" / run_id / "outputs" / f"{bill_id}.json"

    def save(
        self,
        run_id: str,
        bill_id: str,
        refined_outputs: list[RefinedQuadruplet],
    ) -> None:
        """Persist one bill's final refined outputs in normalized JSON form.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.
            refined_outputs (list[RefinedQuadruplet]): Final refined outputs to
                persist for the bill.

        Returns:
            None: This method persists the bill-level final outputs.
        """

        output_path = self._output_path(run_id, bill_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(
                [refined_output.to_dict() for refined_output in refined_outputs],
                handle,
                indent=2,
                ensure_ascii=False,
            )

    def exists(self, run_id: str, bill_id: str) -> bool:
        """Return whether final outputs already exist for the bill.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.

        Returns:
            bool: ``True`` when final bill-level output already exists.
        """

        return self._output_path(run_id, bill_id).exists()

    def load(self, run_id: str, bill_id: str) -> list[RefinedQuadruplet]:
        """Load one bill's final refined outputs.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.

        Returns:
            list[RefinedQuadruplet]: Parsed refined quadruplets for the bill.
        """

        with open(self._output_path(run_id, bill_id), encoding="utf-8") as handle:
            payload = json.load(handle)
        return [RefinedQuadruplet.from_dict(item) for item in payload]

    def load_all(self, run_id: str) -> dict[str, list[RefinedQuadruplet]]:
        """Load all bill-level outputs for a given run id.

        Args:
            run_id (str): Stable run identifier.

        Returns:
            dict[str, list[RefinedQuadruplet]]: Mapping from ``bill_id`` to
                parsed final refined outputs.
        """

        output_dir = self._base_dir / "runs" / run_id / "outputs"
        if not output_dir.exists():
            return {}

        results: dict[str, list[RefinedQuadruplet]] = {}
        for output_path in sorted(output_dir.glob("*.json")):
            with open(output_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            results[output_path.stem] = [
                RefinedQuadruplet.from_dict(item) for item in payload
            ]
        return results

    def delete(self, run_id: str, bill_id: str) -> None:
        """Delete one bill's final outputs so they can be recomputed.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.

        Returns:
            None: This method removes the persisted final output when it exists.
        """

        output_path = self._output_path(run_id, bill_id)
        if output_path.exists():
            output_path.unlink()

