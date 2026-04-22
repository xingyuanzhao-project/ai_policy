"""Persistence for normalized bill-level refined quadruplets.

- Owns run-scoped storage for final bill-level refined outputs.
- Keeps final outputs separate from intermediate candidates and grouping state.
- Wraps each bill's output with source metadata (year, state, source bill id)
  when metadata has been registered for that bill.
- Does not validate artifacts, call the LLM, or orchestrate pipeline stages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
        self._bill_metadata: dict[str, dict[str, str]] = {}

    def register_bill_metadata(
        self,
        bill_id: str,
        metadata: dict[str, str],
    ) -> None:
        """Register source-corpus metadata for a bill so ``save`` can embed it.

        Call this before orchestration begins for each bill. The metadata is
        keyed by the pipeline-unique ``bill_id`` and written into the output
        JSON wrapper alongside the quadruplets.

        Args:
            bill_id (str): Pipeline-unique bill identifier.
            metadata (dict[str, str]): Metadata dict with keys such as
                ``year``, ``state``, ``source_bill_id``.
        """

        self._bill_metadata[bill_id] = metadata

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

        When bill metadata has been registered via
        :meth:`register_bill_metadata`, the output is wrapped in a dict
        containing the metadata and a ``quadruplets`` key. Otherwise the
        output is a bare list for backward compatibility.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.
            refined_outputs (list[RefinedQuadruplet]): Final refined outputs to
                persist for the bill.

        Returns:
            None: This method persists the bill-level final outputs.
        """

        quadruplets = [r.to_dict() for r in refined_outputs]

        metadata = self._bill_metadata.get(bill_id)
        if metadata is not None:
            payload: Any = {
                "bill_id": bill_id,
                **metadata,
                "quadruplets": quadruplets,
            }
        else:
            payload = quadruplets

        output_path = self._output_path(run_id, bill_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

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

        Handles both the new wrapped format (dict with ``quadruplets`` key)
        and the legacy bare-list format.

        Args:
            run_id (str): Stable run identifier.
            bill_id (str): Bill identifier.

        Returns:
            list[RefinedQuadruplet]: Parsed refined quadruplets for the bill.
        """

        with open(self._output_path(run_id, bill_id), encoding="utf-8") as handle:
            payload = json.load(handle)

        items = payload["quadruplets"] if isinstance(payload, dict) else payload
        return [RefinedQuadruplet.from_dict(item) for item in items]

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
            items = payload["quadruplets"] if isinstance(payload, dict) else payload
            results[output_path.stem] = [
                RefinedQuadruplet.from_dict(item) for item in items
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
