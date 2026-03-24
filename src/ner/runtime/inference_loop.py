"""Agent main loop for bill-level NER inference.

- Owns bill-level iteration over chunk building and staged orchestration.
- Owns mapping processed bills back to bill-level final outputs.
- Does not bootstrap runtime dependencies or persist artifacts directly.
"""

from __future__ import annotations

from tqdm.auto import tqdm

from ..orchestration.orchestrator import Orchestrator
from ..schemas.artifacts import BillRecord, RefinedQuadruplet
from ..schemas.inference_unit_builder import InferenceUnitBuilder


class InferenceLoop:
    """Run the staged NER pipeline over bills and preserve chunk boundaries."""

    def __init__(
        self,
        inference_unit_builder: InferenceUnitBuilder,
        orchestrator: Orchestrator,
        max_bill_text_chars: int | None = None,
    ) -> None:
        """Initialize the inference loop.

        Args:
            inference_unit_builder (InferenceUnitBuilder): Builder that derives
                chunks from raw bills.
            orchestrator (Orchestrator): Stage orchestrator for the NER
                pipeline.
            max_bill_text_chars (int | None): Optional bill-text length ceiling
                beyond which bills are skipped before chunking.
        """

        self._inference_unit_builder = inference_unit_builder
        self._orchestrator = orchestrator
        self._max_bill_text_chars = max_bill_text_chars

    async def run_bill(self, bill: BillRecord, resume: bool = True) -> list[RefinedQuadruplet]:
        """Run one bill through chunking, staged inference, and bill-level assembly.

        Args:
            bill (BillRecord): Raw bill record to process.
            resume (bool): Whether persisted stage outputs may be reused.

        Returns:
            list[RefinedQuadruplet]: Final refined quadruplets for the bill.
        """

        if (
            self._max_bill_text_chars is not None
            and len(bill.text) > self._max_bill_text_chars
        ):
            return []

        chunks = self._inference_unit_builder.build(bill)
        return await self._orchestrator.run_bill(
            bill_id=bill.bill_id,
            chunks=chunks,
            resume=resume,
        )

    async def run_corpus(
        self,
        bills: list[BillRecord],
        resume: bool = True,
    ) -> dict[str, list[RefinedQuadruplet]]:
        """Run the staged NER pipeline over a list of bill records.

        Bills are processed sequentially so that the single local vLLM server
        is not overwhelmed.  Within each bill, annotation chunks and refinement
        groups fan out with bounded concurrency.

        Args:
            bills (list[BillRecord]): Raw bill records to process.
            resume (bool): Whether persisted stage outputs may be reused.

        Returns:
            dict[str, list[RefinedQuadruplet]]: Mapping from ``bill_id`` to
                final refined quadruplets.
        """

        results: dict[str, list[RefinedQuadruplet]] = {}
        for bill in tqdm(bills, desc="NER bills", unit="bill"):
            results[bill.bill_id] = await self.run_bill(bill, resume=resume)
        return results

