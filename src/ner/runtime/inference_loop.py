"""Agent main loop for bill-level NER inference.

- Owns bill-level iteration over chunk building and staged orchestration.
- Owns mapping processed bills back to bill-level final outputs.
- Does not bootstrap runtime dependencies or persist artifacts directly.
"""

from __future__ import annotations

import logging
import time

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

from ..orchestration.orchestrator import Orchestrator, StageFailure
from ..runtime.llm_client import CreditsExhaustedError
from ..schemas.artifacts import BillRecord, RefinedQuadruplet
from ..schemas.inference_unit_builder import InferenceUnitBuilder
from ..storage.final_output_store import FinalOutputStore


class InferenceLoop:
    """Run the staged NER pipeline over bills and preserve chunk boundaries."""

    def __init__(
        self,
        inference_unit_builder: InferenceUnitBuilder,
        orchestrator: Orchestrator,
        final_output_store: FinalOutputStore,
        max_bill_text_chars: int | None = None,
    ) -> None:
        """Initialize the inference loop.

        Args:
            inference_unit_builder (InferenceUnitBuilder): Builder that derives
                chunks from raw bills.
            orchestrator (Orchestrator): Stage orchestrator for the NER
                pipeline.
            final_output_store (FinalOutputStore): Store for bill-level final
                outputs; used to register bill metadata before orchestration.
            max_bill_text_chars (int | None): Optional bill-text length ceiling
                beyond which bills are skipped before chunking.
        """

        self._inference_unit_builder = inference_unit_builder
        self._orchestrator = orchestrator
        self._final_output_store = final_output_store
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
            logger.info("Skipping bill=%s  text_len=%d > limit=%d", bill.bill_id, len(bill.text), self._max_bill_text_chars)
            return []

        self._final_output_store.register_bill_metadata(
            bill.bill_id,
            {
                "source_bill_id": bill.source_bill_id,
                "year": bill.year,
                "state": bill.state,
            },
        )

        chunks = self._inference_unit_builder.build(bill)
        logger.info("Starting bill=%s  text_len=%d  chunks=%d", bill.bill_id, len(bill.text), len(chunks))
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

        logger.info("Corpus run starting: %d bills  resume=%s", len(bills), resume)
        corpus_t0 = time.perf_counter()
        results: dict[str, list[RefinedQuadruplet]] = {}
        failed_bills: list[str] = []
        for bill in tqdm(bills, desc="NER bills", unit="bill"):
            try:
                results[bill.bill_id] = await self.run_bill(bill, resume=resume)
            except CreditsExhaustedError:
                logger.error(
                    "Credits exhausted during bill=%s. Completed %d/%d bills. "
                    "Top up and re-run with resume=True to continue.",
                    bill.bill_id, len(results), len(bills),
                )
                break
            except StageFailure as exc:
                failed_bills.append(exc.bill_id)
                logger.error(
                    "Stage failure bill=%s stage=%s: %s. Skipping.",
                    exc.bill_id, exc.stage, exc.detail,
                )
        logger.info(
            "Corpus run complete: %d/%d bills  failed=%d  total_elapsed=%.1fs",
            len(results), len(bills), len(failed_bills),
            time.perf_counter() - corpus_t0,
        )
        if failed_bills:
            logger.warning("Failed bills: %s", failed_bills)
        return results

