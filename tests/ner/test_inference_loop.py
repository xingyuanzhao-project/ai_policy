"""Inference-loop tests for config-driven bill skipping.

- Verifies bill-level runtime filtering happens before chunking.
- Verifies skipped bills do not invoke the orchestrator.
- Does not touch storage or live LLM components.
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, Mock

from src.ner.runtime.inference_loop import InferenceLoop
from src.ner.schemas.artifacts import BillRecord, ContextChunk, RefinedQuadruplet


class InferenceLoopTests(unittest.TestCase):
    """Verify bill-length skip behavior at the runtime boundary."""

    def test_run_bill_processes_bill_at_configured_limit(self) -> None:
        """Verify bills at the configured limit still build chunks and run."""

        bill = _make_bill_record("BILL-AT-LIMIT", "A" * 100)
        chunks = [_make_context_chunk(bill)]
        refined_outputs = [_make_refined_output()]
        inference_unit_builder = Mock()
        inference_unit_builder.build.return_value = chunks
        orchestrator = Mock()
        orchestrator.run_bill = AsyncMock(return_value=refined_outputs)

        final_output_store = Mock()

        inference_loop = InferenceLoop(
            inference_unit_builder=inference_unit_builder,
            orchestrator=orchestrator,
            final_output_store=final_output_store,
            max_bill_text_chars=100,
        )

        result = asyncio.run(inference_loop.run_bill(bill, resume=False))

        self.assertEqual(result, refined_outputs)
        inference_unit_builder.build.assert_called_once_with(bill)
        orchestrator.run_bill.assert_awaited_once_with(
            bill_id=bill.bill_id,
            chunks=chunks,
            resume=False,
        )
        final_output_store.register_bill_metadata.assert_called_once()

    def test_run_bill_skips_bill_above_configured_limit(self) -> None:
        """Verify oversized bills return empty outputs before chunking starts."""

        bill = _make_bill_record("BILL-TOO-LONG", "B" * 101)
        inference_unit_builder = Mock()
        orchestrator = Mock()
        orchestrator.run_bill = AsyncMock()

        final_output_store = Mock()

        inference_loop = InferenceLoop(
            inference_unit_builder=inference_unit_builder,
            orchestrator=orchestrator,
            final_output_store=final_output_store,
            max_bill_text_chars=100,
        )

        result = asyncio.run(inference_loop.run_bill(bill, resume=True))

        self.assertEqual(result, [])
        inference_unit_builder.build.assert_not_called()
        orchestrator.run_bill.assert_not_awaited()
        final_output_store.register_bill_metadata.assert_not_called()


def _make_bill_record(bill_id: str, text: str) -> BillRecord:
    """Build a minimal bill fixture for inference-loop tests."""

    return BillRecord(
        bill_id=bill_id,
        state="CA",
        text=text,
    )


def _make_context_chunk(bill: BillRecord) -> ContextChunk:
    """Build a deterministic chunk fixture spanning the full bill text."""

    return ContextChunk(
        chunk_id=101,
        bill_id=bill.bill_id,
        text=bill.text,
        start_offset=0,
        end_offset=len(bill.text),
    )


def _make_refined_output() -> RefinedQuadruplet:
    """Build a minimal refined-output fixture."""

    return RefinedQuadruplet(
        refined_id=301,
        source_group_id=201,
        source_candidate_ids=[101],
        entity="entity",
    )
