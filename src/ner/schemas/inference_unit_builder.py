"""Inference-unit construction for the NER pipeline.

- Owns conversion from raw `BillRecord` objects into `ContextChunk` objects.
- Owns deterministic chunk-id generation and source-offset preservation.
- Does not load corpus files, call the LLM, or persist artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass

from .artifacts import BillRecord, ContextChunk
from .constants import stable_int_id
from .validation import validate_bill_record, validate_context_chunk

_RECURSIVE_SEPARATORS: tuple[str, str, str] = ("\n\n", "\n", " ")


@dataclass(slots=True)
class ChunkingConfig:
    """Chunking parameters used to derive ``ContextChunk`` objects from bills.

    Attributes:
        chunk_size (int): Maximum number of bill-text characters allowed per
            chunk.
        overlap (int): Number of trailing characters repeated into the next
            chunk.
    """

    chunk_size: int
    overlap: int

    def validate(self) -> None:
        """Validate chunking parameters before chunk generation starts.

        Returns:
            None: This method validates in place and raises on invalid input.

        Raises:
            ValueError: If the chunking parameters are internally inconsistent.
        """

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.overlap < 0:
            raise ValueError("overlap must be >= 0")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")


class InferenceUnitBuilder:
    """Convert raw bill records into schema-valid ``ContextChunk`` objects."""

    def __init__(self, config: ChunkingConfig) -> None:
        """Initialize the inference-unit builder.

        Args:
            config (ChunkingConfig): Chunking configuration used to derive
                context chunks.
        """

        config.validate()
        self._config = config

    def build(self, bill: BillRecord) -> list[ContextChunk]:
        """Build stable chunks for one bill.

        Chunk ids are derived from the bill id plus source offsets so reruns
        regenerate the same ids as long as the bill text and chunking config
        stay unchanged. Chunk boundaries prefer paragraph, then line, then word
        separators before falling back to hard character cuts.

        Args:
            bill (BillRecord): Canonical raw bill record to split into
                inference chunks.

        Returns:
            list[ContextChunk]: Ordered list of schema-valid context chunks for
                the bill.
        """

        validate_bill_record(bill)
        if not bill.text:
            return []

        if len(bill.text) <= self._config.chunk_size:
            return [self._make_chunk(bill, 0, len(bill.text))]

        core_chunk_size = self._config.chunk_size - self._config.overlap
        base_ranges = self._merge_ranges(
            self._split_range_recursively(
                bill.text,
                0,
                len(bill.text),
                core_chunk_size,
                _RECURSIVE_SEPARATORS,
            ),
            core_chunk_size,
        )

        chunks: list[ContextChunk] = []
        for index, (base_start, base_end) in enumerate(base_ranges):
            start_offset = (
                base_start
                if index == 0
                else max(0, base_start - self._config.overlap)
            )
            chunks.append(self._make_chunk(bill, start_offset, base_end))
        return chunks

    def _make_chunk(
        self,
        bill: BillRecord,
        start_offset: int,
        end_offset: int,
    ) -> ContextChunk:
        """Create and validate one deterministic context chunk.

        Args:
            bill (BillRecord): Bill record from which the chunk is derived.
            start_offset (int): Inclusive chunk start offset in the bill text.
            end_offset (int): Exclusive chunk end offset in the bill text.

        Returns:
            ContextChunk: Deterministic validated context chunk.
        """

        chunk = ContextChunk(
            chunk_id=stable_int_id("chunk", bill.bill_id, start_offset, end_offset),
            bill_id=bill.bill_id,
            text=bill.text[start_offset:end_offset],
            start_offset=start_offset,
            end_offset=end_offset,
        )
        validate_context_chunk(chunk)
        return chunk

    def _split_range_recursively(
        self,
        text: str,
        start_offset: int,
        end_offset: int,
        max_length: int,
        separators: tuple[str, ...],
    ) -> list[tuple[int, int]]:
        """Recursively split one text range using progressively weaker boundaries.

        Args:
            text (str): Full source text being chunked.
            start_offset (int): Inclusive start offset of the range to split.
            end_offset (int): Exclusive end offset of the range to split.
            max_length (int): Maximum allowed length for each returned range.
            separators (tuple[str, ...]): Ordered separator preferences used
                before falling back to hard splitting.

        Returns:
            list[tuple[int, int]]: Ordered character ranges produced from the
                requested text span.
        """

        if end_offset - start_offset <= max_length:
            return [(start_offset, end_offset)]
        if not separators:
            return self._hard_split_range(start_offset, end_offset, max_length)

        pieces = self._split_range_on_separator(
            text,
            start_offset,
            end_offset,
            separators[0],
        )
        if len(pieces) == 1:
            return self._split_range_recursively(
                text,
                start_offset,
                end_offset,
                max_length,
                separators[1:],
            )

        split_ranges: list[tuple[int, int]] = []
        for piece_start, piece_end in pieces:
            split_ranges.extend(
                self._split_range_recursively(
                    text,
                    piece_start,
                    piece_end,
                    max_length,
                    separators[1:],
                )
            )
        return split_ranges

    def _split_range_on_separator(
        self,
        text: str,
        start_offset: int,
        end_offset: int,
        separator: str,
    ) -> list[tuple[int, int]]:
        """Split one text range while preserving separator characters in-order.

        Args:
            text (str): Full source text being chunked.
            start_offset (int): Inclusive start offset of the range to split.
            end_offset (int): Exclusive end offset of the range to split.
            separator (str): Separator string used as the split preference.

        Returns:
            list[tuple[int, int]]: Ordered character ranges split on the
                supplied separator.
        """

        ranges: list[tuple[int, int]] = []
        cursor = start_offset
        while True:
            split_index = text.find(separator, cursor, end_offset)
            if split_index < 0:
                break
            split_end = split_index + len(separator)
            if split_end > cursor:
                ranges.append((cursor, split_end))
            cursor = split_end
        if cursor < end_offset:
            ranges.append((cursor, end_offset))
        return ranges or [(start_offset, end_offset)]

    def _hard_split_range(
        self,
        start_offset: int,
        end_offset: int,
        max_length: int,
    ) -> list[tuple[int, int]]:
        """Split a range by fixed width when no semantic separator is available.

        Args:
            start_offset (int): Inclusive start offset of the range to split.
            end_offset (int): Exclusive end offset of the range to split.
            max_length (int): Maximum allowed length for each returned range.

        Returns:
            list[tuple[int, int]]: Fixed-width character ranges covering the
                requested interval.
        """

        return [
            (range_start, min(range_start + max_length, end_offset))
            for range_start in range(start_offset, end_offset, max_length)
        ]

    def _merge_ranges(
        self,
        ranges: list[tuple[int, int]],
        max_length: int,
    ) -> list[tuple[int, int]]:
        """Greedily merge contiguous ranges without exceeding ``max_length``.

        Args:
            ranges (list[tuple[int, int]]): Ordered source ranges to merge.
            max_length (int): Maximum allowed merged range length.

        Returns:
            list[tuple[int, int]]: Greedily merged character ranges.
        """

        if not ranges:
            return []

        merged_ranges: list[tuple[int, int]] = []
        current_start, current_end = ranges[0]
        for range_start, range_end in ranges[1:]:
            if range_end - current_start <= max_length:
                current_end = range_end
                continue
            merged_ranges.append((current_start, current_end))
            current_start, current_end = range_start, range_end
        merged_ranges.append((current_start, current_end))
        return merged_ranges

    def build_many(self, bills: list[BillRecord]) -> dict[str, list[ContextChunk]]:
        """Build chunks for multiple bills keyed by ``bill_id``.

        Args:
            bills (list[BillRecord]): Raw bill records to split into chunks.

        Returns:
            dict[str, list[ContextChunk]]: Mapping from ``bill_id`` to ordered
                context chunks for that bill.
        """

        return {bill.bill_id: self.build(bill) for bill in bills}

