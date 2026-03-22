"""Lazy chunk storage for the hosted QA runtime.

- Keeps chunk payloads on disk as newline-delimited JSON plus a small byte-offset
  index so startup does not need to materialize the full chunk corpus.
- Exposes a sequence-like interface compatible with the retriever.
- Loads only the requested top-k chunk payloads on demand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import overload

import numpy as np

from .artifacts import IndexedChunk


@dataclass(frozen=True, slots=True)
class ChunkStore:
    """Sequence-like on-disk chunk store backed by offsets into one JSONL file."""

    chunks_jsonl_path: Path
    chunk_offsets: np.ndarray

    def __len__(self) -> int:
        """Return the number of persisted chunks."""

        return int(self.chunk_offsets.shape[0])

    @overload
    def __getitem__(self, index: int) -> IndexedChunk: ...

    @overload
    def __getitem__(self, index: slice) -> list[IndexedChunk]: ...

    def __getitem__(self, index: int | slice) -> IndexedChunk | list[IndexedChunk]:
        """Load one chunk or slice of chunks from disk."""

        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]

        normalized_index = int(index)
        if normalized_index < 0:
            normalized_index += len(self)
        if normalized_index < 0 or normalized_index >= len(self):
            raise IndexError("ChunkStore index out of range")

        chunk_offset = int(self.chunk_offsets[normalized_index])
        with open(self.chunks_jsonl_path, "rb") as handle:
            handle.seek(chunk_offset)
            line = handle.readline()
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("Persisted QA chunk line must decode to one object")
        return IndexedChunk.from_dict(payload)


__all__ = ["ChunkStore"]
