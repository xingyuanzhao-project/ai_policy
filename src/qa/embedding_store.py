"""Streamable embedding storage for the hosted QA retriever.

- Keeps the persisted embedding batches on disk rather than concatenating them
  into one in-memory matrix at startup.
- Exposes a small matrix-like surface so the retriever can validate row counts
  while iterating over batches on demand.
- Does not mutate cache files or own retrieval ranking logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True, slots=True)
class EmbeddingBatchSpec:
    """One persisted embedding batch file."""

    path: Path


@dataclass(frozen=True, slots=True)
class EmbeddingStore:
    """Matrix-like embedding store backed by persisted batch files."""

    batch_specs: tuple[EmbeddingBatchSpec, ...]
    total_rows: int
    embedding_dimension: int

    @property
    def shape(self) -> tuple[int, int]:
        """Return the matrix-style shape expected by the retriever."""

        return (self.total_rows, self.embedding_dimension)

    @property
    def ndim(self) -> int:
        """Return the matrix rank expected by the retriever."""

        return 2

    def iter_batches(self) -> Iterator[np.ndarray]:
        """Yield validated embedding batches as read-only memory maps."""

        for batch_spec in self.batch_specs:
            yield np.load(batch_spec.path, mmap_mode="r")


__all__ = ["EmbeddingBatchSpec", "EmbeddingStore"]
