"""Vector retrieval over the persisted local QA index.

- Converts normalized query embeddings into ranked chunk results.
- Preserves a typed retrieval contract for the QA service and UI.
- Does not call provider APIs or mutate persisted index files.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .artifacts import IndexedChunk, RetrievedChunk, validate_retrieved_chunk


class Retriever:
    """Run cosine-style nearest-neighbor retrieval over normalized embeddings."""

    def __init__(self, chunks: Sequence[IndexedChunk], embeddings: np.ndarray) -> None:
        """Initialize the retriever with indexed chunks and aligned embeddings."""

        if embeddings.ndim != 2:
            raise ValueError("Retriever embeddings must be a 2D matrix")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Retriever chunk count must match embedding row count")
        self._chunks = list(chunks)
        self._embeddings = embeddings.astype(np.float32, copy=False)

    def retrieve(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        """Return the highest-scoring retrieved chunks for one query vector."""

        if top_k <= 0:
            raise ValueError("Retriever top_k must be > 0")
        if query_embedding.ndim != 1:
            raise ValueError("Retriever query_embedding must be a 1D vector")
        if self._embeddings.shape[1] != query_embedding.shape[0]:
            raise ValueError("Retriever query embedding dimension did not match index dimension")
        scores = self._embeddings @ query_embedding.astype(np.float32, copy=False)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        results: list[RetrievedChunk] = []
        for rank, row_index in enumerate(ranked_indices, start=1):
            chunk = self._chunks[int(row_index)]
            retrieved = RetrievedChunk(
                rank=rank,
                score=float(scores[int(row_index)]),
                chunk_id=chunk.chunk_id,
                bill_id=chunk.bill_id,
                text=chunk.text,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                state=chunk.state,
                title=chunk.title,
                status=chunk.status,
                summary=chunk.summary,
                bill_url=chunk.bill_url,
            )
            validate_retrieved_chunk(retrieved)
            results.append(retrieved)
        return results


__all__ = ["Retriever"]
