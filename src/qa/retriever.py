"""Vector retrieval over the persisted local QA index.

- Converts normalized query embeddings into ranked chunk results.
- Supports both an in-memory embedding matrix and a streamable on-disk batch
  store so the hosted app can stay within Render's memory limit.
- Preserves a typed retrieval contract for the QA service and UI.
- Does not call provider APIs or mutate persisted index files.
"""

from __future__ import annotations

import heapq
from typing import Sequence

import numpy as np

from .artifacts import IndexedChunk, RetrievedChunk, validate_retrieved_chunk
from .embedding_store import EmbeddingStore


class Retriever:
    """Run cosine-style nearest-neighbor retrieval over normalized embeddings."""

    def __init__(
        self,
        chunks: Sequence[IndexedChunk],
        embeddings: np.ndarray | EmbeddingStore,
    ) -> None:
        """Initialize the retriever with indexed chunks and aligned embeddings."""

        if embeddings.ndim != 2:
            raise ValueError("Retriever embeddings must be a 2D matrix")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Retriever chunk count must match embedding row count")
        self._chunks = list(chunks)
        self._embedding_store = embeddings if isinstance(embeddings, EmbeddingStore) else None
        self._embeddings = (
            None
            if self._embedding_store is not None
            else embeddings.astype(np.float32, copy=False)
        )

    def retrieve(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        """Return the highest-scoring retrieved chunks for one query vector."""

        if top_k <= 0:
            raise ValueError("Retriever top_k must be > 0")
        if query_embedding.ndim != 1:
            raise ValueError("Retriever query_embedding must be a 1D vector")
        embedding_dimension = (
            self._embedding_store.shape[1]
            if self._embedding_store is not None
            else self._embeddings.shape[1]
        )
        if embedding_dimension != query_embedding.shape[0]:
            raise ValueError("Retriever query embedding dimension did not match index dimension")
        normalized_query_embedding = query_embedding.astype(np.float32, copy=False)
        if self._embedding_store is None:
            assert self._embeddings is not None
            return self._retrieve_from_matrix(normalized_query_embedding, top_k)
        return self._retrieve_from_store(normalized_query_embedding, top_k)

    def _retrieve_from_matrix(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Return top-k retrieval results from one in-memory embedding matrix."""

        assert self._embeddings is not None
        scores = self._embeddings @ query_embedding
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        ranked_matches = [
            (float(scores[int(row_index)]), int(row_index)) for row_index in ranked_indices
        ]
        return self._build_results(ranked_matches)

    def _retrieve_from_store(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Return top-k retrieval results by scanning persisted embedding batches."""

        assert self._embedding_store is not None
        top_matches: list[tuple[float, int]] = []
        row_offset = 0
        for batch in self._embedding_store.iter_batches():
            scores = batch @ query_embedding
            for row_index in self._top_row_indices(scores, top_k):
                score = float(scores[int(row_index)])
                global_row_index = row_offset + int(row_index)
                if len(top_matches) < top_k:
                    heapq.heappush(top_matches, (score, global_row_index))
                elif score > top_matches[0][0]:
                    heapq.heapreplace(top_matches, (score, global_row_index))
            row_offset += int(batch.shape[0])
        ranked_matches = sorted(top_matches, key=lambda match: match[0], reverse=True)
        return self._build_results(ranked_matches)

    def _top_row_indices(self, scores: np.ndarray, top_k: int) -> np.ndarray:
        """Return row indices for the strongest scores in one batch."""

        candidate_count = min(top_k, int(scores.shape[0]))
        if candidate_count == int(scores.shape[0]):
            return np.argsort(scores)[::-1]
        top_indices = np.argpartition(scores, -candidate_count)[-candidate_count:]
        return top_indices[np.argsort(scores[top_indices])[::-1]]

    def _build_results(self, ranked_matches: list[tuple[float, int]]) -> list[RetrievedChunk]:
        """Build typed retrieval results from sorted score/index pairs."""

        results: list[RetrievedChunk] = []
        for rank, (score, row_index) in enumerate(ranked_matches, start=1):
            chunk = self._chunks[row_index]
            retrieved = RetrievedChunk(
                rank=rank,
                score=score,
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
