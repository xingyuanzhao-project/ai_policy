"""Vector retrieval over the persisted local QA index.

- Converts normalized query embeddings into ranked chunk results.
- Supports both an in-memory embedding matrix and a streamable on-disk batch
  store so the hosted app can stay within Render's memory limit.
- Preserves a typed retrieval contract for the QA service and UI.
- Does not call provider APIs or mutate persisted index files.
"""

from __future__ import annotations

import heapq
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .artifacts import IndexedChunk, RetrievedChunk, validate_retrieved_chunk
from .chunk_store import ChunkStore
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
        self._chunks = chunks
        self._embedding_store = embeddings if isinstance(embeddings, EmbeddingStore) else None
        self._embeddings = (
            None
            if self._embedding_store is not None
            else embeddings.astype(np.float32, copy=False)
        )
        self._metadata = _build_metadata_arrays(chunks)

    @property
    def chunk_metadata(self) -> "RetrieverMetadata":
        """Expose precomputed filter metadata for callers that enumerate facet values."""

        return self._metadata

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
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
        mask = self._build_mask(filters) if filters else None
        if self._embedding_store is None:
            assert self._embeddings is not None
            return self._retrieve_from_matrix(normalized_query_embedding, top_k, mask)
        return self._retrieve_from_store(normalized_query_embedding, top_k, mask)

    def _retrieve_from_matrix(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        mask: np.ndarray | None,
    ) -> list[RetrievedChunk]:
        """Return top-k retrieval results from one in-memory embedding matrix."""

        assert self._embeddings is not None
        scores = self._embeddings @ query_embedding
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        ranked_matches: list[tuple[float, int]] = []
        for row_index in ranked_indices:
            score = float(scores[int(row_index)])
            if not np.isfinite(score):
                continue
            ranked_matches.append((score, int(row_index)))
        return self._build_results(ranked_matches)

    def _retrieve_from_store(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        mask: np.ndarray | None,
    ) -> list[RetrievedChunk]:
        """Return top-k retrieval results by scanning persisted embedding batches."""

        assert self._embedding_store is not None
        top_matches: list[tuple[float, int]] = []
        row_offset = 0
        for batch in self._embedding_store.iter_batches():
            scores = batch @ query_embedding
            batch_rows = int(batch.shape[0])
            if mask is not None:
                batch_mask = mask[row_offset : row_offset + batch_rows]
                scores = np.where(batch_mask, scores, -np.inf)
            for row_index in self._top_row_indices(scores, top_k):
                score = float(scores[int(row_index)])
                if not np.isfinite(score):
                    continue
                global_row_index = row_offset + int(row_index)
                if len(top_matches) < top_k:
                    heapq.heappush(top_matches, (score, global_row_index))
                elif score > top_matches[0][0]:
                    heapq.heapreplace(top_matches, (score, global_row_index))
            row_offset += batch_rows
        ranked_matches = sorted(top_matches, key=lambda match: match[0], reverse=True)
        return self._build_results(ranked_matches)

    def _build_mask(self, filters: dict) -> np.ndarray:
        """Build a boolean mask selecting chunks that satisfy all filter fields.

        Each scalar field (``year``, ``state``, ``status_bucket``) accepts either
        a single value or a list; a list means OR-within-field (``np.isin``).
        Multiple fields combine with AND. ``topics`` is always list-valued with
        OR-within-field.
        """

        total = self._metadata.total
        mask = np.ones(total, dtype=bool)

        years = _coerce_int_values(filters.get("year"))
        if years:
            mask &= np.isin(self._metadata.years, years)

        states = _coerce_str_values(filters.get("state"))
        if states:
            mask &= np.isin(self._metadata.states, states)

        status_buckets = _coerce_str_values(filters.get("status_bucket"))
        if status_buckets:
            mask &= np.isin(self._metadata.status_buckets, status_buckets)

        topics = _coerce_str_values(filters.get("topics"))
        if topics:
            selected_set = set(topics)
            topic_mask = np.fromiter(
                (bool(selected_set & topic_set) for topic_set in self._metadata.topic_sets),
                dtype=bool,
                count=total,
            )
            mask &= topic_mask

        return mask

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
                year=chunk.year,
                status_bucket=chunk.status_bucket,
                topics_list=list(chunk.topics_list),
            )
            validate_retrieved_chunk(retrieved)
            results.append(retrieved)
        return results


class RetrieverMetadata:
    """Vectorized filter metadata aligned to the retriever's chunk ordering.

    Kept separate from the chunk payloads so the Render path can populate these
    arrays without materializing every chunk's text in memory.
    """

    __slots__ = ("years", "states", "status_buckets", "topic_sets")

    def __init__(
        self,
        years: np.ndarray,
        states: np.ndarray,
        status_buckets: np.ndarray,
        topic_sets: list[frozenset[str]],
    ) -> None:
        self.years = years
        self.states = states
        self.status_buckets = status_buckets
        self.topic_sets = topic_sets

    @property
    def total(self) -> int:
        """Return the number of chunks covered by the metadata arrays."""

        return int(self.years.shape[0])


def _build_metadata_arrays(chunks: Sequence[IndexedChunk]) -> RetrieverMetadata:
    """Extract parallel filter arrays from the chunk sequence once at init time.

    For the in-memory list case this is a simple iteration. For :class:`ChunkStore`
    the jsonl is streamed directly so only metadata fields are parsed, keeping the
    memory footprint minimal even though chunk texts live on disk.
    """

    if isinstance(chunks, ChunkStore):
        return _build_metadata_from_chunk_store(chunks)
    return _build_metadata_from_sequence(chunks)


def _build_metadata_from_sequence(chunks: Sequence[IndexedChunk]) -> RetrieverMetadata:
    years: list[int] = []
    states: list[str] = []
    buckets: list[str] = []
    topic_sets: list[frozenset[str]] = []
    for chunk in chunks:
        years.append(int(chunk.year))
        states.append(str(chunk.state))
        buckets.append(str(chunk.status_bucket))
        topic_sets.append(frozenset(str(topic) for topic in chunk.topics_list))
    return RetrieverMetadata(
        years=np.array(years, dtype=np.int32),
        states=np.array(states, dtype=object),
        status_buckets=np.array(buckets, dtype=object),
        topic_sets=topic_sets,
    )


def _build_metadata_from_chunk_store(store: ChunkStore) -> RetrieverMetadata:
    jsonl_path = Path(store.chunks_jsonl_path)
    expected_total = len(store)
    years: list[int] = []
    states: list[str] = []
    buckets: list[str] = []
    topic_sets: list[frozenset[str]] = []
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for raw_line in _iter_nonempty_lines(handle):
            payload = json.loads(raw_line)
            if not isinstance(payload, dict):
                raise ValueError("Persisted QA chunk line must decode to one object")
            years.append(int(payload.get("year", 0) or 0))
            states.append(str(payload.get("state", "") or ""))
            buckets.append(str(payload.get("status_bucket", "Other") or "Other"))
            raw_topics = payload.get("topics_list", []) or []
            topic_sets.append(
                frozenset(str(topic) for topic in raw_topics if str(topic).strip())
            )
    if len(years) != expected_total:
        raise ValueError(
            "Chunk store metadata row count did not match persisted chunk count"
        )
    return RetrieverMetadata(
        years=np.array(years, dtype=np.int32),
        states=np.array(states, dtype=object),
        status_buckets=np.array(buckets, dtype=object),
        topic_sets=topic_sets,
    )


def _iter_nonempty_lines(handle: Iterable[str]) -> Iterable[str]:
    for raw_line in handle:
        if raw_line.strip():
            yield raw_line


def _coerce_str_values(raw: object) -> list[str]:
    """Normalize a scalar-or-iterable filter value into a list of non-empty strings.

    Used by the retrievers and tool handlers so that
    ``filters={"state": "California"}`` and
    ``filters={"state": ["California", "Texas"]}`` behave identically, with
    the list form giving OR-within-field semantics. Vocabulary-level
    normalization (USPS ``"TX"`` -> ``"Texas"``, case-folding, topic fuzzy
    match) happens upstream in ``filter_normalizers`` / ``_coerce_filters``;
    this helper only handles the shape, not the values.
    """

    if raw is None or raw == 0:
        return []
    items: Iterable[object]
    if isinstance(raw, (list, tuple, set, frozenset)):
        items = raw
    elif isinstance(raw, str):
        items = [raw]
    else:
        items = [raw]
    cleaned: list[str] = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _coerce_int_values(raw: object) -> list[int]:
    """Normalize a scalar-or-iterable year filter into a unique list of ints."""

    if raw is None or raw == 0:
        return []
    items: Iterable[object]
    if isinstance(raw, (list, tuple, set, frozenset)):
        items = raw
    else:
        items = [raw]
    cleaned: list[int] = []
    for item in items:
        if item in (None, "", 0):
            continue
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if value not in cleaned:
            cleaned.append(value)
    return cleaned


__all__ = ["Retriever", "RetrieverMetadata"]
