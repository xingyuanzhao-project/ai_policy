"""Local lexical retrieval over the full bill chunk corpus.

- Builds an in-memory BM25-style inverted index over typed bill chunks.
- Keeps full-corpus retrieval local so the web app can run without a persisted
  hosted-embedding cache.
- Preserves the same `RetrievedChunk` contract used by the vector retriever and
  answer synthesis layers.
- Does not call provider APIs or own Flask route behavior.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from heapq import nlargest
from typing import Sequence

from .artifacts import IndexedChunk, RetrievedChunk, validate_retrieved_chunk
from .retriever import _coerce_int_values, _coerce_str_values

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
_MIN_TOKEN_LENGTH = 2
_BM25_K1 = 1.2
_BM25_B = 0.75
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "what",
    "when",
    "which",
    "who",
    "with",
}


class LexicalRetriever:
    """Run local BM25-style retrieval without a hosted embedding dependency."""

    def __init__(self, chunks: Sequence[IndexedChunk]) -> None:
        """Initialize the lexical retriever over one chunk sequence."""

        if not chunks:
            raise ValueError("LexicalRetriever requires at least one chunk")
        self._chunks = list(chunks)
        self._document_count = len(self._chunks)
        self._doc_lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        total_term_count = 0
        for row_index, chunk in enumerate(self._chunks):
            term_counts = Counter(self._tokenize(chunk.text))
            document_length = sum(term_counts.values())
            self._doc_lengths.append(document_length)
            total_term_count += document_length
            for term, term_frequency in term_counts.items():
                self._postings[term].append((row_index, term_frequency))

        self._average_document_length = max(total_term_count / self._document_count, 1.0)

    @property
    def chunks(self) -> Sequence[IndexedChunk]:
        """Expose the underlying chunk sequence for filter-facet enumeration."""

        return self._chunks

    def retrieve_question(
        self,
        question: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Return the highest-scoring chunk matches for one natural-language question."""

        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("LexicalRetriever question must be a non-empty string")
        if top_k <= 0:
            raise ValueError("LexicalRetriever top_k must be > 0")

        query_terms = Counter(self._tokenize(normalized_question))
        if not query_terms:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term, query_term_count in query_terms.items():
            postings = self._postings.get(term)
            if not postings:
                continue
            document_frequency = len(postings)
            inverse_document_frequency = math.log(
                1.0 + (self._document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            query_weight = 1.0 + math.log(query_term_count)
            for row_index, term_frequency in postings:
                document_length = self._doc_lengths[row_index]
                normalization = _BM25_K1 * (
                    1.0 - _BM25_B + _BM25_B * document_length / self._average_document_length
                )
                score = inverse_document_frequency * (
                    term_frequency * (_BM25_K1 + 1.0) / (term_frequency + normalization)
                )
                scores[row_index] += score * query_weight

        if filters:
            scores = {
                row_index: score
                for row_index, score in scores.items()
                if self._chunk_matches_filters(self._chunks[row_index], filters)
            }
            if not scores:
                return []

        ranked_rows = nlargest(top_k, scores.items(), key=lambda item: item[1])
        results: list[RetrievedChunk] = []
        for rank, (row_index, score) in enumerate(ranked_rows, start=1):
            chunk = self._chunks[row_index]
            retrieved = RetrievedChunk(
                rank=rank,
                score=float(score),
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

    @staticmethod
    def _chunk_matches_filters(chunk: IndexedChunk, filters: dict) -> bool:
        """Return True if the chunk passes every active filter field.

        Each field supports scalar or list input: list means OR-within-field.
        Multiple fields combine with AND.
        """

        years = _coerce_int_values(filters.get("year"))
        if years and int(chunk.year) not in years:
            return False
        states = _coerce_str_values(filters.get("state"))
        if states and str(chunk.state) not in states:
            return False
        status_buckets = _coerce_str_values(filters.get("status_bucket"))
        if status_buckets and str(chunk.status_bucket) not in status_buckets:
            return False
        topics = _coerce_str_values(filters.get("topics"))
        if topics and not (set(topics) & set(chunk.topics_list)):
            return False
        return True

    def _tokenize(self, text: str) -> list[str]:
        """Normalize chunk or query text into lexical retrieval terms."""

        return [
            token
            for token in (
                match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)
            )
            if len(token) >= _MIN_TOKEN_LENGTH and token not in _STOPWORDS
        ]


__all__ = ["LexicalRetriever"]
