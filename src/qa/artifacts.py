"""Typed artifacts for the OpenRouter-first QA pipeline.

- Defines persisted index metadata, indexed chunks, retrieved chunks, and answer
  payloads used by the local RAG app.
- Preserves explicit serialization contracts so cache files, routes, and tests
  all share the same schema.
- Does not perform retrieval, model calls, or file-system orchestration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

INDEX_STATUS_BUILDING = "building"
INDEX_STATUS_READY = "ready"
_VALID_INDEX_STATUSES = frozenset({INDEX_STATUS_BUILDING, INDEX_STATUS_READY})

STATUS_BUCKETS: tuple[str, ...] = ("Enacted", "Failed", "Vetoed", "Pending", "Other")
_VALID_STATUS_BUCKETS = frozenset(STATUS_BUCKETS)


class QAArtifactValidationError(ValueError):
    """Raised when a QA artifact violates its declared schema."""


@dataclass(slots=True)
class IndexedChunk:
    """Persisted chunk payload aligned to one embedding row in the local index."""

    chunk_id: int
    bill_id: str
    text: str
    start_offset: int
    end_offset: int
    state: str
    title: str = ""
    status: str = ""
    summary: str = ""
    bill_url: str = ""
    year: int = 0
    status_bucket: str = "Other"
    topics_list: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the indexed chunk."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IndexedChunk":
        """Build an indexed chunk from stored JSON data."""

        chunk = cls(**payload)
        validate_indexed_chunk(chunk)
        return chunk


@dataclass(slots=True)
class RetrievedChunk:
    """Ranked retrieval result ready for answer synthesis or UI rendering."""

    rank: int
    score: float
    chunk_id: int
    bill_id: str
    text: str
    start_offset: int
    end_offset: int
    state: str
    title: str = ""
    status: str = ""
    summary: str = ""
    bill_url: str = ""
    year: int = 0
    status_bucket: str = "Other"
    topics_list: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the retrieved chunk."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RetrievedChunk":
        """Build a retrieved chunk from stored JSON data."""

        chunk = cls(**payload)
        validate_retrieved_chunk(chunk)
        return chunk


@dataclass(slots=True)
class AnswerResult:
    """Answer payload returned by the QA service and UI routes."""

    question: str
    answer: str
    answer_model: str
    citations: list[RetrievedChunk] = field(default_factory=list)
    applied_filters: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the answer result."""

        payload = asdict(self)
        payload["citations"] = [citation.to_dict() for citation in self.citations]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnswerResult":
        """Build an answer result from stored JSON data."""

        raw_applied = payload.get("applied_filters", {})
        applied_filters = raw_applied if isinstance(raw_applied, dict) else {}
        raw_trace = payload.get("trace", []) or []
        trace = list(raw_trace) if isinstance(raw_trace, list) else []
        result = cls(
            question=payload["question"],
            answer=payload["answer"],
            answer_model=payload["answer_model"],
            citations=[
                RetrievedChunk.from_dict(citation_payload)
                for citation_payload in payload.get("citations", [])
            ],
            applied_filters=dict(applied_filters),
            trace=trace,
        )
        validate_answer_result(result)
        return result


@dataclass(slots=True)
class IndexManifest:
    """Manifest describing the inputs and status of a persisted QA index."""

    index_format_version: int
    status: str
    corpus_path: str
    corpus_fingerprint: str
    chunk_size: int
    overlap: int
    provider_api_base_url: str
    embedding_model: str
    batch_size: int
    total_chunks: int
    completed_batch_count: int
    built_at_utc: str
    bill_limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the index manifest."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IndexManifest":
        """Build an index manifest from stored JSON data."""

        manifest = cls(**payload)
        validate_index_manifest(manifest)
        return manifest


def validate_indexed_chunk(chunk: IndexedChunk) -> None:
    """Validate one indexed chunk."""

    if not isinstance(chunk.chunk_id, int):
        raise QAArtifactValidationError("IndexedChunk.chunk_id must be an integer")
    if not isinstance(chunk.bill_id, str) or not chunk.bill_id.strip():
        raise QAArtifactValidationError("IndexedChunk.bill_id must be a non-empty string")
    if not isinstance(chunk.text, str) or not chunk.text.strip():
        raise QAArtifactValidationError("IndexedChunk.text must be a non-empty string")
    if not isinstance(chunk.state, str) or not chunk.state.strip():
        raise QAArtifactValidationError("IndexedChunk.state must be a non-empty string")
    if not isinstance(chunk.start_offset, int) or chunk.start_offset < 0:
        raise QAArtifactValidationError("IndexedChunk.start_offset must be >= 0")
    if not isinstance(chunk.end_offset, int) or chunk.end_offset < chunk.start_offset:
        raise QAArtifactValidationError(
            "IndexedChunk.end_offset must be >= IndexedChunk.start_offset"
        )
    if not isinstance(chunk.year, int) or chunk.year < 0:
        raise QAArtifactValidationError("IndexedChunk.year must be an integer >= 0")
    if chunk.status_bucket not in _VALID_STATUS_BUCKETS:
        raise QAArtifactValidationError(
            f"IndexedChunk.status_bucket must be one of {sorted(_VALID_STATUS_BUCKETS)}"
        )
    if not isinstance(chunk.topics_list, list) or not all(
        isinstance(topic, str) for topic in chunk.topics_list
    ):
        raise QAArtifactValidationError(
            "IndexedChunk.topics_list must be a list of strings"
        )


def validate_retrieved_chunk(chunk: RetrievedChunk) -> None:
    """Validate one retrieved chunk."""

    if not isinstance(chunk.rank, int) or chunk.rank <= 0:
        raise QAArtifactValidationError("RetrievedChunk.rank must be a positive integer")
    if not isinstance(chunk.score, (int, float)):
        raise QAArtifactValidationError("RetrievedChunk.score must be numeric")
    validate_indexed_chunk(
        IndexedChunk(
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
    )


def validate_answer_result(result: AnswerResult) -> None:
    """Validate one answer result."""

    if not isinstance(result.question, str) or not result.question.strip():
        raise QAArtifactValidationError("AnswerResult.question must be a non-empty string")
    if not isinstance(result.answer, str):
        raise QAArtifactValidationError("AnswerResult.answer must be a string")
    if not isinstance(result.answer_model, str) or not result.answer_model.strip():
        raise QAArtifactValidationError(
            "AnswerResult.answer_model must be a non-empty string"
        )
    if not isinstance(result.citations, list):
        raise QAArtifactValidationError("AnswerResult.citations must be a list")
    for citation in result.citations:
        if not isinstance(citation, RetrievedChunk):
            raise QAArtifactValidationError(
                "AnswerResult.citations must contain only RetrievedChunk values"
            )
        validate_retrieved_chunk(citation)
    if not isinstance(result.applied_filters, dict):
        raise QAArtifactValidationError(
            "AnswerResult.applied_filters must be a dict"
        )
    if not isinstance(result.trace, list):
        raise QAArtifactValidationError("AnswerResult.trace must be a list")


def validate_index_manifest(manifest: IndexManifest) -> None:
    """Validate one persisted index manifest."""

    if not isinstance(manifest.index_format_version, int) or manifest.index_format_version < 1:
        raise QAArtifactValidationError(
            "IndexManifest.index_format_version must be an integer >= 1"
        )
    if manifest.status not in _VALID_INDEX_STATUSES:
        raise QAArtifactValidationError(
            f"IndexManifest.status must be one of {sorted(_VALID_INDEX_STATUSES)}"
        )
    if not isinstance(manifest.corpus_path, str) or not manifest.corpus_path.strip():
        raise QAArtifactValidationError("IndexManifest.corpus_path must be a non-empty string")
    if not isinstance(manifest.corpus_fingerprint, str) or not manifest.corpus_fingerprint.strip():
        raise QAArtifactValidationError(
            "IndexManifest.corpus_fingerprint must be a non-empty string"
        )
    if not isinstance(manifest.embedding_model, str) or not manifest.embedding_model.strip():
        raise QAArtifactValidationError(
            "IndexManifest.embedding_model must be a non-empty string"
        )
    if (
        not isinstance(manifest.provider_api_base_url, str)
        or not manifest.provider_api_base_url.strip()
    ):
        raise QAArtifactValidationError(
            "IndexManifest.provider_api_base_url must be a non-empty string"
        )
    if not isinstance(manifest.chunk_size, int) or manifest.chunk_size <= 0:
        raise QAArtifactValidationError("IndexManifest.chunk_size must be > 0")
    if not isinstance(manifest.overlap, int) or manifest.overlap < 0:
        raise QAArtifactValidationError("IndexManifest.overlap must be >= 0")
    if manifest.overlap >= manifest.chunk_size:
        raise QAArtifactValidationError(
            "IndexManifest.overlap must be smaller than IndexManifest.chunk_size"
        )
    if not isinstance(manifest.batch_size, int) or manifest.batch_size <= 0:
        raise QAArtifactValidationError("IndexManifest.batch_size must be > 0")
    if not isinstance(manifest.total_chunks, int) or manifest.total_chunks < 0:
        raise QAArtifactValidationError("IndexManifest.total_chunks must be >= 0")
    if (
        not isinstance(manifest.completed_batch_count, int)
        or manifest.completed_batch_count < 0
    ):
        raise QAArtifactValidationError(
            "IndexManifest.completed_batch_count must be >= 0"
        )
    if not isinstance(manifest.built_at_utc, str) or not manifest.built_at_utc.strip():
        raise QAArtifactValidationError("IndexManifest.built_at_utc must be a non-empty string")
    if manifest.bill_limit is not None and (
        not isinstance(manifest.bill_limit, int) or manifest.bill_limit <= 0
    ):
        raise QAArtifactValidationError("IndexManifest.bill_limit must be a positive integer or None")


__all__ = [
    "AnswerResult",
    "INDEX_STATUS_BUILDING",
    "INDEX_STATUS_READY",
    "IndexManifest",
    "IndexedChunk",
    "QAArtifactValidationError",
    "RetrievedChunk",
    "STATUS_BUCKETS",
    "validate_answer_result",
    "validate_index_manifest",
    "validate_indexed_chunk",
    "validate_retrieved_chunk",
]
