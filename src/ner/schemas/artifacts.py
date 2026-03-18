"""Canonical artifact schemas for the NER multi-agent pipeline.

- Defines the shared typed artifacts passed between all NER stages.
- Defines serialization helpers that preserve nested evidence structure.
- Does not perform validation, prompting, orchestration, or storage I/O.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .constants import CANONICAL_FIELD_ORDER

EVIDENCE_FIELD_NAMES: tuple[str, str, str, str] = (
    "entity_evidence",
    "type_evidence",
    "attribute_evidence",
    "value_evidence",
)


@dataclass(slots=True)
class BillRecord:
    """Canonical raw-bill schema loaded by the corpus store.

    Attributes:
        bill_id (str): Stable bill identifier from the source corpus.
        state (str): State associated with the bill.
        text (str): Full raw bill text.
        bill_url (str): Source URL for the bill text.
        title (str): Bill title or short caption.
        status (str): Most recent legislative status string.
        date_of_last_action (str): Date string for the latest bill action.
        author (str): Sponsor or author string from the source corpus.
        topics (str): Topic tags associated with the bill.
        summary (str): Short bill summary from the source corpus.
        history (str): Legislative history text preserved from the source
            corpus.
    """

    bill_id: str
    state: str
    text: str
    bill_url: str = ""
    title: str = ""
    status: str = ""
    date_of_last_action: str = ""
    author: str = ""
    topics: str = ""
    summary: str = ""
    history: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the bill record.

        Returns:
            Plain dictionary representation of the bill record.
        """

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BillRecord":
        """Build a bill record from stored JSON data.

        Args:
            payload (dict[str, Any]): Stored dictionary representation of a
                bill record.

        Returns:
            BillRecord: Parsed ``BillRecord`` instance.
        """

        return cls(**payload)


@dataclass(slots=True)
class ContextChunk:
    """Canonical derived inference-unit schema for one bill chunk.

    Attributes:
        chunk_id (int): Stable deterministic identifier for the chunk.
        bill_id (str): Bill identifier from which the chunk was derived.
        text (str): Chunk text slice used for inference.
        start_offset (int): Inclusive start offset of the chunk in the bill
            text.
        end_offset (int): Exclusive end offset of the chunk in the bill text.
    """

    chunk_id: int
    bill_id: str
    text: str
    start_offset: int
    end_offset: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the chunk.

        Returns:
            Plain dictionary representation of the chunk.
        """

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContextChunk":
        """Build a chunk from stored JSON data.

        Args:
            payload (dict[str, Any]): Stored dictionary representation of a
                context chunk.

        Returns:
            ContextChunk: Parsed ``ContextChunk`` instance.
        """

        return cls(**payload)


@dataclass(slots=True)
class SpanRef:
    """Field-linked evidence span with source offsets and chunk provenance.

    Attributes:
        span_id (int): Stable deterministic identifier for the evidence span.
        start (int): Inclusive source-text start offset.
        end (int): Exclusive source-text end offset.
        text (str): Exact evidence text returned by the model.
        chunk_id (int): Stable chunk identifier from which the evidence was
            derived.
    """

    span_id: int
    start: int
    end: int
    text: str
    chunk_id: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the span.

        Returns:
            Plain dictionary representation of the evidence span.
        """

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SpanRef":
        """Build a span reference from stored JSON data.

        Args:
            payload (dict[str, Any]): Stored dictionary representation of an
                evidence span.

        Returns:
            SpanRef: Parsed ``SpanRef`` instance.
        """

        return cls(**payload)


@dataclass(slots=True)
class CandidateQuadruplet:
    """Zero-shot candidate with nullable fields and field-linked evidence.

    Attributes:
        candidate_id (int): Stable deterministic identifier for the candidate.
        entity (str | None): Entity field value, or ``None`` when absent.
        type (str | None): Type field value, or ``None`` when absent.
        attribute (str | None): Attribute field value, or ``None`` when absent.
        value (str | None): Value field value, or ``None`` when absent.
        entity_evidence (list[SpanRef]): Evidence spans linked to the entity
            field.
        type_evidence (list[SpanRef]): Evidence spans linked to the type field.
        attribute_evidence (list[SpanRef]): Evidence spans linked to the
            attribute field.
        value_evidence (list[SpanRef]): Evidence spans linked to the value
            field.
    """

    candidate_id: int
    entity: str | None = None
    type: str | None = None
    attribute: str | None = None
    value: str | None = None
    entity_evidence: list[SpanRef] = field(default_factory=list)
    type_evidence: list[SpanRef] = field(default_factory=list)
    attribute_evidence: list[SpanRef] = field(default_factory=list)
    value_evidence: list[SpanRef] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the candidate.

        Returns:
            Plain dictionary representation of the candidate quadruplet.
        """

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateQuadruplet":
        """Build a candidate quadruplet from stored JSON data.

        Args:
            payload (dict[str, Any]): Stored dictionary representation of a
                candidate quadruplet.

        Returns:
            CandidateQuadruplet: Parsed ``CandidateQuadruplet`` instance.
        """

        data = dict(payload)
        for field_name in EVIDENCE_FIELD_NAMES:
            data[field_name] = [
                SpanRef.from_dict(span_payload)
                for span_payload in data.get(field_name, [])
            ]
        return cls(**data)


@dataclass(slots=True)
class GroupedCandidateSet:
    """Grouped candidate set with its canonical field-score matrix.

    Attributes:
        group_id (int): Stable deterministic identifier for the grouped set.
        candidate_ids (list[int]): Candidate ids in the exact row order of the
            score matrix.
        field_score_matrix (list[list[float | None]]): Field-wise score matrix
            aligned to candidate rows and canonical field-order columns.
        field_order (tuple[str, str, str, str]): Canonical column order for
            ``field_score_matrix``.
    """

    group_id: int
    candidate_ids: list[int]
    field_score_matrix: list[list[float | None]]
    field_order: tuple[str, str, str, str] = CANONICAL_FIELD_ORDER

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the grouped set.

        Returns:
            Plain dictionary representation of the grouped candidate set.
        """

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GroupedCandidateSet":
        """Build a grouped candidate set from stored JSON data.

        Args:
            payload (dict[str, Any]): Stored dictionary representation of a
                grouped set.

        Returns:
            GroupedCandidateSet: Parsed ``GroupedCandidateSet`` instance.
        """

        data = dict(payload)
        data["field_order"] = tuple(data.get("field_order", CANONICAL_FIELD_ORDER))
        return cls(**data)


@dataclass(slots=True)
class RefinedQuadruplet:
    """Final refined output with source links and preserved evidence.

    Attributes:
        refined_id (int): Stable deterministic identifier for the refined
            output.
        source_group_id (int): Group id from which the refined output was
            produced.
        source_candidate_ids (list[int]): Candidate ids contributing to the
            refined output.
        entity (str | None): Final entity field value, or ``None`` when absent.
        type (str | None): Final type field value, or ``None`` when absent.
        attribute (str | None): Final attribute field value, or ``None`` when
            absent.
        value (str | None): Final value field value, or ``None`` when absent.
        entity_evidence (list[SpanRef]): Evidence spans linked to the entity
            field.
        type_evidence (list[SpanRef]): Evidence spans linked to the type field.
        attribute_evidence (list[SpanRef]): Evidence spans linked to the
            attribute field.
        value_evidence (list[SpanRef]): Evidence spans linked to the value
            field.
    """

    refined_id: int
    source_group_id: int
    source_candidate_ids: list[int]
    entity: str | None = None
    type: str | None = None
    attribute: str | None = None
    value: str | None = None
    entity_evidence: list[SpanRef] = field(default_factory=list)
    type_evidence: list[SpanRef] = field(default_factory=list)
    attribute_evidence: list[SpanRef] = field(default_factory=list)
    value_evidence: list[SpanRef] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the refined output.

        Returns:
            Plain dictionary representation of the refined quadruplet.
        """

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RefinedQuadruplet":
        """Build a refined quadruplet from stored JSON data.

        Args:
            payload (dict[str, Any]): Stored dictionary representation of a
                refined quadruplet.

        Returns:
            RefinedQuadruplet: Parsed ``RefinedQuadruplet`` instance.
        """

        data = dict(payload)
        for field_name in EVIDENCE_FIELD_NAMES:
            data[field_name] = [
                SpanRef.from_dict(span_payload)
                for span_payload in data.get(field_name, [])
            ]
        return cls(**data)


@dataclass(slots=True)
class RefinementArtifact:
    """Optional refinement-side relation matrices for one grouped candidate set.

    Attributes:
        group_id (int): Group id associated with the refinement artifact.
        candidate_ids (list[int]): Candidate ids in the exact row and column
            order of every relation matrix.
        entity_relations (list[list[str | None]]): Entity-level relation
            matrix.
        type_relations (list[list[str | None]]): Type-level relation matrix.
        attribute_relations (list[list[str | None]]): Attribute-level relation
            matrix.
        value_relations (list[list[str | None]]): Value-level relation matrix.
    """

    group_id: int
    candidate_ids: list[int]
    entity_relations: list[list[str | None]]
    type_relations: list[list[str | None]]
    attribute_relations: list[list[str | None]]
    value_relations: list[list[str | None]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the refinement artifact.

        Returns:
            Plain dictionary representation of the refinement artifact.
        """

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RefinementArtifact":
        """Build a refinement artifact from stored JSON data.

        Args:
            payload (dict[str, Any]): Stored dictionary representation of a
                refinement artifact.

        Returns:
            RefinementArtifact: Parsed ``RefinementArtifact`` instance.
        """

        return cls(**payload)


def artifact_to_dict(artifact: Any) -> Any:
    """Serialize one artifact or a collection of artifacts for storage/prompting.

    Args:
        artifact (Any): Artifact instance, collection, mapping, or primitive
            value to serialize.

    Returns:
        Any: JSON-serializable version of the supplied artifact payload.
    """

    if isinstance(artifact, list):
        return [artifact_to_dict(item) for item in artifact]
    if isinstance(artifact, dict):
        return {key: artifact_to_dict(value) for key, value in artifact.items()}
    if hasattr(artifact, "to_dict"):
        return artifact.to_dict()
    return artifact

