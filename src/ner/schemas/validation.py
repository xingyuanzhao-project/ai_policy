"""Schema validation for all NER pipeline artifacts.

- Owns schema validation for every canonical artifact type in the NER pipeline.
- Enforces field-order, relation-label, and evidence-shape invariants.
- Does not perform prompting, storage I/O, or orchestration.
"""

from __future__ import annotations

from .artifacts import (
    BillRecord,
    CandidateQuadruplet,
    ContextChunk,
    EVIDENCE_FIELD_NAMES,
    GroupedCandidateSet,
    RefinedQuadruplet,
    RefinementArtifact,
    SpanRef,
)
from .constants import CANONICAL_FIELD_ORDER, CANONICAL_RELATION_LABELS

_CANONICAL_RELATION_LABEL_SET = frozenset(CANONICAL_RELATION_LABELS)


class SchemaValidationError(ValueError):
    """Raised when a parsed or loaded artifact breaks its declared schema."""


def validate_bill_record(record: BillRecord) -> None:
    """Validate the canonical raw-bill schema.

    Args:
        record (BillRecord): Bill record to validate.

    Returns:
        None: This function validates in place and raises on invalid input.

    Raises:
        SchemaValidationError: If the bill record violates the canonical raw
            bill contract.
    """

    if not isinstance(record.bill_id, str) or not record.bill_id.strip():
        raise SchemaValidationError("BillRecord.bill_id must be a non-empty string")
    if not isinstance(record.state, str) or not record.state.strip():
        raise SchemaValidationError("BillRecord.state must be a non-empty string")
    if not isinstance(record.text, str):
        raise SchemaValidationError("BillRecord.text must be a string")


def validate_context_chunk(chunk: ContextChunk) -> None:
    """Validate the canonical derived chunk schema.

    Args:
        chunk (ContextChunk): Context chunk to validate.

    Returns:
        None: This function validates in place and raises on invalid input.

    Raises:
        SchemaValidationError: If the chunk violates the canonical context
            chunk contract.
    """

    if not isinstance(chunk.chunk_id, int):
        raise SchemaValidationError("ContextChunk.chunk_id must be an integer")
    if not isinstance(chunk.bill_id, str) or not chunk.bill_id.strip():
        raise SchemaValidationError("ContextChunk.bill_id must be a non-empty string")
    if not isinstance(chunk.text, str):
        raise SchemaValidationError("ContextChunk.text must be a string")
    if not isinstance(chunk.start_offset, int) or chunk.start_offset < 0:
        raise SchemaValidationError("ContextChunk.start_offset must be >= 0")
    if not isinstance(chunk.end_offset, int) or chunk.end_offset < chunk.start_offset:
        raise SchemaValidationError(
            "ContextChunk.end_offset must be >= ContextChunk.start_offset"
        )


def validate_span_ref(span: SpanRef) -> None:
    """Validate one evidence span.

    Args:
        span (SpanRef): Evidence span to validate.

    Returns:
        None: This function validates in place and raises on invalid input.

    Raises:
        SchemaValidationError: If the span violates the canonical span
            reference contract.
    """

    if not isinstance(span.span_id, int):
        raise SchemaValidationError("SpanRef.span_id must be an integer")
    if not isinstance(span.chunk_id, int):
        raise SchemaValidationError("SpanRef.chunk_id must be an integer")
    if not isinstance(span.start, int) or span.start < 0:
        raise SchemaValidationError("SpanRef.start must be >= 0")
    if not isinstance(span.end, int) or span.end < span.start:
        raise SchemaValidationError("SpanRef.end must be >= SpanRef.start")
    if not isinstance(span.text, str):
        raise SchemaValidationError("SpanRef.text must be a string")


def _validate_evidence_fields(owner: CandidateQuadruplet | RefinedQuadruplet) -> None:
    """Validate evidence fields shared by candidate and refined artifacts.

    Args:
        owner (CandidateQuadruplet | RefinedQuadruplet): Candidate or refined
            artifact whose evidence lists should be validated.

    Returns:
        None: This function validates in place and raises on invalid input.

    Raises:
        SchemaValidationError: If an evidence field is not a list of valid
            ``SpanRef`` objects.
    """

    for field_name in EVIDENCE_FIELD_NAMES:
        evidence = getattr(owner, field_name)
        if not isinstance(evidence, list):
            raise SchemaValidationError(f"{field_name} must be a list of SpanRef")
        for span in evidence:
            if not isinstance(span, SpanRef):
                raise SchemaValidationError(f"{field_name} must contain only SpanRef")
            validate_span_ref(span)


def validate_candidate_quadruplet(candidate: CandidateQuadruplet) -> None:
    """Validate one zero-shot candidate quadruplet.

    Args:
        candidate (CandidateQuadruplet): Candidate quadruplet to validate.

    Returns:
        None: This function validates in place and raises on invalid input.

    Raises:
        SchemaValidationError: If the candidate violates the zero-shot candidate
            contract.
    """

    if not isinstance(candidate.candidate_id, int):
        raise SchemaValidationError("CandidateQuadruplet.candidate_id must be an integer")
    for field_name in CANONICAL_FIELD_ORDER:
        value = getattr(candidate, field_name)
        if value is not None and not isinstance(value, str):
            raise SchemaValidationError(f"{field_name} must be a string or None")
    _validate_evidence_fields(candidate)


def validate_grouped_candidate_set(group: GroupedCandidateSet) -> None:
    """Validate one grouped candidate set and its score matrix.

    Args:
        group (GroupedCandidateSet): Grouped candidate set to validate.

    Returns:
        None: This function validates in place and raises on invalid input.

    Raises:
        SchemaValidationError: If the grouped set violates the canonical
            grouping contract or score-matrix alignment rules.
    """

    if not isinstance(group.group_id, int):
        raise SchemaValidationError("GroupedCandidateSet.group_id must be an integer")
    if not isinstance(group.candidate_ids, list) or not group.candidate_ids:
        raise SchemaValidationError("GroupedCandidateSet.candidate_ids must be a non-empty list")
    if len(set(group.candidate_ids)) != len(group.candidate_ids):
        raise SchemaValidationError("GroupedCandidateSet.candidate_ids must be unique")
    if tuple(group.field_order) != CANONICAL_FIELD_ORDER:
        raise SchemaValidationError(
            "GroupedCandidateSet.field_order must match the canonical field order"
        )
    if not isinstance(group.field_score_matrix, list):
        raise SchemaValidationError("field_score_matrix must be a list of rows")
    if len(group.field_score_matrix) != len(group.candidate_ids):
        raise SchemaValidationError(
            "field_score_matrix row count must match candidate_ids row order"
        )
    for row in group.field_score_matrix:
        if not isinstance(row, list) or len(row) != len(group.field_order):
            raise SchemaValidationError(
                "field_score_matrix column count must match field_order"
            )
        for value in row:
            if value is not None and (
                not isinstance(value, (int, float)) or value < 0.0 or value > 1.0
            ):
                raise SchemaValidationError(
                    "field_score_matrix values must be floats between 0.0 and 1.0 or None"
                )


def validate_refined_quadruplet(refined: RefinedQuadruplet) -> None:
    """Validate one final refined quadruplet.

    Args:
        refined (RefinedQuadruplet): Refined quadruplet to validate.

    Returns:
        None: This function validates in place and raises on invalid input.

    Raises:
        SchemaValidationError: If the refined quadruplet violates the final
            output contract.
    """

    if not isinstance(refined.refined_id, int):
        raise SchemaValidationError("RefinedQuadruplet.refined_id must be an integer")
    if not isinstance(refined.source_group_id, int):
        raise SchemaValidationError("RefinedQuadruplet.source_group_id must be an integer")
    if not isinstance(refined.source_candidate_ids, list) or not refined.source_candidate_ids:
        raise SchemaValidationError(
            "RefinedQuadruplet.source_candidate_ids must be a non-empty list"
        )
    for field_name in CANONICAL_FIELD_ORDER:
        value = getattr(refined, field_name)
        if value is not None and not isinstance(value, str):
            raise SchemaValidationError(f"{field_name} must be a string or None")
    _validate_evidence_fields(refined)


def validate_refinement_artifact(artifact: RefinementArtifact) -> None:
    """Validate one optional refinement-side relation artifact.

    Args:
        artifact (RefinementArtifact): Refinement artifact to validate.

    Returns:
        None: This function validates in place and raises on invalid input.

    Raises:
        SchemaValidationError: If the artifact violates candidate-order or
            canonical relation-label rules.
    """

    if not isinstance(artifact.group_id, int):
        raise SchemaValidationError("RefinementArtifact.group_id must be an integer")
    if not isinstance(artifact.candidate_ids, list):
        raise SchemaValidationError("RefinementArtifact.candidate_ids must be a list")
    candidate_count = len(artifact.candidate_ids)
    for field_name in (
        "entity_relations",
        "type_relations",
        "attribute_relations",
        "value_relations",
    ):
        matrix = getattr(artifact, field_name)
        if not isinstance(matrix, list) or len(matrix) != candidate_count:
            raise SchemaValidationError(
                f"{field_name} must be a square matrix aligned with candidate_ids"
            )
        for row in matrix:
            if not isinstance(row, list) or len(row) != candidate_count:
                raise SchemaValidationError(
                    f"{field_name} must be a square matrix aligned with candidate_ids"
                )
            for value in row:
                if value is not None and value not in _CANONICAL_RELATION_LABEL_SET:
                    raise SchemaValidationError(
                        f"{field_name} contains non-canonical relation label '{value}'"
                    )

