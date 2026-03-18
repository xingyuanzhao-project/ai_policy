"""Shared contracts and validation for the NER pipeline.

- Re-exports canonical artifact schemas, constants, builders, and validators.
- Defines the typed interface between all NER stages.
- Does not perform runtime orchestration or persistence.
"""

from .artifacts import (
    BillRecord,
    CandidateQuadruplet,
    ContextChunk,
    GroupedCandidateSet,
    RefinedQuadruplet,
    RefinementArtifact,
    SpanRef,
)
from .constants import CANONICAL_FIELD_ORDER, CANONICAL_RELATION_LABELS, stable_int_id
from .inference_unit_builder import ChunkingConfig, InferenceUnitBuilder
from .validation import (
    SchemaValidationError,
    validate_bill_record,
    validate_candidate_quadruplet,
    validate_context_chunk,
    validate_grouped_candidate_set,
    validate_refined_quadruplet,
    validate_refinement_artifact,
    validate_span_ref,
)

__all__ = [
    "BillRecord",
    "CandidateQuadruplet",
    "CANONICAL_FIELD_ORDER",
    "CANONICAL_RELATION_LABELS",
    "ChunkingConfig",
    "ContextChunk",
    "GroupedCandidateSet",
    "InferenceUnitBuilder",
    "RefinedQuadruplet",
    "RefinementArtifact",
    "SchemaValidationError",
    "SpanRef",
    "stable_int_id",
    "validate_bill_record",
    "validate_candidate_quadruplet",
    "validate_context_chunk",
    "validate_grouped_candidate_set",
    "validate_refined_quadruplet",
    "validate_refinement_artifact",
    "validate_span_ref",
]

