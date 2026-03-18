# Proposed Architecture

## Zero-shot Annotator

- In:
    - Context Chunks
- Out:
    - candidate quadruplets `<entity, type, attribute, value>`
    - fields can be missing at this stage
    - field-linked evidence map
    - chunk-local candidate ids
    - this stage proposes structured candidates, not finalized quadruplets

@dataclass(slots=True)
class SpanRef:
    span_id: int
    start: int
    end: int
    text: str
    chunk_id: int

@dataclass(slots=True)
class CandidateQuadruplet:
    candidate_id: int

    entity: str | None = None
    type: str | None = None
    attribute: str | None = None
    value: str | None = None

    entity_evidence: list[SpanRef] = field(default_factory=list)
    type_evidence: list[SpanRef] = field(default_factory=list)
    attribute_evidence: list[SpanRef] = field(default_factory=list)
    value_evidence: list[SpanRef] = field(default_factory=list)

## Eval Assembler

- In:
    - candidate quadruplets pool from Self-Annotator
    - each candidate already carries `candidate_id`
    - each candidate already carries field-linked evidence
- Out:
    - grouped candidate sets
        - candidates judged as related are grouped here by the Eval Assembler itself
    - scores on each field
        - `entity`
        - `type`
        - `attribute`
        - `value`
    - candidate ids remain linked to the original candidate pool and its evidence
    - this stage groups and scores candidates, but does not finalize `support / overlap / conflict / duplicate / refinement`

@dataclass(slots=True)
class GroupedCandidateSet:
    group_id: int
    candidate_ids: list[int]
    field_order: tuple[str, str, str, str] = ("entity", "type", "attribute", "value")
    field_score_matrix: list[list[float | None]]
    # every element is a score between 0 and 1
    # row order follows candidate_ids
    # column order follows field_order

## Granularity Refiner

- In:
    - grouped candidate sets
    - candidate quadruplets pool referenced by candidate ids, with field-linked evidence
- Out:
    - refined quadruplets `<entity, type, attribute, value>`
    - field-linked evidence preserved in each refined quadruplet
    - field refinement artifact: field-wise relation matrix
        - `support`
        - `overlap`
        - `conflict`
        - `duplicate`
        - `refinement`
    - this stage refines grouped candidates into final structured outputs without re-reading raw chunks alone

@dataclass(slots=True)
class RefinedQuadruplet:
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

@dataclass(slots=True)
class RefinementArtifact:
    group_id: int
    candidate_ids: list[int]
    entity_relations: list[list[str | None]]
    type_relations: list[list[str | None]]
    attribute_relations: list[list[str | None]]
    value_relations: list[list[str | None]]
    # every non-null relation entry must be one of:
    # support, overlap, conflict, duplicate, refinement
    # row and column order follow candidate_ids
