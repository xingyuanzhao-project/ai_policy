"""Value types shared across the eval pipeline stages.

- ``Quadruplet`` is the normalised per-extraction record that Stage 2 onwards
  consumes, abstracting over minor shape differences between the orchestrated
  and skill-driven run outputs.
- ``EvidenceSpan``, ``JudgeVerdict`` and ``StageResult`` capture the common
  judge output and stage-level aggregate shapes.
- All dataclasses are defined with ``slots=True`` so they are lightweight and
  raise on typos.
- No I/O, judging, or aggregation logic lives here; see ``io.py``,
  ``judge.py`` and the individual stage modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

VerdictLabel = Literal[
    "entailed",
    "neutral",
    "contradicted",
    "covered",
    "partially_covered",
    "not_covered",
    "error",
]


@dataclass(slots=True, frozen=True)
class EvidenceSpan:
    """One character-offset span tying a field to a region of bill text.

    Attributes:
        start: Inclusive character start offset into the bill text.
        end: Exclusive character end offset into the bill text.
        text: The literal substring from the bill at ``[start, end)``.
        chunk_id: Identifier of the text chunk the span came from (may be 0
            for the skill-driven run, which does not chunk).
    """

    start: int
    end: int
    text: str
    chunk_id: int = 0


@dataclass(slots=True)
class Quadruplet:
    """Normalised (entity, type, attribute, value) extraction with evidence.

    A single quadruplet carries its own identifier plus per-field evidence
    span lists so Stage 2 (rule-based plausibility) and Stage 3 (grounding)
    can reason about each field independently.

    Attributes:
        bill_id: Composite bill identifier of the form ``{year}__{state_id}``.
        method: Method name that produced this quadruplet, e.g.
            ``orchestrated`` or ``skill_driven``.
        quadruplet_id: Stable identifier used in cache keys; combined with
            ``bill_id`` to avoid collisions across methods.
        entity: Extracted entity surface form.
        type: Extracted entity type.
        attribute: Extracted attribute label.
        value: Extracted value for that attribute.
        entity_evidence: Evidence spans linked to the entity field.
        type_evidence: Evidence spans linked to the type field.
        attribute_evidence: Evidence spans linked to the attribute field.
        value_evidence: Evidence spans linked to the value field.
    """

    bill_id: str
    method: str
    quadruplet_id: str
    entity: str
    type: str
    attribute: str
    value: str
    entity_evidence: list[EvidenceSpan] = field(default_factory=list)
    type_evidence: list[EvidenceSpan] = field(default_factory=list)
    attribute_evidence: list[EvidenceSpan] = field(default_factory=list)
    value_evidence: list[EvidenceSpan] = field(default_factory=list)

    def all_spans(self) -> list[EvidenceSpan]:
        """Return the flat list of every evidence span attached to this row."""

        return [
            *self.entity_evidence,
            *self.type_evidence,
            *self.attribute_evidence,
            *self.value_evidence,
        ]


@dataclass(slots=True)
class BillRecord:
    """NCSL corpus row joined with its extracted quadruplets.

    Attributes:
        bill_id: Composite identifier matching the NER run output filename
            stem (``{year}__{state_id}``).
        source_bill_id: The bare NCSL ``bill_id`` column value.
        year: Year as an integer (2023, 2024, ...).
        state: Full state name, e.g. ``"Arizona"``.
        text: Raw bill text as stored in the NCSL JSONL.
        title: NCSL title, used in some judge prompts.
        summary: NCSL human summary, used as optional context.
        topics_raw: The raw ``topics`` string from NCSL before splitting.
        topics: Cleaned topic labels after splitting on ``;`` and ``,``.
    """

    bill_id: str
    source_bill_id: str
    year: int
    state: str
    text: str
    title: str
    summary: str
    topics_raw: str
    topics: list[str]


@dataclass(slots=True)
class JudgeVerdict:
    """One judge decision recorded in the per-item cache.

    Attributes:
        verdict: Normalised label the judge returned (see :data:`VerdictLabel`).
        rationale: Short natural-language rationale the judge produced.
        supporting_ids: Quadruplet ids the judge cited as supporting the
            verdict (used by Stage 4 to build the coverage subset).
        raw: The unmodified JSON payload the judge returned, for audit.
        usage: Provider-reported token usage for this call, 8-key format.
    """

    verdict: VerdictLabel
    rationale: str = ""
    supporting_ids: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StageResult:
    """Aggregate artefact a stage writes under ``results/``.

    Attributes:
        stage: Stage number (1..9).
        status: ``completed`` on success, ``skipped`` when the stage opted
            out (e.g. missing expert file for Stage 7), or ``error``.
        summary: Human-readable one-liner; also printed to stdout by the
            orchestrator.
        metrics: Stage-specific metric dictionary (coverage rate, grounding
            proportions, pairwise win rates, ...).
        artifacts: Map of artefact label to filesystem path, rooted under
            ``output/evals/v1/``.
    """

    stage: int
    status: Literal["completed", "skipped", "error"]
    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
