"""Locked prompt templates and JSON schemas for the judge-backed stages.

- Holds the calibrated prompt text used by Stage 3 (per-quadruplet grounding)
  and Stage 4 (set-to-label coverage); Stage 6 pairwise comparisons and
  Stage 8 CALM bias audit reuse the same two base prompts.
- Mirrors the structured-output discipline from Laskar et al. 2025 (see
  ``docs/lit_rev_eval.md``) by returning a JSON object per call rather than
  free text, so verdict parsing is deterministic.
- Does not call the judge, load data, or decide which subset of prompts to
  use; callers render placeholder fields and pass the result to
  ``src.eval.judge.call_judge``.
"""

from __future__ import annotations

from typing import Any

GROUNDING_SYSTEM_PROMPT = """\
You are an expert evaluator of structured information extraction from US state
AI legislation. Your task is to decide whether a single extracted quadruplet
is grounded in the bill text.

Return a strict three-way verdict per the RefChecker/NLI convention:
- "entailed": the bill text (especially the cited evidence spans) clearly
  supports every field of the quadruplet.
- "neutral": the bill text neither clearly supports nor contradicts the
  quadruplet; information is missing or ambiguous.
- "contradicted": the bill text clearly contradicts at least one field.

Judge only by what is present in the bill text supplied to you. Do not use
outside knowledge. Respond as JSON that matches the schema you are given.
"""


GROUNDING_USER_PROMPT_TEMPLATE = """\
Bill id: {bill_id}
State: {state}   Year: {year}

Evidence spans (character offsets into the bill text):
{evidence_block}

Claim under test (quadruplet):
  entity    = {entity}
  type      = {entity_type}
  attribute = {attribute}
  value     = {value}

Relevant bill text (window around the evidence spans):
\"\"\"
{bill_excerpt}
\"\"\"

Decide whether the claim is entailed, neutral, or contradicted by the bill
text above. Cite evidence using a brief rationale (<=30 words).
"""


GROUNDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "rationale"],
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["entailed", "neutral", "contradicted"],
        },
        "rationale": {"type": "string", "maxLength": 500},
    },
}


COVERAGE_SYSTEM_PROMPT = """\
You are an expert evaluator of coverage between a set of fine-grained
information-extraction quadruplets and a single coarse, human-assigned
topic label for a US state AI bill.

You must decide, solely from the quadruplet set and the (optional) bill
summary, whether the set collectively covers the human label:
- "covered": one or more quadruplets in the set directly instantiate the
  topic; you can name the specific quadruplet ids that support coverage.
- "partially_covered": a subset of quadruplets touches the topic but does
  not fully capture its scope (e.g. policy mentioned but missing an actor).
- "not_covered": no quadruplet reasonably instantiates the topic.

Be strict: topic labels are broad policy categories ("Government Use",
"Workforce", "Health", ...). A quadruplet about a tangential technology
should not be treated as coverage of an unrelated topic. When the verdict
is "covered" or "partially_covered", list the quadruplet_ids that support
the verdict in "supporting_ids". Return JSON that matches the schema.
"""


COVERAGE_USER_PROMPT_TEMPLATE = """\
Bill id: {bill_id}
State: {state}   Year: {year}
Bill title: {title}
Human summary: {summary}

Human-assigned topic label under evaluation: "{label}"

Extracted quadruplets (from method "{method}"):
{quadruplet_block}

Decide whether the set collectively covers the topic label. If covered or
partially covered, list the supporting quadruplet ids.
"""


COVERAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "rationale", "supporting_ids"],
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["covered", "partially_covered", "not_covered"],
        },
        "rationale": {"type": "string", "maxLength": 600},
        "supporting_ids": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 20,
        },
    },
}


PAIRWISE_SYSTEM_PROMPT = """\
You are comparing two information-extraction systems (A and B) on a US
state AI bill. Each system produced a set of (entity, type, attribute,
value) quadruplets. Using the human-assigned topic labels as a reference,
decide which system's set more faithfully covers the bill's AI-related
policy content.

Be aware of the MT-Bench pairwise pitfalls: do not prefer a system merely
because its set is longer or more verbose; judge on coverage of the
reference labels and grounded specificity of the quadruplets. Return JSON
matching the schema.
"""


PAIRWISE_USER_PROMPT_TEMPLATE = """\
Bill id: {bill_id}
State: {state}   Year: {year}
Bill title: {title}
Reference (human) topic labels: {topics}

System A quadruplets (n={a_count}):
{a_block}

System B quadruplets (n={b_count}):
{b_block}

Pick one of "A", "B", or "tie". Provide a one-sentence rationale.
"""


PAIRWISE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["winner", "rationale"],
    "properties": {
        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
        "rationale": {"type": "string", "maxLength": 400},
    },
}
