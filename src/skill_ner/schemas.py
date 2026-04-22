"""Output conversion from agent JSON to canonical RefinedQuadruplet dicts.

- Parses the model's final JSON response into a list of dicts compatible
  with ``RefinedQuadruplet.from_dict()``.
- Maps the agentic output format to the canonical NER pipeline schema so
  downstream evaluation (task 1.6) can treat both pipelines identically.
- Does not call the LLM, build prompts, or persist data.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _extract_json_block(text: str) -> str:
    """Extract JSON from the model response, handling fences and preamble.

    Models often return prose followed by a fenced JSON block.  This
    finds the last fenced code block (```...```) and returns its content.
    If no fences are found, returns the original text stripped.

    Args:
        text: Raw response text potentially containing fenced JSON.

    Returns:
        The extracted JSON string, or the original text if no fences found.
    """

    stripped = text.strip()

    fence_start = stripped.rfind("```")
    if fence_start == -1:
        return stripped

    # Walk backwards from the last ``` to find the opening ```
    preceding = stripped[:fence_start].rstrip()
    open_fence = preceding.rfind("```")
    if open_fence == -1:
        return stripped

    inner = preceding[open_fence:]
    first_newline = inner.find("\n")
    if first_newline == -1:
        return stripped

    return inner[first_newline + 1:].strip()


def parse_agent_response(raw_response: str) -> list[dict[str, Any]]:
    """Parse the agent's final JSON into a list of quadruplet dicts.

    The agent is instructed to return a JSON object with a ``"quadruplets"``
    key.  Each item should have at minimum ``entity``, ``type``,
    ``attribute``, ``value`` fields.  This function normalizes the raw output
    into dicts compatible with ``RefinedQuadruplet.from_dict()``.

    Args:
        raw_response: The model's final text response (expected to be JSON).

    Returns:
        List of quadruplet dicts with canonical field names. Each dict has
        ``refined_id``, ``source_group_id``, ``source_candidate_ids``,
        the four quadruplet fields, and four evidence arrays.
    """

    cleaned = _extract_json_block(raw_response)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error("Agent response is not valid JSON; returning empty list")
        return []

    raw_quads: list[dict[str, Any]]
    if isinstance(payload, dict):
        raw_quads = payload.get("quadruplets", [])
        if not isinstance(raw_quads, list):
            logger.error(
                "Expected 'quadruplets' to be a list, got %s",
                type(raw_quads).__name__,
            )
            return []
    elif isinstance(payload, list):
        raw_quads = payload
    else:
        logger.error("Unexpected payload type: %s", type(payload).__name__)
        return []

    results: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_quads):
        if not isinstance(raw, dict):
            logger.warning("Skipping non-dict quadruplet at index %d", idx)
            continue
        results.append(_normalize_quadruplet(raw, idx))

    logger.info("Parsed %d quadruplet(s) from agent response", len(results))
    return results


def _normalize_quadruplet(
    raw: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    """Normalize one raw quadruplet dict to the canonical schema.

    Args:
        raw: Raw quadruplet dict from the agent's JSON output.
        index: Positional index used as a fallback ``refined_id``.

    Returns:
        Dict compatible with ``RefinedQuadruplet.from_dict()``.
    """

    evidence_fields = (
        "entity_evidence",
        "type_evidence",
        "attribute_evidence",
        "value_evidence",
    )

    return {
        "refined_id": raw.get("refined_id", index),
        "source_group_id": raw.get("source_group_id", 0),
        "source_candidate_ids": raw.get("source_candidate_ids", []),
        "entity": raw.get("entity"),
        "type": raw.get("type"),
        "attribute": raw.get("attribute"),
        "value": raw.get("value"),
        **{
            field: _normalize_evidence(raw.get(field, []))
            for field in evidence_fields
        },
    }


def _normalize_evidence(
    raw_evidence: Any,
) -> list[dict[str, Any]]:
    """Normalize an evidence array to canonical SpanRef-compatible dicts.

    Args:
        raw_evidence: Evidence value from the agent output (expected list).

    Returns:
        List of dicts with ``span_id``, ``start``, ``end``, ``text``,
        ``chunk_id`` keys.
    """

    if not isinstance(raw_evidence, list):
        return []

    result: list[dict[str, Any]] = []
    for idx, span in enumerate(raw_evidence):
        if not isinstance(span, dict):
            continue
        result.append(
            {
                "span_id": span.get("span_id", idx),
                "start": span.get("start", span.get("start_offset", 0)),
                "end": span.get("end", span.get("end_offset", 0)),
                "text": span.get("text", ""),
                "chunk_id": span.get("chunk_id", 0),
            }
        )
    return result
