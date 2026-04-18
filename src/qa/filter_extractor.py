"""Self-query filter extractor for the QA pipeline.

- Makes exactly one tool-constrained LLM call to turn a natural-language
  question into ``(semantic_query, filters)`` without any user dropdown input.
- Uses ``tool_choice`` to force a single ``search_corpus`` tool call, parses
  the returned JSON arguments, and validates each filter value against the
  currently-indexed facet values before returning.
- Silently falls back to ``ExtractedQuery(question, {})`` on any API, parsing,
  or validation failure so the QA pipeline can still answer unfiltered.
- Does not perform retrieval, embedding, or answer synthesis itself.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from .artifacts import STATUS_BUCKETS

logger = logging.getLogger(__name__)

_EXTRACTOR_SYSTEM_PROMPT = (
    "You convert a user's natural-language question about United States state "
    "AI legislation into structured search arguments. "
    "You MUST call the `search_corpus` tool exactly once. "
    "Rules:\n"
    "- `semantic_query`: rewrite the question focusing on the topical meaning; "
    "remove filter phrases (state names, years, status words, topic labels) "
    "because those are handled by the structured filters.\n"
    "- Only populate `state`, `year`, `status_bucket`, or `topics` when the "
    "question clearly constrains them. When in doubt, omit the field.\n"
    "- Every filter field is a LIST. Use a single-element list for one value "
    "(e.g. state=[\"CA\"]) and a multi-element list when the user asks for "
    "several alternatives joined by 'or' (e.g. 'California or Texas' -> "
    "state=[\"CA\",\"TX\"]; 'enacted or pending' -> status_bucket=[\"Enacted\","
    "\"Pending\"]). Within a field, multiple values mean OR at retrieval; "
    "across fields values are combined with AND.\n"
    "- Every item in `state` MUST be one of the listed state codes, every "
    "item in `status_bucket` MUST be one of the listed buckets, and every "
    "item in `topics` MUST be an exact match from the listed topic "
    "vocabulary. Never invent new values."
)

_EXTRACTOR_MAX_TOKENS = 256
_TOOL_NAME = "search_corpus"


@dataclass(slots=True)
class ExtractedQuery:
    """Result of running the self-query filter extractor on one question."""

    semantic_query: str
    filters: dict[str, Any] = field(default_factory=dict)


class FilterExtractor:
    """Run one tool-constrained LLM call to extract filters from a question."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        *,
        max_topic_enum_values: int = 200,
    ) -> None:
        """Initialize the extractor with a bound OpenAI-compatible client.

        Args:
            client: Connected OpenAI-compatible client instance.
            model: Model identifier used for the single extraction call.
            max_topic_enum_values: Cap on the topic enum list to keep the tool
                schema under provider payload limits.
        """

        if not model.strip():
            raise ValueError("FilterExtractor.model must be a non-empty string")
        if max_topic_enum_values <= 0:
            raise ValueError("FilterExtractor.max_topic_enum_values must be > 0")
        self._client = client
        self._model = model
        self._max_topic_enum_values = max_topic_enum_values

    @property
    def model(self) -> str:
        """Return the model identifier used for the extraction call."""

        return self._model

    def extract(
        self,
        question: str,
        available_filter_values: dict[str, list],
    ) -> ExtractedQuery:
        """Extract ``semantic_query`` and structured filters from one question.

        Args:
            question: Raw user question.
            available_filter_values: Mapping ``{"year": [...], "state": [...],
                "status_bucket": [...], "topics": [...]}`` used to build the
                tool schema enums and to validate returned values.

        Returns:
            ``ExtractedQuery`` with the cleaned semantic query and a dict of
            validated filters (possibly empty). Any failure path returns the
            original question with no filters so the QA pipeline can still
            answer unfiltered.
        """

        normalized_question = question.strip()
        if not normalized_question:
            return ExtractedQuery(semantic_query=question, filters={})

        try:
            tool_schema = self._build_tool_schema(available_filter_values)
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _EXTRACTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": normalized_question},
                ],
                tools=[tool_schema],
                tool_choice={
                    "type": "function",
                    "function": {"name": _TOOL_NAME},
                },
                max_tokens=_EXTRACTOR_MAX_TOKENS,
                temperature=0.0,
            )
        except Exception:
            logger.warning(
                "FilterExtractor: LLM call failed, falling back to unfiltered query",
                exc_info=True,
            )
            return ExtractedQuery(semantic_query=normalized_question, filters={})

        arguments = _extract_tool_arguments(response)
        if arguments is None:
            logger.warning(
                "FilterExtractor: model did not return a %s tool call", _TOOL_NAME
            )
            return ExtractedQuery(semantic_query=normalized_question, filters={})

        semantic_query = _coerce_semantic_query(arguments, fallback=normalized_question)
        filters = _validate_filters(arguments, available_filter_values)
        return ExtractedQuery(semantic_query=semantic_query, filters=filters)

    def _build_tool_schema(
        self, available_filter_values: dict[str, list]
    ) -> dict[str, Any]:
        """Build the ``search_corpus`` tool schema with dynamic enums."""

        states = [
            str(value).strip()
            for value in available_filter_values.get("state", [])
            if isinstance(value, str) and value.strip()
        ]
        status_buckets = [
            str(value).strip()
            for value in available_filter_values.get("status_bucket", [])
            if isinstance(value, str) and value.strip()
        ] or list(STATUS_BUCKETS)
        topics_all = [
            str(value).strip()
            for value in available_filter_values.get("topics", [])
            if isinstance(value, str) and value.strip()
        ]
        topics = topics_all[: self._max_topic_enum_values]

        properties: dict[str, Any] = {
            "semantic_query": {
                "type": "string",
                "description": (
                    "Rewritten topical query for embedding search, with filter "
                    "phrases (state names, years, status words, topic labels) "
                    "removed."
                ),
            },
            "year": {
                "type": "array",
                "items": {
                    "type": "integer",
                    "minimum": 1900,
                    "maximum": 2100,
                },
                "description": (
                    "One or more four-digit calendar years the bill was "
                    "introduced. Multiple years mean OR at retrieval time "
                    "(e.g. 'bills from 2024 or 2025' -> [2024, 2025])."
                ),
            },
            "status_bucket": {
                "type": "array",
                "items": {"type": "string", "enum": status_buckets},
                "description": (
                    "One or more canonical bill status buckets. Use Enacted "
                    "for passed/signed bills, Failed for dead/adjourned, "
                    "Vetoed for vetoed, Pending for still-active, Other "
                    "otherwise. Multiple buckets mean OR at retrieval."
                ),
            },
            "topics": {
                "type": "array",
                "items": {"type": "string", "enum": topics} if topics else {"type": "string"},
                "description": (
                    "Zero or more topic tags. Each tag MUST be an exact match "
                    "from the enum. Matching uses OR semantics at retrieval time."
                ),
            },
        }
        state_items: dict[str, Any] = {"type": "string"}
        if states:
            state_items["enum"] = states
        properties["state"] = {
            "type": "array",
            "items": state_items,
            "description": (
                "One or more two-letter USPS state codes. Multiple states "
                "mean OR at retrieval (e.g. 'California or Texas' -> "
                "[\"CA\", \"TX\"])."
            ),
        }

        return {
            "type": "function",
            "function": {
                "name": _TOOL_NAME,
                "description": (
                    "Search the indexed state AI bills. Extract structured "
                    "filters from the user's question and a cleaned semantic "
                    "query suitable for embedding search."
                ),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": ["semantic_query"],
                    "additionalProperties": False,
                },
            },
        }


def _extract_tool_arguments(response: Any) -> dict[str, Any] | None:
    """Pull the parsed JSON argument dict from the first ``search_corpus`` call."""

    try:
        choices = response.choices
        if not choices:
            return None
        tool_calls = getattr(choices[0].message, "tool_calls", None) or []
    except Exception:
        return None

    for tool_call in tool_calls:
        function = getattr(tool_call, "function", None)
        if function is None:
            continue
        if getattr(function, "name", "") != _TOOL_NAME:
            continue
        raw_arguments = getattr(function, "arguments", "") or ""
        try:
            parsed = json.loads(raw_arguments)
        except (TypeError, ValueError):
            logger.warning(
                "FilterExtractor: could not JSON-parse tool arguments: %r",
                raw_arguments,
            )
            return None
        if isinstance(parsed, dict):
            return parsed
        return None
    return None


def _coerce_semantic_query(arguments: dict[str, Any], *, fallback: str) -> str:
    """Return the ``semantic_query`` field as a stripped string, with fallback."""

    raw = arguments.get("semantic_query")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return fallback


def _validate_filters(
    arguments: dict[str, Any],
    available_filter_values: dict[str, list],
) -> dict[str, Any]:
    """Keep only filter values present in the available-values vocabulary.

    Every scalar field (``year``, ``state``, ``status_bucket``) accepts either
    a scalar or a list from the LLM. The validator drops unknown values and
    collapses single-value lists back to scalars so downstream consumers that
    pre-date the OR feature still see the old shape; 2+ valid values are
    preserved as a list for OR-within-field matching.
    """

    cleaned: dict[str, Any] = {}

    valid_years = {int(value) for value in available_filter_values.get("year", [])}
    years = _extract_int_field(arguments.get("year"), valid_years)
    if years:
        cleaned["year"] = years[0] if len(years) == 1 else years

    valid_states = {
        str(value).strip() for value in available_filter_values.get("state", [])
    }
    states = _extract_str_field(arguments.get("state"), valid_states)
    if states:
        cleaned["state"] = states[0] if len(states) == 1 else states

    valid_statuses = {
        str(value).strip()
        for value in available_filter_values.get("status_bucket", [])
    } or set(STATUS_BUCKETS)
    statuses = _extract_str_field(arguments.get("status_bucket"), valid_statuses)
    if statuses:
        cleaned["status_bucket"] = statuses[0] if len(statuses) == 1 else statuses

    valid_topics = {
        str(value).strip() for value in available_filter_values.get("topics", [])
    }
    topics = _extract_str_field(arguments.get("topics"), valid_topics)
    if topics:
        cleaned["topics"] = topics

    return cleaned


def _extract_int_field(raw: Any, allowed: set[int]) -> list[int]:
    """Normalize a scalar-or-list LLM payload into a unique ordered list of ints.

    Values outside ``allowed`` are dropped when ``allowed`` is non-empty; when
    ``allowed`` is empty (no vocabulary available) everything that parses as an
    int is kept.
    """

    candidates: list[Any]
    if raw in (None, "", 0):
        candidates = []
    elif isinstance(raw, (list, tuple, set, frozenset)):
        candidates = list(raw)
    else:
        candidates = [raw]
    cleaned: list[int] = []
    for candidate in candidates:
        if candidate in (None, "", 0):
            continue
        try:
            value = int(candidate)
        except (TypeError, ValueError):
            continue
        if allowed and value not in allowed:
            continue
        if value not in cleaned:
            cleaned.append(value)
    return cleaned


def _extract_str_field(raw: Any, allowed: set[str]) -> list[str]:
    """Normalize a scalar-or-list LLM payload into a unique ordered list of strs.

    Strips whitespace and drops entries outside ``allowed`` when ``allowed`` is
    non-empty. Duplicates are dropped while preserving first-seen order so the
    LLM's argument order survives (useful for display).
    """

    candidates: list[Any]
    if raw is None:
        candidates = []
    elif isinstance(raw, (list, tuple, set, frozenset)):
        candidates = list(raw)
    elif isinstance(raw, str):
        candidates = [raw]
    else:
        candidates = [raw]
    cleaned: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        value = candidate.strip()
        if not value:
            continue
        if allowed and value not in allowed:
            continue
        if value not in cleaned:
            cleaned.append(value)
    return cleaned


__all__ = ["ExtractedQuery", "FilterExtractor"]
