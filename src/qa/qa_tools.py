"""Planner tools for the orchestrator-workers QA agent.

- Defines the four tools the planner can call (``list_bills``, ``get_bill_content``,
  ``summarize_bill``, ``compare_bills``) wired against either retrieval backend.
- Exposes ``CitationAccumulator`` so the planner can collect the chunks that
  backed its answer and surface them as ``AnswerResult.citations``.
- Enforces a ``WorkerCallBudget`` so ``summarize_bill`` / ``compare_bills`` calls
  are capped per question via ``AgentConfig.max_worker_calls``.
- Does not call the planner LLM, build prompts for the planner, or own the
  multi-turn loop; that is :mod:`src.qa.planner_agent`'s responsibility.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

import numpy as np

from src.agent.tools import ToolRegistry

from .artifacts import IndexedChunk, RetrievedChunk, STATUS_BUCKETS
from .chunk_store import ChunkStore
from .config import AgentConfig
from .lexical_retriever import LexicalRetriever
from .retriever import Retriever, _coerce_int_values, _coerce_str_values

logger = logging.getLogger(__name__)

_DEFAULT_GET_BILL_MAX_CHARS = 8000
_DEFAULT_SUMMARIZE_MAX_CHARS = 8000
_DEFAULT_COMPARE_MAX_CHARS_PER_BILL = 3500


# ---------------------------------------------------------------------------
# Bill-level metadata index
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BillSummary:
    """Metadata for one bill aggregated from all of its indexed chunks."""

    bill_id: str
    state: str
    year: int
    title: str
    status: str
    status_bucket: str
    topics: tuple[str, ...]
    bill_url: str
    chunk_row_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable summary used in tool results."""

        return {
            "bill_id": self.bill_id,
            "state": self.state,
            "year": self.year,
            "title": self.title,
            "status": self.status,
            "status_bucket": self.status_bucket,
            "topics": list(self.topics),
            "bill_url": self.bill_url,
        }


def build_bill_index(chunks: Sequence[IndexedChunk]) -> dict[str, BillSummary]:
    """Build an in-memory ``bill_id -> BillSummary`` map from the chunk sequence.

    For a ``ChunkStore`` (the production vector path) this streams the backing
    JSONL file line-by-line so only metadata fields are loaded - chunk text is
    not read. For an in-memory list the iteration is a direct pass.
    """

    accumulator: dict[str, dict[str, Any]] = {}
    if isinstance(chunks, ChunkStore):
        jsonl_path = Path(chunks.chunks_jsonl_path)
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for row_index, raw_line in enumerate(handle):
                if not raw_line.strip():
                    continue
                payload = json.loads(raw_line)
                if not isinstance(payload, dict):
                    raise ValueError(
                        "Persisted QA chunk line must decode to one object"
                    )
                _accumulate_bill_payload(accumulator, payload, row_index)
    else:
        for row_index, chunk in enumerate(chunks):
            _accumulate_bill_payload(accumulator, chunk.to_dict(), row_index)

    return {
        bill_id: BillSummary(
            bill_id=bill_id,
            state=str(payload.get("state", "")),
            year=int(payload.get("year", 0) or 0),
            title=str(payload.get("title", "")),
            status=str(payload.get("status", "")),
            status_bucket=str(payload.get("status_bucket", "Other") or "Other"),
            topics=tuple(sorted({str(topic) for topic in payload.get("topics", set())})),
            bill_url=str(payload.get("bill_url", "")),
            chunk_row_indices=tuple(sorted(payload["chunk_row_indices"])),
        )
        for bill_id, payload in accumulator.items()
    }


def _accumulate_bill_payload(
    accumulator: dict[str, dict[str, Any]],
    payload: dict[str, Any],
    row_index: int,
) -> None:
    """Merge one chunk-level payload into the per-bill accumulator."""

    bill_id = str(payload.get("bill_id", "")).strip()
    if not bill_id:
        return
    bucket = accumulator.setdefault(
        bill_id,
        {
            "state": str(payload.get("state", "")),
            "year": int(payload.get("year", 0) or 0),
            "title": str(payload.get("title", "")),
            "status": str(payload.get("status", "")),
            "status_bucket": str(payload.get("status_bucket", "Other") or "Other"),
            "bill_url": str(payload.get("bill_url", "")),
            "topics": set(),
            "chunk_row_indices": [],
        },
    )
    raw_topics = payload.get("topics_list") or payload.get("topics") or []
    if isinstance(raw_topics, list):
        for topic in raw_topics:
            if isinstance(topic, str) and topic.strip():
                bucket["topics"].add(topic.strip())
    bucket["chunk_row_indices"].append(row_index)


# ---------------------------------------------------------------------------
# Citation accumulator and worker-call budget
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CitationAccumulator:
    """Ordered, dedup'd set of chunks the planner used, with a per-bill cap."""

    max_per_bill: int
    _ordered: list[RetrievedChunk] = field(default_factory=list)
    _seen_chunk_ids: set[int] = field(default_factory=set)
    _per_bill_counts: dict[str, int] = field(default_factory=dict)

    def add(self, chunk: RetrievedChunk) -> None:
        """Add one chunk if it is new and the per-bill cap has room."""

        if chunk.chunk_id in self._seen_chunk_ids:
            return
        bill_id = str(chunk.bill_id)
        if self._per_bill_counts.get(bill_id, 0) >= self.max_per_bill:
            return
        self._ordered.append(chunk)
        self._seen_chunk_ids.add(chunk.chunk_id)
        self._per_bill_counts[bill_id] = self._per_bill_counts.get(bill_id, 0) + 1

    def extend(self, chunks: Iterable[RetrievedChunk]) -> None:
        """Add multiple chunks in order, skipping duplicates and over-cap bills."""

        for chunk in chunks:
            self.add(chunk)

    def export(self, max_total: int | None = None) -> list[RetrievedChunk]:
        """Return a ranked, truncated citation list with renumbered ranks."""

        if max_total is not None and max_total > 0:
            slice_source = self._ordered[:max_total]
        else:
            slice_source = list(self._ordered)
        reranked: list[RetrievedChunk] = []
        for new_rank, chunk in enumerate(slice_source, start=1):
            reranked.append(
                RetrievedChunk(
                    rank=new_rank,
                    score=chunk.score,
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
        return reranked


class WorkerBudgetExceededError(RuntimeError):
    """Raised when the planner exhausts its ``max_worker_calls`` budget."""


@dataclass(slots=True)
class WorkerCallBudget:
    """Cumulative cap on ``summarize_bill`` + ``compare_bills`` worker calls."""

    max_calls: int
    _count: int = 0

    def consume(self) -> None:
        """Record one worker call; raise when the cap would be exceeded."""

        if self._count >= self.max_calls:
            raise WorkerBudgetExceededError(
                f"Worker call budget exhausted (max={self.max_calls})"
            )
        self._count += 1

    @property
    def used(self) -> int:
        """Return how many worker calls have been consumed so far."""

        return self._count


# ---------------------------------------------------------------------------
# Search backend adapters (bridge vector vs lexical retrievers)
# ---------------------------------------------------------------------------


class SearchBackend(Protocol):
    """Unified retrieval surface used by ``list_bills`` tool."""

    def search(
        self,
        query_text: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[RetrievedChunk]: ...


@dataclass(slots=True)
class VectorSearchBackend:
    """Adapter that turns a text query into an embedding-based retrieval."""

    retriever: Retriever
    embed_query: Callable[[str], np.ndarray]

    def search(
        self,
        query_text: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        """Run vector retrieval for one free-text query."""

        if not query_text.strip():
            return []
        query_embedding = self.embed_query(query_text)
        return self.retriever.retrieve(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )


@dataclass(slots=True)
class LexicalSearchBackend:
    """Adapter that forwards a text query to the BM25 lexical retriever."""

    retriever: LexicalRetriever

    def search(
        self,
        query_text: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        """Run lexical retrieval for one free-text query."""

        if not query_text.strip():
            return []
        return self.retriever.retrieve_question(
            question=query_text,
            top_k=top_k,
            filters=filters,
        )


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------


_FILTERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "Optional structured filters. Omit fields you do not want to "
        "constrain. Every field is list-valued: multiple values within a "
        "single field mean OR at retrieval (e.g. state=[\"CA\",\"TX\"] "
        "matches either). Multiple fields combine with AND. A single-element "
        "list is equivalent to a scalar and either shape is accepted."
    ),
    "properties": {
        "year": {
            "anyOf": [
                {
                    "type": "integer",
                    "minimum": 1900,
                    "maximum": 2100,
                },
                {
                    "type": "array",
                    "items": {
                        "type": "integer",
                        "minimum": 1900,
                        "maximum": 2100,
                    },
                },
            ],
            "description": (
                "Four-digit calendar year(s) the bill was introduced. "
                "Pass a list for OR (e.g. [2024, 2025])."
            ),
        },
        "state": {
            "anyOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}},
            ],
            "description": (
                "Two-letter USPS state code(s). Pass a list for OR "
                "(e.g. [\"CA\", \"TX\"])."
            ),
        },
        "status_bucket": {
            "anyOf": [
                {"type": "string", "enum": list(STATUS_BUCKETS)},
                {
                    "type": "array",
                    "items": {"type": "string", "enum": list(STATUS_BUCKETS)},
                },
            ],
            "description": (
                "Canonical bill status bucket(s): Enacted, Failed, Vetoed, "
                "Pending, or Other. Pass a list for OR "
                "(e.g. [\"Enacted\",\"Pending\"])."
            ),
        },
        "topics": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Zero or more exact-match topic tags (OR-within-field)."
            ),
        },
    },
    "additionalProperties": False,
}

_LIST_BILLS_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "list_bills",
        "description": (
            "List state AI bills that match optional filters. When "
            "`semantic_query` is provided, bills are ranked by semantic "
            "similarity to the query; otherwise bills are listed by "
            "state+year. Use this first to discover candidate bills before "
            "calling get_bill_content, summarize_bill, or compare_bills."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "semantic_query": {
                    "type": "string",
                    "description": (
                        "Optional topical query for semantic ranking. "
                        "Leave empty for metadata-only listing."
                    ),
                },
                "filters": _FILTERS_SCHEMA,
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Maximum number of bills to return. Capped by the "
                        "system configuration."
                    ),
                },
            },
            "additionalProperties": False,
        },
    },
}

_GET_BILL_CONTENT_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_bill_content",
        "description": (
            "Return the concatenated text of one bill (up to the configured "
            "chunk limit). Use this when you need the bill's exact wording "
            "to answer a question directly or to quote provisions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bill_id": {
                    "type": "string",
                    "description": "Exact bill_id as returned by list_bills.",
                },
                "max_chars": {
                    "type": "integer",
                    "minimum": 200,
                    "description": (
                        "Optional character cap on the returned text. "
                        "Default 8000."
                    ),
                },
            },
            "required": ["bill_id"],
            "additionalProperties": False,
        },
    },
}

_SUMMARIZE_BILL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "summarize_bill",
        "description": (
            "Run a worker LLM to produce a concise structured summary of one "
            "bill. Prefer this over get_bill_content when you only need the "
            "high-level shape of a bill (purpose, scope, obligations). "
            "Counts against the worker-call budget."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bill_id": {
                    "type": "string",
                    "description": "Exact bill_id as returned by list_bills.",
                },
                "focus": {
                    "type": "string",
                    "description": (
                        "Optional short phrase directing the worker to "
                        "emphasize a dimension (e.g. 'exemptions', "
                        "'effective date', 'penalties')."
                    ),
                },
            },
            "required": ["bill_id"],
            "additionalProperties": False,
        },
    },
}

_COMPARE_BILLS_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "compare_bills",
        "description": (
            "Run a worker LLM to compare 2+ bills against each other in the "
            "context of the user's question. Returns a structured comparison "
            "covering commonalities and differences. Counts against the "
            "worker-call budget."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bill_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "description": (
                        "Two or more exact bill_ids to compare, as returned "
                        "by list_bills."
                    ),
                },
                "question": {
                    "type": "string",
                    "description": (
                        "The user-level question the comparison should "
                        "address."
                    ),
                },
            },
            "required": ["bill_ids", "question"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Internal filter utilities
# ---------------------------------------------------------------------------


def _coerce_filters(raw_filters: Any) -> dict[str, Any]:
    """Coerce a raw filters payload from the LLM into a clean dict.

    Accepts each scalar field (``year``, ``state``, ``status_bucket``) either
    as a scalar or as a list and preserves the chosen shape so the planner's
    downstream echo of ``applied_filters`` still reflects what it requested.
    Single-element lists are collapsed to scalars, multi-value lists stay
    lists (OR-within-field at retrieval time).
    """

    if not isinstance(raw_filters, dict):
        return {}
    cleaned: dict[str, Any] = {}

    years = _coerce_int_values(raw_filters.get("year"))
    if years:
        cleaned["year"] = years[0] if len(years) == 1 else years

    states = _coerce_str_values(raw_filters.get("state"))
    if states:
        cleaned["state"] = states[0] if len(states) == 1 else states

    buckets = _coerce_str_values(raw_filters.get("status_bucket"))
    if buckets:
        cleaned["status_bucket"] = buckets[0] if len(buckets) == 1 else buckets

    topics = _coerce_str_values(raw_filters.get("topics"))
    if topics:
        cleaned["topics"] = topics

    return cleaned


def _summary_matches_filters(summary: BillSummary, filters: dict[str, Any]) -> bool:
    """Return True when a bill summary satisfies every active filter field.

    Each field accepts scalar or list; list means OR-within-field.
    """

    years = _coerce_int_values(filters.get("year"))
    if years and int(summary.year) not in years:
        return False
    states = _coerce_str_values(filters.get("state"))
    if states and str(summary.state) not in states:
        return False
    buckets = _coerce_str_values(filters.get("status_bucket"))
    if buckets and str(summary.status_bucket) not in buckets:
        return False
    topics = _coerce_str_values(filters.get("topics"))
    if topics and not (set(topics) & set(summary.topics)):
        return False
    return True


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def _list_bills_handler(
    arguments: dict[str, Any],
    *,
    bill_index: dict[str, BillSummary],
    search_backend: SearchBackend,
    agent_config: AgentConfig,
    accumulator: CitationAccumulator,
) -> dict[str, Any]:
    """Return a ranked or metadata-only list of bills matching filters."""

    raw_limit = arguments.get("limit")
    try:
        requested_limit = int(raw_limit) if raw_limit is not None else agent_config.max_bills_per_list
    except (TypeError, ValueError):
        requested_limit = agent_config.max_bills_per_list
    limit = max(1, min(int(requested_limit), agent_config.max_bills_per_list))

    filters = _coerce_filters(arguments.get("filters"))
    semantic_query = arguments.get("semantic_query")
    semantic_query = semantic_query.strip() if isinstance(semantic_query, str) else ""

    ordered_bill_ids: list[str] = []
    if semantic_query:
        retrieved = search_backend.search(
            query_text=semantic_query,
            top_k=max(limit * 3, limit),
            filters=filters or None,
        )
        accumulator.extend(retrieved)
        for chunk in retrieved:
            bill_id = str(chunk.bill_id)
            if bill_id in bill_index and bill_id not in ordered_bill_ids:
                ordered_bill_ids.append(bill_id)
            if len(ordered_bill_ids) >= limit:
                break
    else:
        filtered_summaries = [
            summary
            for summary in bill_index.values()
            if not filters or _summary_matches_filters(summary, filters)
        ]
        filtered_summaries.sort(
            key=lambda item: (item.state, -int(item.year), item.bill_id)
        )
        for summary in filtered_summaries[:limit]:
            ordered_bill_ids.append(summary.bill_id)

    bills_payload = [
        bill_index[bill_id].to_dict() for bill_id in ordered_bill_ids if bill_id in bill_index
    ]
    return {
        "bills": bills_payload,
        "bill_count": len(bills_payload),
        "applied_filters": filters,
        "semantic_query": semantic_query,
        "limit": limit,
    }


def _get_bill_content_handler(
    arguments: dict[str, Any],
    *,
    chunks: Sequence[IndexedChunk],
    bill_index: dict[str, BillSummary],
    agent_config: AgentConfig,
    accumulator: CitationAccumulator,
) -> dict[str, Any]:
    """Return the concatenated text + metadata for one bill."""

    bill_id = arguments.get("bill_id")
    if not isinstance(bill_id, str) or not bill_id.strip():
        return {"error": "bill_id is required"}
    bill_id = bill_id.strip()
    summary = bill_index.get(bill_id)
    if summary is None:
        return {
            "error": (
                f"Unknown bill_id '{bill_id}'. Call list_bills first to get "
                "valid bill_ids."
            )
        }

    raw_max_chars = arguments.get("max_chars")
    try:
        max_chars = (
            int(raw_max_chars)
            if raw_max_chars is not None
            else _DEFAULT_GET_BILL_MAX_CHARS
        )
    except (TypeError, ValueError):
        max_chars = _DEFAULT_GET_BILL_MAX_CHARS
    max_chars = max(200, max_chars)

    row_indices = list(summary.chunk_row_indices[: agent_config.max_chunks_per_bill])
    fetched_chunks: list[IndexedChunk] = [chunks[row_index] for row_index in row_indices]
    fetched_chunks.sort(key=lambda chunk: chunk.start_offset)

    combined_text_parts: list[str] = []
    total_chars = 0
    kept_chunks: list[IndexedChunk] = []
    for chunk in fetched_chunks:
        remaining = max_chars - total_chars
        if remaining <= 0:
            break
        chunk_text = chunk.text if len(chunk.text) <= remaining else chunk.text[:remaining]
        combined_text_parts.append(chunk_text)
        kept_chunks.append(chunk)
        total_chars += len(chunk_text)

    retrieved_for_citation = [
        _to_retrieved_chunk(chunk, rank=rank_index + 1)
        for rank_index, chunk in enumerate(kept_chunks)
    ]
    accumulator.extend(retrieved_for_citation)

    return {
        "bill": summary.to_dict(),
        "text": "\n".join(combined_text_parts),
        "text_truncated": total_chars >= max_chars and len(fetched_chunks) > len(kept_chunks),
        "chunks_returned": len(kept_chunks),
        "total_chunks_in_bill": len(summary.chunk_row_indices),
    }


def _to_retrieved_chunk(chunk: IndexedChunk, rank: int) -> RetrievedChunk:
    """Wrap an ``IndexedChunk`` in a ``RetrievedChunk`` for citation use."""

    return RetrievedChunk(
        rank=rank,
        score=0.0,
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


def _summarize_bill_handler(
    arguments: dict[str, Any],
    *,
    chunks: Sequence[IndexedChunk],
    bill_index: dict[str, BillSummary],
    worker_client: Any,
    worker_model: str,
    agent_config: AgentConfig,
    accumulator: CitationAccumulator,
    worker_budget: WorkerCallBudget,
) -> dict[str, Any]:
    """Run one worker LLM call to produce a concise bill summary."""

    bill_id = arguments.get("bill_id")
    if not isinstance(bill_id, str) or not bill_id.strip():
        return {"error": "bill_id is required"}
    bill_id = bill_id.strip()
    summary = bill_index.get(bill_id)
    if summary is None:
        return {
            "error": (
                f"Unknown bill_id '{bill_id}'. Call list_bills first to get "
                "valid bill_ids."
            )
        }

    content_payload = _get_bill_content_handler(
        {"bill_id": bill_id, "max_chars": _DEFAULT_SUMMARIZE_MAX_CHARS},
        chunks=chunks,
        bill_index=bill_index,
        agent_config=agent_config,
        accumulator=accumulator,
    )
    if "error" in content_payload:
        return content_payload

    focus = arguments.get("focus")
    focus_value = focus.strip() if isinstance(focus, str) and focus.strip() else ""

    try:
        worker_budget.consume()
    except WorkerBudgetExceededError as exc:
        return {"error": str(exc), "bill_id": bill_id}

    system_prompt = (
        "You are summarizing a United States state AI bill for a policy "
        "researcher. Use only the bill text provided. Do not fabricate "
        "content. Return 6-10 concise bullet points covering: purpose, "
        "scope, covered entities, key obligations, effective date, and "
        "enforcement/penalties. If a focus phrase is provided, emphasize "
        "that dimension. Keep the summary under 250 words."
    )
    user_prompt_lines = [
        f"Bill ID: {summary.bill_id}",
        f"State: {summary.state}",
        f"Year: {summary.year}",
        f"Title: {summary.title}",
        f"Status: {summary.status}",
    ]
    if focus_value:
        user_prompt_lines.append(f"Focus: {focus_value}")
    user_prompt_lines.extend(["", "Full bill text (may be truncated):", content_payload["text"]])

    try:
        response = worker_client.chat.completions.create(
            model=worker_model,
            max_tokens=agent_config.max_worker_tokens,
            temperature=agent_config.worker_temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "\n".join(user_prompt_lines)},
            ],
        )
    except Exception as exc:
        logger.warning("summarize_bill worker call failed: %s", exc)
        return {"error": f"summarize_bill worker call failed: {exc}", "bill_id": bill_id}

    text = _extract_completion_text(response)
    return {
        "bill_id": bill_id,
        "bill": summary.to_dict(),
        "summary": text,
        "worker_model": worker_model,
        "worker_calls_used": worker_budget.used,
        "focus": focus_value,
    }


def _compare_bills_handler(
    arguments: dict[str, Any],
    *,
    chunks: Sequence[IndexedChunk],
    bill_index: dict[str, BillSummary],
    worker_client: Any,
    worker_model: str,
    agent_config: AgentConfig,
    accumulator: CitationAccumulator,
    worker_budget: WorkerCallBudget,
) -> dict[str, Any]:
    """Run one worker LLM call to compare 2+ bills against each other."""

    raw_bill_ids = arguments.get("bill_ids")
    if not isinstance(raw_bill_ids, list) or len(raw_bill_ids) < 2:
        return {"error": "bill_ids must be a list of at least 2 bill_ids"}
    bill_ids = [
        str(raw_id).strip() for raw_id in raw_bill_ids if isinstance(raw_id, str) and raw_id.strip()
    ]
    if len(bill_ids) < 2:
        return {"error": "bill_ids must contain at least 2 non-empty strings"}

    question = arguments.get("question")
    if not isinstance(question, str) or not question.strip():
        return {"error": "question is required"}
    question_value = question.strip()

    bill_contents: list[dict[str, Any]] = []
    for bill_id in bill_ids:
        summary = bill_index.get(bill_id)
        if summary is None:
            return {
                "error": (
                    f"Unknown bill_id '{bill_id}'. Call list_bills first to "
                    "get valid bill_ids."
                ),
                "bill_ids": bill_ids,
            }
        content_payload = _get_bill_content_handler(
            {"bill_id": bill_id, "max_chars": _DEFAULT_COMPARE_MAX_CHARS_PER_BILL},
            chunks=chunks,
            bill_index=bill_index,
            agent_config=agent_config,
            accumulator=accumulator,
        )
        if "error" in content_payload:
            return content_payload
        bill_contents.append(
            {
                "bill": summary.to_dict(),
                "text": content_payload["text"],
            }
        )

    try:
        worker_budget.consume()
    except WorkerBudgetExceededError as exc:
        return {"error": str(exc), "bill_ids": bill_ids}

    system_prompt = (
        "You are a policy analyst comparing United States state AI bills. "
        "Use only the provided bill texts; do not fabricate. Organize the "
        "comparison as: 1) What they have in common, 2) Key differences "
        "(by bill), 3) How this answers the original question. Cite bills "
        "by their bill_id."
    )
    user_prompt_lines = [f"Original question: {question_value}", ""]
    for index, entry in enumerate(bill_contents, start=1):
        bill = entry["bill"]
        user_prompt_lines.append(
            f"Bill {index}: {bill['bill_id']} ({bill['state']}, {bill['year']}, "
            f"status={bill['status']})"
        )
        user_prompt_lines.append(f"Title: {bill['title']}")
        user_prompt_lines.append("Text (may be truncated):")
        user_prompt_lines.append(entry["text"])
        user_prompt_lines.append("")

    try:
        response = worker_client.chat.completions.create(
            model=worker_model,
            max_tokens=agent_config.max_worker_tokens,
            temperature=agent_config.worker_temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "\n".join(user_prompt_lines)},
            ],
        )
    except Exception as exc:
        logger.warning("compare_bills worker call failed: %s", exc)
        return {"error": f"compare_bills worker call failed: {exc}", "bill_ids": bill_ids}

    text = _extract_completion_text(response)
    return {
        "bill_ids": bill_ids,
        "question": question_value,
        "comparison": text,
        "worker_model": worker_model,
        "worker_calls_used": worker_budget.used,
    }


def _extract_completion_text(response: Any) -> str:
    """Pull the assistant text out of a chat completion response safely."""

    try:
        choices = response.choices
        if not choices:
            return ""
        content = getattr(choices[0].message, "content", None)
    except Exception:
        return ""
    if not isinstance(content, str):
        return ""
    return content.strip()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_qa_tool_registry(
    *,
    chunks: Sequence[IndexedChunk],
    bill_index: dict[str, BillSummary],
    search_backend: SearchBackend,
    worker_client: Any,
    worker_model: str,
    agent_config: AgentConfig,
    accumulator: CitationAccumulator,
    worker_budget: WorkerCallBudget,
) -> ToolRegistry:
    """Build a ``ToolRegistry`` wired with the four QA planner tools.

    Each invocation of ``PlannerAgent.answer(...)`` should build its own
    fresh ``accumulator`` and ``worker_budget`` and pass them here so
    citations and worker-call counts do not leak across questions.
    """

    registry = ToolRegistry()

    registry.register(
        name="list_bills",
        handler=lambda arguments: _list_bills_handler(
            arguments,
            bill_index=bill_index,
            search_backend=search_backend,
            agent_config=agent_config,
            accumulator=accumulator,
        ),
        schema=_LIST_BILLS_SCHEMA,
    )
    registry.register(
        name="get_bill_content",
        handler=lambda arguments: _get_bill_content_handler(
            arguments,
            chunks=chunks,
            bill_index=bill_index,
            agent_config=agent_config,
            accumulator=accumulator,
        ),
        schema=_GET_BILL_CONTENT_SCHEMA,
    )
    registry.register(
        name="summarize_bill",
        handler=lambda arguments: _summarize_bill_handler(
            arguments,
            chunks=chunks,
            bill_index=bill_index,
            worker_client=worker_client,
            worker_model=worker_model,
            agent_config=agent_config,
            accumulator=accumulator,
            worker_budget=worker_budget,
        ),
        schema=_SUMMARIZE_BILL_SCHEMA,
    )
    registry.register(
        name="compare_bills",
        handler=lambda arguments: _compare_bills_handler(
            arguments,
            chunks=chunks,
            bill_index=bill_index,
            worker_client=worker_client,
            worker_model=worker_model,
            agent_config=agent_config,
            accumulator=accumulator,
            worker_budget=worker_budget,
        ),
        schema=_COMPARE_BILLS_SCHEMA,
    )
    return registry


__all__ = [
    "BillSummary",
    "CitationAccumulator",
    "LexicalSearchBackend",
    "SearchBackend",
    "VectorSearchBackend",
    "WorkerBudgetExceededError",
    "WorkerCallBudget",
    "build_bill_index",
    "build_qa_tool_registry",
]
