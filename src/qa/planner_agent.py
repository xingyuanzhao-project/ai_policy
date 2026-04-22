"""Orchestrator-workers planner for the QA service.

- Owns the multi-turn tool-calling loop that drives the four planner tools
  (``list_bills``, ``get_bill_content``, ``summarize_bill``, ``compare_bills``).
- Seeds the planner with hints produced by the pre-filter ``FilterExtractor``
  and returns a typed payload containing the final answer text plus the
  citations the planner actually consulted.
- Enforces every per-question cap from :class:`AgentConfig` (planner turns,
  total tool calls, worker-call budget, per-bill citation cap).
- Does not embed queries, build the persisted index, or serve HTTP routes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from openai import OpenAI

from src.agent.loop import CreditsExhaustedError, run_tool_loop
from src.agent.tools import ToolRegistry
from src.agent.usage import UsageStats

from .artifacts import IndexedChunk, RetrievedChunk, STATUS_BUCKETS
from .config import AgentConfig
from .qa_tools import (
    BillSummary,
    CitationAccumulator,
    SearchBackend,
    WorkerCallBudget,
    build_bill_index,
    build_qa_tool_registry,
)
from .quadruplet_store import QuadrupletStore

logger = logging.getLogger(__name__)

_DEFAULT_CITATION_CAP = 10
_TOPIC_HINT_LIMIT = 30


@dataclass(slots=True)
class PlannerAnswer:
    """Final payload returned by :meth:`PlannerAgent.answer`."""

    answer_text: str
    citations: list[RetrievedChunk] = field(default_factory=list)
    planner_turns: int = 0
    tool_calls: int = 0
    worker_calls: int = 0
    routing_path: str = "agent"
    trace: list[dict[str, Any]] = field(default_factory=list)


class PlannerAgent:
    """Run the orchestrator tool-calling loop for one QA question."""

    def __init__(
        self,
        *,
        planner_client: OpenAI,
        worker_client: OpenAI,
        chunks: Sequence[IndexedChunk],
        search_backend: SearchBackend,
        worker_model: str,
        agent_config: AgentConfig,
        bill_index: dict[str, BillSummary] | None = None,
        quadruplet_store: QuadrupletStore | None = None,
    ) -> None:
        """Initialize the planner agent and prebuild the bill-level metadata index.

        Args:
            planner_client: OpenAI-compatible client used for the planner loop.
            worker_client: OpenAI-compatible client used by worker tool calls.
                May be the same instance as ``planner_client``.
            chunks: Full indexed-chunk sequence backing the retriever.
            search_backend: Adapter exposing semantic search against ``chunks``.
            worker_model: Model identifier used by the worker tools.
            agent_config: Resolved agent hyperparameters.
            bill_index: Optional precomputed bill-level metadata map. When
                omitted, one is built from ``chunks`` at construction time.
            quadruplet_store: Optional pre-loaded sidecar backing the
                ``query_quadruplets`` planner tool. When omitted or empty the
                tool is not registered and the planner behaves as if only the
                four legacy tools exist.
        """

        if not worker_model.strip():
            raise ValueError("PlannerAgent.worker_model must be a non-empty string")
        self._planner_client = planner_client
        self._worker_client = worker_client
        self._chunks = chunks
        self._search_backend = search_backend
        self._worker_model = worker_model
        self._agent_config = agent_config
        self._bill_index: dict[str, BillSummary] = (
            dict(bill_index) if bill_index is not None else build_bill_index(chunks)
        )
        self._quadruplet_store = quadruplet_store

    @property
    def bill_index(self) -> dict[str, BillSummary]:
        """Return a read-only view of the preloaded bill-level metadata map."""

        return self._bill_index

    def answer(
        self,
        question: str,
        *,
        semantic_query: str,
        initial_filters: dict[str, Any],
        available_filter_values: dict[str, list],
        planner_model: str,
        citation_cap: int = _DEFAULT_CITATION_CAP,
        capture_trace: bool = False,
        worker_model: str | None = None,
    ) -> PlannerAnswer:
        """Plan and answer one question, returning the final text + citations.

        The ``worker_model`` kwarg, when provided, overrides the agent's
        configured ``worker_model`` for this single question. This lets the
        UI's single model dropdown drive both the planner (via
        ``planner_model``) and the summarize/compare workers with one choice.
        When ``worker_model`` is ``None``, the YAML default captured at
        construction is used.
        """

        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("PlannerAgent.answer question must be non-empty")
        if not planner_model.strip():
            raise ValueError("PlannerAgent.answer planner_model must be non-empty")

        effective_worker_model = (
            worker_model.strip()
            if isinstance(worker_model, str) and worker_model.strip()
            else self._worker_model
        )

        accumulator = CitationAccumulator(
            max_per_bill=self._agent_config.max_citations_per_bill
        )
        worker_budget = WorkerCallBudget(max_calls=self._agent_config.max_worker_calls)
        registry = build_qa_tool_registry(
            chunks=self._chunks,
            bill_index=self._bill_index,
            search_backend=self._search_backend,
            worker_client=self._worker_client,
            worker_model=effective_worker_model,
            agent_config=self._agent_config,
            accumulator=accumulator,
            worker_budget=worker_budget,
            quadruplet_store=self._quadruplet_store,
        )
        counting_executor = _CountingToolExecutor(
            registry=registry,
            max_calls=self._agent_config.max_tool_calls,
        )
        usage_stats = UsageStats()
        messages = self._build_initial_messages(
            question=normalized_question,
            semantic_query=semantic_query,
            initial_filters=initial_filters,
            available_filter_values=available_filter_values,
        )

        trace_sink: list[dict[str, Any]] | None = None
        if capture_trace:
            trace_sink = [
                {
                    "turn": 0,
                    "role": "seed",
                    "messages": [
                        {
                            "role": str(msg.get("role", "")),
                            "content": str(msg.get("content", "")),
                        }
                        for msg in messages
                    ],
                }
            ]

        answer_text = ""
        planner_turns = 0
        fallback_note = ""
        try:
            answer_text = run_tool_loop(
                client=self._planner_client,
                model=planner_model,
                messages=messages,
                tools=registry.definitions(),
                tool_executor=counting_executor,
                usage_stats=usage_stats,
                max_turns=self._agent_config.max_planner_turns,
                temperature=self._agent_config.planner_temperature,
                max_tokens=self._agent_config.max_planner_tokens,
                trace_sink=trace_sink,
            )
            planner_turns = usage_stats.total_calls
        except CreditsExhaustedError:
            logger.error("PlannerAgent aborted: provider credits exhausted")
            fallback_note = (
                "The planner could not complete because the upstream model "
                "provider reported that credits are exhausted."
            )
            planner_turns = usage_stats.total_calls
        except RuntimeError as exc:
            message_text = str(exc)
            if "exceeded max_turns" in message_text:
                logger.warning(
                    "PlannerAgent stopped: exceeded max_turns=%d",
                    self._agent_config.max_planner_turns,
                )
                fallback_note = (
                    "The planner stopped before producing a final answer "
                    f"because it exceeded max_planner_turns="
                    f"{self._agent_config.max_planner_turns}."
                )
                planner_turns = self._agent_config.max_planner_turns
            else:
                raise

        if not answer_text:
            answer_text = fallback_note or (
                "The planner completed without producing a final answer."
            )

        citations = accumulator.export(max_total=citation_cap)
        return PlannerAnswer(
            answer_text=answer_text,
            citations=citations,
            planner_turns=planner_turns,
            tool_calls=counting_executor.call_count,
            worker_calls=worker_budget.used,
            routing_path="agent",
            trace=list(trace_sink) if trace_sink is not None else [],
        )

    def _build_initial_messages(
        self,
        *,
        question: str,
        semantic_query: str,
        initial_filters: dict[str, Any],
        available_filter_values: dict[str, list],
    ) -> list[dict[str, Any]]:
        """Assemble the planner's system + user messages with pre-filter hints."""

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            question=question,
            semantic_query=semantic_query,
            initial_filters=initial_filters,
            available_filter_values=available_filter_values,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _build_system_prompt(self) -> str:
        """Return the planner's static system prompt."""

        has_quadruplets = (
            self._quadruplet_store is not None
            and self._quadruplet_store.total_quadruplets > 0
        )

        tool_lines = [
            "- list_bills(filters?, semantic_query?, limit?): find candidate "
            "bills.",
            "- get_bill_content(bill_id, max_chars?): fetch concatenated "
            "bill text.",
            "- summarize_bill(bill_id, focus?): run a worker LLM to "
            "summarize one bill.",
            "- compare_bills(bill_ids, question): run a worker LLM to "
            "compare 2+ bills.",
        ]
        workflow_lines = [
            "1. Call list_bills with filters (and a semantic_query when "
            "useful) to discover candidate bill_ids.",
            "2. For a direct fact lookup, call get_bill_content on the "
            "relevant bill_id.",
            "3. For high-level summaries, call summarize_bill. For cross-"
            "bill questions (commonalities, differences), call "
            "compare_bills.",
        ]
        if has_quadruplets:
            tool_lines.append(
                "- query_quadruplets(regulated_entity?, entity_type?, "
                "regulatory_mechanism?, provision_contains?, state?, year?, "
                "bill_id?, limit?): search pre-extracted (regulated_entity, "
                "entity_type, regulatory_mechanism, provision_text) tuples."
            )
            workflow_lines.append(
                "4. For questions about WHAT a bill does to something ('which "
                "bills prohibit deepfakes?', 'what are California's "
                "disclosure requirements?', 'list every obligation on "
                "developers'), call query_quadruplets FIRST. Each hit cites "
                "the exact bill_id, and you can then call get_bill_content "
                "on that bill_id to quote the surrounding text."
            )
            workflow_lines.append(
                "5. Stop calling tools once you have enough evidence. Your "
                "final assistant turn (with NO tool calls) must be the full, "
                "self-contained answer."
            )
        else:
            workflow_lines.append(
                "4. Stop calling tools once you have enough evidence. Your "
                "final assistant turn (with NO tool calls) must be the full, "
                "self-contained answer."
            )

        quadruplet_rule = (
            "- When a question targets a specific regulated entity, "
            "regulatory mechanism, or provision keyword, prefer "
            "query_quadruplets over semantic_query on list_bills. Return "
            "the bill_id plus the exact provision_text from each match.\n"
            if has_quadruplets
            else ""
        )

        return (
            "You are a research assistant for United States state AI "
            "legislation. Answer the user's question by calling the tools "
            "below. Never fabricate bill_ids, quotes, or statistics; every "
            "factual claim in your final answer must be grounded in a tool "
            "result.\n"
            "\n"
            "Tools available:\n"
            + "\n".join(tool_lines) + "\n"
            "\n"
            "Typical workflow:\n"
            + "\n".join(workflow_lines) + "\n"
            "\n"
            "Budgets (strictly enforced):\n"
            f"- Max planner turns: {self._agent_config.max_planner_turns}\n"
            f"- Max total tool calls: {self._agent_config.max_tool_calls}\n"
            f"- Max worker LLM calls (summarize/compare combined): "
            f"{self._agent_config.max_worker_calls}\n"
            "\n"
            "Answer-quality rules:\n"
            "- Cite bills inline with their exact bill_id from tool "
            "results (e.g. 'AR HB 1876').\n"
            "- When a question names a count (\"how many\"), report the "
            "exact count returned by list_bills.\n"
            "- When a question says \"list all\", enumerate each matching "
            "bill returned by list_bills.\n"
            + quadruplet_rule
            + "- When the tools return no matching bill, say so explicitly; "
            "do not invent an answer."
        )

    def _build_user_prompt(
        self,
        *,
        question: str,
        semantic_query: str,
        initial_filters: dict[str, Any],
        available_filter_values: dict[str, list],
    ) -> str:
        """Return the planner's user-turn prompt seeded with extractor hints."""

        lines: list[str] = [f"User question: {question}", ""]
        lines.append("Hints from the pre-filter pass (you may override):")
        semantic_hint = semantic_query.strip() if isinstance(semantic_query, str) else ""
        lines.append(f"- Topical rewrite for semantic_query: {semantic_hint or '(none)'}")
        if initial_filters:
            lines.append(f"- Suggested filters: {json.dumps(initial_filters, ensure_ascii=False)}")
        else:
            lines.append("- Suggested filters: (none)")
        lines.append("")
        lines.append("Available filter vocabulary:")
        years = available_filter_values.get("year") or []
        states = available_filter_values.get("state") or []
        status_buckets = available_filter_values.get("status_bucket") or list(STATUS_BUCKETS)
        topics = available_filter_values.get("topics") or []
        lines.append(f"- Years: {', '.join(str(year) for year in sorted(years, reverse=True))}")
        lines.append(f"- States: {', '.join(str(state) for state in sorted(states))}")
        lines.append(f"- Status buckets: {', '.join(str(bucket) for bucket in status_buckets)}")
        topics_preview = list(sorted(topics))[:_TOPIC_HINT_LIMIT]
        topic_preview_text = ", ".join(topics_preview) if topics_preview else "(none)"
        suffix = (
            f" (showing first {_TOPIC_HINT_LIMIT} of {len(topics)})"
            if len(topics) > _TOPIC_HINT_LIMIT
            else ""
        )
        lines.append(f"- Topics: {topic_preview_text}{suffix}")
        lines.append("")
        lines.append(
            "Plan your tool calls, then produce a concise grounded answer. "
            "Your FINAL assistant message must contain the answer itself "
            "with no further tool calls."
        )
        return "\n".join(lines)


class _CountingToolExecutor:
    """Wrap a ``ToolRegistry`` so tool calls are capped per question."""

    __slots__ = ("_registry", "_max_calls", "_count")

    def __init__(self, *, registry: ToolRegistry, max_calls: int) -> None:
        if max_calls <= 0:
            raise ValueError("_CountingToolExecutor.max_calls must be > 0")
        self._registry = registry
        self._max_calls = max_calls
        self._count = 0

    def __call__(self, name: str, arguments: dict[str, Any]) -> str:
        """Dispatch one tool call, refusing once the cap is reached."""

        if self._count >= self._max_calls:
            return json.dumps(
                {
                    "error": (
                        f"Tool-call budget exhausted (max={self._max_calls}). "
                        "Stop calling tools and produce a final answer with "
                        "the evidence already gathered."
                    )
                }
            )
        self._count += 1
        return self._registry.execute(name, arguments)

    @property
    def call_count(self) -> int:
        """Return how many tool calls have been dispatched."""

        return self._count


__all__ = ["PlannerAgent", "PlannerAnswer"]
