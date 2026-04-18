"""Offline tests for the orchestrator-workers planner agent."""

from __future__ import annotations

import json
import unittest
from dataclasses import dataclass, field
from typing import Any

from src.qa.artifacts import IndexedChunk, RetrievedChunk
from src.qa.config import AgentConfig
from src.qa.planner_agent import PlannerAgent, PlannerAnswer


_DEFAULT_PLANNER_MODEL = "google/gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Config & chunk fixtures
# ---------------------------------------------------------------------------


def _make_agent_config(
    *,
    max_planner_turns: int = 3,
    max_tool_calls: int = 6,
    max_worker_calls: int = 2,
) -> AgentConfig:
    """Return an ``AgentConfig`` tuned for fast in-test loops."""

    return AgentConfig(
        max_planner_turns=max_planner_turns,
        max_planner_tokens=512,
        planner_temperature=0.0,
        max_worker_tokens=256,
        worker_temperature=0.0,
        max_tool_calls=max_tool_calls,
        max_worker_calls=max_worker_calls,
        max_bills_per_list=10,
        max_chunks_per_bill=3,
        max_citations_per_bill=2,
    )


def _make_chunks() -> list[IndexedChunk]:
    """Build a small bill corpus usable by the planner tools in tests."""

    return [
        IndexedChunk(
            chunk_id=11,
            bill_id="CA-2024-AI",
            text="California bias-audit requirements.",
            start_offset=0,
            end_offset=40,
            state="CA",
            title="CA AI Act",
            status="Enacted",
            status_bucket="Enacted",
            year=2024,
            topics_list=["Private Sector Use"],
        ),
        IndexedChunk(
            chunk_id=22,
            bill_id="TX-2023-SB",
            text="Texas sandbox definitions.",
            start_offset=0,
            end_offset=28,
            state="TX",
            title="TX Sandbox",
            status="Enacted",
            status_bucket="Enacted",
            year=2023,
            topics_list=["Government Use"],
        ),
    ]


# ---------------------------------------------------------------------------
# Fake OpenAI SDK surface
# ---------------------------------------------------------------------------


@dataclass
class _ToolCallFunction:
    """Stand-in for ``tool_call.function``."""

    name: str
    arguments: str


@dataclass
class _ToolCall:
    """Stand-in for an SDK tool call entry on the assistant message."""

    id: str
    function: _ToolCallFunction
    type: str = "function"


@dataclass
class _AssistantMessage:
    """Stand-in for ``choice.message`` on a chat completion response."""

    content: str | None
    tool_calls: list[_ToolCall] | None = None


@dataclass
class _Choice:
    """Stand-in for one element of ``response.choices``."""

    message: _AssistantMessage
    finish_reason: str = "stop"


@dataclass
class _CompletionResponse:
    """Stand-in for a chat-completion response object."""

    choices: list[_Choice]
    usage: Any = None


class _ScriptedChatCompletions:
    """Return scripted completions in order, recording invocations as we go."""

    def __init__(self, scripted_responses: list[_CompletionResponse]) -> None:
        self._scripted = list(scripted_responses)
        self.calls: list[dict[str, Any]] = []

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> _CompletionResponse:
        """Record the invocation and pop the next scripted response."""

        self.calls.append(
            {
                "model": model,
                "messages": list(messages),
                "tools": list(tools or []),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if not self._scripted:
            raise AssertionError("No scripted chat-completion response remaining")
        return self._scripted.pop(0)


@dataclass
class _FakeOpenAIClient:
    """Minimal OpenAI-shaped client exposing ``chat.completions.create``."""

    chat: Any = field(default=None)

    @classmethod
    def with_scripted(
        cls, scripted_responses: list[_CompletionResponse]
    ) -> "_FakeOpenAIClient":
        """Build a client whose completions replay ``scripted_responses``."""

        completions = _ScriptedChatCompletions(scripted_responses)
        wrapper = type("_ChatWrapper", (), {"completions": completions})
        return cls(chat=wrapper)

    @property
    def completion_calls(self) -> list[dict[str, Any]]:
        """Return the raw recorded invocation list."""

        return self.chat.completions.calls  # type: ignore[no-any-return]


class _ListOnlySearchBackend:
    """Search backend stub that never returns hits (metadata-only listings)."""

    def search(
        self,
        query_text: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        """Return an empty result set regardless of input."""

        return []


# ---------------------------------------------------------------------------
# Helpers to assemble scripted responses
# ---------------------------------------------------------------------------


def _tool_call(
    name: str,
    arguments: dict[str, Any],
    call_id: str = "call-1",
) -> _ToolCall:
    """Assemble one SDK-shaped tool call entry."""

    return _ToolCall(
        id=call_id,
        function=_ToolCallFunction(name=name, arguments=json.dumps(arguments)),
    )


def _response_with_tool_call(tool_call_obj: _ToolCall) -> _CompletionResponse:
    """Return a scripted response that triggers one tool call."""

    return _CompletionResponse(
        choices=[
            _Choice(
                message=_AssistantMessage(content=None, tool_calls=[tool_call_obj]),
                finish_reason="tool_calls",
            )
        ]
    )


def _response_with_final_text(text: str) -> _CompletionResponse:
    """Return a scripted response that finalizes the conversation with ``text``."""

    return _CompletionResponse(
        choices=[_Choice(message=_AssistantMessage(content=text))]
    )


# ---------------------------------------------------------------------------
# PlannerAgent tests
# ---------------------------------------------------------------------------


class PlannerAgentTests(unittest.TestCase):
    """Verify planner orchestration, caps, and output contract."""

    def test_basic_flow_returns_final_text_and_records_citations(self) -> None:
        """Verify a list-then-answer script yields a ``PlannerAnswer`` with citations."""

        scripted = [
            _response_with_tool_call(
                _tool_call("list_bills", {"filters": {"state": "CA"}, "limit": 5})
            ),
            _response_with_tool_call(
                _tool_call(
                    "get_bill_content",
                    {"bill_id": "CA-2024-AI"},
                    call_id="call-2",
                )
            ),
            _response_with_final_text("Final planner answer."),
        ]
        planner_client = _FakeOpenAIClient.with_scripted(scripted)
        worker_client = _FakeOpenAIClient.with_scripted([])
        agent = PlannerAgent(
            planner_client=planner_client,
            worker_client=worker_client,
            chunks=_make_chunks(),
            search_backend=_ListOnlySearchBackend(),
            worker_model=_DEFAULT_PLANNER_MODEL,
            agent_config=_make_agent_config(),
        )

        result = agent.answer(
            "What does CA-2024-AI require?",
            semantic_query="bias audit",
            initial_filters={"state": "CA"},
            available_filter_values={
                "year": [2024, 2023],
                "state": ["CA", "TX"],
                "status_bucket": ["Enacted", "Pending"],
                "topics": ["Private Sector Use", "Government Use"],
            },
            planner_model=_DEFAULT_PLANNER_MODEL,
            citation_cap=5,
        )

        self.assertIsInstance(result, PlannerAnswer)
        self.assertEqual(result.answer_text, "Final planner answer.")
        self.assertEqual(result.routing_path, "agent")
        self.assertEqual(result.tool_calls, 2)
        self.assertEqual(result.worker_calls, 0)
        self.assertEqual(len(planner_client.completion_calls), 3)
        self.assertEqual(
            planner_client.completion_calls[0]["temperature"],
            0.0,
        )
        self.assertGreaterEqual(len(result.citations), 1)
        self.assertEqual(result.citations[0].bill_id, "CA-2024-AI")

    def test_returns_fallback_when_max_planner_turns_exceeded(self) -> None:
        """Verify the agent stops gracefully when the planner loops too long."""

        scripted = [
            _response_with_tool_call(
                _tool_call("list_bills", {"limit": 5}, call_id=f"call-{index}")
            )
            for index in range(5)
        ]
        planner_client = _FakeOpenAIClient.with_scripted(scripted)
        worker_client = _FakeOpenAIClient.with_scripted([])
        agent = PlannerAgent(
            planner_client=planner_client,
            worker_client=worker_client,
            chunks=_make_chunks(),
            search_backend=_ListOnlySearchBackend(),
            worker_model=_DEFAULT_PLANNER_MODEL,
            agent_config=_make_agent_config(max_planner_turns=2),
        )

        result = agent.answer(
            "compare every bill",
            semantic_query="",
            initial_filters={},
            available_filter_values={
                "year": [2024],
                "state": ["CA"],
                "status_bucket": ["Enacted"],
                "topics": ["Private Sector Use"],
            },
            planner_model=_DEFAULT_PLANNER_MODEL,
        )

        self.assertIn("exceeded max_planner_turns", result.answer_text)
        self.assertEqual(result.planner_turns, 2)

    def test_tool_call_budget_is_enforced(self) -> None:
        """Verify the agent refuses tool calls past ``max_tool_calls``."""

        scripted = [
            _response_with_tool_call(
                _tool_call(
                    "list_bills",
                    {"limit": 5},
                    call_id=f"call-{index}",
                )
            )
            for index in range(4)
        ] + [_response_with_final_text("Forced final answer.")]
        planner_client = _FakeOpenAIClient.with_scripted(scripted)
        worker_client = _FakeOpenAIClient.with_scripted([])
        agent = PlannerAgent(
            planner_client=planner_client,
            worker_client=worker_client,
            chunks=_make_chunks(),
            search_backend=_ListOnlySearchBackend(),
            worker_model=_DEFAULT_PLANNER_MODEL,
            agent_config=_make_agent_config(
                max_planner_turns=6,
                max_tool_calls=2,
            ),
        )

        result = agent.answer(
            "list bills",
            semantic_query="",
            initial_filters={},
            available_filter_values={
                "year": [2024],
                "state": ["CA"],
                "status_bucket": ["Enacted"],
                "topics": ["Private Sector Use"],
            },
            planner_model=_DEFAULT_PLANNER_MODEL,
        )

        self.assertEqual(result.answer_text, "Forced final answer.")
        self.assertEqual(result.tool_calls, 2)
        tool_messages = [
            msg
            for call in planner_client.completion_calls
            for msg in call["messages"]
            if msg.get("role") == "tool"
        ]
        self.assertTrue(any("budget exhausted" in msg["content"] for msg in tool_messages))

    def test_user_prompt_carries_extractor_hints(self) -> None:
        """Verify the planner's user prompt includes the semantic query and filters."""

        scripted = [_response_with_final_text("Direct final answer.")]
        planner_client = _FakeOpenAIClient.with_scripted(scripted)
        worker_client = _FakeOpenAIClient.with_scripted([])
        agent = PlannerAgent(
            planner_client=planner_client,
            worker_client=worker_client,
            chunks=_make_chunks(),
            search_backend=_ListOnlySearchBackend(),
            worker_model=_DEFAULT_PLANNER_MODEL,
            agent_config=_make_agent_config(),
        )

        agent.answer(
            "What California AI bills exist?",
            semantic_query="impact disclosure",
            initial_filters={"state": "CA"},
            available_filter_values={
                "year": [2024],
                "state": ["CA"],
                "status_bucket": ["Enacted"],
                "topics": ["Private Sector Use"],
            },
            planner_model=_DEFAULT_PLANNER_MODEL,
        )

        first_call_messages = planner_client.completion_calls[0]["messages"]
        user_message = next(msg for msg in first_call_messages if msg["role"] == "user")
        self.assertIn("impact disclosure", user_message["content"])
        self.assertIn('"state": "CA"', user_message["content"])
        self.assertIn("What California AI bills exist?", user_message["content"])

    def test_planner_tool_schemas_are_exposed(self) -> None:
        """Verify all four planner tools are declared to the planner call."""

        scripted = [_response_with_final_text("Static final answer.")]
        planner_client = _FakeOpenAIClient.with_scripted(scripted)
        worker_client = _FakeOpenAIClient.with_scripted([])
        agent = PlannerAgent(
            planner_client=planner_client,
            worker_client=worker_client,
            chunks=_make_chunks(),
            search_backend=_ListOnlySearchBackend(),
            worker_model=_DEFAULT_PLANNER_MODEL,
            agent_config=_make_agent_config(),
        )

        agent.answer(
            "trivia",
            semantic_query="",
            initial_filters={},
            available_filter_values={
                "year": [],
                "state": [],
                "status_bucket": ["Enacted"],
                "topics": [],
            },
            planner_model=_DEFAULT_PLANNER_MODEL,
        )

        declared_tools = planner_client.completion_calls[0]["tools"]
        tool_names = {tool["function"]["name"] for tool in declared_tools}
        self.assertEqual(
            tool_names,
            {"list_bills", "get_bill_content", "summarize_bill", "compare_bills"},
        )


if __name__ == "__main__":
    unittest.main()
