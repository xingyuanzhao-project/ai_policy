"""Generic multi-turn tool-calling agent loop.

- Runs a conversation with an OpenAI-compatible endpoint using tool calls.
- Dispatches tool calls to a caller-supplied executor and feeds results back.
- Tracks per-turn usage via an external ``UsageStats`` accumulator.
- Does not build prompts, define tools, or persist results.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from openai import APIStatusError, AsyncOpenAI, OpenAI

from .usage import UsageStats


@dataclass
class LoopResult:
    """Return value from the agent loop, carrying the final response and stats."""

    response: str
    turns: int = 0
    tool_calls: int = 0

logger = logging.getLogger(__name__)

_HTTP_PAYMENT_REQUIRED = 402
_MAX_TRACE_RESULT_CHARS = 4000


class CreditsExhaustedError(RuntimeError):
    """Raised when the remote provider returns HTTP 402 (payment required)."""


def run_tool_loop(
    *,
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_executor: Callable[[str, dict[str, Any]], str],
    usage_stats: UsageStats,
    max_turns: int = 50,
    temperature: float = 0.0,
    max_tokens: int = 16384,
    trace_sink: list[dict[str, Any]] | None = None,
) -> str:
    """Run a multi-turn tool-calling conversation until the model stops.

    The loop calls the LLM, checks for tool calls in the response, dispatches
    them via *tool_executor*, appends tool results to the conversation, and
    repeats.  It terminates when the model's ``finish_reason`` is ``"stop"``
    (or equivalent) or when *max_turns* is reached.

    Args:
        client: Connected OpenAI-compatible client instance.
        model: Model identifier passed to the completions endpoint.
        messages: Initial conversation messages (system + user). Modified
            in-place as the loop appends assistant and tool messages.
        tools: List of OpenAI-compatible tool definition dicts.
        tool_executor: Callable(name, arguments_dict) -> result_string.
            Invoked locally for each tool call the model requests.
        usage_stats: Accumulator for per-turn token and cost tracking.
        max_turns: Safety cap on the number of LLM round-trips.
        temperature: Sampling temperature for the completions request.
        max_tokens: Maximum completion tokens per turn.

    Returns:
        The model's final text response (the first turn where it does not
        request any tool calls).

    Raises:
        CreditsExhaustedError: If the provider returns HTTP 402.
        RuntimeError: If max_turns is exceeded without a final response.
    """

    turn = 0
    while turn < max_turns:
        turn += 1
        t0 = time.perf_counter()

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except APIStatusError as exc:
            if exc.status_code == _HTTP_PAYMENT_REQUIRED:
                logger.error("Credits exhausted (HTTP 402). Stopping.")
                raise CreditsExhaustedError(
                    "OpenRouter credits exhausted"
                ) from exc
            raise

        elapsed_ms = (time.perf_counter() - t0) * 1000
        _record_usage(usage_stats, response, elapsed_ms)

        choice = response.choices[0]
        assistant_message = choice.message

        messages.append(_serialize_assistant_message(assistant_message))

        tool_calls = assistant_message.tool_calls
        if not tool_calls:
            if trace_sink is not None:
                trace_sink.append(
                    {
                        "turn": turn,
                        "assistant_content": assistant_message.content or "",
                        "tool_calls": [],
                    }
                )
            logger.info(
                "Agent loop finished: turn=%d  finish_reason=%s",
                turn,
                choice.finish_reason,
            )
            return assistant_message.content or ""

        if trace_sink is not None:
            turn_record: dict[str, Any] = {
                "turn": turn,
                "assistant_content": assistant_message.content or "",
                "tool_calls": [
                    {
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                        "result": None,
                    }
                    for tc in tool_calls
                ],
            }
            trace_sink.append(turn_record)

        # Dispatch each tool call and append results
        for call_index, tc in enumerate(tool_calls):
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            logger.debug(
                "Tool call turn=%d  name=%s  args_keys=%s",
                turn,
                fn_name,
                list(fn_args.keys()),
            )
            try:
                result = tool_executor(fn_name, fn_args)
            except Exception:
                logger.exception("Tool execution failed: %s", fn_name)
                result = json.dumps({"error": f"Tool '{fn_name}' failed"})

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

            if trace_sink is not None:
                trace_sink[-1]["tool_calls"][call_index]["result"] = (
                    result[:_MAX_TRACE_RESULT_CHARS]
                    if isinstance(result, str)
                    else str(result)[:_MAX_TRACE_RESULT_CHARS]
                )

        logger.info(
            "Turn %d: dispatched %d tool call(s)  elapsed=%.0fms",
            turn,
            len(tool_calls),
            elapsed_ms,
        )

    raise RuntimeError(
        f"Agent loop exceeded max_turns={max_turns} without final response"
    )


async def run_tool_loop_async(
    *,
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_executor: Callable[[str, dict[str, Any]], str],
    usage_stats: UsageStats,
    max_turns: int = 50,
    temperature: float = 0.0,
    max_tokens: int = 16384,
    context_label: str = "",
) -> LoopResult:
    """Async variant of :func:`run_tool_loop` for concurrent bill processing.

    Identical logic to the sync version but uses ``AsyncOpenAI`` so multiple
    bill sessions can overlap their HTTP waits via ``asyncio.gather``.
    Tool execution remains synchronous (in-memory operations).

    Args:
        client: Connected async OpenAI-compatible client instance.
        model: Model identifier passed to the completions endpoint.
        messages: Initial conversation messages (system + user). Modified
            in-place as the loop appends assistant and tool messages.
        tools: List of OpenAI-compatible tool definition dicts.
        tool_executor: Callable(name, arguments_dict) -> result_string.
            Invoked locally for each tool call the model requests.
        usage_stats: Accumulator for per-turn token and cost tracking.
        max_turns: Safety cap on the number of LLM round-trips.
        temperature: Sampling temperature for the completions request.
        max_tokens: Maximum completion tokens per turn.
        context_label: Optional label (e.g. bill_id) prepended to log
            lines so interleaved concurrent sessions are distinguishable.

    Returns:
        A :class:`LoopResult` carrying the final text response, total
        turns used, and total tool calls dispatched.

    Raises:
        CreditsExhaustedError: If the provider returns HTTP 402.
        RuntimeError: If max_turns is exceeded without a final response.
    """

    tag = f"[{context_label}] " if context_label else ""

    turn = 0
    total_tool_calls = 0
    while turn < max_turns:
        turn += 1
        t0 = time.perf_counter()

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except APIStatusError as exc:
            if exc.status_code == _HTTP_PAYMENT_REQUIRED:
                logger.error("%sCredits exhausted (HTTP 402). Stopping.", tag)
                raise CreditsExhaustedError(
                    "OpenRouter credits exhausted"
                ) from exc
            raise

        elapsed_ms = (time.perf_counter() - t0) * 1000
        _record_usage(usage_stats, response, elapsed_ms)

        choice = response.choices[0]
        assistant_message = choice.message

        messages.append(_serialize_assistant_message(assistant_message))

        tool_calls = assistant_message.tool_calls
        if not tool_calls:
            content = assistant_message.content or ""
            logger.info(
                "%sAgent loop finished: turn=%d  finish_reason=%s  "
                "response_length=%d  total_tool_calls=%d",
                tag, turn, choice.finish_reason, len(content),
                total_tool_calls,
            )
            return LoopResult(
                response=content, turns=turn, tool_calls=total_tool_calls,
            )

        total_tool_calls += len(tool_calls)
        for tc in tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            try:
                result = tool_executor(fn_name, fn_args)
            except Exception:
                logger.exception("%sTool execution failed: %s", tag, fn_name)
                result = json.dumps({"error": f"Tool '{fn_name}' failed"})

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

            logger.info(
                "%sTurn %d: tool=%s  args=%s  result_length=%d  "
                "elapsed=%.0fms",
                tag, turn, fn_name,
                json.dumps(fn_args, ensure_ascii=False),
                len(result), elapsed_ms,
            )

    raise RuntimeError(
        f"{tag}Agent loop exceeded max_turns={max_turns} without "
        f"final response"
    )


def _serialize_assistant_message(
    message: Any,
) -> dict[str, Any]:
    """Convert the SDK assistant message object to a plain dict for the
    messages list.

    Args:
        message: The ``choices[0].message`` object from the SDK response.

    Returns:
        Dict suitable for appending to the messages list.
    """

    result: dict[str, Any] = {
        "role": "assistant",
        "content": message.content,
    }
    if message.tool_calls:
        result["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]
    return result


def _record_usage(
    stats: UsageStats,
    response: Any,
    elapsed_ms: float,
) -> None:
    """Extract usage fields from the API response and record them.

    Args:
        stats: The usage accumulator to update.
        response: The raw SDK response object.
        elapsed_ms: Wall-clock latency for the request.
    """

    usage = getattr(response, "usage", None)
    if usage is None:
        stats.record(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            elapsed_ms=elapsed_ms,
        )
        return

    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens_val = getattr(usage, "total_tokens", 0) or 0
    cost_usd = float(getattr(usage, "cost", 0) or 0)

    reasoning_tokens = 0
    completion_details = getattr(usage, "completion_tokens_details", None)
    if completion_details is not None:
        reasoning_tokens = int(
            getattr(completion_details, "reasoning_tokens", 0) or 0
        )

    cached_tokens = 0
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    if prompt_details is not None:
        cached_tokens = int(
            getattr(prompt_details, "cached_tokens", 0) or 0
        )

    stats.record(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens_val,
        reasoning_tokens=reasoning_tokens,
        cached_tokens=cached_tokens,
        cost_usd=cost_usd,
        elapsed_ms=elapsed_ms,
    )

    logger.debug(
        "Usage: prompt=%d  completion=%d  reasoning=%d  "
        "cached=%d  cost=$%.6f  elapsed=%.0fms",
        prompt_tokens,
        completion_tokens,
        reasoning_tokens,
        cached_tokens,
        cost_usd,
        elapsed_ms,
    )
