"""Judge LLM client for the nine-stage eval pipeline.

- Wraps :class:`openai.AsyncOpenAI` for the OpenRouter endpoint configured in
  ``settings/config.yml`` and calls ``google/gemini-2.5-pro`` (or whichever
  model the YAML points at) with an OpenAI-standard ``response_format``
  JSON-schema payload so stages can rely on structured output.
- Resolves the API key the same way the NER bootstrap does
  (``OPENROUTER_API_KEY`` env var with keyring fallback), and shares the
  8-key :class:`~src.agent.usage.UsageStats` accumulator so the orchestrator
  can emit a unified ``judge_usage_summary.json`` at the end of the run.
- Retries on transient provider errors (rate limiting, connection resets,
  5xx) with exponential back-off; surfaces HTTP 402 as a hard stop.
- Does not load corpus data, cache verdicts, or run any stage logic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError

from src.agent.usage import UsageStats

from .artifacts import JudgeVerdict, VerdictLabel
from .config import JudgeConfig

logger = logging.getLogger(__name__)

_HTTP_PAYMENT_REQUIRED = 402
_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_RETRYABLE_STATUS: tuple[int, ...] = (408, 409, 429, 500, 502, 503, 504)


class CreditsExhaustedError(RuntimeError):
    """Raised when the provider returns HTTP 402 (credits exhausted)."""


@dataclass(slots=True)
class JudgeConnection:
    """Bundle of state the judge client needs for each call.

    Attributes:
        config: Parsed judge configuration.
        client: Connected :class:`AsyncOpenAI` instance bound to OpenRouter.
        usage_stats: Shared 8-key accumulator updated on every call.
        model: Provider-qualified model id to pass to the chat-completions API.
    """

    config: JudgeConfig
    client: AsyncOpenAI
    usage_stats: UsageStats
    model: str


def build_judge_connection(
    judge_config: JudgeConfig,
    *,
    project_root: Path,
    usage_stats: UsageStats | None = None,
) -> JudgeConnection:
    """Construct the shared :class:`JudgeConnection` for all judge stages.

    Args:
        judge_config: Parsed judge block from the eval config.
        project_root: Project root used to locate ``settings/config.yml`` for
            the base URL and API key env-var fallback.
        usage_stats: Optional pre-built usage accumulator; a new one is
            created when omitted.

    Returns:
        Fully connected :class:`JudgeConnection` ready for
        :func:`call_judge`.

    Raises:
        RuntimeError: If no API key can be resolved.
    """

    provider = judge_config.provider.strip().lower()
    if provider != "openrouter":
        raise ValueError(
            f"Unsupported judge provider: {judge_config.provider!r}; "
            "only 'openrouter' is wired at the moment"
        )

    base_url, api_key_env_var = _read_shared_provider_settings(project_root)
    api_key = _resolve_api_key(api_key_env_var)
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=judge_config.request_timeout_seconds,
    )
    logger.info(
        "Judge connection: model=%s  base_url=%s  timeout=%ss  max_retries=%s",
        judge_config.model,
        base_url,
        judge_config.request_timeout_seconds,
        judge_config.max_retries,
    )
    return JudgeConnection(
        config=judge_config,
        client=client,
        usage_stats=usage_stats or UsageStats(),
        model=judge_config.model,
    )


async def call_judge(
    connection: JudgeConnection,
    *,
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any],
    schema_name: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> JudgeVerdict:
    """Issue one structured judge call and return a parsed verdict.

    The call forces the judge to return a JSON object that conforms to
    ``schema``. The schema's top-level object is expected to include a
    ``verdict`` field whose value is one of :data:`VerdictLabel`; stages
    that need additional fields (``rationale``, ``supporting_ids``) declare
    them in their own schema and the parser reads them opportunistically.

    Args:
        connection: Shared :class:`JudgeConnection` built by
            :func:`build_judge_connection`.
        system_prompt: System role message establishing the judge's task.
        user_prompt: Fully rendered user message describing the specific item.
        schema: JSON-schema dict for ``response_format``.
        schema_name: Identifier passed through as ``json_schema.name``.
        temperature: Optional per-call override of the config default.
        max_tokens: Optional per-call override of the config default.

    Returns:
        Parsed :class:`JudgeVerdict`. On non-retryable parse failure, the
        verdict is returned with ``verdict="error"`` and the raw payload
        preserved on ``raw``.

    Raises:
        CreditsExhaustedError: Propagated on HTTP 402.
        RuntimeError: If every retry attempt fails for a retryable reason.
    """

    payload, response = await _call_judge_raw(
        connection,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=schema,
        schema_name=schema_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if payload is None:
        return JudgeVerdict(
            verdict="error",
            rationale="non-json judge output",
            raw={},
            usage=_usage_dict_from_response(response) if response is not None else {},
        )
    return _verdict_from_payload(payload, response=response)


async def call_judge_json(
    connection: JudgeConnection,
    *,
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any],
    schema_name: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Issue a structured judge call that returns the raw parsed payload.

    Intended for callers whose schemas do not use the ``verdict`` /
    ``supporting_ids`` shape of :class:`JudgeVerdict` (e.g. the Stage 6
    pairwise comparison which returns a ``winner`` field).

    Args:
        connection: Shared :class:`JudgeConnection`.
        system_prompt: System role message.
        user_prompt: User role message.
        schema: JSON-schema dict for ``response_format``.
        schema_name: Identifier passed through as ``json_schema.name``.
        temperature: Optional per-call override of the config default.
        max_tokens: Optional per-call override of the config default.

    Returns:
        ``(payload, usage_dict)`` where ``payload`` is the parsed JSON
        object (``None`` on non-JSON response after retries) and
        ``usage_dict`` is the minimal per-call usage mapping.
    """

    payload, response = await _call_judge_raw(
        connection,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=schema,
        schema_name=schema_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    usage = _usage_dict_from_response(response) if response is not None else {}
    return payload, usage


async def _call_judge_raw(
    connection: JudgeConnection,
    *,
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any],
    schema_name: str,
    temperature: float | None,
    max_tokens: int | None,
) -> tuple[dict[str, Any] | None, Any]:
    """Issue the raw OpenAI-compatible call and return ``(payload, response)``.

    Shared retry and usage-recording code. Both :func:`call_judge` and
    :func:`call_judge_json` wrap this with their own payload parsing.
    """

    kwargs = {
        "model": connection.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": (
            connection.config.temperature if temperature is None else temperature
        ),
        "max_tokens": (
            connection.config.max_tokens if max_tokens is None else max_tokens
        ),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
                "strict": True,
            },
        },
    }

    max_attempts = max(1, connection.config.max_retries)
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        t0 = time.perf_counter()
        try:
            response = await connection.client.chat.completions.create(**kwargs)
        except APIStatusError as exc:
            if exc.status_code == _HTTP_PAYMENT_REQUIRED:
                logger.error("Judge credits exhausted (HTTP 402). Stopping.")
                raise CreditsExhaustedError(
                    "OpenRouter judge credits exhausted"
                ) from exc
            if exc.status_code in _RETRYABLE_STATUS and attempt < max_attempts:
                wait = _retry_wait(attempt)
                logger.warning(
                    "Judge retryable status %d (attempt %d/%d); sleeping %.1fs",
                    exc.status_code, attempt, max_attempts, wait,
                )
                last_error = exc
                await asyncio.sleep(wait)
                continue
            raise
        except (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            asyncio.TimeoutError,
        ) as exc:
            if attempt < max_attempts:
                wait = _retry_wait(attempt)
                logger.warning(
                    "Judge transient error on attempt %d/%d: %s; sleeping %.1fs",
                    attempt, max_attempts, exc, wait,
                )
                last_error = exc
                await asyncio.sleep(wait)
                continue
            raise

        elapsed_ms = (time.perf_counter() - t0) * 1000
        _record_usage(connection.usage_stats, response, elapsed_ms)

        choice = response.choices[0]
        content = choice.message.content or ""
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            if attempt < max_attempts:
                wait = _retry_wait(attempt)
                logger.warning(
                    "Judge returned non-JSON payload (attempt %d/%d); sleeping %.1fs",
                    attempt, max_attempts, wait,
                )
                await asyncio.sleep(wait)
                continue
            logger.error("Judge returned non-JSON after %d attempts", max_attempts)
            return None, response

        return payload, response

    raise RuntimeError(
        f"Judge call failed after {max_attempts} attempts: {last_error!r}"
    )


def close_connection(connection: JudgeConnection) -> None:
    """Close the underlying async HTTP client cleanly.

    The orchestrator calls this at shutdown so ``AsyncOpenAI`` releases its
    connection pool. It is a best-effort helper; any exception is logged
    and swallowed.
    """

    try:
        asyncio.run(connection.client.close())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(connection.client.close())
        finally:
            loop.close()
    except Exception:
        logger.exception("Failed to close judge client cleanly")


def _verdict_from_payload(
    payload: dict[str, Any], *, response: Any
) -> JudgeVerdict:
    """Build a :class:`JudgeVerdict` from a parsed JSON payload."""

    verdict_raw = str(payload.get("verdict") or "").strip().lower()
    verdict: VerdictLabel
    if verdict_raw in {
        "entailed",
        "neutral",
        "contradicted",
        "covered",
        "partially_covered",
        "not_covered",
    }:
        verdict = verdict_raw  # type: ignore[assignment]
    else:
        verdict = "error"

    rationale = str(payload.get("rationale") or "").strip()
    supporting = payload.get("supporting_ids") or payload.get("supporting") or []
    if isinstance(supporting, str):
        supporting = [supporting]
    supporting_ids = [str(x) for x in supporting if x]

    return JudgeVerdict(
        verdict=verdict,
        rationale=rationale,
        supporting_ids=supporting_ids,
        raw=payload,
        usage=_usage_dict_from_response(response),
    )


def _record_usage(usage_stats: UsageStats, response: Any, elapsed_ms: float) -> None:
    """Feed the shared accumulator with this call's 8-key usage tally."""

    usage = getattr(response, "usage", None)
    prompt_tokens = _safe_int(getattr(usage, "prompt_tokens", 0))
    completion_tokens = _safe_int(getattr(usage, "completion_tokens", 0))
    total_tokens = _safe_int(getattr(usage, "total_tokens", 0))

    cost_usd = 0.0
    reasoning_tokens = 0
    cached_tokens = 0
    if usage is not None:
        cost_usd = float(getattr(usage, "cost", 0) or 0)
        completion_details = getattr(usage, "completion_tokens_details", None)
        if completion_details is not None:
            reasoning_tokens = _safe_int(
                getattr(completion_details, "reasoning_tokens", 0)
            )
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        if prompt_details is not None:
            cached_tokens = _safe_int(getattr(prompt_details, "cached_tokens", 0))

    usage_stats.record(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_tokens=cached_tokens,
        cost_usd=cost_usd,
        elapsed_ms=elapsed_ms,
    )


def _usage_dict_from_response(response: Any) -> dict[str, Any]:
    """Return a minimal per-call usage dict we preserve in the cache row."""

    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {
        "prompt_tokens": _safe_int(getattr(usage, "prompt_tokens", 0)),
        "completion_tokens": _safe_int(getattr(usage, "completion_tokens", 0)),
        "total_tokens": _safe_int(getattr(usage, "total_tokens", 0)),
        "cost_usd": float(getattr(usage, "cost", 0) or 0),
    }


def _read_shared_provider_settings(project_root: Path) -> tuple[str, str]:
    """Read ``base_url`` and ``api_key_env_var`` from ``settings/config.yml``."""

    shared_cfg_path = project_root / "settings" / "config.yml"
    if not shared_cfg_path.is_file():
        logger.warning(
            "Shared settings/config.yml not found; defaulting judge base_url to %s",
            _DEFAULT_BASE_URL,
        )
        return _DEFAULT_BASE_URL, "OPENROUTER_API_KEY"

    with shared_cfg_path.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}
    llm_cfg = data.get("llm") or {}
    base_url = str(llm_cfg.get("base_url") or _DEFAULT_BASE_URL)
    env_var = str(llm_cfg.get("api_key_env_var") or "OPENROUTER_API_KEY")
    return base_url, env_var


def _resolve_api_key(env_var: str) -> str:
    """Mirror the NER bootstrap's env-var-then-keyring API key resolver."""

    env_var = env_var.strip()
    if env_var:
        value = os.environ.get(env_var, "").strip()
        if value:
            return value
    try:
        import keyring as _keyring

        secret = _keyring.get_password("ai_policy.qa", "openrouter")
        if secret and secret.strip():
            return secret.strip()
    except Exception:
        logger.debug("Keyring lookup for judge key unavailable", exc_info=True)
    raise RuntimeError(
        "Cannot resolve judge API key. Set $OPENROUTER_API_KEY or store a "
        "keyring entry under service 'ai_policy.qa', username 'openrouter'."
    )


def _retry_wait(attempt: int) -> float:
    """Exponential backoff schedule for retryable provider errors."""

    return min(16.0, 0.5 * (2 ** (attempt - 1)))


def _safe_int(value: Any) -> int:
    """Coerce a provider response attribute to ``int`` with a zero fallback."""

    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
