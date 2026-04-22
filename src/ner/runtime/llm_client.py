"""Shared LLM client for all NER agents.

- Owns OpenAI-compatible connection setup for local vLLM or remote providers
  such as OpenRouter.
- Owns runtime preflight checks and shared structured generation calls.
- Tracks cumulative usage statistics (tokens, cost, latency) across all calls.
- Does not build prompts, validate task-specific schemas, or persist artifacts.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from openai import APIStatusError, OpenAI

logger = logging.getLogger(__name__)

STRUCTURED_OUTPUT_GUIDED_JSON = "guided_json"
STRUCTURED_OUTPUT_JSON_SCHEMA = "json_schema"
_VALID_STRUCTURED_OUTPUT_MODES = {STRUCTURED_OUTPUT_GUIDED_JSON, STRUCTURED_OUTPUT_JSON_SCHEMA}

_HTTP_PAYMENT_REQUIRED = 402


class CreditsExhaustedError(RuntimeError):
    """Raised when the remote provider returns HTTP 402 (payment required)."""


class EmptyCompletionError(RuntimeError):
    """Raised when the LLM returns empty content after all content-level retries.

    Distinct from transport-level errors (handled by the openai SDK's built-in
    retry) and from truncation (``finish_reason='length'``, which is
    deterministic for the same prompt+token budget and is not retried here).
    """


class RefusalError(RuntimeError):
    """Raised when the upstream model explicitly refuses to answer.

    Detected via OpenRouter's ``native_finish_reason == "refusal"`` signal
    (OpenAI's normalized ``finish_reason`` collapses this into ``"stop"``,
    so the native field is the only reliable indicator).  Refusals are
    deterministic for the same prompt+schema at temperature 0 and are not
    retried here; callers should route them to a deterministic fallback or
    drop the work unit.

    Attributes:
        provider (str | None): Upstream provider id reported by OpenRouter
            (e.g. ``"Amazon Bedrock"``, ``"Anthropic"``), when available.
        prompt_tokens (int): Prompt tokens billed for the refused call.
        completion_tokens (int): Completion tokens billed for the refused
            call (typically 1: the stop token).
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


@dataclass
class UsageStats:
    """Thread-safe accumulator for LLM resource consumption across a run.

    All monetary values are in USD as reported by OpenRouter's ``usage.cost``
    field.  Token counts use the provider's native tokenizer.
    """

    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_cached_tokens: int = 0
    total_cost_usd: float = 0.0
    total_elapsed_ms: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        reasoning_tokens: int,
        cached_tokens: int,
        cost_usd: float,
        elapsed_ms: float,
    ) -> None:
        with self._lock:
            self.total_calls += 1
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += total_tokens
            self.total_reasoning_tokens += reasoning_tokens
            self.total_cached_tokens += cached_tokens
            self.total_cost_usd += cost_usd
            self.total_elapsed_ms += elapsed_ms

    def summary_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_calls": self.total_calls,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens,
                "total_reasoning_tokens": self.total_reasoning_tokens,
                "total_cached_tokens": self.total_cached_tokens,
                "total_cost_usd": round(self.total_cost_usd, 8),
                "total_elapsed_ms": round(self.total_elapsed_ms, 0),
            }


@dataclass(slots=True)
class LLMConfig:
    """Connection and generation settings for the shared model client.

    Attributes:
        base_url (str): OpenAI-compatible base URL (local vLLM or remote
            provider such as OpenRouter).
        api_key (str): API key string passed to the OpenAI-compatible client.
        model_name (str): Served model id used for all NER requests.
        temperature (float): Default sampling temperature for requests.
        max_tokens (int): Default completion token limit for requests.
        max_retries (int): Maximum number of client retries per request.
        request_timeout_seconds (float): Request timeout applied to the
            underlying HTTP client.
        structured_output_mode (str): How structured JSON output is enforced.
            ``"guided_json"`` uses vLLM's ``extra_body`` parameter for
            token-level constrained decoding.  ``"json_schema"`` uses the
            OpenAI-standard ``response_format`` parameter supported by
            OpenRouter and other providers.
        skip_model_listing (bool): When ``True``, the preflight check skips
            the ``/v1/models`` listing (useful for remote providers that
            expose thousands of models).
    """

    base_url: str
    api_key: str
    model_name: str
    temperature: float
    max_tokens: int
    max_retries: int = 2
    request_timeout_seconds: float = 60.0
    structured_output_mode: str = STRUCTURED_OUTPUT_GUIDED_JSON
    skip_model_listing: bool = False

    def __post_init__(self) -> None:
        if self.structured_output_mode not in _VALID_STRUCTURED_OUTPUT_MODES:
            raise ValueError(
                f"structured_output_mode must be one of "
                f"{_VALID_STRUCTURED_OUTPUT_MODES}, "
                f"got '{self.structured_output_mode}'"
            )


@dataclass(slots=True)
class LLMPreflightResult:
    """Concrete evidence that the configured model endpoint is reachable and usable.

    Attributes:
        resolved_model_name (str): Model id verified against the server.
        raw_probe_response (str): Raw structured probe response returned during
            preflight.
    """

    resolved_model_name: str
    raw_probe_response: str


class LLMClient:
    """Wrap the OpenAI-compatible client used to talk to a local vLLM server
    or a remote provider such as OpenRouter."""

    def __init__(self, config: LLMConfig) -> None:
        """Initialize the shared LLM client wrapper.

        Args:
            config (LLMConfig): Connection and default generation settings.
        """

        self._config = config
        self._client: OpenAI | None = None
        self._usage_stats = UsageStats()

    @property
    def usage_stats(self) -> UsageStats:
        """Cumulative resource consumption recorded across all ``generate`` calls."""
        return self._usage_stats

    @property
    def config(self) -> LLMConfig:
        """Return the effective shared LLM config.

        Returns:
            LLMConfig: Connection and generation settings for this client
                instance.
        """

        return self._config

    def connect(self) -> None:
        """Instantiate the shared client against the configured endpoint.

        Returns:
            None: This method initializes the underlying OpenAI-compatible
                client in place.
        """

        logger.info(
            "Connecting to %s  model=%s  mode=%s  skip_listing=%s",
            self._config.base_url,
            self._config.model_name,
            self._config.structured_output_mode,
            self._config.skip_model_listing,
        )
        self._client = OpenAI(
            base_url=self._config.base_url,
            api_key=self._config.api_key,
            max_retries=self._config.max_retries,
            timeout=self._config.request_timeout_seconds,
        )

    def close(self) -> None:
        """Dispose of the shared client reference.

        Returns:
            None: This method clears the in-memory client reference.
        """

        self._client = None

    def list_models(self) -> list[str]:
        """List model ids exposed by the configured OpenAI-compatible endpoint.

        Returns:
            list[str]: Model identifiers advertised by the server.

        Raises:
            RuntimeError: If the client has not been connected.
        """

        if self._client is None:
            raise RuntimeError("LLM client is not connected")
        response = self._client.models.list()
        return [model.id for model in response.data]

    def verify_runtime(self) -> LLMPreflightResult:
        """Probe the configured model with a real structured completion request.

        When ``skip_model_listing`` is ``False`` (the default for local vLLM),
        the method first confirms the configured model appears in the
        ``/v1/models`` listing.  Remote providers such as OpenRouter expose
        thousands of models, so setting ``skip_model_listing=True`` bypasses
        that check and relies solely on the structured probe.

        Returns:
            LLMPreflightResult: Preflight evidence confirming the configured
                model is reachable and can produce a structured response.

        Raises:
            RuntimeError: If the configured model is unavailable or the probe
                response does not satisfy the health contract.
        """

        logger.info("Preflight check starting (skip_model_listing=%s)", self._config.skip_model_listing)
        if not self._config.skip_model_listing:
            available_models = self.list_models()
            if self._config.model_name not in available_models:
                raise RuntimeError(
                    f"Configured model '{self._config.model_name}' is not served by "
                    f"{self._config.base_url}. Available models: {available_models}"
                )
            logger.info("Model listing verified: %s found in %d models", self._config.model_name, len(available_models))

        health_schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
            },
            "required": ["status"],
        }
        raw_probe_response = self.generate(
            prompt=(
                "Return a JSON object with exactly one field named 'status' and "
                "set it to the string 'ok'."
            ),
            output_schema=health_schema,
            temperature=0.0,
            max_tokens=32,
        )
        payload = json.loads(raw_probe_response)
        if payload.get("status") != "ok":
            raise RuntimeError(
                "NER LLM preflight structured completion did not return status='ok'"
            )

        logger.info("Preflight OK: model=%s  probe_response=%s", self._config.model_name, raw_probe_response)
        return LLMPreflightResult(
            resolved_model_name=self._config.model_name,
            raw_probe_response=raw_probe_response,
        )

    def generate(
        self,
        prompt: str,
        output_schema: dict[str, Any],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Run one structured prompt against the configured endpoint.

        The structured-output mechanism depends on ``structured_output_mode``:

        * ``"guided_json"`` -- passes the schema via vLLM's ``extra_body``
          parameter for token-level constrained decoding.
        * ``"json_schema"`` -- passes the schema via the OpenAI-standard
          ``response_format`` parameter (supported by OpenRouter and other
          remote providers).

        Args:
            prompt (str): Fully rendered prompt text.
            output_schema (dict[str, Any]): JSON schema used for structured
                decoding.
            temperature (float | None): Optional per-call temperature override.
            max_tokens (int | None): Optional per-call token-limit override.

        Returns:
            str: Raw response text returned by the server.

        Raises:
            CreditsExhaustedError: If the provider returns HTTP 402 (payment
                required / credits exhausted).
            RefusalError: If the upstream model explicitly refuses
                (``native_finish_reason='refusal'``).  Deterministic at
                temperature 0, so not retried here.
            EmptyCompletionError: If the server returns empty content on every
                attempt (up to ``max_retries`` attempts) without a refusal
                flag or truncation.
            RuntimeError: If the client is disconnected or the server returns
                ``finish_reason='length'`` (truncation is deterministic and is
                not retried here).
        """

        if self._client is None:
            raise RuntimeError("LLM client is not connected")

        effective_temperature = (
            self._config.temperature if temperature is None else temperature
        )
        effective_max_tokens = (
            self._config.max_tokens if max_tokens is None else max_tokens
        )

        kwargs: dict[str, Any] = {
            "model": self._config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": effective_temperature,
            "max_tokens": effective_max_tokens,
        }

        if self._config.structured_output_mode == STRUCTURED_OUTPUT_GUIDED_JSON:
            kwargs["extra_body"] = {"guided_json": output_schema}
        else:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "ner_output",
                    "schema": output_schema,
                    "strict": True,
                },
            }

        max_attempts = max(1, self._config.max_retries)
        last_finish_reason: str | None = None
        for attempt in range(1, max_attempts + 1):
            t0 = time.perf_counter()
            try:
                response = self._client.chat.completions.create(**kwargs)
            except APIStatusError as exc:
                if exc.status_code == _HTTP_PAYMENT_REQUIRED:
                    logger.error("Credits exhausted (HTTP 402). Stopping.")
                    raise CreditsExhaustedError("OpenRouter credits exhausted") from exc
                raise
            elapsed_ms = (time.perf_counter() - t0) * 1000

            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0 if usage else 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0 if usage else 0
            total_tokens_val = getattr(usage, "total_tokens", 0) or 0 if usage else 0

            # OpenRouter-specific: cost in USD and token detail breakdowns
            cost_usd = 0.0
            reasoning_tokens = 0
            cached_tokens = 0
            if usage is not None:
                cost_usd = float(getattr(usage, "cost", 0) or 0)

                completion_details = getattr(usage, "completion_tokens_details", None)
                if completion_details is not None:
                    reasoning_tokens = int(getattr(completion_details, "reasoning_tokens", 0) or 0)

                prompt_details = getattr(usage, "prompt_tokens_details", None)
                if prompt_details is not None:
                    cached_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)

            self._usage_stats.record(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens_val,
                reasoning_tokens=reasoning_tokens,
                cached_tokens=cached_tokens,
                cost_usd=cost_usd,
                elapsed_ms=elapsed_ms,
            )

            choice = response.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)

            # OpenRouter preserves the upstream provider's raw finish reason
            # in ``native_finish_reason`` (lives on ``model_extra`` because
            # it is not part of the OpenAI response schema).  OpenAI's
            # normalized ``finish_reason`` collapses "refusal" into "stop",
            # so this native field is the only way to distinguish a genuine
            # stop from an upstream-model refusal.
            choice_extras = getattr(choice, "model_extra", {}) or {}
            native_finish_reason = choice_extras.get("native_finish_reason")

            response_extras = getattr(response, "model_extra", {}) or {}
            provider = response_extras.get("provider")

            logger.info(
                "LLM call  model=%s  provider=%s  mode=%s  attempt=%d/%d  "
                "elapsed=%.0fms  prompt_tokens=%d  completion_tokens=%d  "
                "reasoning_tokens=%d  cached_tokens=%d  cost=$%.6f  "
                "finish_reason=%s  native_finish_reason=%s",
                self._config.model_name,
                provider,
                self._config.structured_output_mode,
                attempt,
                max_attempts,
                elapsed_ms,
                prompt_tokens,
                completion_tokens,
                reasoning_tokens,
                cached_tokens,
                cost_usd,
                finish_reason,
                native_finish_reason,
            )

            # Upstream refusal: explicit safety/content-policy block.
            # Deterministic for the same prompt at temperature 0, so there
            # is no point retrying.  Raise immediately and let the agent
            # decide how to handle it (passthrough vs drop).
            if native_finish_reason == "refusal":
                logger.warning(
                    "Upstream refusal  model=%s  provider=%s  "
                    "prompt_tokens=%d  completion_tokens=%d  "
                    "finish_reason=%s  native_finish_reason=%s",
                    self._config.model_name,
                    provider,
                    prompt_tokens,
                    completion_tokens,
                    finish_reason,
                    native_finish_reason,
                )
                raise RefusalError(
                    f"Upstream model refused the request "
                    f"(provider={provider!r}, finish_reason={finish_reason!r}, "
                    f"native_finish_reason='refusal'). "
                    f"Retrying at temperature 0 is pointless; callers should "
                    f"route to a deterministic fallback.",
                    provider=provider,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

            # Truncation is deterministic for the same prompt + max_tokens.
            # Retrying cannot fix it; raise immediately so the operator can
            # raise max_tokens in config.
            if finish_reason == "length":
                raise RuntimeError(
                    "Structured completion truncated (finish_reason='length'). "
                    f"Used {completion_tokens} completion tokens. Increase max_tokens."
                )

            content = choice.message.content
            if content:
                return content

            last_finish_reason = finish_reason
            suffix = "Retrying." if attempt < max_attempts else "No attempts remaining."
            logger.warning(
                "Empty completion on attempt %d/%d  finish_reason=%r  "
                "native_finish_reason=%r  provider=%s. %s",
                attempt,
                max_attempts,
                finish_reason,
                native_finish_reason,
                provider,
                suffix,
            )

        raise EmptyCompletionError(
            "Structured completion returned empty content after "
            f"{max_attempts} attempts (last finish_reason={last_finish_reason!r})"
        )

