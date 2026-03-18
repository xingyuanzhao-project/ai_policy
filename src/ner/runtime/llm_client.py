"""Shared local-vLLM client for all NER agents.

- Owns OpenAI-compatible connection setup for the local vLLM server.
- Owns runtime preflight checks and shared structured generation calls.
- Does not build prompts, validate task-specific schemas, or persist artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass(slots=True)
class LLMConfig:
    """Connection and generation settings for the shared local model client.

    Attributes:
        base_url (str): OpenAI-compatible base URL exposed by the local vLLM
            server.
        api_key (str): API key string passed to the OpenAI-compatible client.
        model_name (str): Served model id used for all NER requests.
        temperature (float): Default sampling temperature for requests.
        max_tokens (int): Default completion token limit for requests.
        max_retries (int): Maximum number of client retries per request.
        request_timeout_seconds (float): Request timeout applied to the
            underlying HTTP client.
    """

    base_url: str
    api_key: str
    model_name: str
    temperature: float
    max_tokens: int
    max_retries: int = 2
    request_timeout_seconds: float = 60.0


@dataclass(slots=True)
class LLMPreflightResult:
    """Concrete evidence that the configured local model is reachable and usable.

    Attributes:
        resolved_model_name (str): Model id verified against the server.
        raw_probe_response (str): Raw structured probe response returned during
            preflight.
    """

    resolved_model_name: str
    raw_probe_response: str


class LLMClient:
    """Wrap the OpenAI-compatible client used to talk to the local vLLM server."""

    def __init__(self, config: LLMConfig) -> None:
        """Initialize the shared LLM client wrapper.

        Args:
            config (LLMConfig): Connection and default generation settings.
        """

        self._config = config
        self._client: OpenAI | None = None

    @property
    def config(self) -> LLMConfig:
        """Return the effective shared LLM config.

        Returns:
            LLMConfig: Connection and generation settings for this client
                instance.
        """

        return self._config

    def connect(self) -> None:
        """Instantiate the shared client against the configured local endpoint.

        Returns:
            None: This method initializes the underlying OpenAI-compatible
                client in place.
        """

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

        Returns:
            LLMPreflightResult: Preflight evidence confirming the configured
                model is reachable and can produce a structured response.

        Raises:
            RuntimeError: If the configured model is unavailable or the probe
                response does not satisfy the health contract.
        """

        available_models = self.list_models()
        if self._config.model_name not in available_models:
            raise RuntimeError(
                f"Configured model '{self._config.model_name}' is not served by "
                f"{self._config.base_url}. Available models: {available_models}"
            )

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
        """Run one structured prompt against the local vLLM server.

        Args:
            prompt (str): Fully rendered prompt text.
            output_schema (dict[str, Any]): JSON schema used for guided
                structured decoding.
            temperature (float | None): Optional per-call temperature override.
            max_tokens (int | None): Optional per-call token-limit override.

        Returns:
            str: Raw response text returned by the server.

        Raises:
            RuntimeError: If the client is disconnected or the server returns an
                empty response payload.
        """

        if self._client is None:
            raise RuntimeError("LLM client is not connected")

        response = self._client.chat.completions.create(
            model=self._config.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._config.temperature if temperature is None else temperature,
            max_tokens=self._config.max_tokens if max_tokens is None else max_tokens,
            extra_body={"guided_json": output_schema},
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("Structured completion returned empty content")
        return content

