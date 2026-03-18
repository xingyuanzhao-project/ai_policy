"""Shared prompt execution and structured parsing helpers for NER agents.

- Owns prompt execution through the shared LLM client.
- Owns minimal structured-output parsing and prompt serialization helpers.
- Does not define task-specific schemas, storage behavior, or orchestration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..runtime.llm_client import LLMClient
from ..schemas.artifacts import artifact_to_dict
from ..schemas.validation import SchemaValidationError


@dataclass(slots=True)
class AgentExecutionConfig:
    """Shared generation settings loaded from config for one agent.

    Attributes:
        temperature (float): Sampling temperature passed to the LLM request.
        max_tokens (int): Maximum completion tokens allowed for the agent call.
    """

    temperature: float
    max_tokens: int


class PromptExecutor:
    """Run structured prompts through the shared local LLM client."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize the shared prompt executor.

        Args:
            llm_client (LLMClient): Connected shared LLM client used by all
                agents.
        """

        self._llm_client = llm_client

    def execute(
        self,
        prompt: str,
        output_schema: dict[str, Any],
        execution_config: AgentExecutionConfig,
    ) -> str:
        """Execute one prompt and return the raw structured response text.

        Args:
            prompt (str): Fully rendered prompt text sent to the LLM.
            output_schema (dict[str, Any]): JSON schema used for guided
                structured decoding.
            execution_config (AgentExecutionConfig): Per-agent generation
                settings.

        Returns:
            str: Raw response text returned by the LLM backend.
        """

        return self._llm_client.generate(
            prompt=prompt,
            output_schema=output_schema,
            temperature=execution_config.temperature,
            max_tokens=execution_config.max_tokens,
        )


class StructuredOutputParser:
    """Parse agent raw responses and fail closed on invalid JSON structure."""

    @staticmethod
    def parse_object(raw_response: str) -> dict[str, Any]:
        """Parse one raw response as a JSON object.

        Args:
            raw_response (str): Raw text returned by the LLM backend.

        Returns:
            dict[str, Any]: Parsed JSON object decoded from the raw response.

        Raises:
            SchemaValidationError: If the response is not valid JSON or the
                decoded root value is not a JSON object.
        """

        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(
                f"Structured output is not valid JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise SchemaValidationError("Structured output root must be a JSON object")
        return payload


def render_prompt(template: str, **values: object) -> str:
    """Render a prompt template from explicit named values.

    Args:
        template (str): Prompt template containing named format placeholders.
        **values (object): Placeholder values injected into the prompt
            template.

    Returns:
        str: Rendered prompt string ready for execution.
    """

    return template.format(**values)


def serialize_for_prompt(payload: Any) -> str:
    """Serialize artifacts for inclusion in a model prompt.

    Args:
        payload (Any): Artifact, collection of artifacts, or plain value to
            encode.

    Returns:
        str: JSON string representation of the payload for prompt inclusion.
    """

    return json.dumps(artifact_to_dict(payload), indent=2, ensure_ascii=False)

