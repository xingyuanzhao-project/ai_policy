"""Common agent contract for the NER multi-agent pipeline.

- Defines the abstract interface that every NER agent must implement.
- Defines the shared result container returned by agent executions.
- Does not contain prompt rendering, schema validation, or storage behavior.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

InputT = TypeVar("InputT")
ParsedT = TypeVar("ParsedT")


@dataclass(slots=True)
class AgentResult(Generic[ParsedT]):
    """Structured result returned by every NER agent.

    Attributes:
        input_schema_name (str): Name of the schema accepted by the agent.
        output_schema_name (str): Name of the schema emitted by the agent.
        raw_response (str): Raw response text returned by the backing LLM call.
        parsed_response (ParsedT): Parsed and validated artifact payload
            produced by the agent.
    """

    input_schema_name: str
    output_schema_name: str
    raw_response: str
    parsed_response: ParsedT


class BaseAgent(ABC, Generic[InputT, ParsedT]):
    """Shared agent interface with explicit input/output schema names."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable config key and log name for this agent.

        Returns:
            str: Stable agent name used for config lookup and logging.
        """

    @property
    @abstractmethod
    def input_schema_name(self) -> str:
        """Return the explicit input schema name handled by this agent.

        Returns:
            str: Human-readable schema name for the input contract.
        """

    @property
    @abstractmethod
    def output_schema_name(self) -> str:
        """Return the explicit output schema name produced by this agent.

        Returns:
            str: Human-readable schema name for the output contract.
        """

    @abstractmethod
    def run(self, input_data: InputT) -> AgentResult[ParsedT]:
        """Execute the agent and return raw plus parsed response content.

        Args:
            input_data (InputT): Typed input artifact consumed by the agent.

        Returns:
            AgentResult[ParsedT]: Parsed agent result containing both raw and
                structured output.
        """

