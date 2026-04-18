"""Tool registry and built-in tool factories for the agent loop.

- Provides ``ToolRegistry`` for registering, listing, and dispatching tools.
- Provides built-in tool factories (e.g. ``make_read_section_tool``) that
  create closures over caller-supplied data.
- Does not call the LLM, parse model output, or persist data.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Register tool definitions and their Python handler functions.

    Each tool consists of a JSON schema (sent to the LLM as the ``tools``
    parameter) and a Python callable that executes locally when the model
    requests that tool.
    """

    def __init__(self) -> None:
        self._schemas: dict[str, dict[str, Any]] = {}
        self._handlers: dict[str, Callable[..., Any]] = {}

    def register(
        self,
        name: str,
        handler: Callable[..., Any],
        schema: dict[str, Any],
    ) -> None:
        """Register a tool definition and its handler.

        Args:
            name: Unique tool name matching the ``function.name`` in the schema.
            handler: Python callable invoked with the tool arguments dict.
            schema: OpenAI-compatible tool schema (``type: function`` envelope).
        """

        self._schemas[name] = schema
        self._handlers[name] = handler
        logger.debug("Registered tool: %s", name)

    def definitions(self) -> list[dict[str, Any]]:
        """Return the list of tool schemas for the ``tools`` API parameter.

        Returns:
            List of OpenAI-compatible tool definition dicts.
        """

        return list(self._schemas.values())

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Dispatch a tool call to the registered handler.

        Args:
            name: Tool name requested by the model.
            arguments: Parsed argument dict from the model's tool call.

        Returns:
            JSON-encoded string result of the tool execution.

        Raises:
            KeyError: If the tool name is not registered.
        """

        if name not in self._handlers:
            raise KeyError(f"Unknown tool: '{name}'")
        result = self._handlers[name](arguments)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Built-in tool factories
# ---------------------------------------------------------------------------

_READ_SECTION_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_section",
        "description": (
            "Read a section of the document by character offsets. "
            "Returns the substring document[start_offset:end_offset]."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Inclusive start character offset.",
                },
                "end_offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Exclusive end character offset.",
                },
            },
            "required": ["start_offset", "end_offset"],
        },
    },
}


def make_read_section_tool(
    document_text: str,
) -> tuple[dict[str, Any], Callable[[dict[str, Any]], str]]:
    """Create a ``read_section`` tool bound to a specific document.

    The returned handler is a closure over *document_text* that slices it by
    character offsets.  This is generic -- it works for any document, not just
    bills.

    Args:
        document_text: The full document text to bind to the tool.

    Returns:
        Tuple of (OpenAI tool schema dict, handler callable).
    """

    text_length = len(document_text)

    def _handler(arguments: dict[str, Any]) -> str:
        start = max(0, int(arguments["start_offset"]))
        end = min(text_length, int(arguments["end_offset"]))
        if start >= end:
            return ""
        return document_text[start:end]

    return _READ_SECTION_SCHEMA, _handler
