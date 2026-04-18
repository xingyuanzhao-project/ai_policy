"""Reusable mini agent loop for tool-calling conversations.

- Provides a generic multi-turn tool-calling loop over any OpenAI-compatible API.
- Provides a tool registry for registering and dispatching tool definitions.
- Provides built-in tool factories for common document-reading operations.
- Tracks cumulative LLM usage statistics (tokens, cost, latency).
- Does not contain domain-specific logic, prompts, or output schemas.
"""

from .loop import LoopResult, run_tool_loop, run_tool_loop_async
from .tools import ToolRegistry, make_read_section_tool
from .usage import UsageStats

__all__ = [
    "LoopResult",
    "ToolRegistry",
    "UsageStats",
    "make_read_section_tool",
    "run_tool_loop",
    "run_tool_loop_async",
]
