"""Thread-safe LLM usage accumulator.

- Tracks cumulative token counts, cost, and latency across API calls.
- Uses the same 8-key format as ``src/ner/runtime/llm_client.UsageStats`` for
  direct comparability across pipelines.
- Does not make API calls, parse responses, or persist data to disk.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


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
        reasoning_tokens: int = 0,
        cached_tokens: int = 0,
        cost_usd: float = 0.0,
        elapsed_ms: float = 0.0,
    ) -> None:
        """Add one API call's usage to the running totals.

        Args:
            prompt_tokens: Prompt (input) token count for this call.
            completion_tokens: Completion (output) token count for this call.
            total_tokens: Total token count for this call.
            reasoning_tokens: Reasoning token count (if model supports it).
            cached_tokens: Cached prompt tokens reused by the provider.
            cost_usd: Monetary cost in USD reported by the provider.
            elapsed_ms: Wall-clock latency in milliseconds for this call.
        """

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
        """Return the 8-key usage summary for JSON serialization.

        Returns:
            Dictionary with the same keys as the NER pipeline's
            ``usage_summary.json`` format.
        """

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
