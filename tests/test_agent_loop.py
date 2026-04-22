"""Test the generic agent loop: verify multi-turn tool calling works.

Sends a simple arithmetic task to the model with a calculator tool.
The model must call the tool at least once, then return a final answer.
This validates the core loop mechanics independently of NER.

Usage:
    python -m tests.test_agent_loop
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import yaml
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("test_agent_loop")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_api_key() -> str:
    """Resolve API key from env var, keyring, or config literal."""

    config_path = PROJECT_ROOT / "settings" / "config.yml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    llm = config.get("llm", {})

    env_var = str(llm.get("api_key_env_var", "")).strip()
    if env_var:
        api_key = os.environ.get(env_var, "").strip()
        if api_key:
            return api_key

    keyring_service = str(llm.get("keyring_service", "")).strip()
    keyring_username = str(llm.get("keyring_username", "")).strip()
    if keyring_service and keyring_username:
        try:
            import keyring
            secret = keyring.get_password(keyring_service, keyring_username)
            if secret and secret.strip():
                return secret.strip()
        except Exception:
            pass

    literal = str(llm.get("api_key", "")).strip()
    if literal:
        return literal

    raise RuntimeError("No API key found")


def main() -> None:
    from src.agent import ToolRegistry, UsageStats, run_tool_loop

    api_key = _resolve_api_key()
    with open(PROJECT_ROOT / "settings" / "config.yml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    llm_config = config.get("llm", {})

    client = OpenAI(
        base_url=llm_config["base_url"],
        api_key=api_key,
        max_retries=2,
        timeout=60,
    )
    model = llm_config["model_name"]

    # Register a simple calculator tool
    registry = ToolRegistry()

    calc_schema = {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a simple arithmetic expression and return the numeric result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A Python arithmetic expression, e.g. '2 + 3 * 4'.",
                    },
                },
                "required": ["expression"],
            },
        },
    }

    tool_calls_made: list[str] = []

    def calc_handler(args: dict) -> str:
        expr = args["expression"]
        tool_calls_made.append(expr)
        # Safe eval for simple arithmetic
        allowed = set("0123456789+-*/.(). ")
        if not all(c in allowed for c in expr):
            return json.dumps({"error": "Invalid expression"})
        try:
            result = eval(expr)  # noqa: S307 -- intentionally limited to arithmetic
            return json.dumps({"result": result})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    registry.register("calculate", calc_handler, calc_schema)

    messages = [
        {
            "role": "system",
            "content": "You are a math assistant. Use the calculate tool to solve problems. Return the final numeric answer as plain text.",
        },
        {
            "role": "user",
            "content": "What is (17 * 23) + (45 * 12)? Use the calculate tool to compute this.",
        },
    ]

    usage_stats = UsageStats()

    logger.info("Running agent loop with calculator tool...")
    final_response = run_tool_loop(
        client=client,
        model=model,
        messages=messages,
        tools=registry.definitions(),
        tool_executor=registry.execute,
        usage_stats=usage_stats,
        max_turns=10,
        temperature=0.0,
        max_tokens=512,
    )

    # Verify results
    summary = usage_stats.summary_dict()
    print(f"\n{'='*60}", file=sys.stderr)
    print("TEST: Agent Loop (multi-turn tool calling)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  Tool calls made: {len(tool_calls_made)}", file=sys.stderr)
    for i, expr in enumerate(tool_calls_made):
        print(f"    [{i}] {expr}", file=sys.stderr)
    print(f"  Final response: {final_response!r}", file=sys.stderr)
    print(f"  API calls: {summary['total_calls']}", file=sys.stderr)
    print(f"  Total tokens: {summary['total_tokens']}", file=sys.stderr)
    print(f"  Cost: ${summary['total_cost_usd']:.6f}", file=sys.stderr)

    assert len(tool_calls_made) >= 1, "Model should have called the calculate tool at least once"
    assert summary["total_calls"] >= 2, "Should have at least 2 API calls (tool call turn + final response)"
    # (17*23) + (45*12) = 391 + 540 = 931
    assert "931" in final_response, f"Expected '931' in final response, got: {final_response!r}"

    print(f"\n  PASS: Agent loop works correctly", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


if __name__ == "__main__":
    main()
