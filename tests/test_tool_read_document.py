"""Test tool calling: verify the agent can call read_section to read a document.

Gives the model a document (a short bill excerpt) and asks it to read
specific sections using the read_section tool, then answer a question about
the content.  Validates that the built-in read_section tool factory works
end-to-end with a real LLM.

Usage:
    python -m tests.test_tool_read_document
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
logger = logging.getLogger("test_tool_read_document")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_DOCUMENT = """\
SECTION 1. SHORT TITLE.
This Act may be cited as the "Artificial Intelligence Accountability Act of 2025".

SECTION 2. DEFINITIONS.
In this Act:
(1) ARTIFICIAL INTELLIGENCE.—The term "artificial intelligence" means a \
machine-based system that can, for a given set of human-defined objectives, \
make predictions, recommendations, or decisions influencing real or virtual \
environments.
(2) AUTOMATED DECISION SYSTEM.—The term "automated decision system" means \
a system that uses computation to determine, replace, or materially assist \
in government decision-making.
(3) COVERED AGENCY.—The term "covered agency" means any Federal agency \
that uses an automated decision system.

SECTION 3. REQUIREMENTS FOR COVERED AGENCIES.
(a) IMPACT ASSESSMENT.—Each covered agency shall conduct an impact \
assessment before deploying any automated decision system.
(b) PUBLIC NOTICE.—Each covered agency shall provide public notice of \
automated decision systems in use, including a plain-language description.
(c) ANNUAL REPORTING.—Each covered agency shall submit an annual report \
to Congress on the use of automated decision systems.
"""


def _resolve_api_key() -> str:
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
    return str(llm.get("api_key", ""))


def main() -> None:
    from src.agent import ToolRegistry, UsageStats, make_read_section_tool, run_tool_loop

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

    # Register the read_section tool with our sample document
    registry = ToolRegistry()
    schema, handler = make_read_section_tool(SAMPLE_DOCUMENT)

    read_calls: list[dict] = []
    original_handler = handler

    def tracking_handler(args: dict) -> str:
        read_calls.append(args)
        return original_handler(args)

    registry.register("read_section", tracking_handler, schema)

    messages = [
        {
            "role": "system",
            "content": (
                "You have access to a legislative bill via the read_section tool. "
                "The document is {length} characters long. "
                "Use read_section to read the text, then answer the user's question."
            ).format(length=len(SAMPLE_DOCUMENT)),
        },
        {
            "role": "user",
            "content": (
                "Read the document and tell me: "
                "What are the three requirements for covered agencies in Section 3? "
                "List them briefly."
            ),
        },
    ]

    usage_stats = UsageStats()

    logger.info("Running agent loop with read_section tool...")
    final_response = run_tool_loop(
        client=client,
        model=model,
        messages=messages,
        tools=registry.definitions(),
        tool_executor=registry.execute,
        usage_stats=usage_stats,
        max_turns=10,
        temperature=0.0,
        max_tokens=1024,
    )

    summary = usage_stats.summary_dict()
    print(f"\n{'='*60}", file=sys.stderr)
    print("TEST: Tool Read Document (read_section)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  read_section calls: {len(read_calls)}", file=sys.stderr)
    for i, call in enumerate(read_calls):
        print(f"    [{i}] start={call['start_offset']}  end={call['end_offset']}", file=sys.stderr)
    print(f"  Final response length: {len(final_response)} chars", file=sys.stderr)
    print(f"  API calls: {summary['total_calls']}", file=sys.stderr)
    print(f"  Total tokens: {summary['total_tokens']}", file=sys.stderr)
    print(f"  Cost: ${summary['total_cost_usd']:.6f}", file=sys.stderr)
    print(f"\n  Response excerpt: {final_response[:300]}...", file=sys.stderr)

    assert len(read_calls) >= 1, "Model should have called read_section at least once"

    response_lower = final_response.lower()
    assert "impact assessment" in response_lower, (
        f"Response should mention 'impact assessment'; got: {final_response[:200]}"
    )
    assert "public notice" in response_lower or "notice" in response_lower, (
        f"Response should mention 'public notice'; got: {final_response[:200]}"
    )
    assert "annual report" in response_lower or "reporting" in response_lower, (
        f"Response should mention 'annual reporting'; got: {final_response[:200]}"
    )

    print(f"\n  PASS: Agent can read a document via read_section tool", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


if __name__ == "__main__":
    main()
