"""Skill methodology loader and API message builder.

- Loads the extraction methodology from ``settings/skills/ner_extraction.md``.
- Constructs the initial ``messages`` list for the agent loop.
- Contains a minimal hardwired fallback used only if the skill file is missing.
- Does not call the LLM, register tools, or persist data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_SKILL_PATH = "settings/skills/ner_extraction.md"

_FALLBACK_SKILL = """\
You are an expert NER extraction agent for AI policy legislation.

Your task is to read a bill's full text using the read_section tool and extract
structured entity quadruplets (entity, type, attribute, value) with evidence spans.

Return your final output as a JSON object with a single key "quadruplets"
containing a list of extracted items.
"""


def load_skill(
    project_root: Path,
    skill_path: str = _DEFAULT_SKILL_PATH,
) -> str:
    """Read the skill methodology file from disk.

    Falls back to ``_FALLBACK_SKILL`` if the file does not exist, logging a
    warning.

    Args:
        project_root: Project root for resolving relative paths.
        skill_path: Path to the skill markdown file relative to project root.

    Returns:
        The skill methodology text to use as the system prompt.
    """

    full_path = project_root / skill_path
    if not full_path.exists():
        logger.warning(
            "Skill file not found at %s; using hardwired fallback",
            full_path,
        )
        return _FALLBACK_SKILL

    content = full_path.read_text(encoding="utf-8").strip()
    if not content:
        logger.warning(
            "Skill file at %s is empty; using hardwired fallback",
            full_path,
        )
        return _FALLBACK_SKILL

    logger.info("Loaded skill methodology from %s (%d chars)", full_path, len(content))
    return content


def build_messages(
    skill_content: str,
    bill_id: str,
    bill_text_length: int,
) -> list[dict[str, Any]]:
    """Construct the initial messages for the agent loop.

    Args:
        skill_content: The loaded skill methodology text (system prompt).
        bill_id: Identifier of the bill being processed.
        bill_text_length: Total character length of the bill text, so the
            model knows the document bounds for ``read_section`` calls.

    Returns:
        List of message dicts ready for ``run_tool_loop``.
    """

    return [
        {"role": "system", "content": skill_content},
        {
            "role": "user",
            "content": (
                f"Extract all AI-policy entity quadruplets from bill '{bill_id}'.\n"
                f"The bill text is {bill_text_length:,} characters long. "
                f"Use the read_section tool to read portions of the text "
                f"(offsets 0 through {bill_text_length - 1}).\n\n"
                f"When done, return your final answer as a JSON object with key "
                f'"quadruplets" containing a list of extracted items.'
            ),
        },
    ]
