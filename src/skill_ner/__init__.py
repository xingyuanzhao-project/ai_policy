"""Skill-driven NER extraction pipeline.

- Wraps ``src/agent/`` to run agentic entity extraction using a methodology
  prompt loaded from ``settings/skills/``.
- Produces ``RefinedQuadruplet``-compatible output for comparison with the
  orchestrated NER pipeline.
- Persists per-run usage statistics in the same 8-key format as the
  orchestrated pipeline.
- Does not modify or import orchestration logic from ``src/ner/``.
"""

from .runner import run_corpus

__all__ = ["run_corpus"]
