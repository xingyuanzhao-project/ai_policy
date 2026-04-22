"""Nine-stage LLM-as-judge evaluation pipeline for quadruplet extraction runs.

- Loads the two configured NER run outputs (orchestrated multi-turn and
  skill-driven agentic) and judges their quadruplets against NCSL bill-level
  theme / keyword labels using a judge LLM in a different model family from
  the extractor.
- Implements the nine-stage modification of Mahbub et al. 2026 documented in
  ``docs/lit_rev_eval.md``: prompt calibration, rule-based plausibility,
  per-quadruplet grounding, set-to-label coverage, novel-entity bookkeeping,
  pairwise method comparison, expert validation, CALM bias audit, and
  extrinsic-validity hooks.
- Caches per-item judge verdicts to JSONL so runs are resumable at the
  quadruplet / (bill, label) granularity.
- Does not modify the extractor runs; read-only against their output files.
"""

from typing import TYPE_CHECKING, Any

from .config import EvalConfig, load_eval_config

if TYPE_CHECKING:
    from .evals import run_evaluation  # re-exported for typing

__all__ = [
    "EvalConfig",
    "load_eval_config",
    "run_evaluation",
]


def __getattr__(name: str) -> Any:
    """Lazy-load ``run_evaluation`` so ``python -m src.eval.evals`` stays clean.

    Eagerly importing ``.evals`` here would insert it into ``sys.modules``
    before the CLI entry point executes, which triggers a ``RuntimeWarning``
    under ``python -m src.eval.evals``.  Lazy-loading avoids that race while
    still letting library callers do ``from src.eval import run_evaluation``.
    """

    if name == "run_evaluation":
        from .evals import run_evaluation as _run_evaluation

        return _run_evaluation
    raise AttributeError(f"module 'src.eval' has no attribute {name!r}")
