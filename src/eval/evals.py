"""CLI orchestrator for the nine-stage LLM-as-judge evaluation pipeline.

- Loads ``settings/eval/eval.yml`` (or the ``--config`` override), builds a
  single :class:`~src.eval.judge.JudgeConnection` with a shared
  :class:`~src.agent.usage.UsageStats`, and dispatches the requested
  subset of stages in order.
- Stage selection is driven exclusively by the ``enabled`` flag under
  each ``stages.stageN`` block in the YAML. Programmatic callers (e.g.
  ``scripts/run_evals.py``) may also pass ``stage_override`` to
  :func:`run_evaluation`; there is no CLI switch for it by design, to
  keep the run spec reproducible from config. Method selection is still
  driven by ``--methods``.
- The orchestrator owns the shared state between stages: the pipeline
  writes stage-result JSONs under ``output/evals/v1/results/``, a
  corpus-wide ``summary.json`` / ``summary.md``, and a
  ``judge_usage_summary.json`` mirroring the 8-key schema used by the NER
  runs.
- Per-stage resumability is implemented inside each stage via the
  :mod:`src.eval.cache` module; ``--no-resume`` wipes the cache for the
  selected stages and methods before running.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.agent.usage import UsageStats

from .artifacts import StageResult
from .cache import cache_paths_for_stage, wipe_cache
from .config import EvalConfig, load_eval_config
from .judge import CreditsExhaustedError, JudgeConnection, build_judge_connection
from .stages import (
    stage1_calibration,
    stage2_plausibility,
    stage3_grounding,
    stage4_coverage,
    stage5_novelty,
    stage6_pairwise,
    stage7_expert,
    stage8_bias,
    stage9_extrinsic,
)
from .stages._common import StageContext, build_stage_context, results_dir

logger = logging.getLogger(__name__)

_STAGE_MODULES: dict[int, Any] = {
    1: stage1_calibration,
    2: stage2_plausibility,
    3: stage3_grounding,
    4: stage4_coverage,
    5: stage5_novelty,
    6: stage6_pairwise,
    7: stage7_expert,
    8: stage8_bias,
    9: stage9_extrinsic,
}

_JUDGE_STAGES: frozenset[int] = frozenset({3, 4, 6, 8})
_PER_METHOD_CACHE_STAGES: frozenset[int] = frozenset({3, 4, 8})
_CROSS_METHOD_CACHE_STAGES: frozenset[int] = frozenset({6})

_LOG_FORMAT = "%(asctime)s  %(name)s  %(levelname)s  %(message)s"


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m src.eval.evals``.

    Args:
        argv: Optional argument list; defaults to :data:`sys.argv`.

    Returns:
        Process exit code (``0`` on success, non-zero on fatal errors).
    """

    args = _parse_args(argv)
    _configure_logging(args.verbose)
    try:
        summary = run_evaluation(
            config_path=Path(args.config),
            method_override=args.methods,
            sample_bills_override=args.sample_bills,
            no_resume=args.no_resume,
        )
    except CreditsExhaustedError as exc:
        logger.error("Judge credits exhausted: %s", exc)
        return 2
    except Exception:
        logger.exception("Eval pipeline aborted with an unexpected error")
        return 1
    status = summary.get("status", "unknown")
    logger.info(
        "Eval run finished: status=%s completed=%d skipped=%d errored=%d",
        status,
        summary.get("completed_stages", 0),
        summary.get("skipped_stages", 0),
        summary.get("errored_stages", 0),
    )
    return 0 if summary.get("errored_stages", 0) == 0 else 1


def run_evaluation(
    *,
    config_path: Path,
    stage_override: list[int] | None = None,
    method_override: list[str] | None = None,
    sample_bills_override: int | None = None,
    no_resume: bool = False,
) -> dict[str, Any]:
    """Run the full evaluation pipeline and return the summary payload.

    Args:
        config_path: Path to ``eval.yml``.
        stage_override: Optional list of stage numbers to run. ``None``
            means run every stage whose ``enabled`` flag is true.
        method_override: Optional list of method names to keep. ``None``
            means keep every method the config declares.
        sample_bills_override: Optional override for ``sampling.sample_bills``.
        no_resume: When true, every selected stage's cache is wiped
            before the stage runs.

    Returns:
        Summary dictionary identical in shape to the ``summary.json`` file
        written under ``output/evals/v1``.
    """

    config = load_eval_config(config_path)
    if method_override:
        config = _filter_methods(config, method_override)

    selected_stages = _select_stages(config, override=stage_override)
    logger.info(
        "Starting eval run: stages=%s methods=%s no_resume=%s",
        selected_stages,
        list(config.methods.keys()),
        no_resume,
    )

    judge_usage = UsageStats()
    judge: JudgeConnection | None = None
    if any(stage in _JUDGE_STAGES for stage in selected_stages):
        judge = build_judge_connection(
            config.judge,
            project_root=config.project_root,
            usage_stats=judge_usage,
        )

    context = build_stage_context(
        config,
        judge=judge,
        sample_bills_override=sample_bills_override,
    )
    _attach_run_log_handler(context.run_dir)

    if no_resume:
        _wipe_selected_caches(
            context=context, selected_stages=selected_stages
        )

    stage_results: list[StageResult] = []
    run_started_at = time.time()
    overall = tqdm(
        total=len(selected_stages),
        desc="Eval pipeline",
        position=0,
        leave=True,
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
        unit="stage",
    )
    with logging_redirect_tqdm():
        try:
            _run_stages_with_progress(
                context=context,
                selected_stages=selected_stages,
                stage_results=stage_results,
                overall=overall,
            )
        finally:
            overall.close()

    summary_payload = _finalise_run(
        context=context,
        stage_results=stage_results,
        judge_usage=judge_usage,
        started_at=run_started_at,
    )
    return summary_payload


def _run_stages_with_progress(
    *,
    context: StageContext,
    selected_stages: list[int],
    stage_results: list[StageResult],
    overall: tqdm,
) -> None:
    """Iterate ``selected_stages`` while advancing the outer progress bar.

    Factored out of :func:`run_evaluation` so the outer ``tqdm`` bar, the
    :func:`logging_redirect_tqdm` context, and the ``try/finally`` cleanup
    stay legible. Mutates ``stage_results`` in place.
    """

    for stage in selected_stages:
        module = _STAGE_MODULES[stage]
        overall.set_description_str(f"Eval pipeline (stage {stage})")
        logger.info("Running stage %d (%s)", stage, module.__name__)
        t0 = time.perf_counter()
        try:
            result = module.run(context)
        except CreditsExhaustedError:
            logger.error(
                "Stage %d aborted: judge credits exhausted. Halting pipeline.",
                stage,
            )
            stage_results.append(
                StageResult(
                    stage=stage,
                    status="error",
                    summary=f"Stage {stage} aborted: judge credits exhausted",
                    metrics={"reason": "credits_exhausted"},
                )
            )
            overall.update(1)
            break
        except Exception as exc:  # noqa: BLE001 -- surfaced in summary
            logger.exception("Stage %d failed", stage)
            stage_results.append(
                StageResult(
                    stage=stage,
                    status="error",
                    summary=f"Stage {stage} raised: {exc}",
                    metrics={"error_type": type(exc).__name__},
                )
            )
            overall.update(1)
            continue
        elapsed_s = time.perf_counter() - t0
        result.metrics.setdefault("elapsed_seconds", round(elapsed_s, 2))
        stage_results.append(result)
        logger.info(
            "Stage %d: status=%s  elapsed=%.1fs  %s",
            stage, result.status, elapsed_s, result.summary,
        )
        overall.update(1)


def _filter_methods(config: EvalConfig, keep: list[str]) -> EvalConfig:
    """Return a copy of ``config`` with only the requested methods."""

    kept = {}
    for name in keep:
        if name not in config.methods:
            raise ValueError(
                f"Method {name!r} is not in eval config; "
                f"available: {list(config.methods.keys())}"
            )
        kept[name] = config.methods[name]
    if not kept:
        raise ValueError("--methods filter matched no methods")
    return EvalConfig(
        judge=config.judge,
        methods=kept,
        corpus=config.corpus,
        output_run_dir=config.output_run_dir,
        stages=config.stages,
        sampling=config.sampling,
        project_root=config.project_root,
    )


def _select_stages(
    config: EvalConfig, *, override: list[int] | None
) -> list[int]:
    """Resolve which stages run.

    ``override`` is reserved for programmatic callers (a ``.py`` runner
    script that wants to pin the stage list regardless of YAML). It is
    *not* wired to any CLI flag, by design: run specs should be
    reproducible from ``settings/eval/eval.yml`` alone, so operators
    toggle stages via each ``stages.stageN.enabled`` block rather than
    via shell arguments. When ``override`` is ``None`` (the default),
    every stage whose YAML ``enabled`` flag is true -- or which has no
    YAML entry at all -- is selected, in ascending order.
    """

    if override is not None:
        unknown = [s for s in override if s not in _STAGE_MODULES]
        if unknown:
            raise ValueError(
                f"Unknown stage numbers in --stages: {unknown}; "
                f"allowed: {sorted(_STAGE_MODULES.keys())}"
            )
        return sorted(set(override))
    enabled: list[int] = []
    for number in sorted(_STAGE_MODULES.keys()):
        toggle = config.stages.get(number)
        if toggle is None or toggle.enabled:
            enabled.append(number)
    return enabled


def _wipe_selected_caches(
    *, context: StageContext, selected_stages: list[int]
) -> None:
    """Delete cache JSONLs for the stages about to run."""

    for stage in selected_stages:
        if stage in _PER_METHOD_CACHE_STAGES:
            for method_name in context.method_outputs.keys():
                paths = cache_paths_for_stage(
                    run_dir=context.run_dir, stage=stage, method=method_name
                )
                wipe_cache(paths)
                logger.info(
                    "Wiped cache: stage=%d method=%s dir=%s",
                    stage, method_name, paths.base_dir,
                )
        elif stage in _CROSS_METHOD_CACHE_STAGES:
            paths = cache_paths_for_stage(run_dir=context.run_dir, stage=stage)
            wipe_cache(paths)
            logger.info(
                "Wiped cache: stage=%d dir=%s", stage, paths.base_dir
            )


def _finalise_run(
    *,
    context: StageContext,
    stage_results: list[StageResult],
    judge_usage: UsageStats,
    started_at: float,
) -> dict[str, Any]:
    """Write ``summary.json``, ``summary.md``, and judge usage artefacts."""

    usage_dict = judge_usage.summary_dict()
    usage_path = context.run_dir / "judge_usage_summary.json"
    usage_path.write_text(
        json.dumps(usage_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    completed = [r for r in stage_results if r.status == "completed"]
    skipped = [r for r in stage_results if r.status == "skipped"]
    errored = [r for r in stage_results if r.status == "error"]

    summary_payload: dict[str, Any] = {
        "status": "error" if errored else "completed",
        "run_dir": str(context.run_dir),
        "completed_stages": len(completed),
        "skipped_stages": len(skipped),
        "errored_stages": len(errored),
        "elapsed_seconds": round(time.time() - started_at, 2),
        "judge_usage": usage_dict,
        "methods": list(context.method_outputs.keys()),
        "bill_count": len(context.sampled_bill_ids),
        "stages": [
            {
                "stage": r.stage,
                "status": r.status,
                "summary": r.summary,
                "metrics": r.metrics,
                "artifacts": r.artifacts,
            }
            for r in stage_results
        ],
    }

    summary_json = context.run_dir / "summary.json"
    summary_json.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (context.run_dir / "summary.md").write_text(
        _render_markdown_summary(summary_payload),
        encoding="utf-8",
    )
    # Keep a symlink-like pointer that downstream tools can rely on.
    pointer_dir = results_dir(context)
    (pointer_dir / "run_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary_payload


def _render_markdown_summary(payload: dict[str, Any]) -> str:
    """Render a human-readable markdown overview of the run."""

    lines: list[str] = []
    lines.append("# Evaluation run summary\n")
    lines.append(f"- **Status**: {payload['status']}")
    lines.append(f"- **Run dir**: `{payload['run_dir']}`")
    lines.append(f"- **Bills**: {payload['bill_count']}")
    lines.append(f"- **Methods**: {', '.join(payload['methods'])}")
    lines.append(
        f"- **Elapsed**: {payload['elapsed_seconds']:.1f} s; "
        f"completed={payload['completed_stages']} "
        f"skipped={payload['skipped_stages']} "
        f"errored={payload['errored_stages']}"
    )
    usage = payload.get("judge_usage") or {}
    if usage.get("total_calls"):
        lines.append(
            f"- **Judge usage**: calls={usage.get('total_calls', 0)} "
            f"tokens={usage.get('total_tokens', 0)} "
            f"cost=${usage.get('total_cost_usd', 0):.4f}"
        )
    lines.append("")
    lines.append("## Stages\n")
    lines.append("| Stage | Status | Summary |")
    lines.append("| ---: | :--- | :--- |")
    for row in payload.get("stages", []):
        summary = str(row.get("summary", "")).replace("|", "\\|")
        lines.append(
            f"| {row['stage']} | {row['status']} | {summary} |"
        )
    lines.append("")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Build and parse the orchestrator CLI arguments."""

    parser = argparse.ArgumentParser(
        prog="python -m src.eval.evals",
        description=(
            "Run the nine-stage LLM-as-judge evaluation over the "
            "configured NER method outputs."
        ),
    )
    parser.add_argument(
        "--config",
        default="settings/eval/eval.yml",
        help="Path to the eval YAML config (default: settings/eval/eval.yml)",
    )
    parser.add_argument(
        "--sample-bills",
        type=int,
        default=None,
        help=(
            "Override sampling.sample_bills; caps the bill intersection "
            "used by every stage."
        ),
    )
    parser.add_argument(
        "--methods",
        default=None,
        help=(
            "Comma-separated method names to keep (e.g. 'orchestrated,skill_driven'). "
            "Defaults to every method in the config."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Wipe each selected stage's cache before running.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase log verbosity (repeat for DEBUG).",
    )
    ns = parser.parse_args(argv)
    ns.methods = _parse_csv_strs(ns.methods)
    return ns


def _parse_csv_strs(raw: str | None) -> list[str] | None:
    """Parse a comma-separated string list, tolerating whitespace."""

    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts or None


def _configure_logging(verbosity: int) -> None:
    """Configure root logging honouring the ``-v`` / ``-vv`` CLI flags."""

    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        stream=sys.stdout,
        force=True,
    )


def _attach_run_log_handler(run_dir: Path) -> None:
    """Tee logs into ``<run_dir>/run.log`` for post-hoc inspection."""

    log_path = run_dir / "run.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root = logging.getLogger()
    root.addHandler(handler)
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)


if __name__ == "__main__":
    raise SystemExit(main())
