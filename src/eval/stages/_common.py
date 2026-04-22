"""Shared helpers for stage modules.

- Provides a single :class:`StageContext` that carries the eval config,
  loaded method outputs, loaded bill records, optional judge connection,
  and the concrete run-output directory.
- Helper functions here handle repeated operations like sampling bill
  ids, locating Stage 2's pass set, and writing stage result JSONs.
- Does not execute any stage itself; each stage module imports what it
  needs from here.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from collections.abc import Awaitable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm.asyncio import tqdm as tqdm_asyncio

from ..artifacts import BillRecord, Quadruplet, StageResult
from ..config import EvalConfig
from ..io import (
    available_bill_ids_for_method,
    intersect_bill_ids,
    load_bill_records,
    load_run_outputs,
)
from ..judge import JudgeConnection

logger = logging.getLogger(__name__)

_RESULTS_SUBDIR = "results"
_PROMPTS_SUBDIR = "prompts"
_PROGRESS_STAGE_POSITION = 1


@dataclass(slots=True)
class StageContext:
    """Lazy bundle of everything a stage needs at run time.

    Attributes:
        config: Parsed :class:`EvalConfig`.
        judge: Optional judge connection; rule-based stages may run without.
        run_dir: Absolute path to ``output/evals/v1/`` (or whatever the
            config picks).
        sampled_bill_ids: Bill ids retained after sampling, shared across
            stages.
        method_outputs: Map ``method_name -> {bill_id: [Quadruplet]}``.
        bill_records: Map ``bill_id -> BillRecord`` (NCSL join).
    """

    config: EvalConfig
    judge: JudgeConnection | None
    run_dir: Path
    sampled_bill_ids: list[str]
    method_outputs: dict[str, dict[str, list[Quadruplet]]]
    bill_records: dict[str, BillRecord]


def build_stage_context(
    config: EvalConfig,
    *,
    judge: JudgeConnection | None,
    sample_bills_override: int | None = None,
) -> StageContext:
    """Load extractor outputs and the corpus once for reuse across stages.

    Args:
        config: Parsed eval configuration.
        judge: Optional judge connection; rule-based stages may pass
            ``None`` here. Judge-backed stages require it.
        sample_bills_override: Optional override for ``sampling.sample_bills``;
            when non-``None``, overrides the YAML value (used by the CLI
            ``--sample-bills`` flag).

    Returns:
        Fully populated :class:`StageContext`.
    """

    run_dir = _prepare_run_dir(config.output_run_dir)
    sampled = _select_sampled_bill_ids(
        config, override=sample_bills_override
    )
    method_outputs: dict[str, dict[str, list[Quadruplet]]] = {}
    for method in config.methods.values():
        method_outputs[method.name] = load_run_outputs(method, bill_ids=sampled)
    bill_records = load_bill_records(config.corpus, bill_ids=sampled)
    logger.info(
        "Stage context ready: methods=%s  bills=%d  corpus_rows=%d  run_dir=%s",
        list(method_outputs.keys()),
        len(sampled),
        len(bill_records),
        run_dir,
    )
    return StageContext(
        config=config,
        judge=judge,
        run_dir=run_dir,
        sampled_bill_ids=sampled,
        method_outputs=method_outputs,
        bill_records=bill_records,
    )


def results_dir(ctx: StageContext) -> Path:
    """Return (and create) the results subdirectory for stage JSON outputs."""

    path = ctx.run_dir / _RESULTS_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def prompts_dir(ctx: StageContext) -> Path:
    """Return (and create) the prompts subdirectory for Stage 1 artefacts."""

    path = ctx.run_dir / _PROMPTS_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_result_json(
    ctx: StageContext, *, filename: str, payload: dict[str, Any]
) -> Path:
    """Write a stage result JSON file under ``results/``."""

    path = results_dir(ctx) / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def read_stage2_pass_set(ctx: StageContext) -> dict[str, dict[str, set[str]]]:
    """Return the ``{method: {bill_id: {quadruplet_ids}}}`` pass map.

    Stages 3 and 5 depend on this; they surface a clean failure if Stage 2
    has not been run yet.
    """

    path = results_dir(ctx) / "stage2_plausibility.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Stage 2 output not found at {path}; run stage 2 first."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    result: dict[str, dict[str, set[str]]] = {}
    for method, bills in (payload.get("passed") or {}).items():
        bill_map: dict[str, set[str]] = {}
        for bill_id, ids in bills.items():
            bill_map[bill_id] = set(ids)
        result[method] = bill_map
    return result


def result_exists(ctx: StageContext, filename: str) -> bool:
    """Return whether a stage has already written its results file."""

    return (results_dir(ctx) / filename).is_file()


def skipped_result(
    stage: int, *, summary: str, reason: str, filename: str | None = None
) -> StageResult:
    """Build a standard ``skipped`` :class:`StageResult`."""

    return StageResult(
        stage=stage,
        status="skipped",
        summary=summary,
        metrics={"reason": reason},
        artifacts={"result_file": filename} if filename else {},
    )


async def gather_with_progress(
    tasks: Iterable[Awaitable[Any]], *, desc: str
) -> list[Any]:
    """Await ``tasks`` while rendering a single tqdm progress bar.

    - Uses :class:`tqdm.asyncio.tqdm` so the bar advances as tasks
      complete (ETA reflects live throughput).
    - Rendered at ``position=1`` so it nests cleanly under the
      orchestrator's outer per-stage bar at ``position=0``.
    - ``leave=False`` clears the bar when the stage finishes so later
      log lines stay on a clean line.
    - Auto-disables when stderr is not a TTY (e.g. piped into a file),
      matching the behaviour of the outer bar.

    Args:
        tasks: Awaitables to run concurrently. An empty iterable is a
            no-op and returns ``[]`` without rendering a bar.
        desc: Short human-readable label for the progress bar (e.g.
            ``"Stage 3 (grounding)"``). Both methods' tasks share the
            same bar, so this label stays method-agnostic.

    Returns:
        The list of awaited results in submission order, matching
        :func:`asyncio.gather`'s contract.
    """

    task_list = list(tasks)
    if not task_list:
        return []
    disable = not sys.stderr.isatty()
    return await tqdm_asyncio.gather(
        *task_list,
        desc=desc,
        total=len(task_list),
        position=_PROGRESS_STAGE_POSITION,
        leave=False,
        dynamic_ncols=True,
        mininterval=0.3,
        disable=disable,
    )


def _prepare_run_dir(run_dir: Path) -> Path:
    """Create the run output directory eagerly so stages never race on mkdir."""

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _select_sampled_bill_ids(
    config: EvalConfig, *, override: int | None
) -> list[str]:
    """Pick the bill ids every stage will see."""

    cap = override if override is not None else config.sampling.sample_bills
    intersection = intersect_bill_ids(config.methods.values())
    if not intersection:
        available = {
            name: len(available_bill_ids_for_method(method))
            for name, method in config.methods.items()
        }
        raise ValueError(
            f"No bill ids are present across every method; per-method counts: {available}"
        )
    if cap is None or cap >= len(intersection):
        logger.info("Using full bill intersection: %d bills", len(intersection))
        return intersection
    rng = random.Random(config.sampling.seed)
    sample = sorted(rng.sample(intersection, cap))
    logger.info("Sampled %d bills (seed=%d)", len(sample), config.sampling.seed)
    return sample
