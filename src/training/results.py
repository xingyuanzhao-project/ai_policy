"""Aggregate per-run JSONs into the four result CSVs.

Walks ``output/training/runs/<model>/<run_dir>/run.json`` and writes:

* ``results.csv`` -- tidy long-form: one row per ``(model, fold, seed,
  lr, split, metric)``.
* ``results_wide.csv`` -- one row per ``(model, fold, seed, lr)`` with
  every metric as a column (test split).
* ``results_summary.csv`` -- per-model mean and std of each metric over
  folds and seeds (test split, full mode only).
* ``pairwise_delta_ci.csv`` -- mean AUROC delta and 95% bootstrap CI
  between every ordered pair of models (full mode only).
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .metrics import paired_delta_bootstrap_ci


_RUN_FILENAME = "run.json"
_TIDY_NAME = "results.csv"
_WIDE_NAME = "results_wide.csv"
_SUMMARY_NAME = "results_summary.csv"
_PAIRWISE_NAME = "pairwise_delta_ci.csv"
_FULL_MODE_PREFIX = "full__"
_FULL_MODE = "full"
_PAIRWISE_COLUMNS = [
    "mode",
    "model_a",
    "model_b",
    "n_folds",
    "delta_mean",
    "ci_low",
    "ci_high",
]


def aggregate_results(
    output_root: Path,
    *,
    allowed_models: set[str] | None = None,
) -> dict[str, Path]:
    """Aggregate every ``run.json`` under ``output_root/runs`` into CSVs.

    Args:
        output_root: ``paths.output_root`` from the training config.
        allowed_models: Optional whitelist of model names to include.
            When provided, ``run.json`` files whose parent model
            directory is not in this set are ignored. ``None`` keeps
            the legacy behaviour of ingesting every model directory on
            disk. The orchestrator passes ``set(cfg.models.keys())`` so
            commenting a model out of the YAML drops it from the CSVs
            and downstream plots without touching disk.

    Returns:
        Map from logical name (``tidy``, ``wide``, ``summary``,
        ``pairwise``) to the written file path.
    """

    runs_root = output_root / "runs"
    if not runs_root.is_dir():
        raise FileNotFoundError(f"runs directory missing: {runs_root}")

    records = list(_iter_run_records(runs_root, allowed_models=allowed_models))
    if not records:
        scope = (
            f" with allowed_models={sorted(allowed_models)!r}"
            if allowed_models is not None
            else ""
        )
        raise RuntimeError(f"no run.json files found under {runs_root}{scope}")

    tidy_df = _build_tidy(records)
    wide_df = _build_wide(records)
    summary_df = _build_summary(wide_df)
    pairwise_df = _build_pairwise(wide_df)

    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "tidy": output_root / _TIDY_NAME,
        "wide": output_root / _WIDE_NAME,
        "summary": output_root / _SUMMARY_NAME,
        "pairwise": output_root / _PAIRWISE_NAME,
    }
    tidy_df.to_csv(paths["tidy"], index=False)
    wide_df.to_csv(paths["wide"], index=False)
    summary_df.to_csv(paths["summary"], index=False)
    pairwise_df.to_csv(paths["pairwise"], index=False)
    return paths


def _iter_run_records(
    runs_root: Path,
    *,
    allowed_models: set[str] | None = None,
) -> Iterable[dict[str, Any]]:
    """Yield one record per ``run.json`` found under ``runs_root``.

    The on-disk layout is ``runs_root/<model>/<run_dir>/run.json`` so
    the model name is the first path component below ``runs_root``.
    Filtering on the directory avoids loading JSON for excluded models
    when ``allowed_models`` is set.
    """

    for run_json in runs_root.glob("*/*/run.json"):
        model_dir = run_json.parent.parent.name
        if allowed_models is not None and model_dir not in allowed_models:
            continue
        with run_json.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["_run_dir"] = run_json.parent.name
        payload["_mode"] = (
            "full" if run_json.parent.name.startswith(_FULL_MODE_PREFIX) else "test"
        )
        yield payload


def _build_tidy(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Tidy long-form dataframe over (model, fold, seed, lr, split, metric)."""

    rows: list[dict[str, Any]] = []
    for record in records:
        for split_name in ("val", "test"):
            metrics = record["metrics"][split_name]
            for metric_name, value in metrics.items():
                rows.append(
                    {
                        "mode": record["_mode"],
                        "model": record["model"],
                        "fold": record["fold"],
                        "seed": record["seed"],
                        "learning_rate": record["learning_rate"],
                        "split": split_name,
                        "metric": metric_name,
                        "value": float(value)
                        if value is not None and np.isfinite(value)
                        else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def _build_wide(records: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per (mode, model, fold, seed, lr) with all test metrics."""

    rows: list[dict[str, Any]] = []
    for record in records:
        test = record["metrics"]["test"]
        rows.append(
            {
                "mode": record["_mode"],
                "model": record["model"],
                "fold": record["fold"],
                "seed": record["seed"],
                "learning_rate": record["learning_rate"],
                "n_test": record.get("n_test"),
                **{f"test_{k}": float(v) for k, v in test.items()},
            }
        )
    return pd.DataFrame(rows)


def _build_summary(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Per-mode, per-model mean/std summary over folds and seeds.

    Includes every mode that produced runs. Test-mode rows are kept so a
    smoke-only state still produces a non-empty summary; the ``mode``
    column makes the distinction explicit so test results cannot be
    mistaken for reportable full-sweep results.
    """

    if wide_df.empty:
        return pd.DataFrame(
            columns=["mode", "model", "n_runs", "metric", "mean", "std"]
        )
    metric_cols = [c for c in wide_df.columns if c.startswith("test_")]
    rows: list[dict[str, Any]] = []
    for (mode_name, model_name), group in wide_df.groupby(["mode", "model"]):
        for col in metric_cols:
            values = group[col].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            rows.append(
                {
                    "mode": mode_name,
                    "model": model_name,
                    "n_runs": int(values.size),
                    "metric": col.removeprefix("test_"),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _build_pairwise(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise AUROC delta + 95% bootstrap CI between models.

    Prefers full-mode rows when present so reportable deltas never get
    contaminated by smoke runs. Falls back to whatever modes exist so a
    smoke-only state still produces non-empty output. The ``mode`` column
    records which scope was used.
    """

    if wide_df.empty or "test_auroc" not in wide_df.columns:
        return pd.DataFrame(columns=_PAIRWISE_COLUMNS)
    if (wide_df["mode"] == _FULL_MODE).any():
        scope = wide_df[wide_df["mode"] == _FULL_MODE]
        scope_label = _FULL_MODE
    else:
        scope = wide_df
        scope_label = "+".join(sorted(scope["mode"].dropna().unique().tolist()))
    per_seed_avg = (
        scope.groupby(["model", "fold"])["test_auroc"].mean().reset_index()
    )
    pivot = per_seed_avg.pivot(index="fold", columns="model", values="test_auroc")
    rows: list[dict[str, Any]] = []
    for model_a, model_b in combinations(sorted(pivot.columns), 2):
        paired = pivot[[model_a, model_b]].dropna()
        if paired.empty:
            continue
        deltas = (paired[model_a] - paired[model_b]).to_numpy(dtype=float)
        mean_delta, ci_low, ci_high = paired_delta_bootstrap_ci(deltas)
        rows.append(
            {
                "mode": scope_label,
                "model_a": model_a,
                "model_b": model_b,
                "n_folds": int(deltas.size),
                "delta_mean": mean_delta,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    if not rows:
        return pd.DataFrame(columns=_PAIRWISE_COLUMNS)
    return pd.DataFrame(rows, columns=_PAIRWISE_COLUMNS)
