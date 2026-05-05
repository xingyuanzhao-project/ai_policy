"""Render the six summary PNGs from the aggregated CSVs.

Matches the matplotlib-only convention used by ``scripts/desc_plots.py``.
All figures are written to ``plotting.figures_dir`` from the YAML.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import paired_delta_bootstrap_ci


_DPI = 300
_BAR_COLOR = "#6baed6"
_BAR_EDGE_COLOR = "#1f4e79"
_REF_LINE_COLOR = "#c1121f"
_BOOTSTRAP_ITERS = 2000
_FULL_MODE = "full"


def _make_per_model_axes(n_models: int, *, figsize_per: tuple[float, float], sharey: bool = False):
    """Return a (fig, axes_list) tuple where ``axes_list`` is always a list."""

    n = max(n_models, 1)
    fig, axes = plt.subplots(
        1, n, figsize=(figsize_per[0] * n, figsize_per[1]), sharey=sharey
    )
    if n == 1:
        axes = [axes]
    else:
        axes = list(axes)
    return fig, axes


def _empty_placeholder(figures_dir: Path, filename: str, message: str) -> Path:
    """Render a placeholder PNG with the given message at the given path."""

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    out_path = figures_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    return out_path


def render_plots(
    *,
    output_root: Path,
    figures_dir: Path,
    primary_metric: str = "auroc",
) -> dict[str, Path]:
    """Render the six comparison plots from the aggregated results.

    Args:
        output_root: ``paths.output_root`` (where ``results_wide.csv``
            and the per-run JSONs live).
        figures_dir: Output directory for the PNGs.
        primary_metric: Metric used by ``01_model_comparison_*``; the
            other plots draw the metrics they need directly.

    Returns:
        Map from short logical name to written PNG path.
    """

    figures_dir.mkdir(parents=True, exist_ok=True)
    wide_path = output_root / "results_wide.csv"
    if not wide_path.is_file():
        raise FileNotFoundError(
            f"results_wide.csv missing; run aggregate_results first ({wide_path})"
        )
    wide_df = pd.read_csv(wide_path)
    wide_df, mode_label = _select_plot_rows(wide_df)

    runs_root = output_root / "runs"
    produced: dict[str, Path] = {}

    produced["01"] = _plot_model_comparison(
        wide_df=wide_df,
        figures_dir=figures_dir,
        primary_metric=primary_metric,
        mode_label=mode_label,
    )
    produced["02"] = _plot_seed_stability(
        wide_df=wide_df,
        figures_dir=figures_dir,
        primary_metric=primary_metric,
        mode_label=mode_label,
    )
    produced["03"] = _plot_confusion_matrices(
        runs_root=runs_root,
        wide_df=wide_df,
        figures_dir=figures_dir,
        mode_label=mode_label,
    )
    produced["04"] = _plot_per_fold_curves(
        runs_root=runs_root,
        wide_df=wide_df,
        figures_dir=figures_dir,
        mode_label=mode_label,
    )
    produced["05"] = _plot_calibration(
        runs_root=runs_root,
        wide_df=wide_df,
        figures_dir=figures_dir,
        mode_label=mode_label,
    )
    produced["06"] = _plot_pairwise_delta(
        wide_df=wide_df,
        figures_dir=figures_dir,
        mode_label=mode_label,
    )
    return produced


def _select_plot_rows(wide_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Pick which rows to plot and return a label describing them.

    Prefers full-mode runs when present so reportable plots never get
    contaminated by smoke runs. Falls back to whatever modes exist so
    a smoke-only state still produces non-empty plots.
    """

    if wide_df.empty:
        return wide_df, "no runs"
    if (wide_df["mode"] == _FULL_MODE).any():
        return wide_df[wide_df["mode"] == _FULL_MODE].copy(), "full"
    modes = sorted(wide_df["mode"].dropna().unique().tolist())
    return wide_df.copy(), "+".join(modes) if modes else "unknown"


def _plot_model_comparison(
    *, wide_df: pd.DataFrame, figures_dir: Path, primary_metric: str, mode_label: str
) -> Path:
    """Plot 01: bar chart of mean primary-metric per model with 95% CI."""

    if wide_df.empty:
        return _empty_placeholder(
            figures_dir,
            "01_model_comparison_auroc.png",
            "No runs available",
        )
    metric_col = f"test_{primary_metric}"
    grouped = wide_df.groupby("model")[metric_col].apply(list)
    models = sorted(grouped.index)
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for model in models:
        values = np.asarray(grouped[model], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            means.append(float("nan"))
            lows.append(float("nan"))
            highs.append(float("nan"))
            continue
        rng = np.random.default_rng(20260504)
        idx = rng.integers(low=0, high=values.size, size=(_BOOTSTRAP_ITERS, values.size))
        sample_means = values[idx].mean(axis=1)
        means.append(float(values.mean()))
        lows.append(float(np.quantile(sample_means, 0.025)))
        highs.append(float(np.quantile(sample_means, 0.975)))

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(models))
    err_low = [m - lo for m, lo in zip(means, lows)]
    err_high = [hi - m for hi, m in zip(highs, means)]
    ax.bar(
        x,
        means,
        yerr=[err_low, err_high],
        color=_BAR_COLOR,
        edgecolor=_BAR_EDGE_COLOR,
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel(f"Test {primary_metric.upper()} (mean ± 95% bootstrap CI)")
    ax.set_title(
        f"Model comparison on test {primary_metric.upper()} (mode={mode_label})"
    )
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    out_path = figures_dir / "01_model_comparison_auroc.png"
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    return out_path


def _plot_seed_stability(
    *, wide_df: pd.DataFrame, figures_dir: Path, primary_metric: str, mode_label: str
) -> Path:
    """Plot 02: per-model strip of (fold, seed) test metric values."""

    if wide_df.empty:
        return _empty_placeholder(
            figures_dir, "02_seed_stability.png", "No runs available"
        )
    metric_col = f"test_{primary_metric}"
    models = sorted(wide_df["model"].unique())
    fig, axes = _make_per_model_axes(len(models), figsize_per=(4, 5), sharey=True)
    rng = np.random.default_rng(20260504)
    for ax, model in zip(axes, models):
        values = wide_df.loc[wide_df["model"] == model, metric_col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            ax.set_title(f"{model}\n(no data)")
            continue
        jitter = rng.uniform(-0.05, 0.05, size=values.size)
        ax.scatter(jitter, values, color=_BAR_EDGE_COLOR, alpha=0.7)
        ax.axhline(values.mean(), color=_REF_LINE_COLOR, linestyle="--", linewidth=1)
        ax.set_title(f"{model}\nmean={values.mean():.3f}")
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
    axes[0].set_ylabel(f"Test {primary_metric.upper()}")
    fig.suptitle(f"Seed stability across folds and seeds (mode={mode_label})")
    fig.tight_layout()
    out_path = figures_dir / "02_seed_stability.png"
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    return out_path


def _plot_confusion_matrices(
    *, runs_root: Path, wide_df: pd.DataFrame, figures_dir: Path, mode_label: str
) -> Path:
    """Plot 03: confusion matrix at the best (model, fold, seed) on test."""

    if wide_df.empty:
        return _empty_placeholder(
            figures_dir, "03_confusion_matrices.png", "No runs available"
        )
    models = sorted(wide_df["model"].unique())
    fig, axes = _make_per_model_axes(len(models), figsize_per=(4, 4))
    for ax, model in zip(axes, models):
        sub = wide_df[wide_df["model"] == model]
        if sub.empty:
            ax.set_title(f"{model}\n(no data)")
            continue
        best_row = sub.loc[sub["test_auroc"].idxmax()]
        record = _load_run_record(
            runs_root,
            model=model,
            fold=int(best_row["fold"]),
            seed=int(best_row["seed"]),
        )
        cm = _confusion_matrix_from_record(record)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred 0", "pred 1"])
        ax.set_yticklabels(["true 0", "true 1"])
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                )
        ax.set_title(
            f"{model}\nfold={int(best_row['fold'])}, seed={int(best_row['seed'])}"
        )
    fig.suptitle(
        f"Best-run confusion matrices, row-normalised (mode={mode_label})"
    )
    fig.tight_layout()
    out_path = figures_dir / "03_confusion_matrices.png"
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    return out_path


def _plot_per_fold_curves(
    *, runs_root: Path, wide_df: pd.DataFrame, figures_dir: Path, mode_label: str
) -> Path:
    """Plot 04: per-epoch train loss + val AUROC trace per (fold, seed)."""

    if wide_df.empty:
        return _empty_placeholder(
            figures_dir, "04_per_fold_curves.png", "No runs available"
        )
    models = sorted(wide_df["model"].unique())
    fig, axes = _make_per_model_axes(len(models), figsize_per=(5, 4))
    for ax, model in zip(axes, models):
        sub = wide_df[wide_df["model"] == model]
        if sub.empty:
            ax.set_title(f"{model}\n(no data)")
            continue
        for _, row in sub.iterrows():
            record = _load_run_record(
                runs_root,
                model=model,
                fold=int(row["fold"]),
                seed=int(row["seed"]),
            )
            epochs, val_auroc = _val_auroc_history(record)
            if epochs:
                ax.plot(epochs, val_auroc, alpha=0.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel("val AUROC")
        ax.set_title(model)
    fig.suptitle(
        f"Validation AUROC per epoch, all (fold, seed) cells (mode={mode_label})"
    )
    fig.tight_layout()
    out_path = figures_dir / "04_per_fold_curves.png"
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    return out_path


def _plot_calibration(
    *, runs_root: Path, wide_df: pd.DataFrame, figures_dir: Path, mode_label: str
) -> Path:
    """Plot 05: per-model ECE + reliability summary using stored metrics."""

    if wide_df.empty:
        return _empty_placeholder(
            figures_dir, "05_calibration_reliability.png", "No runs available"
        )
    models = sorted(wide_df["model"].unique())
    fig, axes = _make_per_model_axes(len(models), figsize_per=(4, 4))
    for ax, model in zip(axes, models):
        values = wide_df.loc[wide_df["model"] == model, "test_ece"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            ax.set_title(f"{model}\n(no data)")
            continue
        ax.bar([0], [values.mean()], color=_BAR_COLOR, edgecolor=_BAR_EDGE_COLOR)
        ax.set_xticks([0])
        ax.set_xticklabels(["mean ECE"])
        ax.set_ylim(0, max(0.2, values.max() * 1.2))
        ax.set_title(f"{model}\nmean ECE={values.mean():.3f}")
    fig.suptitle(
        f"Calibration summary, lower ECE = better (mode={mode_label})"
    )
    fig.tight_layout()
    out_path = figures_dir / "05_calibration_reliability.png"
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    return out_path


def _plot_pairwise_delta(
    *, wide_df: pd.DataFrame, figures_dir: Path, mode_label: str
) -> Path:
    """Plot 06: pairwise AUROC delta + 95% CI between every model pair."""

    if wide_df.empty:
        return _empty_placeholder(
            figures_dir, "06_pairwise_delta_ci.png", "No runs available"
        )
    models = sorted(wide_df["model"].unique())
    per_seed_avg = (
        wide_df.groupby(["model", "fold"])["test_auroc"].mean().reset_index()
    )
    pivot = per_seed_avg.pivot(index="fold", columns="model", values="test_auroc")

    pair_labels: list[str] = []
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for model_a, model_b in combinations(models, 2):
        if model_a not in pivot.columns or model_b not in pivot.columns:
            continue
        paired = pivot[[model_a, model_b]].dropna()
        if paired.empty:
            continue
        deltas = (paired[model_a] - paired[model_b]).to_numpy(dtype=float)
        mean_delta, ci_low, ci_high = paired_delta_bootstrap_ci(deltas)
        pair_labels.append(f"{model_a}\n−\n{model_b}")
        means.append(mean_delta)
        lows.append(ci_low)
        highs.append(ci_high)

    fig, ax = plt.subplots(figsize=(max(5, 2 * len(pair_labels)), 5))
    if not pair_labels:
        ax.text(
            0.5,
            0.5,
            "No model pairs available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        x = np.arange(len(pair_labels))
        err_low = [m - lo for m, lo in zip(means, lows)]
        err_high = [hi - m for hi, m in zip(highs, means)]
        ax.bar(
            x,
            means,
            yerr=[err_low, err_high],
            color=_BAR_COLOR,
            edgecolor=_BAR_EDGE_COLOR,
            capsize=4,
        )
        ax.axhline(0.0, color=_REF_LINE_COLOR, linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(pair_labels)
    ax.set_ylabel("Test AUROC delta (model_A − model_B)")
    ax.set_title(
        f"Pairwise model comparison with 95% bootstrap CI (mode={mode_label})"
    )
    fig.tight_layout()
    out_path = figures_dir / "06_pairwise_delta_ci.png"
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    return out_path


def _load_run_record(
    runs_root: Path, *, model: str, fold: int, seed: int
) -> dict:
    """Load the matching ``run.json`` for a (model, fold, seed) cell."""

    for run_json in (runs_root / model).glob("*/run.json"):
        with run_json.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if int(payload["fold"]) == fold and int(payload["seed"]) == seed:
            return payload
    raise FileNotFoundError(
        f"no run.json for model={model} fold={fold} seed={seed}"
    )


def _confusion_matrix_from_record(record: dict) -> np.ndarray:
    """Return the row-normalised test confusion matrix stored in the run."""

    raw = record.get("test_confusion_matrix")
    if raw is None:
        return np.zeros((2, 2))
    matrix = np.asarray(raw, dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums


def _val_auroc_history(record: dict) -> tuple[list[float], list[float]]:
    """Extract per-epoch validation AUROC trace from the run log history."""

    epochs: list[float] = []
    auroc: list[float] = []
    for entry in record.get("log_history", []):
        if "eval_auroc" in entry and "epoch" in entry:
            epochs.append(float(entry["epoch"]))
            auroc.append(float(entry["eval_auroc"]))
    return epochs, auroc
