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
from matplotlib.lines import Line2D

from .metrics import paired_delta_bootstrap_ci


_DPI = 300
_BAR_COLOR = "#6baed6"
_BAR_EDGE_COLOR = "#1f4e79"
_REF_LINE_COLOR = "#c1121f"
_BOOTSTRAP_ITERS = 2000
_FULL_MODE = "full"
_FOOTER_FONTSIZE = 8
_FOOTER_COLOR = "#666666"
_LEGEND_FONTSIZE = 8
_SEED_LINESTYLES = ("-", "--", ":", "-.")


def _add_mode_footer(fig, mode_label: str) -> None:
    """Annotate the figure with the run mode in a bottom-right footnote.

    The mode is metadata about which subset of runs the figure was built
    from (``full`` for reportable, ``test`` for smoke). It belongs in
    the figure metadata band, not in any axes title.
    """

    fig.text(
        0.99,
        0.005,
        f"mode={mode_label}",
        ha="right",
        va="bottom",
        fontsize=_FOOTER_FONTSIZE,
        color=_FOOTER_COLOR,
    )


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
    ax.set_ylabel(f"Test {primary_metric.upper()}")
    ax.set_title(f"Model comparison on test {primary_metric.upper()}")
    ax.set_ylim(0, 1.0)
    bar_handle = Line2D(
        [0], [0], color=_BAR_COLOR, marker="s", linestyle="", markersize=10
    )
    ax.legend(
        [bar_handle],
        ["mean ± 95% bootstrap CI (over fold × seed)"],
        loc="upper right",
        fontsize=_LEGEND_FONTSIZE,
    )
    fig.tight_layout()
    _add_mode_footer(fig, mode_label)
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
        ax.set_title(model)
        if values.size == 0:
            ax.text(
                0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes
            )
            continue
        jitter = rng.uniform(-0.05, 0.05, size=values.size)
        scatter_handle = ax.scatter(
            jitter, values, color=_BAR_EDGE_COLOR, alpha=0.7
        )
        mean_value = float(values.mean())
        mean_handle = ax.axhline(
            mean_value, color=_REF_LINE_COLOR, linestyle="--", linewidth=1
        )
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.legend(
            [scatter_handle, mean_handle],
            [f"(fold, seed) cell  (n={values.size})", f"mean = {mean_value:.3f}"],
            loc="lower right",
            fontsize=_LEGEND_FONTSIZE,
        )
    axes[0].set_ylabel(f"Test {primary_metric.upper()}")
    fig.suptitle("Seed stability of test AUROC")
    fig.tight_layout()
    _add_mode_footer(fig, mode_label)
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
        ax.set_title(model)
        if sub.empty:
            ax.text(
                0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes
            )
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
        ax.set_xlabel(
            f"best (fold, seed) = ({int(best_row['fold'])}, {int(best_row['seed'])})"
        )
    fig.suptitle("Best-run confusion matrix")
    fig.tight_layout()
    _add_mode_footer(fig, mode_label)
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
    folds = sorted(int(f) for f in wide_df["fold"].unique())
    seeds = sorted(int(s) for s in wide_df["seed"].unique())
    fold_to_color = {
        fold: plt.cm.viridis(t)
        for fold, t in zip(folds, np.linspace(0.15, 0.85, max(len(folds), 1)))
    }
    seed_to_linestyle = {
        seed: _SEED_LINESTYLES[i % len(_SEED_LINESTYLES)]
        for i, seed in enumerate(seeds)
    }
    fig, axes = _make_per_model_axes(len(models), figsize_per=(5, 4))
    for ax, model in zip(axes, models):
        sub = wide_df[wide_df["model"] == model]
        ax.set_title(model)
        ax.set_xlabel("epoch")
        ax.set_ylabel("val AUROC")
        if sub.empty:
            ax.text(
                0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes
            )
            continue
        ax_epochs: set[int] = set()
        for _, row in sub.iterrows():
            fold = int(row["fold"])
            seed = int(row["seed"])
            record = _load_run_record(
                runs_root, model=model, fold=fold, seed=seed
            )
            epochs, val_auroc = _val_auroc_history(record)
            if epochs:
                ax.plot(
                    epochs,
                    val_auroc,
                    color=fold_to_color[fold],
                    linestyle=seed_to_linestyle[seed],
                    linewidth=1.4,
                    alpha=0.85,
                )
                ax_epochs.update(
                    int(round(e)) for e in epochs if abs(e - round(e)) < 1e-6
                )
        if ax_epochs:
            ax.set_xticks(sorted(ax_epochs))
    fold_handles = [
        Line2D([0], [0], color=fold_to_color[f], lw=2, label=f"fold {f}")
        for f in folds
    ]
    seed_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=seed_to_linestyle[s],
            lw=1.4,
            label=f"seed {s}",
        )
        for s in seeds
    ]
    axes[-1].legend(
        handles=fold_handles + seed_handles,
        loc="lower right",
        fontsize=_LEGEND_FONTSIZE,
        ncol=2,
        framealpha=0.9,
    )
    fig.suptitle("Validation AUROC per epoch")
    fig.tight_layout()
    _add_mode_footer(fig, mode_label)
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
        ax.set_title(model)
        if values.size == 0:
            ax.text(
                0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes
            )
            continue
        mean_ece = float(values.mean())
        std_ece = float(values.std(ddof=1)) if values.size > 1 else 0.0
        bar_handle = ax.bar(
            [0], [mean_ece], color=_BAR_COLOR, edgecolor=_BAR_EDGE_COLOR
        )
        ax.set_xticks([0])
        ax.set_xticklabels(["mean ECE"])
        y_top = max(0.2, float(values.max()) * 1.2)
        ax.set_ylim(0, y_top)
        ax.text(
            0,
            mean_ece + 0.01 * y_top,
            f"{mean_ece:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax.legend(
            [bar_handle],
            [f"mean ± std = {mean_ece:.3f} ± {std_ece:.3f}  (n={values.size})"],
            loc="upper right",
            fontsize=_LEGEND_FONTSIZE,
        )
    fig.suptitle("Calibration summary")
    fig.tight_layout()
    _add_mode_footer(fig, mode_label)
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
            "No model pairs available\n(at least 2 models required)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        x = np.arange(len(pair_labels))
        err_low = [m - lo for m, lo in zip(means, lows)]
        err_high = [hi - m for hi, m in zip(highs, means)]
        bar_handle = ax.bar(
            x,
            means,
            yerr=[err_low, err_high],
            color=_BAR_COLOR,
            edgecolor=_BAR_EDGE_COLOR,
            capsize=4,
        )
        zero_handle = ax.axhline(
            0.0, color=_REF_LINE_COLOR, linestyle="--", linewidth=1
        )
        ax.set_xticks(x)
        ax.set_xticklabels(pair_labels)
        ax.legend(
            [bar_handle, zero_handle],
            ["mean Δ ± 95% bootstrap CI", "no-difference reference"],
            loc="best",
            fontsize=_LEGEND_FONTSIZE,
        )
    ax.set_ylabel("Test AUROC delta (model A − model B)")
    ax.set_title("Pairwise model comparison")
    fig.tight_layout()
    _add_mode_footer(fig, mode_label)
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
