"""Publication-quality plots for the nine-stage eval run.

- Reads the stage result JSONs emitted by :mod:`src.eval.evals` under
  ``output/evals/v1/results/`` and renders six focused plots to
  ``output/evals/v1/plots/``.
- Each plot answers one decision-relevant question; nothing decorative is
  added.  Captions in the docstring of each helper describe the exact
  claim the chart supports.
- Uses Matplotlib with the ``Agg`` backend so the script runs headless.
- Does not call the judge, does not regenerate any stage result, and
  never mutates input JSONs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 -- ``use`` must precede pyplot import

from .config import load_eval_config

logger = logging.getLogger(__name__)

_METHOD_COLOURS: dict[str, str] = {
    "orchestrated": "#1f4e79",
    "skill_driven": "#c1121f",
}
_FALLBACK_COLOURS = [
    "#1f4e79", "#c1121f", "#5b8e3a", "#b08400", "#6a4c93", "#8b6c42",
]
_FIG_W = 11.0
_FIG_H = 5.5
_TITLE_FONTSIZE = 13
_LABEL_FONTSIZE = 11
_ANNOT_FONTSIZE = 10


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m src.eval.eval_plots``.

    Args:
        argv: Optional argument list; defaults to :data:`sys.argv`.

    Returns:
        Process exit code.  ``0`` on success.
    """

    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        stream=sys.stdout,
    )
    run_dir = _resolve_run_dir(args.config, args.run_dir)
    render_all_plots(run_dir=run_dir)
    return 0


def render_all_plots(*, run_dir: Path) -> dict[str, Path]:
    """Render the six eval plots for a completed run directory.

    Args:
        run_dir: Absolute path to the run directory (``output/evals/v1``).

    Returns:
        Map of plot name to the absolute path written.
    """

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir = run_dir / "results"

    produced: dict[str, Path] = {}

    stage3 = _load_json(results_dir / "stage3_grounding.json")
    stage4 = _load_json(results_dir / "stage4_coverage.json")
    stage5 = _load_json(results_dir / "stage5_novelty.json")
    stage6 = _load_json(results_dir / "stage6_pairwise.json")
    stage8 = _load_json(results_dir / "stage8_bias.json")
    judge_usage = _load_json(run_dir / "judge_usage_summary.json")

    if stage4 is not None:
        produced["coverage"] = _plot_coverage_by_method(
            stage4, out_path=plots_dir / "01_coverage_by_method.png"
        )
    if stage3 is not None:
        produced["grounding"] = _plot_grounding_distribution(
            stage3, out_path=plots_dir / "02_grounding_distribution.png"
        )
    if stage4 is not None and judge_usage is not None:
        produced["cost"] = _plot_cost_vs_coverage(
            stage4=stage4,
            judge_usage=judge_usage,
            run_dir=run_dir,
            out_path=plots_dir / "03_cost_vs_coverage.png",
        )
    if stage6 is not None:
        produced["pairwise"] = _plot_pairwise_winrate(
            stage6, out_path=plots_dir / "04_pairwise_winrate.png"
        )
    if stage5 is not None:
        produced["novel_types"] = _plot_novel_type_breakdown(
            stage5, out_path=plots_dir / "05_novel_type_breakdown.png"
        )
    if stage8 is not None:
        produced["bias"] = _plot_bias_scorecard(
            stage8, out_path=plots_dir / "06_bias_scorecard.png"
        )

    logger.info(
        "Rendered %d plot(s) to %s", len(produced), plots_dir
    )
    return produced


def _plot_coverage_by_method(stage4: dict[str, Any], *, out_path: Path) -> Path:
    """Stage 4: coverage rate per method (covered / total judged labels).

    Plots two bars per method: ``strict`` (supporting_ids all survived
    Stage 3 grounding) and ``lenient`` (``covered`` verdict regardless of
    grounding).  Both denominators are the total count of label-level
    judgements the judge returned for that method.
    """

    per_method = stage4.get("per_method") or {}
    methods = list(per_method.keys())
    counts = [per_method[m].get("counts", {}) for m in methods]
    totals = [int(per_method[m].get("total_judged", 0)) for m in methods]
    strict_counts = [
        int(per_method[m].get("strict_covered_count", 0)) for m in methods
    ]
    lenient_counts = [int(c.get("covered", 0)) for c in counts]
    strict_rates = [
        strict_counts[i] / totals[i] if totals[i] else 0.0
        for i in range(len(methods))
    ]
    lenient_rates = [
        lenient_counts[i] / totals[i] if totals[i] else 0.0
        for i in range(len(methods))
    ]

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    x = range(len(methods))
    width = 0.36
    strict_bars = ax.bar(
        [i - width / 2 for i in x],
        strict_rates,
        width=width,
        color=[_colour_for(m) for m in methods],
        edgecolor="black",
        linewidth=0.7,
        label="Strict (grounded supporting ids)",
    )
    lenient_bars = ax.bar(
        [i + width / 2 for i in x],
        lenient_rates,
        width=width,
        color=[_shade(_colour_for(m), 0.55) for m in methods],
        edgecolor="black",
        linewidth=0.7,
        label="Lenient (judge verdict only)",
    )
    max_rate = max(strict_rates + lenient_rates, default=0.0)
    ax.set_ylim(0, max(0.05, max_rate * 1.25))
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods)
    ax.set_ylabel(
        "Coverage rate (covered / judged labels)", fontsize=_LABEL_FONTSIZE,
    )
    ax.set_title(
        "Stage 4 -- Label coverage by extraction method",
        fontsize=_TITLE_FONTSIZE,
    )
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _pos: f"{v:.0%}")
    )
    for bar, rate, count, total in zip(
        strict_bars, strict_rates, strict_counts, totals
    ):
        ax.annotate(
            f"{rate:.1%}\n({count}/{total})",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=_ANNOT_FONTSIZE,
        )
    for bar, rate, count, total in zip(
        lenient_bars, lenient_rates, lenient_counts, totals
    ):
        ax.annotate(
            f"{rate:.1%}\n({count}/{total})",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=_ANNOT_FONTSIZE,
        )
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _plot_grounding_distribution(
    stage3: dict[str, Any], *, out_path: Path
) -> Path:
    """Stage 3: entailed / neutral / contradicted proportions per method."""

    per_method = stage3.get("per_method") or {}
    methods = list(per_method.keys())
    labels = ("entailed", "neutral", "contradicted")
    label_colours = {
        "entailed": "#2a9d8f",
        "neutral": "#b8b8b8",
        "contradicted": "#c1121f",
    }

    counts = {
        label: [int(per_method[m].get("counts", {}).get(label, 0)) for m in methods]
        for label in labels
    }
    totals = [sum(counts[label][i] for label in labels) for i in range(len(methods))]
    percentages = {
        label: [
            counts[label][i] / totals[i] if totals[i] else 0.0
            for i in range(len(methods))
        ]
        for label in labels
    }

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    left = [0.0] * len(methods)
    for label in labels:
        widths = percentages[label]
        bars = ax.barh(
            methods,
            widths,
            left=left,
            color=label_colours[label],
            edgecolor="black",
            linewidth=0.5,
            label=label,
        )
        for bar, pct, count in zip(bars, widths, counts[label]):
            if pct < 0.04:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.0%}\n({count})",
                ha="center",
                va="center",
                fontsize=_ANNOT_FONTSIZE,
                color="white" if label != "neutral" else "black",
            )
        left = [l + w for l, w in zip(left, widths)]
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _pos: f"{v:.0%}")
    )
    ax.set_xlabel(
        "Share of Stage 2 survivors (n = bar total)",
        fontsize=_LABEL_FONTSIZE,
    )
    ax.set_title(
        "Stage 3 -- Per-quadruplet grounding verdict by method",
        fontsize=_TITLE_FONTSIZE,
    )
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_cost_vs_coverage(
    *,
    stage4: dict[str, Any],
    judge_usage: dict[str, Any],
    run_dir: Path,
    out_path: Path,
) -> Path:
    """Cost-per-covered-label, combining Stage 4 and the extractor usage files."""

    per_method = stage4.get("per_method") or {}
    methods = list(per_method.keys())
    covered_counts = [
        int(per_method[m].get("strict_covered_count", 0)) for m in methods
    ]

    extractor_cost = _load_extractor_costs(run_dir=run_dir, methods=methods)

    judge_cost_total = float(judge_usage.get("total_cost_usd", 0.0))
    total_covered = sum(covered_counts) or 1
    judge_cost_per_covered = [
        judge_cost_total * (c / total_covered) if total_covered else 0.0
        for c in covered_counts
    ]
    extractor_cost_per_covered = [
        extractor_cost.get(m, 0.0) / c if c else 0.0
        for m, c in zip(methods, covered_counts)
    ]

    fig, ax_left = plt.subplots(figsize=(_FIG_W, _FIG_H))
    x = range(len(methods))
    width = 0.38
    ax_left.bar(
        [i - width / 2 for i in x],
        extractor_cost_per_covered,
        width=width,
        color="#1f4e79",
        label="Extractor $ per covered label",
    )
    ax_left.bar(
        [i + width / 2 for i in x],
        judge_cost_per_covered,
        width=width,
        color="#b08400",
        label="Judge $ per covered label (amortised)",
    )
    ax_left.set_xticks(list(x))
    ax_left.set_xticklabels(methods)
    ax_left.set_ylabel("USD per covered label", fontsize=_LABEL_FONTSIZE)
    ax_left.set_title(
        "Cost-adjusted quality -- lower is better",
        fontsize=_TITLE_FONTSIZE,
    )
    for xa, xb, extr, judge in zip(
        [i - width / 2 for i in x],
        [i + width / 2 for i in x],
        extractor_cost_per_covered,
        judge_cost_per_covered,
    ):
        if extr > 0:
            ax_left.annotate(
                f"${extr:.3f}", xy=(xa, extr), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=_ANNOT_FONTSIZE,
            )
        if judge > 0:
            ax_left.annotate(
                f"${judge:.3f}", xy=(xb, judge), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=_ANNOT_FONTSIZE,
            )
    ax_left.grid(axis="y", linestyle=":", alpha=0.4)
    ax_left.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _plot_pairwise_winrate(
    stage6: dict[str, Any], *, out_path: Path
) -> Path:
    """Stage 6 swap-averaged win rates for one pair of methods.

    The chart shows the swap-averaged verdict distribution (A wins / tie /
    B wins) and also overlays the count-normalised points -- the latter
    corrects for the fact that a method which simply returns more
    quadruplets would otherwise accumulate more wins mechanically.
    """

    a = str(stage6.get("a_method") or "A")
    b = str(stage6.get("b_method") or "B")
    swap = stage6.get("swap_averaged_winrate") or {}
    if not swap:
        return _empty_plot(out_path, title="Stage 6 -- No pairwise comparisons ran")
    win_a = float(swap.get(a, 0.0))
    win_b = float(swap.get(b, 0.0))
    tie = float(swap.get("tie", max(0.0, 1.0 - win_a - win_b)))

    normalised = stage6.get("count_normalised_points") or {}
    norm_total = max(1.0, sum(float(v) for v in normalised.values()))
    norm_a = float(normalised.get(a, 0.0)) / norm_total
    norm_b = float(normalised.get(b, 0.0)) / norm_total

    labels = [f"{a} wins", "Tie", f"{b} wins"]
    win_rates = [win_a, tie, win_b]
    normalised_rates = [norm_a, 0.0, norm_b]
    colours = [_colour_for(a), "#b8b8b8", _colour_for(b)]

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    x = range(len(labels))
    width = 0.36
    bars = ax.bar(
        [i - width / 2 for i in x],
        win_rates,
        width=width,
        color=colours,
        edgecolor="black",
        linewidth=0.6,
        label="Swap-averaged win rate",
    )
    ax.bar(
        [i + width / 2 for i in x],
        normalised_rates,
        width=width,
        color=[_shade(c, 0.55) for c in colours],
        edgecolor="black",
        linewidth=0.6,
        label="Count-normalised share",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _pos: f"{v:.0%}")
    )
    ax.set_ylabel("Share of judged pairs", fontsize=_LABEL_FONTSIZE)
    n_pairs = int(stage6.get("n_pairs", 0))
    ax.set_title(
        f"Stage 6 -- Pairwise winners ({a} vs {b}), swap-averaged (n={n_pairs})",
        fontsize=_TITLE_FONTSIZE,
    )
    for bar, value in zip(bars, win_rates):
        ax.annotate(
            f"{value:.1%}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=_ANNOT_FONTSIZE,
        )
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _plot_novel_type_breakdown(
    stage5: dict[str, Any], *, out_path: Path
) -> Path:
    """Stage 5 top-10 novel entity types per method (grouped horizontal bars)."""

    per_method = stage5.get("per_method") or {}
    if not per_method:
        return _empty_plot(out_path, title="Stage 5 -- No novelty data")

    all_types: list[str] = []
    seen: set[str] = set()
    for method_name, payload in per_method.items():
        top = payload.get("by_type_top10") or []
        for entry in top:
            t = str(entry[0])
            if t not in seen:
                seen.add(t)
                all_types.append(t)
    if not all_types:
        return _empty_plot(out_path, title="Stage 5 -- No novel quadruplets")
    all_types = all_types[:10]
    methods = list(per_method.keys())

    data: dict[str, list[int]] = {}
    for m in methods:
        lookup = dict(per_method[m].get("by_type_top10") or [])
        data[m] = [int(lookup.get(t, 0)) for t in all_types]

    fig, ax = plt.subplots(figsize=(_FIG_W, max(_FIG_H, len(all_types) * 0.45)))
    y = list(range(len(all_types)))
    bar_h = 0.8 / max(1, len(methods))
    for i, m in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * bar_h
        bars = ax.barh(
            [yi - offset for yi in y],
            data[m],
            height=bar_h,
            color=_colour_for(m),
            edgecolor="black",
            linewidth=0.5,
            label=m,
        )
        for bar, val in zip(bars, data[m]):
            if val <= 0:
                continue
            ax.annotate(
                f"{val}",
                xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(3, 0),
                textcoords="offset points",
                va="center",
                fontsize=_ANNOT_FONTSIZE,
            )
    ax.set_yticks(y)
    ax.set_yticklabels(all_types)
    ax.invert_yaxis()
    ax.set_xlabel("Novel quadruplet count", fontsize=_LABEL_FONTSIZE)
    ax.set_title(
        "Stage 5 -- Top-10 novel entity types by method",
        fontsize=_TITLE_FONTSIZE,
    )
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _plot_bias_scorecard(
    stage8: dict[str, Any], *, out_path: Path
) -> Path:
    """Stage 8 CALM bias-flip rates per (method, bias)."""

    per_method = stage8.get("per_method") or {}
    biases = stage8.get("biases") or []
    if not per_method or not biases:
        return _empty_plot(out_path, title="Stage 8 -- Bias audit not run")

    methods = list(per_method.keys())
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    width = 0.8 / max(1, len(methods))
    x = range(len(biases))
    for i, m in enumerate(methods):
        rates: list[float] = []
        for bias in biases:
            counters = (per_method[m].get(bias) or {})
            total = int(counters.get("total", 0))
            flips = int(counters.get("flips", 0))
            rates.append(flips / total if total else 0.0)
        bars = ax.bar(
            [xi + (i - (len(methods) - 1) / 2) * width for xi in x],
            rates,
            width=width,
            color=_colour_for(m),
            edgecolor="black",
            linewidth=0.5,
            label=m,
        )
        for bar, rate in zip(bars, rates):
            ax.annotate(
                f"{rate:.0%}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=_ANNOT_FONTSIZE,
            )
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(b).replace("_", " ") for b in biases])
    ax.set_ylabel("Verdict flip rate", fontsize=_LABEL_FONTSIZE)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _pos: f"{v:.0%}")
    )
    ax.set_title(
        "Stage 8 -- Judge bias flip rate by CALM perturbation",
        fontsize=_TITLE_FONTSIZE,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _empty_plot(out_path: Path, *, title: str) -> Path:
    """Render a single-panel "no data" placeholder plot."""

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    ax.text(
        0.5, 0.5, title, ha="center", va="center",
        fontsize=_TITLE_FONTSIZE,
        transform=ax.transAxes,
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, returning ``None`` when it is absent."""

    if not path.is_file():
        logger.warning("Missing artefact: %s", path)
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_extractor_costs(
    *, run_dir: Path, methods: Iterable[str]
) -> dict[str, float]:
    """Look up the extractor ``total_cost_usd`` for each method."""

    config_path = _find_eval_config(run_dir)
    if config_path is None:
        return {}
    config = load_eval_config(config_path)
    costs: dict[str, float] = {}
    for method_name in methods:
        method = config.methods.get(method_name)
        if method is None or not method.usage_summary.is_file():
            continue
        with method.usage_summary.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        costs[method_name] = float(data.get("total_cost_usd") or 0.0)
    return costs


def _find_eval_config(run_dir: Path) -> Path | None:
    """Locate ``settings/eval/eval.yml`` relative to the run directory."""

    for parent in [run_dir, *run_dir.parents]:
        candidate = parent / "settings" / "eval" / "eval.yml"
        if candidate.is_file():
            return candidate
    return None


def _resolve_run_dir(config_path: str, run_dir_arg: str | None) -> Path:
    """Resolve which run directory the plots should read from."""

    if run_dir_arg:
        path = Path(run_dir_arg)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()
    config = load_eval_config(Path(config_path))
    return config.output_run_dir


def _colour_for(method: str) -> str:
    """Look up a stable colour for a method name, falling back deterministically."""

    if method in _METHOD_COLOURS:
        return _METHOD_COLOURS[method]
    index = abs(hash(method)) % len(_FALLBACK_COLOURS)
    return _FALLBACK_COLOURS[index]


def _shade(hex_colour: str, factor: float) -> str:
    """Lighten a ``#RRGGBB`` colour by ``factor`` (0.0=same, 1.0=white)."""

    h = hex_colour.lstrip("#")
    if len(h) != 6:
        return hex_colour
    factor = max(0.0, min(1.0, factor))
    rgb = tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
    lightened = tuple(
        int(c + (255 - c) * factor) for c in rgb
    )
    return "#{:02x}{:02x}{:02x}".format(*lightened)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Build and parse the plot-only CLI arguments."""

    parser = argparse.ArgumentParser(
        prog="python -m src.eval.eval_plots",
        description=(
            "Render the six summary plots for a completed eval run."
        ),
    )
    parser.add_argument(
        "--config",
        default="settings/eval/eval.yml",
        help="Path to the eval YAML config (used to locate the run dir).",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Explicit run directory (overrides the config's output.run_dir). "
            "Pass when plotting an older archived run."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
