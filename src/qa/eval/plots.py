"""Comparison plots for V1 / V2 / V3 QA eval runs.

- Reads each version's ``results.json`` and renders five decision-oriented
  figures into ``output/evals_app/_comparison/`` plus one per-version mini
  figure so every version directory is self-contained.
- Uses Matplotlib with the ``Agg`` backend so the script runs headless.
- Does not re-grade questions or mutate input JSONs; pure read + render.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 -- ``use`` must precede pyplot import

_FIG_SIZE_WIDE = (10.0, 5.5)
_FIG_SIZE_TALL = (10.0, 7.0)
_DIFFICULTY_ORDER = ("easy", "medium", "hard", "very_hard")


def render_comparison_plots(
    *,
    version_results: dict[str, Path],
    output_dir: Path,
    per_version_output_root: Path | None = None,
) -> None:
    """Render all comparison plots plus per-version mini figures.

    Args:
        version_results: Mapping of version key (e.g. ``"v1"``) to that
            version's ``results.json`` path.
        output_dir: Directory that will receive the cross-version PNGs.
        per_version_output_root: Optional parent directory whose ``<version>``
            subfolders should each receive a single-version ``pass_by_difficulty.png``.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, dict[str, Any]] = {
        version: _load_results(path) for version, path in version_results.items()
    }
    versions = list(loaded.keys())

    _plot_overall_pass_rate(loaded, versions, output_dir / "01_overall_pass_rate.png")
    _plot_pass_rate_by_difficulty(loaded, versions, output_dir / "02_pass_rate_by_difficulty.png")
    _plot_mean_scores(loaded, versions, output_dir / "03_mean_scores.png")
    _plot_per_question_heatmap(loaded, versions, output_dir / "04_per_question_heatmap.png")
    _plot_latency_box(loaded, versions, output_dir / "05_latency_box.png")

    if per_version_output_root is not None:
        for version, payload in loaded.items():
            target_dir = per_version_output_root / version
            target_dir.mkdir(parents=True, exist_ok=True)
            _plot_single_version_difficulty_bars(
                version=version,
                payload=payload,
                output_path=target_dir / "pass_by_difficulty.png",
            )


def _load_results(path: Path) -> dict[str, Any]:
    """Load one version's ``results.json`` payload."""

    return json.loads(path.read_text(encoding="utf-8"))


def _plot_overall_pass_rate(
    loaded: dict[str, dict[str, Any]],
    versions: list[str],
    output_path: Path,
) -> None:
    """Grouped bar of overall pass rate across versions."""

    labels = [_version_display_label(loaded[v], v) for v in versions]
    pass_rates = [
        float(loaded[v].get("summary", {}).get("overall_pass_rate", 0.0))
        for v in versions
    ]
    n_questions = [
        int(loaded[v].get("summary", {}).get("n_questions", 0))
        for v in versions
    ]

    fig, ax = plt.subplots(figsize=_FIG_SIZE_WIDE)
    positions = range(len(versions))
    bars = ax.bar(positions, pass_rates)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("Pass rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Overall pass rate by QA version")
    for rect, rate, n in zip(bars, pass_rates, n_questions):
        ax.annotate(
            f"{rate:.0%}\n(n={n})",
            xy=(rect.get_x() + rect.get_width() / 2, rate),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_pass_rate_by_difficulty(
    loaded: dict[str, dict[str, Any]],
    versions: list[str],
    output_path: Path,
) -> None:
    """Grouped bar of pass rate by difficulty, versions side by side per difficulty."""

    difficulties = _collect_difficulties(loaded)
    if not difficulties:
        return

    fig, ax = plt.subplots(figsize=_FIG_SIZE_WIDE)
    bar_width = 0.8 / max(len(versions), 1)
    for version_index, version in enumerate(versions):
        by_diff = loaded[version].get("summary", {}).get("by_difficulty", {})
        rates = [float(by_diff.get(d, {}).get("pass_rate", 0.0)) for d in difficulties]
        positions = [
            difficulty_index + version_index * bar_width
            for difficulty_index in range(len(difficulties))
        ]
        label = _version_display_label(loaded[version], version)
        ax.bar(positions, rates, width=bar_width, label=label)

    center_offset = bar_width * (len(versions) - 1) / 2
    ax.set_xticks([i + center_offset for i in range(len(difficulties))])
    ax.set_xticklabels(difficulties)
    ax.set_ylabel("Pass rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Pass rate by difficulty x QA version")
    ax.legend(loc="best")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_mean_scores(
    loaded: dict[str, dict[str, Any]],
    versions: list[str],
    output_path: Path,
) -> None:
    """Grouped bar of mean keyword score, mean citation F1, mean count score."""

    metrics = [
        ("mean_keyword_score", "Mean keyword score"),
        ("mean_citation_f1", "Mean citation F1"),
        ("mean_count_score", "Mean count score"),
    ]

    fig, ax = plt.subplots(figsize=_FIG_SIZE_WIDE)
    bar_width = 0.8 / max(len(versions), 1)
    for version_index, version in enumerate(versions):
        by_diff = loaded[version].get("summary", {}).get("by_difficulty", {})
        values = [
            _mean_over_difficulties(by_diff, metric_key)
            for metric_key, _ in metrics
        ]
        positions = [metric_index + version_index * bar_width for metric_index in range(len(metrics))]
        label = _version_display_label(loaded[version], version)
        ax.bar(positions, values, width=bar_width, label=label)

    center_offset = bar_width * (len(versions) - 1) / 2
    ax.set_xticks([i + center_offset for i in range(len(metrics))])
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Mean score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Mean quality metrics by QA version")
    ax.legend(loc="best")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_per_question_heatmap(
    loaded: dict[str, dict[str, Any]],
    versions: list[str],
    output_path: Path,
) -> None:
    """Pass/fail matrix -- rows = question ids, cols = versions."""

    question_order = _collect_question_order(loaded)
    if not question_order:
        return

    matrix: list[list[float]] = []
    row_labels: list[str] = []
    for qid in question_order:
        row: list[float] = []
        for version in versions:
            flag = _lookup_pass_flag(loaded[version], qid)
            row.append(1.0 if flag is True else 0.0 if flag is False else float("nan"))
        matrix.append(row)
        row_labels.append(qid)

    fig_height = max(_FIG_SIZE_TALL[1], 0.32 * len(question_order) + 1.5)
    fig, ax = plt.subplots(figsize=(_FIG_SIZE_TALL[0], fig_height))
    im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="RdYlGn")
    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels(
        [_version_display_label(loaded[v], v) for v in versions],
        rotation=0,
        ha="center",
    )
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title("Per-question pass/fail across QA versions (1 = pass, 0 = fail)")
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            text = "YES" if value == 1.0 else ("no" if value == 0.0 else "-")
            ax.text(
                col_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    fig.colorbar(im, ax=ax, shrink=0.6, label="passed")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_latency_box(
    loaded: dict[str, dict[str, Any]],
    versions: list[str],
    output_path: Path,
) -> None:
    """Box plot of per-question latency by version."""

    latencies_by_version: list[list[float]] = []
    labels: list[str] = []
    for version in versions:
        latencies = [
            float(score.get("latency_seconds", 0.0))
            for score in loaded[version].get("scores", [])
            if isinstance(score, dict)
        ]
        latencies_by_version.append(latencies)
        labels.append(_version_display_label(loaded[version], version))

    if not any(latencies_by_version):
        return

    fig, ax = plt.subplots(figsize=_FIG_SIZE_WIDE)
    ax.boxplot(latencies_by_version, tick_labels=labels, showmeans=True)
    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Per-question latency by QA version")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_single_version_difficulty_bars(
    *,
    version: str,
    payload: dict[str, Any],
    output_path: Path,
) -> None:
    """Per-version pass-rate-by-difficulty bar chart (self-contained)."""

    by_diff = payload.get("summary", {}).get("by_difficulty", {})
    difficulties = [d for d in _DIFFICULTY_ORDER if d in by_diff] + [
        d for d in by_diff.keys() if d not in _DIFFICULTY_ORDER
    ]
    if not difficulties:
        return
    rates = [float(by_diff.get(d, {}).get("pass_rate", 0.0)) for d in difficulties]
    counts = [int(by_diff.get(d, {}).get("count", 0)) for d in difficulties]

    fig, ax = plt.subplots(figsize=_FIG_SIZE_WIDE)
    positions = range(len(difficulties))
    bars = ax.bar(positions, rates)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(difficulties)
    ax.set_ylabel("Pass rate")
    ax.set_ylim(0.0, 1.05)
    label = _version_display_label(payload, version)
    ax.set_title(f"Pass rate by difficulty -- {label}")
    for rect, rate, n in zip(bars, rates, counts):
        ax.annotate(
            f"{rate:.0%}\n(n={n})",
            xy=(rect.get_x() + rect.get_width() / 2, rate),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _version_display_label(payload: dict[str, Any], version: str) -> str:
    """Return a compact bar/legend label built from version + brief label."""

    label = str(payload.get("version_label", "")).strip()
    if not label:
        return version
    short = label.split("--")[0].strip() or label
    if version.lower() not in short.lower():
        return f"{version} ({short})"
    return short


def _mean_over_difficulties(by_diff: dict[str, dict[str, Any]], key: str) -> float:
    """Average a per-difficulty metric across difficulties present."""

    values = [
        float(stats.get(key, 0.0))
        for stats in by_diff.values()
        if key in stats
    ]
    return sum(values) / len(values) if values else 0.0


def _collect_difficulties(loaded: dict[str, dict[str, Any]]) -> list[str]:
    """Union of difficulties seen across versions, preferred order first."""

    seen: list[str] = []
    for version_payload in loaded.values():
        by_diff = version_payload.get("summary", {}).get("by_difficulty", {})
        for difficulty in by_diff.keys():
            if difficulty not in seen:
                seen.append(difficulty)
    return [d for d in _DIFFICULTY_ORDER if d in seen] + [
        d for d in seen if d not in _DIFFICULTY_ORDER
    ]


def _collect_question_order(loaded: dict[str, dict[str, Any]]) -> list[str]:
    """Union of question ids across versions, preserving first-seen order."""

    seen: list[str] = []
    for version_payload in loaded.values():
        for score in version_payload.get("scores", []):
            qid = score.get("question_id")
            if qid and qid not in seen:
                seen.append(qid)
    return seen


def _lookup_pass_flag(payload: dict[str, Any], qid: str) -> bool | None:
    """Return the pass flag for ``qid`` from one version payload, or ``None``."""

    for score in payload.get("scores", []):
        if score.get("question_id") == qid:
            return bool(score.get("passed", False))
    return None


__all__ = ["render_comparison_plots"]
