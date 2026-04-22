"""Descriptive plots for the NCSL state AI legislation corpus.

Reads the NCSL metadata CSV and the NCSL text CSV, then produces five
PNGs under ``plot/`` describing the corpus along four dimensions
(year, state, status, topic) plus the text-length distribution.

Run from the project root:

    .\\.venv\\Scripts\\python.exe scripts\\desc_plots.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
META_CSV_PATH = PROJECT_ROOT / "data" / "ncsl" / "us_ai_legislation_ncsl_meta.csv"
TEXT_CSV_PATH = PROJECT_ROOT / "data" / "ncsl" / "us_ai_legislation_ncsl_text.csv"
OUTPUT_DIR = PROJECT_ROOT / "plot"

YEAR_PLOT_PATH = OUTPUT_DIR / "desc_bill_count_by_year.png"
STATE_PLOT_PATH = OUTPUT_DIR / "desc_bill_count_by_state.png"
STATUS_PLOT_PATH = OUTPUT_DIR / "desc_bill_status.png"
TOPIC_PLOT_PATH = OUTPUT_DIR / "desc_topic_coverage.png"
TEXT_LENGTH_PLOT_PATH = OUTPUT_DIR / "desc_text_length_distribution.png"

BILL_ID_COLUMN = "bill_id"
YEAR_COLUMN = "year"
STATE_COLUMN = "state"
STATUS_COLUMN = "status"
TOPICS_COLUMN = "topics"

TEXT_COLUMN = "text"
TEXT_LENGTH_COLUMN = "text_len"

TOPIC_SEPARATOR_PATTERN = r"[;,]"
STATUS_CATEGORY_SEPARATOR = " - "
STATUS_KEEP_TOP_N = 7
STATUS_MAX_UNIQUE_BEFORE_OTHER = 8
OTHER_CATEGORY_LABEL = "Other"
TOPIC_TOP_N = 15

BAR_COLOR = "#6baed6"
BAR_EDGE_COLOR = "#1f4e79"
MEAN_LINE_COLOR = "#c1121f"
MEDIAN_LINE_COLOR = "#003049"
ZOOM_LINE_COLOR = "#6a4c93"

HISTOGRAM_BIN_COUNT = 50
ZOOM_QUANTILE = 0.95
MIN_LOG_TEXT_LENGTH = 1

DPI = 300
COUNT_LABEL_OFFSET = 0.01


def _load_meta_frame(csv_path: Path) -> pd.DataFrame:
    """Load the NCSL meta CSV and drop rows with missing/empty ``bill_id``."""
    frame = pd.read_csv(csv_path)
    frame[BILL_ID_COLUMN] = frame[BILL_ID_COLUMN].astype("string").str.strip()
    valid_mask = frame[BILL_ID_COLUMN].notna() & (frame[BILL_ID_COLUMN] != "")
    return frame.loc[valid_mask].reset_index(drop=True)


def _ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def _annotate_vertical_bars(axis: plt.Axes, counts: pd.Series) -> None:
    """Draw the count above each vertical bar."""
    y_max = counts.max()
    offset = y_max * COUNT_LABEL_OFFSET
    for x_position, count in enumerate(counts.values):
        axis.text(
            x_position,
            count + offset,
            f"{int(count):,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def _annotate_horizontal_bars(axis: plt.Axes, counts: pd.Series) -> None:
    """Draw the count to the right of each horizontal bar."""
    x_max = counts.max()
    offset = x_max * COUNT_LABEL_OFFSET
    for y_position, count in enumerate(counts.values):
        axis.text(
            count + offset,
            y_position,
            f"{int(count):,}",
            ha="left",
            va="center",
            fontsize=9,
        )


def plot_bill_count_by_year(meta_frame: pd.DataFrame, output_path: Path) -> bool:
    """Bar chart of bill count per year."""
    if YEAR_COLUMN not in meta_frame.columns:
        print(f"Warning: '{YEAR_COLUMN}' column missing. Skipping year plot.")
        return False

    year_series = pd.to_numeric(meta_frame[YEAR_COLUMN], errors="coerce").dropna().astype(int)
    if year_series.empty:
        print(f"Warning: no usable '{YEAR_COLUMN}' values. Skipping year plot.")
        return False

    counts = year_series.value_counts().sort_index()

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        counts.index.astype(str),
        counts.values,
        color=BAR_COLOR,
        edgecolor=BAR_EDGE_COLOR,
    )
    _annotate_vertical_bars(axis, counts)
    axis.set_title("State AI bill count by year")
    axis.set_xlabel("Year")
    axis.set_ylabel("Bill count")
    axis.set_ylim(top=counts.max() * 1.10)
    axis.grid(axis="y", alpha=0.2)
    axis.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y):,}"))

    figure.tight_layout()
    figure.savefig(output_path, dpi=DPI)
    plt.close(figure)
    return True


def plot_bill_count_by_state(meta_frame: pd.DataFrame, output_path: Path) -> bool:
    """Horizontal bar chart of bill count per state, sorted descending."""
    if STATE_COLUMN not in meta_frame.columns:
        print(f"Warning: '{STATE_COLUMN}' column missing. Skipping state plot.")
        return False

    state_series = meta_frame[STATE_COLUMN].astype("string").str.strip()
    state_series = state_series[state_series.notna() & (state_series != "")]
    if state_series.empty:
        print(f"Warning: no usable '{STATE_COLUMN}' values. Skipping state plot.")
        return False

    counts = state_series.value_counts().sort_values(ascending=True)

    figure_height = max(8, 0.28 * len(counts))
    figure, axis = plt.subplots(figsize=(10, figure_height))
    axis.barh(
        counts.index,
        counts.values,
        color=BAR_COLOR,
        edgecolor=BAR_EDGE_COLOR,
    )
    _annotate_horizontal_bars(axis, counts)
    axis.set_title("State AI bill count by state")
    axis.set_xlabel("Bill count")
    axis.set_ylabel("State")
    axis.set_xlim(right=counts.max() * 1.10)
    axis.grid(axis="x", alpha=0.2)
    axis.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    figure.tight_layout()
    figure.savefig(output_path, dpi=DPI)
    plt.close(figure)
    return True


def _normalize_status(raw_status: pd.Series) -> pd.Series:
    """Collapse raw statuses to their high-level category (before ``" - "``)."""
    head = raw_status.fillna("").str.split(STATUS_CATEGORY_SEPARATOR, n=1).str[0]
    head = head.str.strip()
    return head.str.capitalize()


def _collapse_rare_statuses(counts: pd.Series) -> pd.Series:
    """Keep the top-N statuses and roll the rest into an ``Other`` bucket."""
    if len(counts) <= STATUS_MAX_UNIQUE_BEFORE_OTHER:
        return counts

    top = counts.iloc[:STATUS_KEEP_TOP_N]
    other_total = int(counts.iloc[STATUS_KEEP_TOP_N:].sum())
    collapsed = top.copy()
    collapsed[OTHER_CATEGORY_LABEL] = other_total
    return collapsed


def plot_bill_status(meta_frame: pd.DataFrame, output_path: Path) -> bool:
    """Bar chart of bill count per status category."""
    if STATUS_COLUMN not in meta_frame.columns:
        print(f"Warning: '{STATUS_COLUMN}' column missing. Skipping status plot.")
        return False

    status_series = meta_frame[STATUS_COLUMN].astype("string")
    status_series = status_series[status_series.notna() & (status_series.str.strip() != "")]
    if status_series.empty:
        print(f"Warning: no usable '{STATUS_COLUMN}' values. Skipping status plot.")
        return False

    normalized = _normalize_status(status_series)
    counts = normalized.value_counts()
    counts = _collapse_rare_statuses(counts)

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(
        counts.index,
        counts.values,
        color=BAR_COLOR,
        edgecolor=BAR_EDGE_COLOR,
    )
    _annotate_vertical_bars(axis, counts)
    axis.set_title("State AI bill count by status")
    axis.set_xlabel("Status category")
    axis.set_ylabel("Bill count")
    axis.set_ylim(top=counts.max() * 1.10)
    axis.grid(axis="y", alpha=0.2)
    axis.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
    plt.setp(axis.get_xticklabels(), rotation=20, ha="right")

    figure.tight_layout()
    figure.savefig(output_path, dpi=DPI)
    plt.close(figure)
    return True


def _explode_topics(topics: pd.Series) -> pd.Series:
    """Split the ``topics`` field on ``,`` or ``;`` into one row per topic."""
    return (
        topics.fillna("")
        .str.split(TOPIC_SEPARATOR_PATTERN, regex=True)
        .explode()
        .str.strip()
        .pipe(lambda series: series[series != ""])
    )


def plot_topic_coverage(meta_frame: pd.DataFrame, output_path: Path) -> bool:
    """Horizontal bar chart of the top-N NCSL topic labels by frequency."""
    if TOPICS_COLUMN not in meta_frame.columns:
        print(f"Warning: '{TOPICS_COLUMN}' column missing. Skipping topic plot.")
        return False

    topic_series = _explode_topics(meta_frame[TOPICS_COLUMN].astype("string"))
    if topic_series.empty:
        print(f"Warning: no usable '{TOPICS_COLUMN}' values. Skipping topic plot.")
        return False

    counts = topic_series.value_counts().head(TOPIC_TOP_N).sort_values(ascending=True)

    figure, axis = plt.subplots(figsize=(10, 8))
    axis.barh(
        counts.index,
        counts.values,
        color=BAR_COLOR,
        edgecolor=BAR_EDGE_COLOR,
    )
    _annotate_horizontal_bars(axis, counts)
    axis.set_title("Top NCSL topic labels across the corpus (2023-2025)")
    axis.set_xlabel("Topic mentions (a bill with N topics counts N times)")
    axis.set_ylabel("Topic")
    axis.set_xlim(right=counts.max() * 1.12)
    axis.grid(axis="x", alpha=0.2)
    axis.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    figure.tight_layout()
    figure.savefig(output_path, dpi=DPI)
    plt.close(figure)
    return True


def _load_text_lengths(csv_path: Path) -> pd.Series:
    """Load bill text lengths, falling back to computing them from raw text.

    Mirrors the loader in ``scripts/desc_stat.py`` so this script stays
    self-contained and does not depend on that module at import time.
    """
    text_lengths = pd.Series(dtype="float64")

    try:
        text_length_frame = pd.read_csv(csv_path, usecols=[TEXT_LENGTH_COLUMN])
    except ValueError:
        text_length_frame = None
    else:
        text_lengths = pd.to_numeric(
            text_length_frame[TEXT_LENGTH_COLUMN],
            errors="coerce",
        )

    if text_lengths.empty or text_lengths.isna().any():
        try:
            text_frame = pd.read_csv(csv_path, usecols=[TEXT_COLUMN])
        except ValueError as error:
            if text_lengths.empty:
                raise KeyError(
                    f"CSV must contain '{TEXT_COLUMN}' or '{TEXT_LENGTH_COLUMN}' columns."
                ) from error
        else:
            computed_lengths = text_frame[TEXT_COLUMN].fillna("").str.len()
            if text_lengths.empty:
                text_lengths = computed_lengths
            else:
                text_lengths = text_lengths.fillna(computed_lengths)

    cleaned_text_lengths = text_lengths.dropna().astype(int)
    if cleaned_text_lengths.empty:
        raise ValueError(f"No valid text lengths were found in '{csv_path}'.")

    return cleaned_text_lengths


def plot_text_length_distribution(csv_path: Path, output_path: Path) -> bool:
    """Two-panel histogram of text length (full log-scale + 95%-zoom)."""
    if not csv_path.exists():
        print(f"Warning: text CSV '{csv_path}' missing. Skipping text-length plot.")
        return False

    try:
        text_lengths = _load_text_lengths(csv_path)
    except (KeyError, ValueError) as error:
        print(f"Warning: could not load text lengths ({error}). Skipping text-length plot.")
        return False

    mean_length = text_lengths.mean()
    median_length = text_lengths.median()
    zoom_limit = int(text_lengths.quantile(ZOOM_QUANTILE))
    log_safe_lengths = text_lengths.clip(lower=MIN_LOG_TEXT_LENGTH)
    zoomed_text_lengths = text_lengths[text_lengths <= zoom_limit]

    figure, (full_axis, zoom_axis) = plt.subplots(ncols=2, figsize=(16, 6))

    full_axis.hist(
        log_safe_lengths,
        bins=HISTOGRAM_BIN_COUNT,
        color=BAR_COLOR,
        edgecolor=BAR_EDGE_COLOR,
        alpha=0.9,
    )
    full_axis.set_xscale("log")
    full_axis.axvline(
        max(mean_length, MIN_LOG_TEXT_LENGTH),
        color=MEAN_LINE_COLOR,
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_length:,.0f}",
    )
    full_axis.axvline(
        max(median_length, MIN_LOG_TEXT_LENGTH),
        color=MEDIAN_LINE_COLOR,
        linestyle="-.",
        linewidth=2,
        label=f"Median: {median_length:,.0f}",
    )
    full_axis.set_title("Full Distribution (log x-axis)")
    full_axis.set_xlabel("Text length (characters, log scale)")
    full_axis.set_ylabel("Bill count")
    full_axis.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    full_axis.grid(axis="y", alpha=0.2)
    full_axis.legend()

    zoom_axis.hist(
        zoomed_text_lengths,
        bins=HISTOGRAM_BIN_COUNT,
        color=BAR_COLOR,
        edgecolor=BAR_EDGE_COLOR,
        alpha=0.9,
    )
    zoom_axis.axvline(
        median_length,
        color=MEDIAN_LINE_COLOR,
        linestyle="-.",
        linewidth=2,
        label=f"Median: {median_length:,.0f}",
    )
    zoom_axis.axvline(
        zoom_limit,
        color=ZOOM_LINE_COLOR,
        linestyle=":",
        linewidth=2,
        label=f"{ZOOM_QUANTILE:.0%} percentile: {zoom_limit:,.0f}",
    )
    zoom_axis.set_title(f"Distribution Through {ZOOM_QUANTILE:.0%} Percentile")
    zoom_axis.set_xlabel("Text length (characters)")
    zoom_axis.set_ylabel("Bill count")
    zoom_axis.set_xlim(left=0)
    zoom_axis.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    zoom_axis.grid(axis="y", alpha=0.2)
    zoom_axis.legend()

    figure.suptitle("NCSL AI Legislation Text Length Distribution")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=DPI)
    plt.close(figure)
    return True


def _relative_to_project(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def main() -> None:
    _ensure_output_dir(OUTPUT_DIR)

    if not META_CSV_PATH.exists():
        raise FileNotFoundError(f"Meta CSV not found: {META_CSV_PATH}")

    meta_frame = _load_meta_frame(META_CSV_PATH)

    plot_results = [
        (YEAR_PLOT_PATH, plot_bill_count_by_year(meta_frame, YEAR_PLOT_PATH)),
        (STATE_PLOT_PATH, plot_bill_count_by_state(meta_frame, STATE_PLOT_PATH)),
        (STATUS_PLOT_PATH, plot_bill_status(meta_frame, STATUS_PLOT_PATH)),
        (TOPIC_PLOT_PATH, plot_topic_coverage(meta_frame, TOPIC_PLOT_PATH)),
        (
            TEXT_LENGTH_PLOT_PATH,
            plot_text_length_distribution(TEXT_CSV_PATH, TEXT_LENGTH_PLOT_PATH),
        ),
    ]

    for output_path, succeeded in plot_results:
        relative = _relative_to_project(output_path)
        if succeeded:
            print(f"Plot saved: {relative}")
        else:
            print(f"Plot NOT saved: {relative}")


if __name__ == "__main__":
    main()
