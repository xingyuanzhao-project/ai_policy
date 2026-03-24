"""Read the NCSL legislation CSV and plot the text-length distribution."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "ncsl" / "us_ai_legislation_ncsl_text.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "ncsl" / "us_ai_legislation_ncsl_text_len_dist.png"
TEXT_COLUMN = "text"
TEXT_LENGTH_COLUMN = "text_len"
HISTOGRAM_BIN_COUNT = 50
FIGURE_WIDTH = 16
FIGURE_HEIGHT = 6
BAR_COLOR = "#6baed6"
BAR_EDGE_COLOR = "#1f4e79"
MEAN_LINE_COLOR = "#c1121f"
MEDIAN_LINE_COLOR = "#003049"
ZOOM_LINE_COLOR = "#6a4c93"
ZOOM_QUANTILE = 0.95
MIN_LOG_TEXT_LENGTH = 1


def load_text_lengths(csv_path: Path) -> pd.Series:
    """Load bill text lengths, falling back to computing them from raw text."""
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


def plot_text_length_distribution(text_lengths: pd.Series, output_path: Path) -> None:
    """Build and save a histogram of legislation text lengths."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mean_length = text_lengths.mean()
    median_length = text_lengths.median()
    zoom_limit = int(text_lengths.quantile(ZOOM_QUANTILE))
    log_safe_lengths = text_lengths.clip(lower=MIN_LOG_TEXT_LENGTH)
    zoomed_text_lengths = text_lengths[text_lengths <= zoom_limit]

    figure, (full_axis, zoom_axis) = plt.subplots(
        ncols=2,
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
    )

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
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def main() -> None:
    """Load the CSV, plot the distribution, and print a short summary."""
    text_lengths = load_text_lengths(CSV_PATH)
    plot_text_length_distribution(text_lengths, OUTPUT_PATH)

    print(f"Rows plotted: {len(text_lengths):,}")
    print(f"Mean text length: {text_lengths.mean():,.2f}")
    print(f"Median text length: {text_lengths.median():,.0f}")
    print(f"Min text length: {text_lengths.min():,}")
    print(f"Max text length: {text_lengths.max():,}")
    print(f"{ZOOM_QUANTILE:.0%} percentile: {text_lengths.quantile(ZOOM_QUANTILE):,.0f}")
    print(f"Plot saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
