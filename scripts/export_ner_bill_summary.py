"""Export one NER run's final bill outputs into a DataFrame and CSV.

Usage:
    python -m scripts.export_ner_bill_summary
    python scripts/export_ner_bill_summary.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import pandas as pd

# Add project root to path for src import when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ner.schemas.artifacts import RefinedQuadruplet
from src.ner.storage.final_output_store import FinalOutputStore


# ===
# Configures
# ===
RUN_ID = "run_full_20260320"
NER_STORAGE_BASE_DIR = PROJECT_ROOT / "data" / "ner_runs"
RUN_DIRECTORY = NER_STORAGE_BASE_DIR / "runs" / RUN_ID
PANDAS_OUTPUT_PATH = RUN_DIRECTORY / "final_outputs_by_bill.pkl"
CSV_OUTPUT_PATH = RUN_DIRECTORY / "final_outputs_by_bill.csv"


# ===
# Operations
# ===
def build_bill_dataframe(
    outputs_by_bill: dict[str, list[RefinedQuadruplet]],
) -> pd.DataFrame:
    """Build one row per bill from final refined outputs."""
    rows = [
        build_bill_row(bill_id=bill_id, refined_outputs=refined_outputs)
        for bill_id, refined_outputs in sorted(outputs_by_bill.items())
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "bill_id",
            "entity_quad",
            "entities",
            "type_count",
            "attribute_count",
            "value_count",
        ],
    )


def build_bill_row(
    bill_id: str,
    refined_outputs: list[RefinedQuadruplet],
) -> dict[str, object]:
    """Build the bill-level export row for one bill."""
    entity_quad = [
        (
            refined_output.entity,
            refined_output.type,
            refined_output.attribute,
            refined_output.value,
        )
        for refined_output in refined_outputs
    ]

    entities = unique_non_null_values(
        refined_output.entity for refined_output in refined_outputs
    )
    type_count = count_non_null_values(
        refined_output.type for refined_output in refined_outputs
    )
    attribute_count = count_non_null_values(
        refined_output.attribute for refined_output in refined_outputs
    )
    value_count = count_non_null_values(
        refined_output.value for refined_output in refined_outputs
    )

    return {
        "bill_id": bill_id,
        "entity_quad": entity_quad,
        "entities": entities,
        "type_count": type_count,
        "attribute_count": attribute_count,
        "value_count": value_count,
    }


def unique_non_null_values(values: object) -> list[str]:
    """Return unique non-empty strings in first-seen order."""
    unique_values: list[str] = []
    seen_values: set[str] = set()
    for value in values:
        if value is None:
            continue
        normalized_value = value.strip()
        if not normalized_value or normalized_value in seen_values:
            continue
        seen_values.add(normalized_value)
        unique_values.append(normalized_value)
    return unique_values


def count_non_null_values(values: object) -> dict[str, int]:
    """Count non-empty strings and return a stable dict."""
    counter: Counter[str] = Counter()
    for value in values:
        if value is None:
            continue
        normalized_value = value.strip()
        if not normalized_value:
            continue
        counter[normalized_value] += 1
    return dict(sorted(counter.items()))


def load_final_outputs(run_id: str) -> dict[str, list[RefinedQuadruplet]]:
    """Load only final bill-level outputs for one run id."""
    output_store = FinalOutputStore(NER_STORAGE_BASE_DIR)
    outputs_by_bill = output_store.load_all(run_id)
    if not outputs_by_bill:
        raise FileNotFoundError(
            f"No final outputs were found for run_id '{run_id}' in '{RUN_DIRECTORY / 'outputs'}'."
        )
    return outputs_by_bill


# ===
# Storages
# ===
def save_outputs(dataframe: pd.DataFrame) -> None:
    """Persist the DataFrame and CSV into the run directory."""
    dataframe.to_pickle(PANDAS_OUTPUT_PATH)
    dataframe.to_csv(CSV_OUTPUT_PATH, index=False, encoding="utf-8")


def main() -> None:
    """Export one run's final bill outputs into bill-level tabular artifacts."""
    outputs_by_bill = load_final_outputs(RUN_ID)
    dataframe = build_bill_dataframe(outputs_by_bill)
    save_outputs(dataframe)
    print(f"Bills exported: {len(dataframe)}")
    print(f"DataFrame saved to: {PANDAS_OUTPUT_PATH}")
    print(f"CSV saved to: {CSV_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
