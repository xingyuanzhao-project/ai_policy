from __future__ import annotations

import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV_PATH = REPO_ROOT / "data" / "ncsl" / "us_ai_legislation_full.csv"
OUTPUT_CSV_PATH = REPO_ROOT / "data" / "ncsl" / "us_ai_legislation_annotated.csv"
OUTCOME_COLUMN_NAME = "outcome"

# Treat only final positive statuses as outcome=1.
PASSED_STATUS_PREFIXES = (
    "enacted",
    "adopted",
    "passed",
    "approved",
    "signed",
    "chaptered",
    "ratified",
)

PASSED_STATUS_SUBSTRINGS = (
    "became law",
    "signed by governor",
    "approved by governor",
)


def status_indicates_passed(status_value: str | None) -> bool:
    normalized_status = (status_value or "").strip().lower()
    if not normalized_status:
        return False

    if normalized_status.startswith(PASSED_STATUS_PREFIXES):
        return True

    return any(substring in normalized_status for substring in PASSED_STATUS_SUBSTRINGS)


def annotate_outcomes(input_csv_path: Path, output_csv_path: Path) -> int:
    with input_csv_path.open("r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)

        if reader.fieldnames is None:
            raise ValueError(f"No header row found in {input_csv_path}")
        if "status" not in reader.fieldnames:
            raise ValueError(f"'status' column not found in {input_csv_path}")

        output_fieldnames = [*reader.fieldnames, OUTCOME_COLUMN_NAME]

        with output_csv_path.open("w", newline="", encoding="utf-8-sig") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=output_fieldnames)
            writer.writeheader()

            written_rows = 0
            for row in reader:
                row[OUTCOME_COLUMN_NAME] = 1 if status_indicates_passed(row.get("status")) else 0
                writer.writerow(row)
                written_rows += 1

    return written_rows


def main() -> None:
    written_rows = annotate_outcomes(INPUT_CSV_PATH, OUTPUT_CSV_PATH)
    print(f"Wrote {written_rows} rows to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
