"""Add the ``entities`` column to the annotated bills CSV from the quadruplet sidecar.

- Reads ``data/qa_cache/quadruplets.jsonl`` (produced by ``build_quadruplet_sidecar.py``).
- Groups full quadruplet records by ``bill_id`` and stores them as a
  JSON-encoded list of dicts per row in the new ``entities`` column,
  e.g. ``[{"regulated_entity": ..., "entity_type": ..., "regulatory_mechanism": ..., "provision_text": ...}, ...]``.
- Bills absent from the sidecar receive an empty JSON list ``[]``.
- Overwrites the annotated CSV in place. Re-running is safe: if ``entities``
  already exists, the values are overwritten and no duplicate column is added.

Usage:
    python -m scripts.adding_quad
    python -m scripts.adding_quad --sidecar data/qa_cache/quadruplets.jsonl
    python -m scripts.adding_quad --annotated data/ncsl/us_ai_legislation_annotated.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.qa.config import load_qa_config

_DEFAULT_ANNOTATED_PATH = Path("data/ncsl/us_ai_legislation_annotated.csv")
_ENTITIES_COLUMN = "entities"
_QUADRUPLET_FIELDS = (
    "regulated_entity",
    "entity_type",
    "regulatory_mechanism",
    "provision_text",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join quadruplet entities onto the annotated bills CSV."
    )
    parser.add_argument(
        "--sidecar",
        default=None,
        help=(
            "Path to the quadruplet sidecar JSONL. "
            "Defaults to <qa_cache>/quadruplets.jsonl resolved from qa_config.yml."
        ),
    )
    parser.add_argument(
        "--annotated",
        default=None,
        help=(
            "Path to the annotated bills CSV. "
            f"Defaults to {_DEFAULT_ANNOTATED_PATH}."
        ),
    )
    return parser.parse_args()


def _resolve_sidecar_path(args: argparse.Namespace) -> Path:
    """Resolve the sidecar JSONL path from the CLI arg or qa_config.yml."""

    if args.sidecar:
        p = Path(args.sidecar)
        return p if p.is_absolute() else _PROJECT_ROOT / p
    config = load_qa_config(_PROJECT_ROOT)
    return config.resolve_cache_dir(_PROJECT_ROOT) / "quadruplets.jsonl"


def _resolve_annotated_path(args: argparse.Namespace) -> Path:
    """Resolve the annotated CSV path from the CLI arg or the project default."""

    if args.annotated:
        p = Path(args.annotated)
        return p if p.is_absolute() else _PROJECT_ROOT / p
    return _PROJECT_ROOT / _DEFAULT_ANNOTATED_PATH


def _build_quadruplet_index(sidecar_path: Path) -> dict[str, list[dict[str, str]]]:
    """Read the sidecar and return bill_id → ordered list of quadruplet dicts.

    Each value preserves source order from the sidecar JSONL and contains the
    four user-facing quadruplet fields: ``regulated_entity``, ``entity_type``,
    ``regulatory_mechanism``, ``provision_text``. Bill-level fields
    (``bill_id``, ``state``, ``year``) are omitted because they are already
    present as columns on the annotated CSV row. Span fields are omitted
    because they are not part of the four-field quadruplet.
    """

    index: dict[str, list[dict[str, str]]] = defaultdict(list)

    with sidecar_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            bill_id = str(record.get("bill_id", "")).strip()
            if not bill_id:
                continue
            quadruplet = {
                field: str(record.get(field, "")).strip()
                for field in _QUADRUPLET_FIELDS
            }
            if not all(quadruplet.values()):
                continue
            index[bill_id].append(quadruplet)

    return dict(index)


def main() -> None:
    args = parse_args()
    sidecar_path = _resolve_sidecar_path(args)
    annotated_path = _resolve_annotated_path(args)

    if not sidecar_path.is_file():
        raise SystemExit(f"Sidecar JSONL not found: {sidecar_path}")
    if not annotated_path.is_file():
        raise SystemExit(f"Annotated CSV not found: {annotated_path}")

    print(f"Sidecar:   {sidecar_path}")
    print(f"Annotated: {annotated_path}")

    quadruplet_index = _build_quadruplet_index(sidecar_path)
    print(f"Bills with quadruplets in sidecar: {len(quadruplet_index)}")

    rows: list[dict] = []
    with annotated_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise SystemExit("Annotated CSV has no header row.")
        original_fieldnames = list(reader.fieldnames)
        for row in reader:
            rows.append(row)

    # Append the new column only when it is not already present (idempotent re-run).
    if _ENTITIES_COLUMN in original_fieldnames:
        output_fieldnames = original_fieldnames
    else:
        output_fieldnames = [*original_fieldnames, _ENTITIES_COLUMN]

    # The sidecar's bill_id is already namespaced as ``<year>__<bill_id>``
    # (see build_quadruplet_sidecar.py output), while the annotated CSV stores
    # ``bill_id`` and ``year`` in separate columns. Recompose the composite
    # key from the CSV side so the join is unambiguous across years.
    bills_matched = 0
    bills_empty = 0
    for row in rows:
        bill_id = str(row.get("bill_id", "")).strip()
        year = str(row.get("year", "")).strip()
        lookup_key = f"{year}__{bill_id}" if year else bill_id
        quadruplets = quadruplet_index.get(lookup_key, [])
        row[_ENTITIES_COLUMN] = json.dumps(quadruplets, ensure_ascii=False)
        if quadruplets:
            bills_matched += 1
        else:
            bills_empty += 1

    with annotated_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Rows written:               {len(rows)}")
    print(f"Bills matched to sidecar:   {bills_matched}")
    print(f"Bills with no quadruplets:  {bills_empty}")
    print(f"Output: {annotated_path}")


if __name__ == "__main__":
    main()
