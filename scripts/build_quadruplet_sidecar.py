"""Build the quadruplet sidecar JSONL used by the QA ``query_quadruplets`` tool.

- Reads the per-bill JSON outputs produced by the skill-driven NER pipeline
  under ``data/skill_ner_runs/runs/<run_id>/outputs/*.json``.
- Flattens each quadruplet into one JSONL line keyed on ``bill_id`` and
  carrying the user-facing fields (``regulated_entity``, ``entity_type``,
  ``regulatory_mechanism``, ``provision_text``) plus the entity and provision
  character spans.
- Writes the sidecar to ``<cache_dir>/quadruplets.jsonl`` so the QA runtime
  discovers it next to ``chunks.jsonl`` both locally and on Render.

Usage:
    python -m scripts.build_quadruplet_sidecar
    python -m scripts.build_quadruplet_sidecar --run-id skill_full_20260416_v2
    python -m scripts.build_quadruplet_sidecar --output data/qa_cache/quadruplets.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.qa.config import load_qa_config

_DEFAULT_RUN_ID = "skill_full_20260416_v2"
_DEFAULT_INPUT_BASE = Path("data/skill_ner_runs/runs")


def parse_args() -> argparse.Namespace:
    """Parse command-line flags for the sidecar builder."""

    parser = argparse.ArgumentParser(
        description=(
            "Flatten skill-NER quadruplet output into the QA sidecar JSONL."
        )
    )
    parser.add_argument(
        "--run-id",
        default=_DEFAULT_RUN_ID,
        help=(
            "Skill-NER run id under data/skill_ner_runs/runs/<run_id>/outputs."
        ),
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help=(
            "Optional direct path to the outputs directory. When set, "
            "--run-id is ignored."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output JSONL path. Defaults to "
            "<qa_cache>/quadruplets.jsonl resolved from qa_config.yml."
        ),
    )
    return parser.parse_args()


def _resolve_input_dir(args: argparse.Namespace) -> Path:
    """Resolve the directory containing per-bill skill-NER JSON outputs."""

    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_absolute():
            input_dir = _PROJECT_ROOT / input_dir
        return input_dir
    return _PROJECT_ROOT / _DEFAULT_INPUT_BASE / args.run_id / "outputs"


def _resolve_output_path(args: argparse.Namespace) -> Path:
    """Resolve where the flattened JSONL should be written."""

    if args.output:
        output = Path(args.output)
        return output if output.is_absolute() else _PROJECT_ROOT / output
    config = load_qa_config(_PROJECT_ROOT)
    cache_dir = config.resolve_cache_dir(_PROJECT_ROOT)
    return cache_dir / "quadruplets.jsonl"


def _first_evidence_span(evidence: Any) -> dict[str, Any] | None:
    """Return the first evidence span as a plain dict, or ``None`` if absent."""

    if not isinstance(evidence, list) or not evidence:
        return None
    first = evidence[0]
    if not isinstance(first, dict):
        return None
    try:
        start = int(first.get("start"))
        end = int(first.get("end"))
    except (TypeError, ValueError):
        return None
    text = first.get("text")
    if not isinstance(text, str):
        text = ""
    return {"start": start, "end": end, "text": text}


def _safe_year(value: Any) -> int:
    """Coerce a ``year`` payload into a non-negative integer, 0 when unknown."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _flatten_quadruplet(
    *,
    bill_id: str,
    state: str,
    year: int,
    raw_quadruplet: dict[str, Any],
) -> dict[str, Any] | None:
    """Convert one raw skill-NER quadruplet into its sidecar record.

    Returns ``None`` when the quadruplet is missing one of the four required
    string fields (``entity``, ``type``, ``attribute``, ``value``) or when any
    of them is empty after stripping. Skipping keeps the sidecar clean so the
    planner tool never surfaces rows that cannot be grounded.
    """

    entity = raw_quadruplet.get("entity")
    entity_type = raw_quadruplet.get("type")
    attribute = raw_quadruplet.get("attribute")
    value = raw_quadruplet.get("value")
    required = {
        "entity": entity,
        "type": entity_type,
        "attribute": attribute,
        "value": value,
    }
    for field_name, field_value in required.items():
        if not isinstance(field_value, str) or not field_value.strip():
            return None

    entity_span = _first_evidence_span(raw_quadruplet.get("entity_evidence"))
    provision_span = _first_evidence_span(raw_quadruplet.get("value_evidence"))

    record: dict[str, Any] = {
        "bill_id": bill_id,
        "state": state,
        "year": year,
        "regulated_entity": entity.strip(),
        "entity_type": entity_type.strip(),
        "regulatory_mechanism": attribute.strip(),
        "provision_text": value.strip(),
    }
    if entity_span is not None:
        record["entity_span"] = entity_span
    if provision_span is not None:
        record["provision_span"] = provision_span
    return record


def main() -> None:
    """Walk the skill-NER outputs and write the flattened sidecar JSONL."""

    args = parse_args()
    input_dir = _resolve_input_dir(args)
    output_path = _resolve_output_path(args)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    input_files = sorted(input_dir.glob("*.json"))
    if not input_files:
        raise SystemExit(f"No *.json files found in {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_bills = 0
    bills_with_quadruplets = 0
    bills_with_zero_quadruplets = 0
    total_quadruplets_raw = 0
    total_quadruplets_written = 0
    skipped_missing_fields = 0
    skipped_bad_payloads = 0
    type_counter: Counter[str] = Counter()
    mechanism_counter: Counter[str] = Counter()

    with output_path.open("w", encoding="utf-8") as handle:
        for input_file in input_files:
            total_bills += 1
            try:
                payload = json.loads(input_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                print(f"WARN: failed to read {input_file.name}: {error}")
                skipped_bad_payloads += 1
                continue
            if not isinstance(payload, dict):
                skipped_bad_payloads += 1
                continue
            bill_id = str(payload.get("bill_id", "")).strip()
            state = str(payload.get("state", "")).strip()
            year = _safe_year(payload.get("year"))
            if not bill_id:
                skipped_bad_payloads += 1
                continue
            quadruplets = payload.get("quadruplets")
            if not isinstance(quadruplets, list) or not quadruplets:
                bills_with_zero_quadruplets += 1
                continue
            bills_with_quadruplets += 1
            for raw_quadruplet in quadruplets:
                if not isinstance(raw_quadruplet, dict):
                    skipped_bad_payloads += 1
                    continue
                total_quadruplets_raw += 1
                record = _flatten_quadruplet(
                    bill_id=bill_id,
                    state=state,
                    year=year,
                    raw_quadruplet=raw_quadruplet,
                )
                if record is None:
                    skipped_missing_fields += 1
                    continue
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
                total_quadruplets_written += 1
                type_counter[record["entity_type"]] += 1
                mechanism_counter[record["regulatory_mechanism"]] += 1

    print(f"Input directory: {input_dir}")
    print(f"Output file:     {output_path}")
    print(f"Bills scanned:                 {total_bills}")
    print(f"Bills with quadruplets:        {bills_with_quadruplets}")
    print(f"Bills with zero quadruplets:   {bills_with_zero_quadruplets}")
    print(f"Raw quadruplets encountered:   {total_quadruplets_raw}")
    print(f"Quadruplets written:           {total_quadruplets_written}")
    print(f"Skipped (missing fields):      {skipped_missing_fields}")
    print(f"Skipped (malformed payload):   {skipped_bad_payloads}")
    print(f"Unique entity_type values:     {len(type_counter)}")
    print(f"Unique regulatory_mechanism:   {len(mechanism_counter)}")
    top_types = type_counter.most_common(10)
    if top_types:
        print("Top entity_type values:")
        for name, count in top_types:
            print(f"  {count:5d}  {name}")


if __name__ == "__main__":
    main()
