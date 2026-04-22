"""Backfill year / status_bucket / topics_list into an existing QA chunks.jsonl.

This one-shot script enriches a persisted QA cache in place after the chunk schema
grew three new filter fields. It joins each chunk row with the source corpus on
``bill_id`` and rewrites both ``chunks.jsonl`` and ``chunk_offsets.npy``. The
embeddings matrix and manifest are left untouched because embeddings depend on
chunk text, not on chunk metadata.

Usage::

    python -m scripts.backfill_qa_chunks_metadata
    python -m scripts.backfill_qa_chunks_metadata --cache-dir data/qa_cache
    python -m scripts.backfill_qa_chunks_metadata --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from src.ner.storage.corpus_store import CorpusStore
from src.qa.config import load_qa_config
from src.qa.indexer import _normalize_status, _split_topics

_CHUNKS_FILENAME = "chunks.jsonl"
_CHUNK_OFFSETS_FILENAME = "chunk_offsets.npy"


def parse_args() -> argparse.Namespace:
    """Parse command-line flags for the backfill script."""

    parser = argparse.ArgumentParser(
        description="Backfill year/status_bucket/topics_list into QA chunks.jsonl"
    )
    parser.add_argument(
        "--corpus-path",
        default=None,
        help="Corpus JSONL path (default: value from settings/qa_config.yml)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="QA cache directory (default: value from settings/qa_config.yml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned changes without writing any files",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip saving chunks.jsonl.bak alongside the rewrite",
    )
    return parser.parse_args()


def main() -> None:
    """Rewrite chunks.jsonl in place with the three new filter fields populated."""

    args = parse_args()
    config = load_qa_config(_PROJECT_ROOT)
    corpus_path = Path(args.corpus_path) if args.corpus_path else (_PROJECT_ROOT / config.corpus_path)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (_PROJECT_ROOT / config.index.cache_dir)

    chunks_path = cache_dir / _CHUNKS_FILENAME
    offsets_path = cache_dir / _CHUNK_OFFSETS_FILENAME
    if not chunks_path.exists():
        raise SystemExit(f"No chunks.jsonl found at {chunks_path}")
    if not offsets_path.exists():
        raise SystemExit(f"No chunk_offsets.npy found at {offsets_path}")
    if not corpus_path.exists():
        raise SystemExit(f"Corpus file not found at {corpus_path}")

    print(f"Corpus:    {corpus_path}")
    print(f"Cache dir: {cache_dir}")

    bills_by_id = _load_bill_metadata(corpus_path)
    print(f"Loaded {len(bills_by_id)} bills from corpus")

    stats = _process_chunks(
        chunks_path=chunks_path,
        offsets_path=offsets_path,
        bills_by_id=bills_by_id,
        dry_run=args.dry_run,
        no_backup=args.no_backup,
    )

    print(
        "Chunks processed: "
        f"{stats['total']} total, "
        f"{stats['enriched']} enriched, "
        f"{stats['already_populated']} already populated, "
        f"{stats['missing_bill']} missing bill_id in corpus"
    )
    if args.dry_run:
        print("Dry run complete. No files modified.")
    else:
        print(f"Rewrote {chunks_path} and {offsets_path}.")


def _load_bill_metadata(corpus_path: Path) -> dict[str, dict[str, object]]:
    """Return a lookup mapping year-qualified bill_id to its enrichment fields."""

    store = CorpusStore(corpus_path)
    records = store.load()
    bills: dict[str, dict[str, object]] = {}
    for bill in records:
        year_raw = bill.year if isinstance(bill.year, str) else str(bill.year or "")
        year_value = int(year_raw) if year_raw.strip().isdigit() else 0
        bills[bill.bill_id] = {
            "year": year_value,
            "status_bucket": _normalize_status(bill.status),
            "topics_list": _split_topics(bill.topics),
        }
    return bills


def _process_chunks(
    *,
    chunks_path: Path,
    offsets_path: Path,
    bills_by_id: dict[str, dict[str, object]],
    dry_run: bool,
    no_backup: bool,
) -> dict[str, int]:
    """Rewrite chunks.jsonl line-by-line and regenerate the byte-offset index."""

    stats = {"total": 0, "enriched": 0, "already_populated": 0, "missing_bill": 0}

    temp_chunks_path = chunks_path.with_suffix(chunks_path.suffix + ".tmp")
    offsets: list[int] = []
    current_offset = 0

    with open(chunks_path, "r", encoding="utf-8") as source:
        output_handle = None if dry_run else open(temp_chunks_path, "wb")
        try:
            for raw_line in source:
                if not raw_line.strip():
                    continue
                stats["total"] += 1
                payload = json.loads(raw_line)
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"chunks.jsonl line {stats['total']} did not decode to an object"
                    )

                had_all_fields = all(
                    key in payload for key in ("year", "status_bucket", "topics_list")
                )
                enrichment = bills_by_id.get(str(payload.get("bill_id", "")))
                if enrichment is None:
                    stats["missing_bill"] += 1
                    payload.setdefault("year", 0)
                    payload.setdefault("status_bucket", "Other")
                    payload.setdefault("topics_list", [])
                else:
                    payload["year"] = enrichment["year"]
                    payload["status_bucket"] = enrichment["status_bucket"]
                    payload["topics_list"] = list(enrichment["topics_list"])  # type: ignore[arg-type]

                if had_all_fields and enrichment is None:
                    stats["already_populated"] += 1
                else:
                    stats["enriched"] += 1

                if output_handle is not None:
                    offsets.append(current_offset)
                    line_bytes = (
                        json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
                    )
                    output_handle.write(line_bytes)
                    current_offset += len(line_bytes)
        finally:
            if output_handle is not None:
                output_handle.close()

    if dry_run:
        return stats

    if not no_backup:
        shutil.copy2(chunks_path, chunks_path.with_suffix(chunks_path.suffix + ".bak"))
    temp_chunks_path.replace(chunks_path)

    offsets_array = np.asarray(offsets, dtype=np.int64)
    np.save(offsets_path, offsets_array)

    return stats


if __name__ == "__main__":
    main()
