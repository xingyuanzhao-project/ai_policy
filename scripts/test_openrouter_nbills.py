"""Test entry point: run N bills through the NER pipeline on OpenRouter.

Selects N small AI-relevant bills, runs the full 3-stage pipeline, and prints
a per-bill summary table with timing and token counts.

Usage:
    python scripts/test_openrouter_nbills.py              # default 5 bills
    python scripts/test_openrouter_nbills.py --n 20       # 20 bills
    python scripts/test_openrouter_nbills.py --n 5 --run-id my_test_run
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("test_openrouter_nbills")

import yaml

from src.ner.runtime.pipeline_api import run_full_corpus

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_JSONL = PROJECT_ROOT / "data" / "ncsl" / "us_ai_legislation_ncsl_text.jsonl"
MAX_TEXT_LEN = 15000
MIN_TEXT_LEN = 800


def select_ai_bills(n: int) -> list[dict]:
    """Select up to *n* small AI-relevant bills from the corpus."""

    selected: list[dict] = []
    with open(SOURCE_JSONL, encoding="utf-8") as handle:
        for raw_line in handle:
            row = json.loads(raw_line)
            text = row.get("text", "")
            title = str(row.get("title", "")).lower()
            summary = str(row.get("summary", "")).lower()
            if not any(
                "artificial intelligence" in field
                for field in (title, summary, text.lower())
            ):
                continue
            if not (MIN_TEXT_LEN < len(text) < MAX_TEXT_LEN):
                continue
            selected.append(row)
            if len(selected) == n:
                break

    if len(selected) < n:
        logger.warning("Only found %d eligible bills (requested %d)", len(selected), n)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Test NER pipeline on N bills via OpenRouter")
    parser.add_argument("--n", type=int, default=5, help="Number of bills to process")
    parser.add_argument("--concurrency", type=int, default=None, help="Override concurrency")
    parser.add_argument("--run-id", type=str, default=None, help="Override run_id")
    args = parser.parse_args()

    bills = select_ai_bills(args.n)
    if not bills:
        logger.error("No eligible bills found")
        sys.exit(1)

    logger.info("Selected %d bills for test run", len(bills))
    for row in bills:
        logger.info("  bill_id=%-20s  text_len=%5d  title=%s", row["bill_id"], len(row.get("text", "")), row.get("title", "")[:60])

    subset_jsonl = PROJECT_ROOT / "data" / "ner_runs" / "test_subset.jsonl"
    subset_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(subset_jsonl, "w", encoding="utf-8") as handle:
        for row in bills:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(PROJECT_ROOT / "settings" / "config.yml", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)
    base_config["input_path"] = str(subset_jsonl)

    with open(PROJECT_ROOT / "settings" / "ner_config.yml", encoding="utf-8") as handle:
        ner_config = yaml.safe_load(handle)
    if args.concurrency is not None:
        ner_config["runtime"]["concurrency"] = args.concurrency

    effective_concurrency = ner_config["runtime"].get("concurrency", 4)

    tmp_path = PROJECT_ROOT / "data" / "ner_runs" / "_tmp_configs"
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config.yml"
    ner_config_path = tmp_path / "ner_config.yml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False)
    with open(ner_config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(ner_config, handle, sort_keys=False)

    run_id = args.run_id or f"test_{args.n}bills_c{effective_concurrency}"
    logger.info("Starting run_id=%s  concurrency=%d", run_id, effective_concurrency)
    t0 = time.perf_counter()

    results = run_full_corpus(
        project_root=PROJECT_ROOT,
        config_path=config_path,
        ner_config_path=ner_config_path,
        prompt_config_path="settings/ner_prompts.json",
        run_id=run_id,
        resume=False,
    )

    total_elapsed = time.perf_counter() - t0

    print(file=sys.stderr)
    print("=" * 90, file=sys.stderr)
    print(f"  TEST COMPLETE: {len(results)} bills  concurrency={effective_concurrency}  total={total_elapsed:.1f}s", file=sys.stderr)
    print("=" * 90, file=sys.stderr)
    print(f"  {'bill_id':<28s} {'text_len':>8s} {'chunks':>6s} {'refined':>7s}", file=sys.stderr)
    print("-" * 90, file=sys.stderr)

    total_refined = 0
    for row in bills:
        raw_bid = str(row["bill_id"])
        year = str(row.get("year", ""))
        pipeline_bid = f"{year}__{raw_bid}" if year else raw_bid
        refined = results.get(pipeline_bid, [])
        text_len = len(row.get("text", ""))
        chunk_count = max(1, (text_len - 300) // (3000 - 300)) if text_len > 0 else 0
        total_refined += len(refined)
        print(f"  {pipeline_bid:<28s} {text_len:>8d} {chunk_count:>6d} {len(refined):>7d}", file=sys.stderr)

    print("-" * 90, file=sys.stderr)
    print(f"  TOTALS: {len(results)} bills, {total_refined} refined outputs, {total_elapsed:.1f}s wall-clock", file=sys.stderr)
    print(f"  AVG per bill: {total_elapsed / max(len(results), 1):.1f}s", file=sys.stderr)
    print(f"  Throughput: {len(results) / max(total_elapsed, 0.1) * 60:.1f} bills/min", file=sys.stderr)
    run_dir = PROJECT_ROOT / "data" / "ner_runs" / "runs" / run_id
    print(f"  Intermediates: {run_dir}", file=sys.stderr)
    print("=" * 90, file=sys.stderr)


if __name__ == "__main__":
    main()
