"""Smoke test: run one small bill through the NER pipeline on OpenRouter.

Usage:
    python -m tests.smoke_test_openrouter
"""

import json
import logging
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    stream=sys.stderr,
)

from src.ner.runtime.pipeline_api import run_single_bill

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_JSONL = PROJECT_ROOT / "data" / "ncsl" / "us_ai_legislation_ncsl_text.jsonl"


def find_small_ai_bill() -> tuple[str, dict]:
    """Find one small AI-relevant bill from the corpus."""

    with open(SOURCE_JSONL, encoding="utf-8") as handle:
        for raw_line in handle:
            row = json.loads(raw_line)
            text = row.get("text", "")
            title = str(row.get("title", "")).lower()
            if "artificial intelligence" in title and 1000 < len(text) < 10000:
                return str(row["bill_id"]), row
    raise RuntimeError("No suitable small AI bill found in corpus")


def main() -> None:
    bill_id, row = find_small_ai_bill()
    print(f"Selected bill: {bill_id}  title={row.get('title', '')}  text_len={len(row.get('text', ''))}", file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        subset_jsonl = tmp_path / "subset.jsonl"
        with open(subset_jsonl, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        import yaml
        with open(PROJECT_ROOT / "settings" / "config.yml", encoding="utf-8") as handle:
            base_config = yaml.safe_load(handle)
        base_config["input_path"] = str(subset_jsonl)

        with open(PROJECT_ROOT / "settings" / "ner_config.yml", encoding="utf-8") as handle:
            ner_config = yaml.safe_load(handle)
        ner_config["storage"]["base_dir"] = str(tmp_path / "ner_runs")

        config_path = tmp_path / "config.yml"
        ner_config_path = tmp_path / "ner_config.yml"
        with open(config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(base_config, handle, sort_keys=False)
        with open(ner_config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(ner_config, handle, sort_keys=False)

        refined_outputs = run_single_bill(
            project_root=PROJECT_ROOT,
            bill_id=bill_id,
            run_id="smoke_openrouter",
            resume=False,
            config_path=config_path,
            ner_config_path=ner_config_path,
        )

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"RESULT: bill={bill_id}  refined_outputs={len(refined_outputs)}", file=sys.stderr)
        for i, output in enumerate(refined_outputs):
            print(
                f"  [{i}] entity={output.entity!r}  type={output.type!r}  "
                f"attribute={output.attribute!r}  value={output.value!r}",
                file=sys.stderr,
            )
        print(f"{'='*60}", file=sys.stderr)

        run_dir = tmp_path / "ner_runs" / "runs" / "smoke_openrouter"
        preflight_path = run_dir / "preflight.json"
        if preflight_path.exists():
            with open(preflight_path, encoding="utf-8") as handle:
                preflight = json.load(handle)
            print(f"Preflight: {json.dumps(preflight, indent=2)}", file=sys.stderr)

        print("\nSMOKE TEST PASSED" if refined_outputs else "\nWARNING: zero outputs (may be valid for short bill)", file=sys.stderr)


if __name__ == "__main__":
    main()
