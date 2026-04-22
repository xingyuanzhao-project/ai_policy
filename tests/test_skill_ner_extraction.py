"""Test skill-driven NER: verify the full pipeline processes a document.

Runs the skill_ner runner on a single synthetic bill to validate the
end-to-end flow: skill loading, tool registration, agent loop, output
parsing, and file persistence.

Usage:
    python -m tests.test_skill_ner_extraction
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("test_skill_ner_extraction")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SYNTHETIC_BILL = {
    "bill_id": "TEST_SKILL_001",
    "state": "TestState",
    "year": "2025",
    "title": "Artificial Intelligence Regulation Test Act",
    "text": (
        "SECTION 1. SHORT TITLE.\n"
        'This Act may be cited as the "Artificial Intelligence Regulation Test Act".\n\n'
        "SECTION 2. DEFINITIONS.\n"
        '(1) "Artificial intelligence system" means a machine-based system that, '
        "for a given set of human-defined objectives, generates outputs such as "
        "predictions, recommendations, or decisions.\n"
        '(2) "High-risk AI system" means an artificial intelligence system that '
        "makes or materially supports decisions affecting individual rights.\n"
        '(3) "Deployer" means any person or entity that deploys an AI system.\n\n'
        "SECTION 3. REQUIREMENTS.\n"
        "(a) IMPACT ASSESSMENT.--Before deploying any high-risk AI system, "
        "each deployer shall conduct and document an algorithmic impact assessment.\n"
        "(b) TRANSPARENCY.--Each deployer of a high-risk AI system shall provide "
        "clear and accessible notice to individuals subject to the system.\n"
        "(c) BIAS TESTING.--Each deployer shall conduct regular bias testing on "
        "high-risk AI systems to ensure they do not discriminate.\n"
        "(d) DATA GOVERNANCE.--Deployers shall maintain data governance practices "
        "including data quality standards and documentation.\n\n"
        "SECTION 4. ENFORCEMENT.\n"
        "The Attorney General may bring a civil action against any deployer that "
        "violates the requirements of this Act.\n"
    ),
    "bill_url": "",
    "status": "Introduced",
    "date_of_last_action": "2025-01-01",
    "author": "Test Author",
    "topics": "artificial intelligence",
    "summary": "Regulates the deployment of high-risk AI systems.",
    "history": "",
}


def main() -> None:
    from src.skill_ner.runner import run_corpus

    tmp_path = Path(tempfile.mkdtemp(prefix="skill_ner_test_"))
    logger.info("Temp directory: %s", tmp_path)

    # Write synthetic corpus
    corpus_path = tmp_path / "test_corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(SYNTHETIC_BILL, ensure_ascii=False) + "\n")

    # Create a modified config pointing to our test corpus and temp storage
    with open(PROJECT_ROOT / "settings" / "config.yml", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)
    base_config["input_path"] = str(corpus_path)

    config_path = tmp_path / "config.yml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(base_config, f, sort_keys=False)

    # Create a modified skill_ner_config pointing to temp storage
    with open(PROJECT_ROOT / "settings" / "skill_ner_config.yml", encoding="utf-8") as f:
        skill_config = yaml.safe_load(f)
    skill_config["storage"] = {"base_dir": str(tmp_path / "skill_ner_runs")}

    skill_config_path = tmp_path / "skill_ner_config.yml"
    with open(skill_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(skill_config, f, sort_keys=False)

    # Monkey-patch _load_config and _load_base_config to use temp files
    import src.skill_ner.runner as runner_mod

    original_load_config = runner_mod._load_config
    original_load_base_config = runner_mod._load_base_config

    def patched_load_config(project_root: Path) -> dict:
        with open(skill_config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def patched_load_base_config(project_root: Path) -> dict:
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    runner_mod._load_config = patched_load_config
    runner_mod._load_base_config = patched_load_base_config

    # Use the cheap model from base config for testing
    test_model = base_config.get("llm", {}).get("model_name", "google/gemini-2.5-flash")

    try:
        logger.info("Running skill NER on synthetic bill with model=%s...", test_model)
        run_dir = run_corpus(
            project_root=PROJECT_ROOT,
            model=test_model,
            run_id="test_skill_ner",
            max_bills=1,
            resume=False,
        )
    finally:
        runner_mod._load_config = original_load_config
        runner_mod._load_base_config = original_load_base_config

    # Verify outputs
    print(f"\n{'='*60}", file=sys.stderr)
    print("TEST: Skill NER End-to-End Extraction", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  Run directory: {run_dir}", file=sys.stderr)

    expected_files = [
        "config_snapshot.yml",
        "skill_snapshot.md",
        "preflight.json",
        "run.log",
        "usage_summary.json",
    ]
    for fname in expected_files:
        fpath = run_dir / fname
        exists = fpath.exists()
        size = fpath.stat().st_size if exists else 0
        status = f"OK ({size} bytes)" if exists else "MISSING"
        print(f"  {fname}: {status}", file=sys.stderr)
        assert exists, f"Expected file {fname} in run directory"

    outputs_dir = run_dir / "outputs"
    assert outputs_dir.exists(), "outputs/ directory should exist"
    output_files = list(outputs_dir.glob("*.json"))
    print(f"  Output files: {len(output_files)}", file=sys.stderr)
    assert len(output_files) >= 1, "Should have at least 1 output file"

    for output_file in output_files:
        with open(output_file, encoding="utf-8") as f:
            quadruplets = json.load(f)
        print(f"    {output_file.name}: {len(quadruplets)} quadruplet(s)", file=sys.stderr)
        for i, q in enumerate(quadruplets[:3]):
            print(
                f"      [{i}] entity={q.get('entity')!r}  "
                f"type={q.get('type')!r}  "
                f"attribute={q.get('attribute')!r}",
                file=sys.stderr,
            )

    raw_dir = run_dir / "raw_responses"
    assert raw_dir.exists(), "raw_responses/ directory should exist"
    raw_files = list(raw_dir.glob("*.json"))
    print(f"  Raw response files: {len(raw_files)}", file=sys.stderr)
    assert len(raw_files) >= 1, "Should have at least 1 raw response file"

    with open(run_dir / "usage_summary.json", encoding="utf-8") as f:
        usage = json.load(f)
    print(f"  Usage: calls={usage['total_calls']}  "
          f"tokens={usage['total_tokens']}  "
          f"cost=${usage['total_cost_usd']:.6f}", file=sys.stderr)
    assert usage["total_calls"] >= 1, "Should have at least 1 API call"
    assert usage["total_tokens"] > 0, "Should have used some tokens"

    skill_snapshot = (run_dir / "skill_snapshot.md").read_text(encoding="utf-8")
    assert len(skill_snapshot) > 100, "Skill snapshot should have meaningful content"
    print(f"  Skill snapshot: {len(skill_snapshot)} chars", file=sys.stderr)

    with open(output_files[0], encoding="utf-8") as f:
        first_output = json.load(f)
    if first_output:
        q = first_output[0]
        assert "entity" in q, "Quadruplet should have 'entity' field"
        assert "type" in q, "Quadruplet should have 'type' field"
        assert "attribute" in q, "Quadruplet should have 'attribute' field"
        assert "value" in q, "Quadruplet should have 'value' field"
        print(f"  Output schema: validated (has entity/type/attribute/value)", file=sys.stderr)

    print(f"\n  PASS: Skill NER pipeline works end-to-end", file=sys.stderr)
    print(f"  Temp dir (manual cleanup): {tmp_path}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


if __name__ == "__main__":
    main()
