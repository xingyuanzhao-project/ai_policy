"""Real live-vLLM integration tests for the NER pipeline.

- Verifies end-to-end runtime behavior against the configured local vLLM
  server.
- Verifies artifact persistence and stage-state outputs for real runs.
- Does not mock the online model interface.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import yaml

from src.ner.runtime.pipeline_api import run_full_corpus, run_single_bill
from src.ner.storage.final_output_store import FinalOutputStore

LIVE_TESTS_ENABLED = os.environ.get("NER_RUN_LIVE_TESTS") == "1"


@unittest.skipUnless(LIVE_TESTS_ENABLED, "Set NER_RUN_LIVE_TESTS=1 to run live vLLM integration tests")
class LivePipelineIntegrationTests(unittest.TestCase):
    """End-to-end validation against the configured local vLLM server."""

    def test_full_pipeline_runs_on_existing_bill_subset_and_persists_artifacts(self) -> None:
        """Verify the full online pipeline persists stage and output artifacts."""

        project_root = Path(__file__).resolve().parents[3]
        source_jsonl = project_root / "data" / "ncsl" / "us_ai_legislation_ncsl_text.jsonl"
        prompt_config_path = project_root / "settings" / "ner_prompts.json"

        with tempfile.TemporaryDirectory() as temporary_directory:
            temp_dir = Path(temporary_directory)
            subset_jsonl = temp_dir / "subset.jsonl"
            selected_bill_ids = _write_bill_subset(source_jsonl, subset_jsonl, max_bills=1)
            config_path, ner_config_path, storage_base_dir = _write_runtime_config_copies(
                project_root=project_root,
                temporary_directory=temp_dir,
                input_path=subset_jsonl,
            )

            results = run_full_corpus(
                project_root=project_root,
                run_id="live_full_pipeline_test",
                resume=False,
                config_path=config_path,
                ner_config_path=ner_config_path,
                prompt_config_path=prompt_config_path,
            )

            self.assertEqual(set(results.keys()), set(selected_bill_ids))
            run_dir = storage_base_dir / "runs" / "live_full_pipeline_test"
            self.assertTrue((run_dir / "preflight.json").exists())
            self.assertTrue((run_dir / "outputs").exists())

            for bill_id in selected_bill_ids:
                self.assertTrue((run_dir / "grouped" / f"{bill_id}.json").exists())
                self.assertTrue((run_dir / "stage_state" / bill_id / "annotation.json").exists())
                self.assertTrue((run_dir / "stage_state" / bill_id / "grouping.json").exists())
                self.assertTrue((run_dir / "stage_state" / bill_id / "refinement.json").exists())

            output_store = FinalOutputStore(storage_base_dir)
            loaded_outputs = output_store.load_all("live_full_pipeline_test")
            self.assertEqual(set(loaded_outputs.keys()), set(selected_bill_ids))
            for bill_outputs in loaded_outputs.values():
                for refined_output in bill_outputs:
                    self.assertIsInstance(refined_output.source_group_id, int)
                    self.assertGreaterEqual(len(refined_output.source_candidate_ids), 1)
                    for evidence_field in (
                        refined_output.entity_evidence,
                        refined_output.type_evidence,
                        refined_output.attribute_evidence,
                        refined_output.value_evidence,
                    ):
                        for span in evidence_field:
                            self.assertIsInstance(span.chunk_id, int)

    def test_single_bill_entry_point_runs_on_one_existing_bill(self) -> None:
        """Verify the single-bill online entry point persists its artifacts."""

        project_root = Path(__file__).resolve().parents[3]
        source_jsonl = project_root / "data" / "ncsl" / "us_ai_legislation_ncsl_text.jsonl"
        prompt_config_path = project_root / "settings" / "ner_prompts.json"

        with tempfile.TemporaryDirectory() as temporary_directory:
            temp_dir = Path(temporary_directory)
            subset_jsonl = temp_dir / "subset.jsonl"
            [selected_bill_id] = _write_bill_subset(source_jsonl, subset_jsonl, max_bills=1)
            config_path, ner_config_path, storage_base_dir = _write_runtime_config_copies(
                project_root=project_root,
                temporary_directory=temp_dir,
                input_path=subset_jsonl,
            )

            refined_outputs = run_single_bill(
                project_root=project_root,
                bill_id=selected_bill_id,
                run_id="live_single_bill_test",
                resume=False,
                config_path=config_path,
                ner_config_path=ner_config_path,
                prompt_config_path=prompt_config_path,
            )

            run_dir = storage_base_dir / "runs" / "live_single_bill_test"
            self.assertTrue((run_dir / "outputs" / f"{selected_bill_id}.json").exists())
            self.assertTrue((run_dir / "grouped" / f"{selected_bill_id}.json").exists())
            self.assertIsInstance(refined_outputs, list)


def _write_bill_subset(source_jsonl: Path, destination_jsonl: Path, max_bills: int) -> list[str]:
    """Write a small AI-policy JSONL subset used by live integration tests.

    Args:
        source_jsonl: Source corpus JSONL path.
        destination_jsonl: Destination JSONL subset path.
        max_bills: Maximum number of eligible bills to copy into the subset.

    Returns:
        Bill ids written into the temporary subset file.

    Raises:
        RuntimeError: If not enough eligible AI-policy bills are found.
    """

    selected_rows: list[dict[str, object]] = []
    with open(source_jsonl, encoding="utf-8") as source_handle:
        for raw_line in source_handle:
            row = json.loads(raw_line)
            title = str(row.get("title", "")).lower()
            summary = str(row.get("summary", "")).lower()
            topics = str(row.get("topics", "")).lower()
            text = str(row.get("text", ""))
            text_lower = text.lower()
            ai_signal = (
                "artificial intelligence" in title
                or "artificial intelligence" in summary
                or "artificial intelligence" in text_lower
                or " ai " in f" {title} "
                or " ai " in f" {summary} "
                or " ai " in f" {topics} "
            )
            if not ai_signal:
                continue
            if len(text) > 18000:
                continue
            selected_rows.append(row)
            if len(selected_rows) == max_bills:
                break

    if len(selected_rows) != max_bills:
        raise RuntimeError(f"Unable to select {max_bills} existing AI-policy bills")

    with open(destination_jsonl, "w", encoding="utf-8") as destination_handle:
        for row in selected_rows:
            destination_handle.write(json.dumps(row, ensure_ascii=False))
            destination_handle.write("\n")

    qualified_ids: list[str] = []
    for row in selected_rows:
        raw_bid = str(row["bill_id"])
        year = str(row.get("year", ""))
        qualified_ids.append(f"{year}__{raw_bid}" if year else raw_bid)
    return qualified_ids


def _write_runtime_config_copies(
    project_root: Path,
    temporary_directory: Path,
    input_path: Path,
    model_name: str | None = None,
    storage_base_dir_override: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Write temporary config copies pointing integration tests at temp paths.

    Args:
        project_root: Project root containing the source config files.
        temporary_directory: Temporary directory used for derived config files.
        input_path: Temporary corpus path used by the live integration test.
        model_name: Optional LLM model name override.
        storage_base_dir_override: Optional explicit storage base directory.
            When omitted, storage is placed under ``temporary_directory``.

    Returns:
        Tuple of temporary project config path, temporary NER config path, and
        storage base directory.
    """

    with open(project_root / "settings" / "config.yml", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)
    with open(project_root / "settings" / "ner_config.yml", encoding="utf-8") as handle:
        ner_config = yaml.safe_load(handle)

    storage_base_dir = storage_base_dir_override or (temporary_directory / "ner_runs")
    base_config["input_path"] = str(input_path)
    if model_name is not None:
        base_config["llm"]["model_name"] = model_name
    ner_config["storage"]["base_dir"] = str(storage_base_dir)

    config_path = temporary_directory / "config.yml"
    ner_config_path = temporary_directory / "ner_config.yml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False)
    with open(ner_config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(ner_config, handle, sort_keys=False)
    return config_path, ner_config_path, storage_base_dir


@unittest.skipUnless(LIVE_TESTS_ENABLED, "Set NER_RUN_LIVE_TESTS=1 to run live vLLM integration tests")
class ModelComparisonTests(unittest.TestCase):
    """Run a small bill subset through different LLM models for comparison.

    Artifacts are persisted to ``data/ner_runs`` so results can be inspected
    and compared after the run.
    """

    _N_BILLS = 5

    def test_claude_sonnet_5bills(self) -> None:
        """Run 5 AI-relevant bills through Claude Sonnet 4.5 on OpenRouter."""

        self._run_model_test(
            model_name="anthropic/claude-sonnet-4.5",
            run_id="test_sonnet45_5bills",
        )

    def _run_model_test(self, model_name: str, run_id: str) -> None:
        project_root = Path(__file__).resolve().parents[3]
        source_jsonl = project_root / "data" / "ncsl" / "us_ai_legislation_ncsl_text.jsonl"
        prompt_config_path = project_root / "settings" / "ner_prompts.json"

        storage_base_dir = project_root / "data" / "ner_runs"
        config_dir = storage_base_dir / "_tmp_configs"
        config_dir.mkdir(parents=True, exist_ok=True)

        subset_jsonl = config_dir / "model_test_subset.jsonl"
        selected_bill_ids = _write_bill_subset(source_jsonl, subset_jsonl, max_bills=self._N_BILLS)
        config_path, ner_config_path, _ = _write_runtime_config_copies(
            project_root=project_root,
            temporary_directory=config_dir,
            input_path=subset_jsonl,
            model_name=model_name,
            storage_base_dir_override=storage_base_dir,
        )

        import sys
        import time

        t0 = time.perf_counter()
        results = run_full_corpus(
            project_root=project_root,
            run_id=run_id,
            resume=False,
            config_path=config_path,
            ner_config_path=ner_config_path,
            prompt_config_path=prompt_config_path,
        )
        elapsed = time.perf_counter() - t0

        self.assertEqual(set(results.keys()), set(selected_bill_ids))

        total_refined = sum(len(v) for v in results.values())
        print(file=sys.stderr)
        print("=" * 90, file=sys.stderr)
        print(f"  MODEL TEST: {model_name}  run_id={run_id}", file=sys.stderr)
        print(f"  {len(results)} bills, {total_refined} refined, {elapsed:.1f}s", file=sys.stderr)
        print("=" * 90, file=sys.stderr)
        for bid in selected_bill_ids:
            n = len(results.get(bid, []))
            print(f"  {bid:<30s}  refined={n}", file=sys.stderr)
        run_dir = storage_base_dir / "runs" / run_id
        print(f"  Artifacts: {run_dir}", file=sys.stderr)
        if (run_dir / "usage_summary.json").exists():
            with open(run_dir / "usage_summary.json", encoding="utf-8") as handle:
                usage = json.load(handle)
            print(
                f"  Usage: calls={usage['total_calls']}  "
                f"tokens={usage['total_tokens']}  "
                f"cost=${usage['total_cost_usd']:.6f}",
                file=sys.stderr,
            )
        print("=" * 90, file=sys.stderr)


if __name__ == "__main__":
    unittest.main()

