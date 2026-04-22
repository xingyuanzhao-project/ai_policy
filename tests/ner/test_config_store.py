"""Config-store tests for runtime bill-length skipping settings.

- Verifies the runtime skip threshold accepts valid values.
- Verifies invalid threshold values fail fast during config loading.
- Does not bootstrap the full runtime.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.ner.storage.config_store import ConfigStore, ConfigValidationError


class ConfigStoreTests(unittest.TestCase):
    """Verify validation of optional runtime config values."""

    def test_load_accepts_positive_max_bill_text_chars(self) -> None:
        """Verify a positive skip threshold is accepted and preserved."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            base_config_path, ner_config_path, prompt_config_path = _write_config_files(
                Path(temporary_directory),
                max_bill_text_chars=500_000,
            )
            config_store = ConfigStore()
            config_store.load(
                base_config_path=base_config_path,
                ner_config_path=ner_config_path,
                prompt_config_path=prompt_config_path,
            )

        self.assertEqual(config_store.runtime_config["max_bill_text_chars"], 500_000)

    def test_load_rejects_non_positive_max_bill_text_chars(self) -> None:
        """Verify zero and negative thresholds are rejected."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            base_config_path, ner_config_path, prompt_config_path = _write_config_files(
                Path(temporary_directory),
                max_bill_text_chars=0,
            )
            config_store = ConfigStore()
            with self.assertRaises(ConfigValidationError):
                config_store.load(
                    base_config_path=base_config_path,
                    ner_config_path=ner_config_path,
                    prompt_config_path=prompt_config_path,
                )

    def test_load_rejects_non_integer_max_bill_text_chars(self) -> None:
        """Verify non-integer thresholds are rejected."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            base_config_path, ner_config_path, prompt_config_path = _write_config_files(
                Path(temporary_directory),
                max_bill_text_chars="too-long",
            )
            config_store = ConfigStore()
            with self.assertRaises(ConfigValidationError):
                config_store.load(
                    base_config_path=base_config_path,
                    ner_config_path=ner_config_path,
                    prompt_config_path=prompt_config_path,
                )


def _write_config_files(
    temp_dir: Path,
    max_bill_text_chars: object,
) -> tuple[Path, Path, Path]:
    """Write minimal config fixtures for ConfigStore tests."""

    base_config_path = temp_dir / "config.yml"
    ner_config_path = temp_dir / "ner_config.yml"
    prompt_config_path = temp_dir / "ner_prompts.json"

    base_config = {
        "input_path": "data/test.jsonl",
        "llm": {
            "base_url": "http://localhost:8000/v1",
            "api_key": "test-key",
            "model_name": "test-model",
            "temperature": 0.0,
            "max_tokens": 256,
        },
    }
    ner_config = {
        "chunking": {"chunk_size": 3000, "overlap": 300},
        "storage": {"base_dir": "data/ner_runs"},
        "runtime": {"max_bill_text_chars": max_bill_text_chars},
        "agents": {
            "zero_shot_annotator": {"temperature": 0.0, "max_tokens": 256},
            "eval_assembler": {"temperature": 0.0, "max_tokens": 256},
            "granularity_refiner": {"temperature": 0.0, "max_tokens": 256},
        },
    }
    prompt_config = {
        "zero_shot_annotator": {
            "prompt_template": "{text}",
            "output_schema": {"type": "object"},
        },
        "eval_assembler": {
            "prompt_template": "{text}",
            "output_schema": {"type": "object"},
        },
        "granularity_refiner": {
            "prompt_template": "{text}",
            "output_schema": {"type": "object"},
        },
    }

    with open(base_config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False)
    with open(ner_config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(ner_config, handle, sort_keys=False)
    with open(prompt_config_path, "w", encoding="utf-8") as handle:
        json.dump(prompt_config, handle, indent=2)

    return base_config_path, ner_config_path, prompt_config_path
