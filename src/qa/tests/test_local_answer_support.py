"""Offline tests for config-driven local answer-model support."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.qa.local_answer_support import LocalAnswerSupport


class LocalAnswerSupportTests(unittest.TestCase):
    """Verify local answer support uses only the configured local runtime."""

    def test_from_project_root_exposes_configured_local_model_when_runtime_is_healthy(
        self,
    ) -> None:
        """Verify a healthy configured local runtime yields one local dropdown option."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            _write_base_config(
                project_root,
                model_name="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
            )
            with patch("src.qa.local_answer_support.LLMClient") as mock_llm_client:
                mock_llm_client.return_value.list_models.return_value = [
                    "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
                ]

                support = LocalAnswerSupport.from_project_root(project_root)

            self.assertEqual(len(support.answer_model_options), 1)
            option = support.answer_model_options[0]
            self.assertEqual(
                option.option_id,
                "local::hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
            )
            self.assertEqual(
                option.label,
                "Local / hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
            )
            resolved_target = support.resolve_answer_model(option.option_id)
            assert resolved_target is not None
            self.assertEqual(
                resolved_target.raw_model_name,
                "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
            )
            support.close()

    def test_from_project_root_ignores_local_runtime_when_configured_model_is_missing(
        self,
    ) -> None:
        """Verify the helper stays disabled when the configured local model is absent."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            _write_base_config(
                project_root,
                model_name="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
            )
            with patch("src.qa.local_answer_support.LLMClient") as mock_llm_client:
                mock_llm_client.return_value.list_models.return_value = [
                    "some-other-model"
                ]

                support = LocalAnswerSupport.from_project_root(project_root)

            self.assertEqual(support.answer_model_options, ())
            self.assertIsNone(
                support.resolve_answer_model(
                    "local::hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
                )
            )


def _write_base_config(project_root: Path, *, model_name: str) -> None:
    """Write the minimal project config block used by local-answer detection."""

    settings_dir = project_root / "settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    (settings_dir / "config.yml").write_text(
        "\n".join(
            [
                "llm:",
                "  base_url: http://localhost:8000/v1",
                "  api_key: dummy",
                f"  model_name: {model_name}",
                "  temperature: 0.0",
                "  max_tokens: 1024",
                "  max_retries: 2",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
