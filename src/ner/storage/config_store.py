"""NER configuration loading, validation, and run snapshotting.

- Loads project config, NER runtime config, and NER prompt config.
- Validates that required sections and keys exist before runtime bootstrap.
- Writes run-scoped config snapshots for reproducibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

REQUIRED_LLM_KEYS: tuple[str, str, str, str, str] = (
    "base_url",
    "api_key",
    "model_name",
    "temperature",
    "max_tokens",
)
REQUIRED_NER_SECTIONS: tuple[str, str, str] = ("chunking", "storage", "agents")
REQUIRED_AGENT_NAMES: tuple[str, str, str] = (
    "zero_shot_annotator",
    "eval_assembler",
    "granularity_refiner",
)


class ConfigValidationError(ValueError):
    """Raised when required NER configuration keys or sections are missing."""


class ConfigStore:
    """Load, validate, and snapshot the NER pipeline configuration."""

    def __init__(self) -> None:
        """Initialize an empty configuration store."""

        self._base_config: dict[str, Any] = {}
        self._ner_config: dict[str, Any] = {}
        self._prompt_config: dict[str, Any] = {}
        self._base_config_path: Path | None = None
        self._ner_config_path: Path | None = None
        self._prompt_config_path: Path | None = None

    def load(
        self,
        base_config_path: Path,
        ner_config_path: Path,
        prompt_config_path: Path,
    ) -> None:
        """Load base config, NER config, and prompt config from disk.

        Args:
            base_config_path (Path): Project-level config path.
            ner_config_path (Path): NER-specific runtime config path.
            prompt_config_path (Path): NER prompt and output-schema config
                path.

        Returns:
            None: This method loads config state into the store in place.
        """

        self._base_config_path = base_config_path
        self._ner_config_path = ner_config_path
        self._prompt_config_path = prompt_config_path

        with open(base_config_path, encoding="utf-8") as handle:
            self._base_config = yaml.safe_load(handle) or {}
        with open(ner_config_path, encoding="utf-8") as handle:
            self._ner_config = yaml.safe_load(handle) or {}
        with open(prompt_config_path, encoding="utf-8") as handle:
            self._prompt_config = json.load(handle)

        self.validate()

    def validate(self) -> None:
        """Validate the required configuration surface before any agent runs.

        Returns:
            None: This method validates in place and raises on invalid config.

        Raises:
            ConfigValidationError: If required config sections or keys are
                missing from the loaded configuration files.
        """

        llm_config = self._base_config.get("llm", {})
        missing_llm_keys = [key for key in REQUIRED_LLM_KEYS if key not in llm_config]
        if missing_llm_keys:
            raise ConfigValidationError(
                f"Missing required llm config keys: {missing_llm_keys}"
            )

        missing_sections = [
            section for section in REQUIRED_NER_SECTIONS if section not in self._ner_config
        ]
        if missing_sections:
            raise ConfigValidationError(
                f"Missing required ner config sections: {missing_sections}"
            )

        missing_agents = [
            agent_name
            for agent_name in REQUIRED_AGENT_NAMES
            if agent_name not in self._ner_config.get("agents", {})
        ]
        if missing_agents:
            raise ConfigValidationError(
                f"Missing required agent config blocks: {missing_agents}"
            )

        missing_prompt_blocks = [
            agent_name
            for agent_name in REQUIRED_AGENT_NAMES
            if agent_name not in self._prompt_config
        ]
        if missing_prompt_blocks:
            raise ConfigValidationError(
                f"Missing required prompt blocks: {missing_prompt_blocks}"
            )

        for agent_name in REQUIRED_AGENT_NAMES:
            prompt_block = self._prompt_config[agent_name]
            if "prompt_template" not in prompt_block:
                raise ConfigValidationError(
                    f"Prompt block '{agent_name}' is missing prompt_template"
                )
            if "output_schema" not in prompt_block:
                raise ConfigValidationError(
                    f"Prompt block '{agent_name}' is missing output_schema"
                )

        runtime_config = self._ner_config.get("runtime", {})
        if "max_bill_text_chars" in runtime_config:
            try:
                max_bill_text_chars = int(runtime_config["max_bill_text_chars"])
            except (TypeError, ValueError) as exc:
                raise ConfigValidationError(
                    "runtime.max_bill_text_chars must be an integer > 0"
                ) from exc
            if max_bill_text_chars <= 0:
                raise ConfigValidationError(
                    "runtime.max_bill_text_chars must be an integer > 0"
                )

    @property
    def base_config(self) -> dict[str, Any]:
        """Return the loaded project-level config.

        Returns:
            dict[str, Any]: Parsed project-level configuration dictionary.
        """

        return self._base_config

    @property
    def llm_config(self) -> dict[str, Any]:
        """Return the configured shared LLM settings.

        Returns:
            dict[str, Any]: Parsed LLM configuration dictionary.
        """

        return self._base_config["llm"]

    @property
    def chunking_config(self) -> dict[str, Any]:
        """Return the configured chunking settings.

        Returns:
            dict[str, Any]: Parsed chunking configuration dictionary.
        """

        return self._ner_config["chunking"]

    @property
    def storage_config(self) -> dict[str, Any]:
        """Return the configured storage settings.

        Returns:
            dict[str, Any]: Parsed storage configuration dictionary.
        """

        return self._ner_config["storage"]

    @property
    def runtime_config(self) -> dict[str, Any]:
        """Return runtime-only execution settings for the NER pipeline.

        Returns:
            dict[str, Any]: Parsed runtime configuration dictionary.
        """

        return self._ner_config.get("runtime", {})

    def agent_config(self, agent_name: str) -> dict[str, Any]:
        """Return the configured execution settings for one named agent.

        Args:
            agent_name (str): Stable agent config key.

        Returns:
            dict[str, Any]: Parsed execution-settings dictionary for the
                requested agent.
        """

        return self._ner_config["agents"][agent_name]

    def prompt_config(self, agent_name: str) -> dict[str, Any]:
        """Return the configured prompt block for one named agent.

        Args:
            agent_name (str): Stable agent config key.

        Returns:
            dict[str, Any]: Parsed prompt block for the requested agent.
        """

        return self._prompt_config[agent_name]

    def input_path(self) -> str:
        """Return the configured bill-corpus path.

        `input_path` is the preferred key for the NER pipeline because the real
        corpus may be CSV or JSONL. `input_csv` remains supported as a fallback
        so existing project config does not break.

        Returns:
            str: Configured filesystem path to the bill corpus.
        """

        if "input_path" in self._base_config:
            return self._base_config["input_path"]
        return self._base_config["input_csv"]

    def snapshot(self, run_dir: Path) -> None:
        """Persist config snapshots so each run can be reproduced exactly.

        Args:
            run_dir (Path): Run-specific directory in which snapshots should be
                stored.

        Returns:
            None: This method writes configuration snapshots to disk.
        """

        run_dir.mkdir(parents=True, exist_ok=True)
        snapshots = (
            ("config_snapshot.yml", self._base_config),
            ("ner_config_snapshot.yml", self._ner_config),
        )
        for filename, payload in snapshots:
            with open(run_dir / filename, "w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False)

        with open(run_dir / "ner_prompts_snapshot.json", "w", encoding="utf-8") as handle:
            json.dump(self._prompt_config, handle, indent=2, ensure_ascii=False)

