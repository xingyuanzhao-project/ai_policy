"""Local-answer support for the QA browser app.

- Reads the configured local OpenAI-compatible runtime from `settings/config.yml`.
- Probes only that configured host and exposes the configured local model when
  the runtime is healthy.
- Owns local answer-model option metadata and local answer-client resolution so
  the rest of the QA stack stays thin.
- Does not affect retrieval, embeddings, or index cache semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

from src.ner.runtime.llm_client import LLMClient, LLMConfig

from .gemini_client import OpenAICompatibleClient

_LOCAL_OPTION_PREFIX = "local::"
_LOCAL_OPTION_LABEL_PREFIX = "Local / "
_LOCAL_CONFIG_FILENAME = "config.yml"
_LOCAL_SETTINGS_DIRNAME = "settings"
_DEFAULT_LOCAL_MAX_RETRIES = 2


@dataclass(frozen=True, slots=True)
class AnswerModelOption:
    """One answer-model option shown in the QA dropdown."""

    option_id: str
    label: str


@dataclass(slots=True)
class LocalAnswerTarget:
    """Resolved local answer target for one dropdown option."""

    option: AnswerModelOption
    raw_model_name: str
    client: OpenAICompatibleClient


class LocalAnswerSupport:
    """Detect and expose the configured local answer model for QA."""

    def __init__(self, answer_targets: Sequence[LocalAnswerTarget] | None = None) -> None:
        """Initialize the local-answer support wrapper."""

        self._answer_targets = tuple(answer_targets or ())
        self._targets_by_option_id = {
            target.option.option_id: target for target in self._answer_targets
        }
        if len(self._targets_by_option_id) != len(self._answer_targets):
            raise ValueError("Local answer option ids must be unique")

    @classmethod
    def from_project_root(cls, project_root: Path) -> LocalAnswerSupport:
        """Build local-answer support from the configured local runtime."""

        config_path = project_root / _LOCAL_SETTINGS_DIRNAME / _LOCAL_CONFIG_FILENAME
        try:
            payload = _load_yaml(config_path)
        except (FileNotFoundError, OSError, ValueError):
            return cls()

        llm_payload = payload.get("llm")
        if not isinstance(llm_payload, dict):
            return cls()

        local_runtime_config = _read_local_runtime_config(llm_payload)
        if local_runtime_config is None:
            return cls()
        if not _configured_model_is_available(local_runtime_config):
            return cls()

        option = AnswerModelOption(
            option_id=f"{_LOCAL_OPTION_PREFIX}{local_runtime_config.model_name}",
            label=f"{_LOCAL_OPTION_LABEL_PREFIX}{local_runtime_config.model_name}",
        )
        client = OpenAICompatibleClient(
            api_key=local_runtime_config.api_key,
            api_base_url=local_runtime_config.base_url,
            embedding_model=local_runtime_config.model_name,
            answer_model=local_runtime_config.model_name,
        )
        return cls(
            [
                LocalAnswerTarget(
                    option=option,
                    raw_model_name=local_runtime_config.model_name,
                    client=client,
                )
            ]
        )

    @property
    def answer_model_options(self) -> tuple[AnswerModelOption, ...]:
        """Return local answer-model options that passed runtime health checks."""

        return tuple(target.option for target in self._answer_targets)

    def resolve_answer_model(self, option_id: str) -> LocalAnswerTarget | None:
        """Return the local target for one dropdown option id."""

        return self._targets_by_option_id.get(option_id)

    def label_for(self, option_id: str) -> str | None:
        """Return the display label for one local option id."""

        target = self.resolve_answer_model(option_id)
        if target is None:
            return None
        return target.option.label

    def close(self) -> None:
        """Close any local answer clients owned by the support wrapper."""

        for target in self._answer_targets:
            target.client.close()


@dataclass(frozen=True, slots=True)
class _LocalRuntimeConfig:
    """Configured local OpenAI-compatible runtime settings."""

    base_url: str
    api_key: str
    model_name: str
    temperature: float
    max_tokens: int
    max_retries: int


def _read_local_runtime_config(payload: dict[str, Any]) -> _LocalRuntimeConfig | None:
    """Parse one local runtime config block from the project settings file."""

    base_url = str(payload.get("base_url", "")).strip()
    api_key = str(payload.get("api_key", "")).strip()
    model_name = str(payload.get("model_name", "")).strip()
    if not base_url or not api_key or not model_name:
        return None
    return _LocalRuntimeConfig(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        temperature=float(payload.get("temperature", 0.0)),
        max_tokens=int(payload.get("max_tokens", 1024)),
        max_retries=int(payload.get("max_retries", _DEFAULT_LOCAL_MAX_RETRIES)),
    )


def _configured_model_is_available(local_runtime_config: _LocalRuntimeConfig) -> bool:
    """Return whether the configured local model is served by the configured host."""

    probe_client = LLMClient(
        LLMConfig(
            base_url=local_runtime_config.base_url,
            api_key=local_runtime_config.api_key,
            model_name=local_runtime_config.model_name,
            temperature=local_runtime_config.temperature,
            max_tokens=local_runtime_config.max_tokens,
            max_retries=local_runtime_config.max_retries,
        )
    )
    try:
        probe_client.connect()
        return local_runtime_config.model_name in probe_client.list_models()
    except Exception:
        return False
    finally:
        probe_client.close()


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML file into a dictionary."""

    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file '{path}' must contain a mapping")
    return payload


__all__ = ["AnswerModelOption", "LocalAnswerSupport", "LocalAnswerTarget"]
