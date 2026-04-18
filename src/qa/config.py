"""Configuration loading and validation for the local QA app.

- Loads QA-specific settings from YAML while reusing project-level corpus and
  chunking defaults when helpful.
- Resolves relative filesystem paths against the project root and supports
  secure provider-key loading.
- Does not build indexes, call the provider API, or serve the web app.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .secrets import SecretStoreError, load_secret


class QAConfigValidationError(ValueError):
    """Raised when required QA configuration or environment values are missing."""


@dataclass(slots=True)
class QAChunkingConfig:
    """Chunking settings used for QA retrieval rather than NER extraction."""

    chunk_size: int
    overlap: int

    def validate(self) -> None:
        """Validate QA chunking settings."""

        if self.chunk_size <= 0:
            raise QAConfigValidationError("QA chunk_size must be > 0")
        if self.overlap < 0:
            raise QAConfigValidationError("QA overlap must be >= 0")
        if self.overlap >= self.chunk_size:
            raise QAConfigValidationError("QA overlap must be smaller than chunk_size")


@dataclass(slots=True)
class QAIndexConfig:
    """Index storage and retrieval settings for the QA app."""

    cache_dir: str
    batch_size: int
    retrieval_top_k: int

    def validate(self) -> None:
        """Validate index settings."""

        if not self.cache_dir.strip():
            raise QAConfigValidationError("QA cache_dir must be a non-empty string")
        if self.batch_size <= 0:
            raise QAConfigValidationError("QA batch_size must be > 0")
        if self.retrieval_top_k <= 0:
            raise QAConfigValidationError("QA retrieval_top_k must be > 0")


@dataclass(slots=True)
class ProviderConfig:
    """Provider settings for the OpenAI-compatible model endpoint."""

    api_base_url: str
    api_key_env_var: str
    keyring_service: str
    keyring_username: str

    def validate(self) -> None:
        """Validate provider endpoint and secret-loading settings."""

        required_values = {
            "api_base_url": self.api_base_url,
            "api_key_env_var": self.api_key_env_var,
            "keyring_service": self.keyring_service,
            "keyring_username": self.keyring_username,
        }
        missing = [key for key, value in required_values.items() if not value.strip()]
        if missing:
            raise QAConfigValidationError(
                f"Missing required provider settings: {missing}"
            )


@dataclass(slots=True)
class ModelConfig:
    """Embedding and answer model settings for the QA app."""

    embedding_model: str
    answer_model: str
    available_answer_models: tuple[str, ...]
    filter_extractor_model: str
    worker_model: str

    def validate(self) -> None:
        """Validate configured embedding and answer models."""

        required_values = {
            "embedding_model": self.embedding_model,
            "answer_model": self.answer_model,
            "filter_extractor_model": self.filter_extractor_model,
            "worker_model": self.worker_model,
        }
        missing = [key for key, value in required_values.items() if not value.strip()]
        if missing:
            raise QAConfigValidationError(
                f"Missing required model settings: {missing}"
            )
        if not self.available_answer_models:
            raise QAConfigValidationError(
                "ModelConfig.available_answer_models must contain at least one model id"
            )
        if any(not model_name.strip() for model_name in self.available_answer_models):
            raise QAConfigValidationError(
                "ModelConfig.available_answer_models cannot contain empty model ids"
            )
        if self.answer_model not in self.available_answer_models:
            raise QAConfigValidationError(
                "ModelConfig.answer_model must be included in available_answer_models"
            )


@dataclass(slots=True)
class AgentConfig:
    """Hyperparameters for the planner-workers agent path.

    The planner model is NOT configured here; it is the currently-selected
    ``answer_model`` at query time so the UI dropdown controls the orchestrator.
    """

    max_planner_turns: int
    max_planner_tokens: int
    planner_temperature: float
    max_worker_tokens: int
    worker_temperature: float
    max_tool_calls: int
    max_worker_calls: int
    max_bills_per_list: int
    max_chunks_per_bill: int
    max_citations_per_bill: int

    def validate(self) -> None:
        """Validate agent hyperparameters."""

        positive_int_fields = {
            "max_planner_turns": self.max_planner_turns,
            "max_planner_tokens": self.max_planner_tokens,
            "max_worker_tokens": self.max_worker_tokens,
            "max_tool_calls": self.max_tool_calls,
            "max_worker_calls": self.max_worker_calls,
            "max_bills_per_list": self.max_bills_per_list,
            "max_chunks_per_bill": self.max_chunks_per_bill,
            "max_citations_per_bill": self.max_citations_per_bill,
        }
        for field_name, value in positive_int_fields.items():
            if not isinstance(value, int) or value <= 0:
                raise QAConfigValidationError(
                    f"AgentConfig.{field_name} must be a positive integer"
                )
        for temperature_name, temperature_value in (
            ("planner_temperature", self.planner_temperature),
            ("worker_temperature", self.worker_temperature),
        ):
            if (
                not isinstance(temperature_value, (int, float))
                or temperature_value < 0.0
                or temperature_value > 2.0
            ):
                raise QAConfigValidationError(
                    f"AgentConfig.{temperature_name} must be within [0.0, 2.0]"
                )


@dataclass(slots=True)
class QAAppConfig:
    """Host and port settings for the local browser app."""

    host: str
    port: int
    show_trace: bool = False

    def validate(self) -> None:
        """Validate browser-app host and port settings."""

        if not self.host.strip():
            raise QAConfigValidationError("QA app host must be a non-empty string")
        if self.port <= 0 or self.port > 65535:
            raise QAConfigValidationError("QA app port must be between 1 and 65535")
        if not isinstance(self.show_trace, bool):
            raise QAConfigValidationError("QA app show_trace must be a boolean")


@dataclass(slots=True)
class QAConfig:
    """Resolved runtime configuration for the local QA app."""

    corpus_path: str
    chunking: QAChunkingConfig
    index: QAIndexConfig
    provider: ProviderConfig
    models: ModelConfig
    app: QAAppConfig
    agent: AgentConfig

    def validate(self) -> None:
        """Validate the full resolved QA configuration."""

        if not self.corpus_path.strip():
            raise QAConfigValidationError("QA corpus_path must be a non-empty string")
        self.chunking.validate()
        self.index.validate()
        self.provider.validate()
        self.models.validate()
        self.app.validate()
        self.agent.validate()

    def resolve_corpus_path(self, project_root: Path) -> Path:
        """Resolve the configured corpus path relative to the project root."""

        return _resolve_project_path(project_root, self.corpus_path)

    def resolve_cache_dir(self, project_root: Path) -> Path:
        """Resolve the configured cache directory relative to the project root."""

        return _resolve_project_path(project_root, self.index.cache_dir)


def load_qa_config(
    project_root: Path,
    qa_config_path: Path | None = None,
    base_config_path: Path | None = None,
    ner_config_path: Path | None = None,
) -> QAConfig:
    """Load the QA app configuration from YAML files."""

    settings_dir = project_root / "settings"
    qa_payload = _load_yaml(qa_config_path or settings_dir / "qa_config.yml")
    base_payload = _load_yaml(base_config_path or settings_dir / "config.yml")
    ner_payload = _load_yaml(ner_config_path or settings_dir / "ner_config.yml")

    base_corpus_path = str(
        base_payload.get("input_path")
        or base_payload.get("input_csv")
        or ""
    )
    ner_chunking = ner_payload.get("chunking", {})
    qa_chunking = qa_payload.get("chunking", {})
    qa_index = qa_payload.get("index", {})
    qa_provider = qa_payload.get("provider", {})
    qa_models = qa_payload.get("models", {})
    qa_app = qa_payload.get("app", {})
    qa_agent = qa_payload.get("agent", {})

    config = QAConfig(
        corpus_path=str(
            os.environ.get("QA_CORPUS_PATH")
            or qa_payload.get("corpus", {}).get("path")
            or base_corpus_path
        ),
        chunking=QAChunkingConfig(
            chunk_size=int(qa_chunking.get("chunk_size", ner_chunking.get("chunk_size", 1200))),
            overlap=int(qa_chunking.get("overlap", ner_chunking.get("overlap", 150))),
        ),
        index=QAIndexConfig(
            cache_dir=str(os.environ.get("QA_CACHE_DIR") or qa_index.get("cache_dir", "data/qa_cache")),
            batch_size=int(qa_index.get("batch_size", 32)),
            retrieval_top_k=int(qa_index.get("retrieval_top_k", 5)),
        ),
        provider=ProviderConfig(
            api_base_url=str(qa_provider.get("api_base_url", "")).strip(),
            api_key_env_var=str(qa_provider.get("api_key_env_var", "OPENROUTER_API_KEY")).strip(),
            keyring_service=str(qa_provider.get("keyring_service", "ai_policy.qa")).strip(),
            keyring_username=str(qa_provider.get("keyring_username", "openrouter")).strip(),
        ),
        models=ModelConfig(
            embedding_model=str(qa_models.get("embedding_model", "")).strip(),
            answer_model=str(qa_models.get("answer_model", "")).strip(),
            available_answer_models=tuple(
                str(model_name).strip()
                for model_name in (
                    qa_models.get("available_answer_models")
                    or [qa_models.get("answer_model", "")]
                )
            ),
            filter_extractor_model=str(
                qa_models.get("filter_extractor_model")
                or qa_models.get("answer_model", "")
            ).strip(),
            worker_model=str(
                qa_models.get("worker_model")
                or qa_models.get("answer_model", "")
            ).strip(),
        ),
        app=QAAppConfig(
            host=str(qa_app.get("host", "127.0.0.1")).strip(),
            port=int(qa_app.get("port", 5050)),
            show_trace=bool(qa_app.get("show_trace", False)),
        ),
        agent=AgentConfig(
            max_planner_turns=int(qa_agent.get("max_planner_turns", 8)),
            max_planner_tokens=int(qa_agent.get("max_planner_tokens", 4096)),
            planner_temperature=float(qa_agent.get("planner_temperature", 0.0)),
            max_worker_tokens=int(qa_agent.get("max_worker_tokens", 1024)),
            worker_temperature=float(qa_agent.get("worker_temperature", 0.0)),
            max_tool_calls=int(qa_agent.get("max_tool_calls", 16)),
            max_worker_calls=int(qa_agent.get("max_worker_calls", 6)),
            max_bills_per_list=int(qa_agent.get("max_bills_per_list", 50)),
            max_chunks_per_bill=int(qa_agent.get("max_chunks_per_bill", 6)),
            max_citations_per_bill=int(qa_agent.get("max_citations_per_bill", 2)),
        ),
    )
    config.validate()
    return config


def load_provider_api_key(
    provider_config: ProviderConfig,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Load the provider API key from the environment or OS keyring."""

    env = environment or os.environ
    api_key = str(env.get(provider_config.api_key_env_var, "")).strip()
    if api_key:
        return api_key
    try:
        return load_secret(
            service=provider_config.keyring_service,
            username=provider_config.keyring_username,
        )
    except SecretStoreError as error:
        raise QAConfigValidationError(
            f"{provider_config.api_key_env_var} is not set and no keyring secret was found "
            f"for service='{provider_config.keyring_service}' "
            f"username='{provider_config.keyring_username}'."
        ) from error


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML file into a dictionary."""

    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise QAConfigValidationError(f"Config file '{path}' must contain a mapping")
    return payload


def _resolve_project_path(project_root: Path, configured_path: str) -> Path:
    """Resolve relative config paths against the project root."""

    path = Path(configured_path)
    return path if path.is_absolute() else project_root / path


__all__ = [
    "AgentConfig",
    "ModelConfig",
    "ProviderConfig",
    "QAAppConfig",
    "QAChunkingConfig",
    "QAConfig",
    "QAConfigValidationError",
    "QAIndexConfig",
    "load_provider_api_key",
    "load_qa_config",
]
