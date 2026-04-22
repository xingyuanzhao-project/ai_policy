"""Pipeline bootstrap for the NER runtime.

- Owns runtime wiring of config, corpus, storage, chunking, agents, and
  orchestrator.
- Owns run-scoped preflight evidence and config snapshot creation.
- Owns provider-aware API key resolution (env var, then keyring fallback).
- Sets up run-scoped file logging so every log message is persisted to
  ``{run_dir}/run.log`` alongside the existing stderr stream.
- Does not execute bill processing loops itself.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_RUN_LOG_FORMAT = "%(asctime)s  %(name)s  %(levelname)s  %(message)s"

from ..agents import (
    AgentExecutionConfig,
    EvalAssembler,
    GranularityRefiner,
    PromptExecutor,
    ZeroShotAnnotator,
)
from ..orchestration.orchestrator import Orchestrator
from ..schemas.inference_unit_builder import ChunkingConfig, InferenceUnitBuilder
from ..storage import ArtifactStore, ConfigStore, CorpusStore, FinalOutputStore
from .llm_client import (
    STRUCTURED_OUTPUT_GUIDED_JSON,
    LLMClient,
    LLMConfig,
)


@dataclass(slots=True)
class PipelineContext:
    """Fully initialized runtime context ready for inference.

    Attributes:
        project_root (Path): Project-root path used for resolving relative
            config paths.
        run_id (str): Stable run identifier for persisted artifacts.
        config_store (ConfigStore): Loaded NER configuration store.
        corpus_store (CorpusStore): Corpus store used to load raw bill records.
        artifact_store (ArtifactStore): Store used for intermediate artifacts
            and stage state.
        final_output_store (FinalOutputStore): Store used for bill-level final
            outputs.
        llm_client (LLMClient): Connected shared LLM client.
        inference_unit_builder (InferenceUnitBuilder): Chunk builder used to
            derive context chunks.
        max_bill_text_chars (int | None): Optional bill-text length ceiling
            beyond which whole bills are skipped before chunking.
        orchestrator (Orchestrator): Stage orchestrator for annotation,
            grouping, and refinement.
        run_log_handler (logging.Handler | None): File handler writing to
            ``{run_dir}/run.log``, attached to the root logger.
    """

    project_root: Path
    run_id: str
    config_store: ConfigStore
    corpus_store: CorpusStore
    artifact_store: ArtifactStore
    final_output_store: FinalOutputStore
    llm_client: LLMClient
    inference_unit_builder: InferenceUnitBuilder
    max_bill_text_chars: int | None
    orchestrator: Orchestrator
    run_log_handler: logging.Handler | None = None

    def detach_run_log(self) -> None:
        """Remove the run-scoped file handler from the root logger."""
        if self.run_log_handler is not None:
            logging.getLogger().removeHandler(self.run_log_handler)
            if hasattr(self.run_log_handler, "close"):
                self.run_log_handler.close()
            self.run_log_handler = None


def bootstrap(
    project_root: Path,
    run_id: str,
    config_path: str | Path,
    ner_config_path: str | Path,
    prompt_config_path: str | Path,
) -> PipelineContext:
    """Bootstrap the full NER pipeline from config, stores, and shared runtime state.

    Args:
        project_root (Path): Project root used for resolving relative paths.
        run_id (str): Stable run identifier for persisted artifacts.
        config_path (str | Path): Project-level config file (relative to
            *project_root* or absolute).
        ner_config_path (str | Path): NER runtime config file.
        prompt_config_path (str | Path): NER prompt / schema config file.

    Returns:
        PipelineContext: Fully initialized pipeline context ready for
            inference.
    """

    config_store = ConfigStore()
    config_store.load(
        base_config_path=_resolve_project_path(project_root, str(config_path)),
        ner_config_path=_resolve_project_path(project_root, str(ner_config_path)),
        prompt_config_path=_resolve_project_path(project_root, str(prompt_config_path)),
    )

    storage_dir = _resolve_project_path(
        project_root,
        config_store.storage_config["base_dir"],
    )
    artifact_store = ArtifactStore(storage_dir)
    final_output_store = FinalOutputStore(storage_dir)
    run_dir = artifact_store.run_dir(run_id)
    config_store.snapshot(run_dir)
    raw_max_bill_text_chars = config_store.runtime_config.get("max_bill_text_chars")
    max_bill_text_chars = (
        None if raw_max_bill_text_chars is None else int(raw_max_bill_text_chars)
    )

    api_key = _resolve_api_key(config_store.llm_config)
    logger.info(
        "Bootstrap: run_id=%s  base_url=%s  model=%s  mode=%s  "
        "temperature=%s  max_tokens=%s  max_retries=%s  timeout=%ss  "
        "api_key_source=%s",
        run_id,
        config_store.llm_config["base_url"],
        config_store.llm_config["model_name"],
        config_store.llm_config.get("structured_output_mode", "guided_json"),
        config_store.llm_config.get("temperature"),
        config_store.llm_config.get("max_tokens"),
        config_store.runtime_config.get(
            "max_retries", config_store.llm_config.get("max_retries", 2)
        ),
        config_store.runtime_config.get("request_timeout_seconds", 60),
        "env_var" if os.environ.get(str(config_store.llm_config.get("api_key_env_var", "")), "") else "config/keyring",
    )
    logger.info(
        "Bootstrap runtime: concurrency=%s  chunk_size=%s  overlap=%s  "
        "max_bill_text_chars=%s  storage_dir=%s",
        config_store.runtime_config.get("concurrency"),
        config_store.chunking_config.get("chunk_size"),
        config_store.chunking_config.get("overlap"),
        max_bill_text_chars,
        storage_dir,
    )
    for agent_name in ("zero_shot_annotator", "eval_assembler", "granularity_refiner"):
        agent_cfg = config_store.agent_config(agent_name)
        logger.info(
            "Bootstrap agent %s: temperature=%s  max_tokens=%s",
            agent_name,
            agent_cfg.get("temperature"),
            agent_cfg.get("max_tokens"),
        )

    llm_client = LLMClient(
        LLMConfig(
            base_url=config_store.llm_config["base_url"],
            api_key=api_key,
            model_name=config_store.llm_config["model_name"],
            temperature=float(config_store.llm_config["temperature"]),
            max_tokens=int(config_store.llm_config["max_tokens"]),
            max_retries=int(
                config_store.runtime_config.get(
                    "max_retries",
                    config_store.llm_config.get("max_retries", 2),
                )
            ),
            request_timeout_seconds=float(
                config_store.runtime_config.get("request_timeout_seconds", 60)
            ),
            structured_output_mode=str(
                config_store.llm_config.get(
                    "structured_output_mode", STRUCTURED_OUTPUT_GUIDED_JSON
                )
            ),
            skip_model_listing=bool(
                config_store.llm_config.get("skip_model_listing", False)
            ),
        )
    )
    llm_client.connect()
    preflight_result = llm_client.verify_runtime()
    with open(run_dir / "preflight.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "resolved_model_name": preflight_result.resolved_model_name,
                "raw_probe_response": preflight_result.raw_probe_response,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    run_log_handler = _attach_run_log_handler(run_dir)
    logger.info("Run log attached: %s", run_dir / "run.log")

    corpus_store = CorpusStore(
        _resolve_project_path(project_root, config_store.input_path())
    )
    inference_unit_builder = InferenceUnitBuilder(
        ChunkingConfig(
            chunk_size=int(config_store.chunking_config["chunk_size"]),
            overlap=int(config_store.chunking_config["overlap"]),
        )
    )
    prompt_executor = PromptExecutor(llm_client)

    annotator = ZeroShotAnnotator(
        prompt_template=config_store.prompt_config("zero_shot_annotator")[
            "prompt_template"
        ],
        output_schema=config_store.prompt_config("zero_shot_annotator")[
            "output_schema"
        ],
        execution_config=AgentExecutionConfig(
            **config_store.agent_config("zero_shot_annotator")
        ),
        prompt_executor=prompt_executor,
    )
    assembler = EvalAssembler(
        prompt_template=config_store.prompt_config("eval_assembler")["prompt_template"],
        output_schema=config_store.prompt_config("eval_assembler")["output_schema"],
        execution_config=AgentExecutionConfig(
            **config_store.agent_config("eval_assembler")
        ),
        prompt_executor=prompt_executor,
    )
    refiner = GranularityRefiner(
        prompt_template=config_store.prompt_config("granularity_refiner")[
            "prompt_template"
        ],
        output_schema=config_store.prompt_config("granularity_refiner")[
            "output_schema"
        ],
        execution_config=AgentExecutionConfig(
            **config_store.agent_config("granularity_refiner")
        ),
        prompt_executor=prompt_executor,
    )
    orchestrator = Orchestrator(
        run_id=run_id,
        annotator=annotator,
        assembler=assembler,
        refiner=refiner,
        artifact_store=artifact_store,
        final_output_store=final_output_store,
        concurrency=int(config_store.runtime_config.get("concurrency", 4)),
    )

    return PipelineContext(
        project_root=project_root,
        run_id=run_id,
        config_store=config_store,
        corpus_store=corpus_store,
        artifact_store=artifact_store,
        final_output_store=final_output_store,
        llm_client=llm_client,
        inference_unit_builder=inference_unit_builder,
        max_bill_text_chars=max_bill_text_chars,
        orchestrator=orchestrator,
        run_log_handler=run_log_handler,
    )


def _attach_run_log_handler(run_dir: Path) -> logging.FileHandler:
    """Attach a file handler to the root logger that writes to ``run.log``.

    Args:
        run_dir: Run directory where the log file is created.

    Returns:
        The handler instance (stored so it can be removed on teardown).
    """

    log_path = run_dir / "run.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_RUN_LOG_FORMAT))
    root = logging.getLogger()
    root.addHandler(handler)
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)
    return handler


def _resolve_api_key(llm_config: dict[str, Any]) -> str:
    """Resolve the effective API key from config, env var, or keyring.

    Resolution order:

    1. If ``api_key_env_var`` is set and non-empty, read the key from that
       environment variable.
    2. If the env var is unset and ``keyring_service`` / ``keyring_username``
       are configured, attempt a keyring lookup.
    3. Otherwise fall back to the literal ``api_key`` value in config (the
       original local-vLLM path where ``api_key: dummy`` is typical).

    Args:
        llm_config: The ``llm`` block from the loaded project config.

    Returns:
        str: Resolved API key string.

    Raises:
        RuntimeError: If the env var is configured but empty/unset and the
            keyring lookup also fails.
    """

    api_key_env_var = str(llm_config.get("api_key_env_var", "")).strip()
    if not api_key_env_var:
        return str(llm_config["api_key"])

    api_key = os.environ.get(api_key_env_var, "").strip()
    if api_key:
        return api_key

    keyring_service = str(llm_config.get("keyring_service", "")).strip()
    keyring_username = str(llm_config.get("keyring_username", "")).strip()
    if keyring_service and keyring_username:
        try:
            import keyring as _keyring

            secret = _keyring.get_password(keyring_service, keyring_username)
            if secret and secret.strip():
                return secret.strip()
        except Exception:
            pass

    raise RuntimeError(
        f"API key could not be resolved: environment variable "
        f"'{api_key_env_var}' is not set and no keyring secret was found "
        f"(service='{keyring_service}', username='{keyring_username}')"
    )


def _resolve_project_path(project_root: Path, configured_path: str) -> Path:
    """Resolve a configured path relative to the project root when needed.

    Args:
        project_root (Path): Project root used to resolve relative paths.
        configured_path (str): Path string read from config.

    Returns:
        Path: Absolute path for the configured filesystem location.
    """

    candidate = Path(configured_path)
    if candidate.is_absolute():
        return candidate
    return project_root / candidate

