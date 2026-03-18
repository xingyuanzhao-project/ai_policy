"""Pipeline bootstrap for the NER runtime.

- Owns runtime wiring of config, corpus, storage, chunking, agents, and
  orchestrator.
- Owns run-scoped preflight evidence and config snapshot creation.
- Does not execute bill processing loops itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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
from .llm_client import LLMClient, LLMConfig


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
        orchestrator (Orchestrator): Stage orchestrator for annotation,
            grouping, and refinement.
    """

    project_root: Path
    run_id: str
    config_store: ConfigStore
    corpus_store: CorpusStore
    artifact_store: ArtifactStore
    final_output_store: FinalOutputStore
    llm_client: LLMClient
    inference_unit_builder: InferenceUnitBuilder
    orchestrator: Orchestrator


def bootstrap(
    project_root: Path,
    run_id: str,
    config_path: Path | None = None,
    ner_config_path: Path | None = None,
    prompt_config_path: Path | None = None,
) -> PipelineContext:
    """Bootstrap the full NER pipeline from config, stores, and shared runtime state.

    Args:
        project_root (Path): Project root used for resolving relative paths.
        run_id (str): Stable run identifier for persisted artifacts.
        config_path (Path | None): Optional override for the project-level
            config file.
        ner_config_path (Path | None): Optional override for the NER runtime
            config file.
        prompt_config_path (Path | None): Optional override for the NER prompt
            config file.

    Returns:
        PipelineContext: Fully initialized pipeline context ready for
            inference.
    """

    settings_dir = project_root / "settings"
    config_store = ConfigStore()
    config_store.load(
        base_config_path=config_path or settings_dir / "config.yml",
        ner_config_path=ner_config_path or settings_dir / "ner_config.yml",
        prompt_config_path=prompt_config_path or settings_dir / "ner_prompts.json",
    )

    storage_dir = _resolve_project_path(
        project_root,
        config_store.storage_config["base_dir"],
    )
    artifact_store = ArtifactStore(storage_dir)
    final_output_store = FinalOutputStore(storage_dir)
    run_dir = artifact_store.run_dir(run_id)
    config_store.snapshot(run_dir)

    llm_client = LLMClient(
        LLMConfig(
            base_url=config_store.llm_config["base_url"],
            api_key=config_store.llm_config["api_key"],
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
        orchestrator=orchestrator,
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

