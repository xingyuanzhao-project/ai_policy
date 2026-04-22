"""Configuration loading for the nine-stage eval pipeline.

- Loads ``settings/eval/eval.yml`` into a single ``EvalConfig`` dataclass that
  all stages consume.
- Validates that the two method run directories and the NCSL metadata file
  referenced by the config exist before any judge call is issued.
- Does not read secrets; the judge client resolves API keys the same way the
  extractor runs do (``settings/config.yml`` plus ``OPENROUTER_API_KEY``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class JudgeConfig:
    """Judge LLM connection settings.

    Attributes:
        provider: Provider key (currently only ``openrouter`` is supported).
        model: Provider-qualified model id, e.g. ``google/gemini-2.5-pro``.
        temperature: Sampling temperature for judge calls.
        max_tokens: Completion token ceiling per judge call.
        request_timeout_seconds: Per-call wall-clock timeout.
        max_retries: Retry count on transient provider errors.
    """

    provider: str
    model: str
    temperature: float
    max_tokens: int
    request_timeout_seconds: int
    max_retries: int


@dataclass(slots=True)
class MethodConfig:
    """Location of one extractor run's artefacts.

    Attributes:
        name: Stable identifier used in cache paths and result files.
        outputs_dir: Directory of per-bill quadruplet JSON files.
        usage_summary: Path to the 8-key ``usage_summary.json`` for that run.
    """

    name: str
    outputs_dir: Path
    usage_summary: Path


@dataclass(slots=True)
class CorpusConfig:
    """Corpus file locations shared by all stages.

    Attributes:
        ncsl_metadata: CSV with bill-level metadata (topics column drives
            Stage 4 coverage).
        ncsl_text: JSONL with one record per bill containing the raw text.
    """

    ncsl_metadata: Path
    ncsl_text: Path


@dataclass(slots=True)
class StageToggle:
    """Common shape for a per-stage configuration block.

    Attributes:
        enabled: Whether the stage runs when the pipeline is invoked
            without explicit ``--stages``.
        params: Stage-specific parameters as a flat dict (e.g.
            ``max_concurrency``, ``sample_items``, ``expert_file``).
    """

    enabled: bool
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SamplingConfig:
    """Corpus-level sampling knobs applied before any stage runs.

    Attributes:
        sample_bills: Optional cap on the number of bills per method. ``None``
            means full corpus.
        seed: Random seed for reproducible sampling.
    """

    sample_bills: int | None
    seed: int


@dataclass(slots=True)
class EvalConfig:
    """Top-level configuration for the eval pipeline.

    Attributes:
        judge: Judge LLM connection settings.
        methods: Ordered dict-like mapping of method name to
            :class:`MethodConfig`. Orders determine A/B assignment in the
            Stage 6 pairwise comparison.
        corpus: Shared corpus file locations.
        output_run_dir: Absolute path to the evaluation run output directory
            (``output/evals/v1`` by default).
        stages: Per-stage toggle + parameter blocks keyed by stage number.
        sampling: Corpus-level sampling knobs.
        project_root: Project root used to resolve relative paths.
    """

    judge: JudgeConfig
    methods: dict[str, MethodConfig]
    corpus: CorpusConfig
    output_run_dir: Path
    stages: dict[int, StageToggle]
    sampling: SamplingConfig
    project_root: Path


def load_eval_config(
    config_path: Path,
    *,
    project_root: Path | None = None,
) -> EvalConfig:
    """Load and validate the eval YAML config.

    Args:
        config_path: Path to ``settings/eval/eval.yml`` (or override).
        project_root: Optional explicit project root. Defaults to two levels
            above ``src/eval/`` so library usage matches CLI usage.

    Returns:
        Fully populated :class:`EvalConfig` with all paths resolved to
        absolute locations.

    Raises:
        FileNotFoundError: If the config file itself is missing.
        ValueError: If required keys are missing or referenced data files
            do not exist on disk.
    """

    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Eval config not found: {config_path}")

    root = _resolve_project_root(project_root)
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    judge = _parse_judge(raw.get("judge", {}))
    methods = _parse_methods(raw.get("methods", {}), root)
    corpus = _parse_corpus(raw.get("corpus", {}), root)
    output_run_dir = _resolve_path(
        raw.get("output", {}).get("run_dir", "output/evals/v1"), root
    )
    stages = _parse_stages(raw.get("stages", {}))
    sampling = _parse_sampling(raw.get("sampling", {}))

    _validate_inputs_exist(methods=methods, corpus=corpus)

    return EvalConfig(
        judge=judge,
        methods=methods,
        corpus=corpus,
        output_run_dir=output_run_dir,
        stages=stages,
        sampling=sampling,
        project_root=root,
    )


def _resolve_project_root(explicit: Path | None) -> Path:
    """Derive the project root used to resolve relative config paths."""

    if explicit is not None:
        return Path(explicit).resolve()
    return Path(__file__).resolve().parents[2]


def _resolve_path(raw: str | Path, root: Path) -> Path:
    """Resolve a config path string against the project root when relative."""

    p = Path(raw)
    return p if p.is_absolute() else (root / p).resolve()


def _parse_judge(raw: dict[str, Any]) -> JudgeConfig:
    """Pull the judge block out of the raw YAML dict."""

    required = ("provider", "model")
    for key in required:
        if key not in raw:
            raise ValueError(f"judge config missing required key: {key}")
    return JudgeConfig(
        provider=str(raw["provider"]),
        model=str(raw["model"]),
        temperature=float(raw.get("temperature", 0.0)),
        max_tokens=int(raw.get("max_tokens", 2048)),
        request_timeout_seconds=int(raw.get("request_timeout_seconds", 120)),
        max_retries=int(raw.get("max_retries", 3)),
    )


def _parse_methods(
    raw: dict[str, Any], root: Path
) -> dict[str, MethodConfig]:
    """Parse the ``methods`` block preserving declaration order."""

    if not raw:
        raise ValueError("methods block is empty; at least one method required")
    parsed: dict[str, MethodConfig] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"methods.{name} must be a mapping")
        if "outputs_dir" not in entry or "usage_summary" not in entry:
            raise ValueError(
                f"methods.{name} missing outputs_dir or usage_summary"
            )
        parsed[name] = MethodConfig(
            name=name,
            outputs_dir=_resolve_path(entry["outputs_dir"], root),
            usage_summary=_resolve_path(entry["usage_summary"], root),
        )
    return parsed


def _parse_corpus(raw: dict[str, Any], root: Path) -> CorpusConfig:
    """Parse the ``corpus`` block."""

    for key in ("ncsl_metadata", "ncsl_text"):
        if key not in raw:
            raise ValueError(f"corpus.{key} is required")
    return CorpusConfig(
        ncsl_metadata=_resolve_path(raw["ncsl_metadata"], root),
        ncsl_text=_resolve_path(raw["ncsl_text"], root),
    )


def _parse_stages(raw: dict[str, Any]) -> dict[int, StageToggle]:
    """Parse stage blocks keyed by ``stage<N>`` into an integer-keyed map."""

    stages: dict[int, StageToggle] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.startswith("stage"):
            raise ValueError(f"stages key must look like 'stageN', got {key!r}")
        try:
            number = int(key.removeprefix("stage"))
        except ValueError as exc:
            raise ValueError(f"invalid stage key {key!r}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"stages.{key} must be a mapping")
        enabled = bool(value.pop("enabled", True))
        stages[number] = StageToggle(enabled=enabled, params=dict(value))
    return stages


def _parse_sampling(raw: dict[str, Any]) -> SamplingConfig:
    """Parse the ``sampling`` block with safe defaults."""

    sample_bills = raw.get("sample_bills")
    if sample_bills is not None:
        sample_bills = int(sample_bills)
    return SamplingConfig(
        sample_bills=sample_bills,
        seed=int(raw.get("seed", 20260417)),
    )


def _validate_inputs_exist(
    *, methods: dict[str, MethodConfig], corpus: CorpusConfig
) -> None:
    """Fail loud if any referenced input file or directory is missing."""

    missing: list[str] = []
    for method in methods.values():
        if not method.outputs_dir.is_dir():
            missing.append(f"outputs_dir for {method.name}: {method.outputs_dir}")
        if not method.usage_summary.is_file():
            missing.append(
                f"usage_summary for {method.name}: {method.usage_summary}"
            )
    if not corpus.ncsl_metadata.is_file():
        missing.append(f"ncsl_metadata: {corpus.ncsl_metadata}")
    if not corpus.ncsl_text.is_file():
        missing.append(f"ncsl_text: {corpus.ncsl_text}")
    if missing:
        joined = "\n  - ".join(missing)
        raise ValueError(f"Missing eval inputs:\n  - {joined}")
