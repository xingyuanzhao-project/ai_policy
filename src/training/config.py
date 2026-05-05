"""Configuration loading for the training pipeline.

Loads ``settings/training/training.yml`` into a single ``TrainingConfig``
dataclass that every module under ``src/training/`` consumes. Mirrors the
shape and validation discipline of :mod:`src.eval.config`: explicit
slotted dataclasses, project-root-relative path resolution, and fail-loud
input validation before any heavy work begins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class SplitConfig:
    """Cross-validation split parameters.

    Attributes:
        strategy: Identifier for the splitter; only ``stratified_kfold`` is
            implemented.
        n_splits: Number of outer folds.
        stratify_on: Column names to stratify on; the implementation uses
            the first column only.
        val_fraction_of_train_fold: Fraction of each train fold reserved
            as a stratified validation carve-out.
        cv_seed: Seed used to build outer folds and the val carve-out.
    """

    strategy: str
    n_splits: int
    stratify_on: list[str]
    val_fraction_of_train_fold: float
    cv_seed: int


@dataclass(slots=True)
class DataConfig:
    """Data-source configuration.

    Attributes:
        annotated_csv: Path to the annotated bill CSV produced by
            :mod:`scripts.outcome_annotation`.
        text_columns: Columns whose values are concatenated into the model
            input text.
        metadata_columns: Columns appended to the input as ``key: value``
            metadata fields.
        label_column: Name of the binary outcome column.
        split: Split strategy block.
        subsample: Optional cap on the number of rows loaded; ``None``
            uses the full dataset. Used by ``mode='test'`` smoke runs.
    """

    annotated_csv: Path
    text_columns: list[str]
    metadata_columns: list[str]
    label_column: str
    split: SplitConfig
    subsample: int | None


@dataclass(slots=True)
class ModelConfig:
    """Per-model training configuration.

    Attributes:
        name: YAML key (e.g. ``deberta-v3-base``).
        hf_id: HuggingFace repo id.
        family: Model family; only ``encoder`` is supported in this run.
        optimizer: Optimizer name passed to ``TrainingArguments.optim``.
        max_length: Tokenizer truncation length.
        per_device_train_batch_size: Train batch per device.
        per_device_eval_batch_size: Eval batch per device.
        epochs: Maximum number of training epochs.
        weight_decay: AdamW weight decay.
        warmup_ratio: Linear warmup fraction.
        gradient_checkpointing: Whether to enable activation checkpointing.
        bf16: Use bfloat16 mixed precision when supported by the GPU.
    """

    name: str
    hf_id: str
    family: str
    optimizer: str
    max_length: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    epochs: int
    weight_decay: float
    warmup_ratio: float
    gradient_checkpointing: bool
    bf16: bool


@dataclass(slots=True)
class SweepConfig:
    """Sweep orchestration parameters.

    Attributes:
        seeds: List of training seeds applied per fold.
        learning_rate: Map from model name to the fixed full-run learning
            rate.
        early_stopping_patience: Patience for the HF ``EarlyStoppingCallback``.
        metric_for_best_model: Metric key the trainer uses to select the
            best checkpoint on the validation split.
    """

    seeds: list[int]
    learning_rate: dict[str, float]
    early_stopping_patience: int
    metric_for_best_model: str


@dataclass(slots=True)
class PathConfig:
    """Output and cache locations.

    Attributes:
        output_root: Root for ``runs/<model>/<run_dir>/run.json`` and the
            aggregated CSVs.
        hf_cache: Project-local HuggingFace cache directory.
        log_subdir: Sub-directory name appended to per-run output dirs
            for HF Trainer logs.
    """

    output_root: Path
    hf_cache: Path
    log_subdir: str


@dataclass(slots=True)
class PlotConfig:
    """Plot output configuration.

    Attributes:
        figures_dir: Directory for the six rendered PNGs.
        primary_metric: Metric used as the headline comparison axis.
    """

    figures_dir: Path
    primary_metric: str


@dataclass(slots=True)
class TrainingConfig:
    """Top-level training pipeline configuration.

    Attributes:
        data: Data source + split block.
        models: Ordered map from model name to :class:`ModelConfig`.
        sweep: Sweep orchestration block.
        paths: Output/cache paths.
        plotting: Plot rendering block.
        project_root: Resolved project root used for path resolution.
    """

    data: DataConfig
    models: dict[str, ModelConfig]
    sweep: SweepConfig
    paths: PathConfig
    plotting: PlotConfig
    project_root: Path


def load_training_config(
    config_path: Path,
    *,
    project_root: Path | None = None,
) -> TrainingConfig:
    """Load and validate the training YAML config.

    Args:
        config_path: Path to ``settings/training/training.yml`` (or
            override).
        project_root: Optional explicit project root. Defaults to two
            levels above ``src/training/``.

    Returns:
        Fully populated :class:`TrainingConfig` with absolute paths.

    Raises:
        FileNotFoundError: If the config file is missing.
        ValueError: On missing required keys or unknown model family.
    """

    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    root = _resolve_project_root(project_root)
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    data = _parse_data(raw.get("data", {}), root)
    models = _parse_models(raw.get("models", {}))
    sweep = _parse_sweep(raw.get("sweep", {}), known_models=set(models))
    paths = _parse_paths(raw.get("paths", {}), root)
    plotting = _parse_plotting(raw.get("plotting", {}), root)

    _validate_inputs_exist(data=data)

    return TrainingConfig(
        data=data,
        models=models,
        sweep=sweep,
        paths=paths,
        plotting=plotting,
        project_root=root,
    )


def _resolve_project_root(explicit: Path | None) -> Path:
    """Derive the project root used to resolve relative config paths."""

    if explicit is not None:
        return Path(explicit).resolve()
    return Path(__file__).resolve().parents[2]


def _resolve_path(raw: str | Path, root: Path) -> Path:
    """Resolve a path string against the project root when relative."""

    p = Path(raw)
    return p if p.is_absolute() else (root / p).resolve()


def _parse_data(raw: dict[str, Any], root: Path) -> DataConfig:
    """Parse the ``data`` block."""

    for key in ("annotated_csv", "text_columns", "label_column", "split"):
        if key not in raw:
            raise ValueError(f"data.{key} is required")
    split_raw = raw["split"]
    split = SplitConfig(
        strategy=str(split_raw.get("strategy", "stratified_kfold")),
        n_splits=int(split_raw.get("n_splits", 5)),
        stratify_on=list(split_raw.get("stratify_on", ["outcome"])),
        val_fraction_of_train_fold=float(
            split_raw.get("val_fraction_of_train_fold", 0.15)
        ),
        cv_seed=int(split_raw.get("cv_seed", 20260504)),
    )
    subsample = raw.get("subsample")
    if subsample is not None:
        subsample = int(subsample)
    return DataConfig(
        annotated_csv=_resolve_path(raw["annotated_csv"], root),
        text_columns=list(raw["text_columns"]),
        metadata_columns=list(raw.get("metadata_columns", [])),
        label_column=str(raw["label_column"]),
        split=split,
        subsample=subsample,
    )


_SUPPORTED_FAMILIES = {"encoder"}


def _parse_models(raw: dict[str, Any]) -> dict[str, ModelConfig]:
    """Parse the ``models`` block preserving declaration order."""

    if not raw:
        raise ValueError("models block is empty; at least one model required")
    parsed: dict[str, ModelConfig] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"models.{name} must be a mapping")
        family = str(entry.get("family", "encoder"))
        if family not in _SUPPORTED_FAMILIES:
            raise ValueError(
                f"models.{name}.family={family!r} is not supported in this "
                f"run; supported families are {sorted(_SUPPORTED_FAMILIES)}"
            )
        parsed[name] = ModelConfig(
            name=name,
            hf_id=str(entry["hf_id"]),
            family=family,
            optimizer=str(entry.get("optimizer", "adamw_torch")),
            max_length=int(entry.get("max_length", 256)),
            per_device_train_batch_size=int(
                entry.get("per_device_train_batch_size", 16)
            ),
            per_device_eval_batch_size=int(
                entry.get("per_device_eval_batch_size", 32)
            ),
            epochs=int(entry.get("epochs", 6)),
            weight_decay=float(entry.get("weight_decay", 0.01)),
            warmup_ratio=float(entry.get("warmup_ratio", 0.06)),
            gradient_checkpointing=bool(entry.get("gradient_checkpointing", False)),
            bf16=bool(entry.get("bf16", True)),
        )
    return parsed


def _parse_sweep(raw: dict[str, Any], *, known_models: set[str]) -> SweepConfig:
    """Parse the ``sweep`` block."""

    full = raw.get("full", {})
    if not full:
        raise ValueError("sweep.full block is required")
    lr_map = full.get("learning_rate") or {}
    if not isinstance(lr_map, dict):
        raise ValueError("sweep.full.learning_rate must be a mapping")
    missing_lrs = [m for m in known_models if m not in lr_map or lr_map[m] is None]
    if missing_lrs:
        raise ValueError(
            "sweep.full.learning_rate missing values for: "
            + ", ".join(sorted(missing_lrs))
        )
    return SweepConfig(
        seeds=[int(s) for s in full.get("seeds", [13, 42, 1729])],
        learning_rate={k: float(v) for k, v in lr_map.items() if k in known_models},
        early_stopping_patience=int(raw.get("early_stopping_patience", 2)),
        metric_for_best_model=str(raw.get("metric_for_best_model", "val_auroc")),
    )


def _parse_paths(raw: dict[str, Any], root: Path) -> PathConfig:
    """Parse the ``paths`` block."""

    return PathConfig(
        output_root=_resolve_path(raw.get("output_root", "output/training"), root),
        hf_cache=_resolve_path(raw.get("hf_cache", ".hf_cache"), root),
        log_subdir=str(raw.get("log_subdir", "logs")),
    )


def _parse_plotting(raw: dict[str, Any], root: Path) -> PlotConfig:
    """Parse the ``plotting`` block."""

    return PlotConfig(
        figures_dir=_resolve_path(
            raw.get("figures_dir", "docs/figures/training"), root
        ),
        primary_metric=str(raw.get("primary_metric", "auroc")),
    )


def _validate_inputs_exist(*, data: DataConfig) -> None:
    """Fail loud if the annotated CSV is missing."""

    if not data.annotated_csv.is_file():
        raise ValueError(f"data.annotated_csv missing on disk: {data.annotated_csv}")
