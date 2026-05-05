"""Top-level training orchestrator.

``run_training_sweep(config_path, model_name, mode)`` is the single entry
point. Two modes are supported:

* ``mode="test"``: a non-reportable wiring check that overrides the
  config in-memory to a tiny subsample, two outer folds, and a single
  seed. Useful for verifying the end-to-end pipeline before paying for
  the full sweep.
* ``mode="full"``: the reportable sweep that uses the YAML's fold count,
  seeds, and pre-registered learning rate.

The sweep is resumable: any ``run.json`` already on disk is skipped so
re-invocations only train the missing cells.
"""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import torch

from .config import (
    DataConfig,
    ModelConfig,
    PathConfig,
    SplitConfig,
    SweepConfig,
    TrainingConfig,
    load_training_config,
)
from .data import load_dataframe, make_stratified_kfold, to_hf_dataset
from .models import load_encoder
from .trainer import train_one_run


Mode = Literal["test", "full"]
CellCallback = Callable[[dict[str, Any]], None]

_TEST_SUBSAMPLE = 50
_TEST_N_SPLITS = 2
_TEST_SEEDS = [13]
_TEST_EPOCHS = 1
_RUN_FILENAME = "run.json"
_CHECKPOINT_PREFIX = "checkpoint-"


def _completed_epochs(run_dir: Path) -> int:
    """Return the highest integer epoch present in this cell's run.json.

    Used by the sweep to decide between skip / resume / fresh-train for
    a given cell. The HF ``Trainer`` log_history records one entry per
    eval (and one per logging step); eval entries land at integer
    epochs, so the max integer-valued epoch in the log is the number of
    epochs that were *fully trained and evaluated* before the run was
    last persisted. Returns 0 when the cell has never produced a
    run.json or when its log_history is missing/empty.
    """

    run_json_path = run_dir / _RUN_FILENAME
    if not run_json_path.is_file():
        return 0
    try:
        with run_json_path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return 0
    max_epoch = 0
    for entry in record.get("log_history", []):
        epoch_val = entry.get("epoch")
        if epoch_val is None:
            continue
        if abs(epoch_val - round(epoch_val)) < 1e-6:
            max_epoch = max(max_epoch, int(round(epoch_val)))
    return max_epoch


def _find_latest_checkpoint(run_dir: Path) -> Path | None:
    """Return the highest-step ``checkpoint-N`` subdir, or ``None``.

    The cell's run dir may contain up to two checkpoint subdirs once
    ``trainer.py`` switches to ``save_total_limit=2`` (best + latest).
    The latest is the one with the largest step count; that is the
    correct restore point when extending training by additional epochs.
    Older runs that pre-date the policy change have only the best
    checkpoint on disk; in that case the "latest" found here equals the
    best, and HF will resume from that step (which is earlier than the
    last trained epoch). The caller is responsible for understanding
    that consequence -- it is unavoidable for legacy runs.
    """

    if not run_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in run_dir.iterdir():
        if not path.is_dir() or not path.name.startswith(_CHECKPOINT_PREFIX):
            continue
        try:
            step = int(path.name.removeprefix(_CHECKPOINT_PREFIX))
        except ValueError:
            continue
        candidates.append((step, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def run_training_sweep(
    *,
    config_path: Path,
    model_name: str,
    mode: Mode,
    on_cell_done: CellCallback | None = None,
) -> list[Path]:
    """Run all (fold, seed) cells for one model and return the run dirs.

    Per-cell action is one of three:

    * ``skip``   -- ``run.json`` exists and recorded ``epochs_completed
      >= model_cfg.epochs``. Nothing is trained; the run dir is still
      returned so aggregation can pick it up.
    * ``resume`` -- ``run.json`` exists but recorded
      ``epochs_completed < model_cfg.epochs``. A ``checkpoint-N`` dir
      under the run dir is found and HF ``Trainer`` is restarted with
      ``resume_from_checkpoint=<that path>``, continuing from the saved
      step until the new ``num_train_epochs`` target. Used to extend a
      completed sweep by a few more epochs without re-running the ones
      already done.
    * ``fresh``  -- no ``run.json`` (or a corrupt one). Train from
      scratch, same as the legacy path.

    Args:
        config_path: Path to ``settings/training/training.yml``.
        model_name: Key into ``models:`` block of the YAML.
        mode: ``"test"`` for a quick wiring check, ``"full"`` for the
            reportable sweep.
        on_cell_done: Optional callback invoked once per cell after the
            cell finishes (whether trained, resumed, or skipped).
            Receives a dict with keys ``model``, ``fold``, ``seed``,
            ``skipped`` (bool), ``resumed`` (bool), ``elapsed_s``
            (float, 0.0 when skipped), and ``run_dir`` (Path). Used by
            the global pipeline orchestrator to drive an end-to-end
            tqdm/ETA display.

    Returns:
        List of run-output directories (one per (fold, seed) cell).
    """

    config = load_training_config(config_path)
    if model_name not in config.models:
        raise ValueError(
            f"model {model_name!r} not in configured models "
            f"{sorted(config.models)!r}"
        )

    config = _apply_mode(config, mode=mode, model_name=model_name)
    model_cfg = config.models[model_name]
    sweep_cfg = config.sweep
    paths = config.paths
    learning_rate = sweep_cfg.learning_rate[model_name]

    df = load_dataframe(config.data)
    run_dirs: list[Path] = []
    model_root = paths.output_root / "runs" / model_name
    model_root.mkdir(parents=True, exist_ok=True)

    target_epochs = model_cfg.epochs
    for fold_index, train_df, val_df, test_df in make_stratified_kfold(
        df, data_cfg=config.data
    ):
        for seed in sweep_cfg.seeds:
            run_dir = model_root / _run_dir_name(
                mode=mode,
                fold_index=fold_index,
                seed=seed,
                learning_rate=learning_rate,
            )
            run_dirs.append(run_dir)

            completed = _completed_epochs(run_dir)
            resume_from: Path | None = None
            if completed >= target_epochs:
                action = "skip"
            elif completed > 0:
                resume_from = _find_latest_checkpoint(run_dir)
                action = "resume" if resume_from is not None else "fresh"
            else:
                action = "fresh"

            if action == "skip":
                if on_cell_done is not None:
                    on_cell_done(
                        {
                            "model": model_name,
                            "fold": fold_index,
                            "seed": seed,
                            "skipped": True,
                            "resumed": False,
                            "elapsed_s": 0.0,
                            "run_dir": run_dir,
                        }
                    )
                continue

            cell_t0 = time.perf_counter()
            _train_single_cell(
                model_cfg=model_cfg,
                sweep_cfg=sweep_cfg,
                paths=paths,
                data_cfg=config.data,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                seed=seed,
                learning_rate=learning_rate,
                fold_index=fold_index,
                run_dir=run_dir,
                log_subdir=paths.log_subdir,
                resume_from_checkpoint=resume_from,
            )
            elapsed = time.perf_counter() - cell_t0
            if on_cell_done is not None:
                on_cell_done(
                    {
                        "model": model_name,
                        "fold": fold_index,
                        "seed": seed,
                        "skipped": False,
                        "resumed": action == "resume",
                        "elapsed_s": elapsed,
                        "run_dir": run_dir,
                    }
                )
    return run_dirs


def _train_single_cell(
    *,
    model_cfg: ModelConfig,
    sweep_cfg: SweepConfig,
    paths: PathConfig,
    data_cfg: DataConfig,
    train_df,
    val_df,
    test_df,
    seed: int,
    learning_rate: float,
    fold_index: int,
    run_dir: Path,
    log_subdir: str,
    resume_from_checkpoint: Path | None = None,
) -> None:
    """Load fresh model + tokenizer, train one cell, and free memory.

    When ``resume_from_checkpoint`` is set, the freshly-loaded model is
    overwritten by HF Trainer's resume logic before training continues
    from the saved step. Loading the encoder still pays an HF cache /
    safetensors hit, but that cost is small relative to a single epoch
    of training.
    """

    model, tokenizer = load_encoder(model_cfg, hf_cache=paths.hf_cache)
    train_ds = to_hf_dataset(
        train_df, tokenizer=tokenizer, data_cfg=data_cfg, model_cfg=model_cfg
    )
    val_ds = to_hf_dataset(
        val_df, tokenizer=tokenizer, data_cfg=data_cfg, model_cfg=model_cfg
    )
    test_ds = to_hf_dataset(
        test_df, tokenizer=tokenizer, data_cfg=data_cfg, model_cfg=model_cfg
    )
    train_one_run(
        model=model,
        tokenizer=tokenizer,
        model_cfg=model_cfg,
        sweep_cfg=sweep_cfg,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        seed=seed,
        learning_rate=learning_rate,
        fold_index=fold_index,
        output_dir=run_dir,
        log_subdir=log_subdir,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    del model, tokenizer, train_ds, val_ds, test_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _apply_mode(
    config: TrainingConfig, *, mode: Mode, model_name: str
) -> TrainingConfig:
    """Return a copy of ``config`` with ``mode='test'`` overrides applied."""

    if mode == "full":
        return config
    if mode != "test":
        raise ValueError(f"mode must be 'test' or 'full', got {mode!r}")
    new_split = replace(
        config.data.split,
        n_splits=_TEST_N_SPLITS,
    )
    new_data = replace(
        config.data,
        split=new_split,
        subsample=_TEST_SUBSAMPLE,
    )
    new_models = dict(config.models)
    new_models[model_name] = replace(
        config.models[model_name],
        epochs=_TEST_EPOCHS,
        bf16=False,
    )
    new_sweep = replace(
        config.sweep,
        seeds=list(_TEST_SEEDS),
    )
    return replace(
        config,
        data=new_data,
        models=new_models,
        sweep=new_sweep,
    )


def _run_dir_name(
    *, mode: Mode, fold_index: int, seed: int, learning_rate: float
) -> str:
    """Stable on-disk run-directory name for a (mode, fold, seed) cell."""

    lr_str = f"{learning_rate:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    return f"{mode}__fold_{fold_index}__seed_{seed}__lr_{lr_str}"


def _unused(_: SplitConfig) -> None:  # pragma: no cover - kept for IDE help
    """No-op reference so editors do not flag SplitConfig as unused."""
