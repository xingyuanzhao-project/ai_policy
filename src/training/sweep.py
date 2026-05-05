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


def run_training_sweep(
    *,
    config_path: Path,
    model_name: str,
    mode: Mode,
    on_cell_done: CellCallback | None = None,
) -> list[Path]:
    """Run all (fold, seed) cells for one model and return the run dirs.

    Args:
        config_path: Path to ``settings/training/training.yml``.
        model_name: Key into ``models:`` block of the YAML.
        mode: ``"test"`` for a quick wiring check, ``"full"`` for the
            reportable sweep.
        on_cell_done: Optional callback invoked once per cell after the
            cell finishes (or is skipped because its ``run.json``
            already exists). Receives a dict with keys ``model``,
            ``fold``, ``seed``, ``skipped`` (bool), ``elapsed_s``
            (float, 0.0 when skipped), and ``run_dir`` (Path). Used by
            the global pipeline orchestrator to drive an end-to-end
            tqdm/ETA display.

    Returns:
        List of run-output directories (one per (fold, seed) cell). All
        cells whose ``run.json`` already exists are included for
        downstream aggregation but skipped during training.
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
            already_done = (run_dir / "run.json").is_file()
            if already_done:
                if on_cell_done is not None:
                    on_cell_done(
                        {
                            "model": model_name,
                            "fold": fold_index,
                            "seed": seed,
                            "skipped": True,
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
            )
            elapsed = time.perf_counter() - cell_t0
            if on_cell_done is not None:
                on_cell_done(
                    {
                        "model": model_name,
                        "fold": fold_index,
                        "seed": seed,
                        "skipped": False,
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
) -> None:
    """Load fresh model + tokenizer, train one cell, and free memory."""

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
