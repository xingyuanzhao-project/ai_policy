"""Run the training sweep for one model.

Edit the constant block below, then run:

    .\\.venv\\Scripts\\python.exe scripts\\run_training.py

Pipeline order (run each step from the project root):

    1. scripts/install_training_deps.py      one-time setup
    2. scripts/run_training.py               this file, once per MODEL_NAME
    3. scripts/aggregate_training_results.py builds the four CSVs
    4. scripts/plot_training_results.py      renders the six PNGs

Constants in this file
----------------------

    MODEL_NAME : str
        Key into the ``models:`` block of the YAML. Currently the only
        active model is ``"modernbert-base"``; the ``"deberta-v3-base"``
        block in the YAML is commented out. Setting MODEL_NAME to a key
        that is not active in the YAML raises a ValueError at startup.
        The sweep is resumable: cells whose ``run.json`` already records
        ``epochs >= model_cfg.epochs`` are skipped, and cells whose
        ``run.json`` records fewer epochs than the current YAML target
        are resumed from their latest on-disk checkpoint.
    MODE : {"test", "full"}
        ``"test"`` overrides the YAML in-memory to subsample=50, 2 folds,
        seeds=[13], 1 epoch, bf16=False. Non-reportable wiring check.
        ``"full"`` uses the YAML as-written: pre-registered LR, all
        seeds, all folds, all epochs, bf16 from the model block.

Where every other hyperparameter lives
--------------------------------------

settings/training/training.yml
    data.subsample                   row cap for full corpus (null = all)
    data.split.n_splits              number of CV folds (>= 2)
    data.split.val_fraction_of_train_fold
    data.split.cv_seed               fold + val carve seed
    models.<name>.epochs             max epochs per cell
    models.<name>.per_device_train_batch_size
    models.<name>.per_device_eval_batch_size
    models.<name>.weight_decay       AdamW weight decay
    models.<name>.warmup_ratio       linear warmup fraction
    models.<name>.optimizer          e.g. adamw_torch
    models.<name>.max_length         tokenizer truncation length
    models.<name>.bf16               mixed precision on CUDA
    models.<name>.gradient_checkpointing
    sweep.full.seeds                 list of training seeds, one cell each
    sweep.full.learning_rate.<name>  fixed full-run LR per model
    sweep.early_stopping_patience    HF EarlyStoppingCallback patience
    sweep.metric_for_best_model      val metric for best-checkpoint pick

src/training/sweep.py (test-mode-only overrides)
    _TEST_SUBSAMPLE = 50
    _TEST_N_SPLITS  = 2
    _TEST_SEEDS     = [13]
    _TEST_EPOCHS    = 1

How to run X folds x Y seeds x Z models
---------------------------------------

Smoke (cheapest wiring check, both models):
    1. set MODE = "test" below
    2. set MODEL_NAME = "deberta-v3-base", run this script
    3. set MODEL_NAME = "modernbert-base", run this script
    4. run aggregate + plot
    Yields 2 folds x 1 seed x 2 models = 4 cells (~40 s wall-clock).

Single 1-fold x 1-seed x 1-model cell (fastest minimal real run):
    1. in the YAML, set sweep.full.seeds: [13] (1 seed)
       and data.split.n_splits: 2          (KFold minimum)
    2. set MODE = "full" and MODEL_NAME = "<your model>" below
    3. run this script; press Ctrl-C as soon as the first run.json
       appears under output/training/runs/<model>/full__fold_0__*/
    4. run aggregate + plot
    Note: there is no built-in "stop after fold 0" knob, so the Ctrl-C
    step is the supported workaround for an exact 1-fold run.

Full reportable sweep (the plan's target: 5 folds x 3 seeds x 2 models):
    1. confirm YAML has data.split.n_splits: 5 and
       sweep.full.seeds: [13, 42, 1729]
    2. set MODE = "full" below
    3. set MODEL_NAME = "deberta-v3-base", run this script
    4. set MODEL_NAME = "modernbert-base", run this script
    5. run aggregate + plot
    Yields 5 x 3 x 2 = 30 cells (~3 hours on the 4090 Laptop GPU).
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training import run_training_sweep


CONFIG_PATH = PROJECT_ROOT / "settings" / "training" / "training.yml"
MODEL_NAME = "modernbert-base"
MODE = "test"


def main() -> None:
    run_dirs = run_training_sweep(
        config_path=CONFIG_PATH, model_name=MODEL_NAME, mode=MODE
    )
    print(
        f"[train] model={MODEL_NAME} mode={MODE} cells={len(run_dirs)} "
        f"output_root={run_dirs[0].parent.parent if run_dirs else 'n/a'}"
    )


if __name__ == "__main__":
    main()
