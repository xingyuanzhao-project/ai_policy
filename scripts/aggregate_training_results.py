"""Aggregate every per-run JSON into the four results CSVs.

    .\\.venv\\Scripts\\python.exe scripts\\aggregate_training_results.py

Pipeline order (run each step from the project root):

    1. scripts/install_training_deps.py      one-time setup
    2. scripts/run_training.py               once per MODEL_NAME
    3. scripts/aggregate_training_results.py this file
    4. scripts/plot_training_results.py      renders the six PNGs

Constants in this file
----------------------

    CONFIG_PATH : Path to settings/training/training.yml. Used only to
                  resolve ``paths.output_root`` (where the per-run JSONs
                  live and where the four CSVs are written).

Outputs written to ``paths.output_root``
----------------------------------------

    results.csv             tidy long-form: one row per (mode, model,
                            fold, seed, lr, split, metric).
    results_wide.csv        one row per (mode, model, fold, seed, lr)
                            with every test-split metric as a column.
    results_summary.csv     per (mode, model) mean/std over folds and
                            seeds; full-mode rows are preferred when
                            present, smoke-only state still aggregates.
    pairwise_delta_ci.csv   model-vs-model AUROC delta and 95% CI.

Where every other hyperparameter lives
--------------------------------------

This script does not own any training hyperparameter. All training
knobs live in settings/training/training.yml and src/training/sweep.py;
see the docstring of scripts/run_training.py for the full map.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training import aggregate_results
from src.training.config import load_training_config


CONFIG_PATH = PROJECT_ROOT / "settings" / "training" / "training.yml"


def main() -> None:
    config = load_training_config(CONFIG_PATH)
    paths = aggregate_results(config.paths.output_root)
    for name, path in paths.items():
        print(f"[aggregate] {name}: {path}")


if __name__ == "__main__":
    main()
