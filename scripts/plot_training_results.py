"""Render the six comparison PNGs from the aggregated CSVs.

    .\\.venv\\Scripts\\python.exe scripts\\plot_training_results.py

Pipeline order (run each step from the project root):

    1. scripts/install_training_deps.py      one-time setup
    2. scripts/run_training.py               once per MODEL_NAME
    3. scripts/aggregate_training_results.py builds the four CSVs
    4. scripts/plot_training_results.py      this file

Constants in this file
----------------------

    CONFIG_PATH : Path to settings/training/training.yml. Used to
                  resolve ``paths.output_root`` (input CSVs) and
                  ``plotting.figures_dir`` (output PNGs).

Mode handling (see src/training/plots.py:_select_plot_rows)
-----------------------------------------------------------

If any full-mode rows exist in results_wide.csv, only those are
plotted; otherwise every available mode is plotted. The plot titles
include the mode label so test-mode runs cannot be mistaken for
reportable full-sweep results.

Outputs written to ``plotting.figures_dir``
-------------------------------------------

    01_model_comparison_auroc.png     bars of per-model mean test AUROC
                                      with 95% bootstrap CI.
    02_seed_stability.png             per-model strip of (fold, seed)
                                      test AUROC values.
    03_confusion_matrices.png         row-normalised confusion matrix at
                                      the best (fold, seed) per model.
    04_per_fold_curves.png            per-epoch validation AUROC trace
                                      for every (fold, seed).
    05_calibration_reliability.png    per-model mean ECE.
    06_pairwise_delta_ci.png          pairwise AUROC delta + 95% CI
                                      between every model pair.

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

from src.training import render_plots
from src.training.config import load_training_config


CONFIG_PATH = PROJECT_ROOT / "settings" / "training" / "training.yml"


def main() -> None:
    config = load_training_config(CONFIG_PATH)
    produced = render_plots(
        output_root=config.paths.output_root,
        figures_dir=config.plotting.figures_dir,
        primary_metric=config.plotting.primary_metric,
    )
    for name, path in produced.items():
        print(f"[plot] {name}: {path}")


if __name__ == "__main__":
    main()
