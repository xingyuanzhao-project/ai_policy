"""End-to-end orchestrator: train every model, aggregate, then plot.

One command runs the entire reportable sweep from raw data to PNGs and
shows a single global ETA across all (model, fold, seed) cells plus the
two post-processing stages, instead of forcing the user to track three
scripts and read the per-cell HF Trainer tqdm bars in isolation.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\run_full_pipeline.py

Constants in this file
----------------------

    MODE : {"test", "full"}
        Forwarded verbatim to ``run_training_sweep``. ``"test"`` overrides
        the YAML in-memory to subsample=50, 2 folds, seeds=[13], 1 epoch
        (~40 s wall-clock for a wiring check). ``"full"`` uses the YAML
        as-written (currently K=5, S=3, E=5 per model).
    INCLUDE_AGGREGATE / INCLUDE_PLOT : bool
        Toggle the two post-training stages independently. Both default
        to True. Set to False if you only want to re-train cells without
        regenerating CSVs / PNGs.

Everything else lives in ``settings/training/training.yml``.

Pipeline order (executed in one process):

    Stage 1/3 : train every model declared in ``models:`` (resumable;
                cells with an existing ``run.json`` are counted as done
                without retraining).
    Stage 2/3 : aggregate per-cell ``run.json`` files into the four
                CSVs under ``paths.output_root``.
    Stage 3/3 : render the six comparison PNGs under
                ``plotting.figures_dir``.

Progress display
----------------

    A single ``tqdm`` bar tracks the K * S * (#models) training cells.
    The bar's elapsed/ETA fields apply to the training stage; per-cell
    wall-clock and the running mean are printed to the bar's postfix.
    Stage 2 and stage 3 each print a one-line "stage done in X.X s"
    summary so the overall pipeline elapsed time is always visible.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training import (
    aggregate_results,
    render_plots,
    run_training_sweep,
)
from src.training.config import load_training_config


CONFIG_PATH = PROJECT_ROOT / "settings" / "training" / "training.yml"
MODE = "full"
INCLUDE_AGGREGATE = True
INCLUDE_PLOT = True


def main() -> None:
    cfg = load_training_config(CONFIG_PATH)
    model_names = list(cfg.models.keys())
    if MODE == "full":
        n_folds = cfg.data.split.n_splits
        n_seeds = len(cfg.sweep.seeds)
    else:
        from src.training.sweep import _TEST_N_SPLITS, _TEST_SEEDS

        n_folds = _TEST_N_SPLITS
        n_seeds = len(_TEST_SEEDS)
    total_cells = n_folds * n_seeds * len(model_names)

    print(
        f"[pipeline] mode={MODE} models={model_names} "
        f"K={n_folds} S={n_seeds} -> {total_cells} cells; "
        f"output_root={cfg.paths.output_root}"
    )

    pipeline_t0 = time.perf_counter()
    cell_times: list[float] = []
    bar = tqdm(
        total=total_cells,
        desc=f"Stage 1/3 train",
        unit="cell",
        dynamic_ncols=True,
        smoothing=0.0,
    )

    def on_cell_done(info: dict) -> None:
        if not info["skipped"]:
            cell_times.append(info["elapsed_s"])
        mean_s = (
            sum(cell_times) / len(cell_times) if cell_times else 0.0
        )
        bar.set_postfix(
            model=info["model"],
            fold=info["fold"],
            seed=info["seed"],
            cell=f"{info['elapsed_s']:.1f}s"
            if not info["skipped"]
            else "skip",
            mean=f"{mean_s:.1f}s",
            refresh=False,
        )
        bar.update(1)

    train_t0 = time.perf_counter()
    for model_name in model_names:
        run_training_sweep(
            config_path=CONFIG_PATH,
            model_name=model_name,
            mode=MODE,
            on_cell_done=on_cell_done,
        )
    bar.close()
    train_elapsed = time.perf_counter() - train_t0
    print(f"[pipeline] Stage 1/3 (train) done in {train_elapsed:.1f}s")

    if INCLUDE_AGGREGATE:
        agg_t0 = time.perf_counter()
        aggregate_results(output_root=cfg.paths.output_root)
        print(
            f"[pipeline] Stage 2/3 (aggregate) done in "
            f"{time.perf_counter() - agg_t0:.1f}s"
        )

    if INCLUDE_PLOT:
        plot_t0 = time.perf_counter()
        render_plots(
            output_root=cfg.paths.output_root,
            figures_dir=cfg.plotting.figures_dir,
            primary_metric=cfg.plotting.primary_metric,
        )
        print(
            f"[pipeline] Stage 3/3 (plot) done in "
            f"{time.perf_counter() - plot_t0:.1f}s"
        )

    total_elapsed = time.perf_counter() - pipeline_t0
    print(
        f"[pipeline] ALL DONE in {total_elapsed:.1f}s "
        f"({total_elapsed / 60.0:.1f} min)"
    )


if __name__ == "__main__":
    main()
