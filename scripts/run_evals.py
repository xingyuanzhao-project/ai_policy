"""Runner script for the nine-stage LLM-as-judge evaluation pipeline.

Usage:
    python -m scripts.run_evals

Runs every enabled stage in ``settings/eval/eval.yml`` against the configured
NER method outputs, writes per-stage result JSONs under ``output/evals/v1/``,
and then renders the six summary PNGs into ``output/evals/v1/plots/``.
Resumable: per-item judge caches under ``output/evals/v1/cache/`` mean
re-invocations only spend tokens on items that have not finished yet.
"""

from pathlib import Path

from src.eval.eval_plots import render_all_plots
from src.eval.evals import run_evaluation

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "settings" / "eval" / "eval.yml"

    summary = run_evaluation(config_path=config_path)
    run_dir = Path(summary["run_dir"])

    produced = render_all_plots(run_dir=run_dir)
    print(
        f"Done. Stages completed: {summary['completed_stages']}"
        f"  errored: {summary['errored_stages']}"
        f"  plots rendered: {len(produced)}"
        f"  run_dir: {run_dir}"
    )
