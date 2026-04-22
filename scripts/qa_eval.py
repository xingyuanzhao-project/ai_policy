"""Runner script for the QA app eval harness across V1, V2, and V3.

Usage:
    python -m scripts.qa_eval
"""

from pathlib import Path

from src.qa.eval.pipeline import run_all_versions_and_compare

if __name__ == "__main__":
    run_all_versions_and_compare(
        project_root=Path(__file__).resolve().parents[1],
        ground_truth_path=Path("src/qa/eval/ground_truth.json"),
        output_root=Path("output/evals_app"),
        versions=("v1", "v2", "v3"),
    )
    print("Done. V1, V2, V3 evals + comparison plots written.")
