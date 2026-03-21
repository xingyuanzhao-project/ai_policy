"""Runner script for the NER multi-agent pipeline.

Usage:
    python -m scripts.run_ner
"""

from pathlib import Path

from src.ner.runtime.entry_points import run_full_corpus

if __name__ == "__main__":
    results = run_full_corpus(
        project_root=Path(__file__).resolve().parents[1],
        run_id="run_full_20260320",
        resume=True,
    )
    print(f"Done. {len(results)} bills processed.")
