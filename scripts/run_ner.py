"""Runner script for the NER multi-agent pipeline.

Usage:
    python -m scripts.run_ner
"""

from pathlib import Path

from src.ner.runtime.pipeline_api import run_full_corpus

if __name__ == "__main__":
    results = run_full_corpus(
        project_root=Path(__file__).resolve().parents[1],
        config_path="settings/config.yml",
        ner_config_path="settings/ner_config.yml",
        prompt_config_path="settings/ner_prompts.json",
        run_id="run_sonnet_full_20260416",
        resume=True,
    )
    print(f"Done. {len(results)} bills processed.")
