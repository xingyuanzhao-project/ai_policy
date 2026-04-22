"""Runner script for the skill-driven NER pipeline.

Usage:
    python -m scripts.run_skill_ner
"""

from pathlib import Path

from src.skill_ner.runner import run_corpus

if __name__ == "__main__":
    run_dir = run_corpus(
        project_root=Path(__file__).resolve().parents[1],
        config_path="settings/skill_ner_config.yml",
        base_config_path="settings/config.yml",
        skill_path="settings/skills/ner_extraction.md",
        run_id="skill_full_20260416_v2",
        resume=True,
    )
    print(f"Done. Run directory: {run_dir}")
