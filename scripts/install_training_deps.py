"""Install training pipeline dependencies and prefetch HF model weights.

Run once before any training:

    .\\.venv\\Scripts\\python.exe scripts\\install_training_deps.py

Pipeline order (run each step from the project root):

    1. scripts/install_training_deps.py      this file, one-time setup
    2. scripts/run_training.py               once per MODEL_NAME
    3. scripts/aggregate_training_results.py builds the four CSVs
    4. scripts/plot_training_results.py      renders the six PNGs

Steps performed by this script
------------------------------

1. Verify the project ``.venv`` is active (or use its pip directly).
2. Install ``EXTRA_PACKAGES`` into the venv via the explicit pip path.
3. Prefetch tokenizer + base model weights for every model in
   ``CONFIG_PATH`` into ``paths.hf_cache``.
4. Print CUDA availability and per-device VRAM.

Constants in this file
----------------------

    CONFIG_PATH    : Path to settings/training/training.yml; the model
                     list is read from there so adding a new model only
                     requires editing the YAML.
    VENV_PIP       : Hard-coded path to .venv\\Scripts\\pip.exe.
    EXTRA_PACKAGES : Tuple of pip specs installed in addition to
                     anything already pinned in requirements.txt.

Where every other hyperparameter lives
--------------------------------------

This script does not own any training hyperparameter. All training
knobs live in settings/training/training.yml and src/training/sweep.py;
see the docstring of scripts/run_training.py for the full map.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "settings" / "training" / "training.yml"
VENV_PIP = PROJECT_ROOT / ".venv" / "Scripts" / "pip.exe"

EXTRA_PACKAGES: tuple[str, ...] = (
    "transformers>=4.45.0",
    "peft>=0.13.0",
    "bitsandbytes>=0.44.0",
    "accelerate>=1.0.0",
    "datasets>=3.0.0",
    "scikit-learn>=1.5.0",
    "sentencepiece>=0.2.0",
    "protobuf",
    "tiktoken",
)


def main() -> None:
    _verify_venv()
    _install_packages(EXTRA_PACKAGES)
    _prefetch_models()
    _print_gpu_status()


def _verify_venv() -> None:
    """Ensure we are using the project's venv pip, not system pip."""

    if not VENV_PIP.is_file():
        raise SystemExit(
            f"venv pip not found at {VENV_PIP}. Create the venv first."
        )
    pip_on_path = shutil.which("pip")
    if pip_on_path is None or Path(pip_on_path).resolve() != VENV_PIP.resolve():
        print(
            f"[install] system pip is {pip_on_path}; using {VENV_PIP} explicitly"
        )


def _install_packages(packages: tuple[str, ...]) -> None:
    """Install the configured package list into the project venv."""

    cmd = [str(VENV_PIP), "install", *packages]
    print("[install] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _prefetch_models() -> None:
    """Prefetch tokenizer + model weights for every model in the YAML."""

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.training.config import load_training_config
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    config = load_training_config(CONFIG_PATH)
    config.paths.hf_cache.mkdir(parents=True, exist_ok=True)
    cache_dir = str(config.paths.hf_cache)
    for name, model_cfg in config.models.items():
        print(f"[install] prefetching {name} ({model_cfg.hf_id}) -> {cache_dir}")
        AutoTokenizer.from_pretrained(
            model_cfg.hf_id, cache_dir=cache_dir, use_fast=True
        )
        AutoModelForSequenceClassification.from_pretrained(
            model_cfg.hf_id, cache_dir=cache_dir, num_labels=2
        )


def _print_gpu_status() -> None:
    """Print torch CUDA availability and per-device VRAM."""

    import torch

    print(f"[install] torch={torch.__version__}, cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024 ** 3)
            print(f"[install]  device {i}: {props.name}  VRAM={vram_gb:.1f} GiB")


if __name__ == "__main__":
    main()
