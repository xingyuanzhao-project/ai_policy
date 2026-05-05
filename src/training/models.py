"""Model + tokenizer factory for the encoder family.

Gemma / causal-LM was dropped from this run, so the only factory exposed
is :func:`load_encoder`. The HuggingFace cache is forced to the project's
``.hf_cache`` directory via the ``cache_dir`` argument so all weights
remain repo-local.
"""

from __future__ import annotations

from pathlib import Path

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .config import ModelConfig


def load_encoder(
    model_cfg: ModelConfig,
    *,
    hf_cache: Path,
    num_labels: int = 2,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load an encoder model + tokenizer from the project HF cache.

    Args:
        model_cfg: Resolved per-model configuration.
        hf_cache: Project-local ``.hf_cache`` directory used for both
            tokenizer and model weight resolution.
        num_labels: Number of classification classes.

    Returns:
        Tuple ``(model, tokenizer)`` ready for the HF ``Trainer``.
    """

    if model_cfg.family != "encoder":
        raise ValueError(
            f"load_encoder only supports family='encoder', got "
            f"{model_cfg.family!r}"
        )
    cache_dir = str(hf_cache)
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.hf_id,
        cache_dir=cache_dir,
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg.hf_id,
        cache_dir=cache_dir,
        num_labels=num_labels,
    )
    return model, tokenizer
