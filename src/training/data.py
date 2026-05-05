"""Data loading, splitting, and tokenisation for the training pipeline.

Contract for downstream modules:

* :func:`load_dataframe` returns the annotated CSV unfiltered (the
  Pending-as-zero policy is owned by ``scripts/outcome_annotation.py``).
* :func:`make_stratified_kfold` yields ``(fold_index, train_df, val_df,
  test_df)`` tuples with the val carve-out drawn stratified from the
  train fold.
* :func:`build_input_text` produces the single string that the encoder
  tokeniser sees, concatenating configured text columns and
  ``key: value`` metadata fields.
* :func:`to_hf_dataset` produces a ``datasets.Dataset`` ready for the HF
  ``Trainer`` (encoder family).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import DataConfig, ModelConfig


def load_dataframe(data_cfg: DataConfig) -> pd.DataFrame:
    """Load the annotated CSV with no row filtering.

    Args:
        data_cfg: Resolved data configuration.

    Returns:
        Full dataframe with the label column coerced to ``int`` and an
        optional row cap applied if ``data_cfg.subsample`` is set.

    Raises:
        ValueError: If the configured label column is missing.
    """

    df = pd.read_csv(data_cfg.annotated_csv)
    if data_cfg.label_column not in df.columns:
        raise ValueError(
            f"label_column {data_cfg.label_column!r} not in CSV columns "
            f"{list(df.columns)!r}"
        )
    df[data_cfg.label_column] = df[data_cfg.label_column].astype(int)
    if data_cfg.subsample is not None and len(df) > data_cfg.subsample:
        df = df.sample(
            n=data_cfg.subsample,
            random_state=data_cfg.split.cv_seed,
        ).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df


def make_stratified_kfold(
    df: pd.DataFrame,
    *,
    data_cfg: DataConfig,
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Yield outer-fold (train, val, test) triples.

    The outer split uses :class:`sklearn.model_selection.StratifiedKFold`
    on the first column listed in ``data_cfg.split.stratify_on``. The
    val carve-out is a stratified slice of each train fold, sized by
    ``data_cfg.split.val_fraction_of_train_fold``.

    Args:
        df: Full dataframe from :func:`load_dataframe`.
        data_cfg: Resolved data configuration.

    Yields:
        Tuples ``(fold_index, train_df, val_df, test_df)``. The frames
        share columns with ``df`` and have reset indices.
    """

    label = data_cfg.label_column
    stratify_col = (data_cfg.split.stratify_on or [label])[0]
    if stratify_col not in df.columns:
        raise ValueError(
            f"stratify column {stratify_col!r} not in dataframe columns"
        )

    skf = StratifiedKFold(
        n_splits=data_cfg.split.n_splits,
        shuffle=True,
        random_state=data_cfg.split.cv_seed,
    )
    val_fraction = data_cfg.split.val_fraction_of_train_fold
    for fold_index, (train_idx, test_idx) in enumerate(
        skf.split(df, df[stratify_col])
    ):
        train_full = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        train_df, val_df = _stratified_train_val_split(
            train_full,
            stratify_col=stratify_col,
            val_fraction=val_fraction,
            seed=data_cfg.split.cv_seed + fold_index,
        )
        yield fold_index, train_df, val_df, test_df


def _stratified_train_val_split(
    train_full: pd.DataFrame,
    *,
    stratify_col: str,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a stratified val slice off the train fold."""

    train_df, val_df = train_test_split(
        train_full,
        test_size=val_fraction,
        stratify=train_full[stratify_col],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def build_input_text(
    row: pd.Series,
    *,
    text_columns: list[str],
    metadata_columns: list[str],
) -> str:
    """Concatenate text + metadata columns into the encoder input string.

    The format is ``"<text1>\\n<text2>\\n...\\n<key1>: <val1> | <key2>: ..."``.
    Missing values are skipped so the string remains compact even when
    sparse metadata columns are configured.

    Args:
        row: One dataframe row.
        text_columns: Free-text columns concatenated with newlines.
        metadata_columns: Metadata columns appended as ``key: value``.

    Returns:
        Single input string for the tokeniser.
    """

    text_parts: list[str] = []
    for col in text_columns:
        value = row.get(col)
        if pd.isna(value):
            continue
        text_parts.append(str(value).strip())
    metadata_parts: list[str] = []
    for col in metadata_columns:
        value = row.get(col)
        if pd.isna(value):
            continue
        metadata_parts.append(f"{col}: {str(value).strip()}")
    sections: list[str] = []
    if text_parts:
        sections.append("\n".join(text_parts))
    if metadata_parts:
        sections.append(" | ".join(metadata_parts))
    return "\n".join(sections)


def to_hf_dataset(
    df: pd.DataFrame,
    *,
    tokenizer,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
) -> Dataset:
    """Build a tokenised :class:`datasets.Dataset` for the encoder family.

    Args:
        df: Source dataframe slice (train/val/test).
        tokenizer: Loaded HF tokenizer.
        data_cfg: Resolved data configuration.
        model_cfg: Resolved model configuration; ``max_length`` is used
            for truncation and ``family`` is checked.

    Returns:
        ``datasets.Dataset`` with ``input_ids``, ``attention_mask``,
        and ``labels`` columns.
    """

    if model_cfg.family != "encoder":
        raise ValueError(
            f"to_hf_dataset only supports encoder family in this run, got "
            f"{model_cfg.family!r}"
        )
    texts = [
        build_input_text(
            row,
            text_columns=data_cfg.text_columns,
            metadata_columns=data_cfg.metadata_columns,
        )
        for _, row in df.iterrows()
    ]
    labels = df[data_cfg.label_column].astype(int).tolist()
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=model_cfg.max_length,
        padding=False,
    )
    return Dataset.from_dict(
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }
    )


def class_weight_vector(labels: list[int] | np.ndarray) -> np.ndarray:
    """Inverse-frequency class weights for binary labels."""

    arr = np.asarray(labels, dtype=int)
    counts = np.bincount(arr, minlength=2).astype(float)
    counts[counts == 0] = 1.0
    weights = arr.shape[0] / (2.0 * counts)
    return weights.astype(np.float32)


def discover_csv_path(data_cfg: DataConfig) -> Path:
    """Return the absolute CSV path; primarily used by tests/diagnostics."""

    return data_cfg.annotated_csv
