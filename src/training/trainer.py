"""Per-run training driver.

Defines a :class:`WeightedLossTrainer` that injects class weights into
the cross-entropy loss and a :func:`train_one_run` orchestrator that
sets seeds, builds the HF :class:`~transformers.Trainer`, picks the
val-optimal Youden-J threshold, evaluates on test, and writes a
``run.json`` artefact for downstream aggregation.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from .config import ModelConfig, SweepConfig
from .data import class_weight_vector
from .metrics import compute_classification_metrics, select_threshold_youden


_RUN_FILENAME = "run.json"
_LABEL_COLUMN = "labels"


class WeightedLossTrainer(Trainer):
    """HF ``Trainer`` subclass that applies class-weighted cross-entropy.

    The fold-local class weight tensor is provided via
    ``set_class_weights`` so the same trainer class works for every fold
    and seed without re-instantiation logic in the caller.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._class_weights: torch.Tensor | None = None

    def set_class_weights(self, weights: np.ndarray) -> None:
        """Register the class-weight tensor used by ``compute_loss``."""

        self._class_weights = torch.as_tensor(weights, dtype=torch.float32)

    def compute_loss(  # type: ignore[override]
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = (
            self._class_weights.to(device=logits.device, dtype=logits.dtype)
            if self._class_weights is not None
            else None
        )
        loss = nn.functional.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def train_one_run(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    model_cfg: ModelConfig,
    sweep_cfg: SweepConfig,
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    seed: int,
    learning_rate: float,
    fold_index: int,
    output_dir: Path,
    log_subdir: str,
) -> dict[str, Any]:
    """Train one (model, fold, seed) cell and write its ``run.json``.

    Returns the run-record dict that is also persisted to disk so the
    sweep loop can keep state without re-reading the file.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(seed)

    train_labels = [int(x) for x in train_ds[_LABEL_COLUMN]]
    weights = class_weight_vector(train_labels)

    args = _build_training_arguments(
        model_cfg=model_cfg,
        sweep_cfg=sweep_cfg,
        learning_rate=learning_rate,
        seed=seed,
        output_dir=output_dir,
        log_subdir=log_subdir,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = WeightedLossTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=sweep_cfg.early_stopping_patience
            )
        ],
    )
    trainer.set_class_weights(weights)

    trainer.train()

    val_pred = trainer.predict(val_ds)
    val_prob = _positive_class_prob(val_pred.predictions)
    val_labels = np.asarray(val_pred.label_ids, dtype=int)
    threshold = select_threshold_youden(val_labels, val_prob)
    val_metrics = compute_classification_metrics(
        val_labels, val_prob, threshold=threshold
    )

    test_pred = trainer.predict(test_ds)
    test_prob = _positive_class_prob(test_pred.predictions)
    test_labels = np.asarray(test_pred.label_ids, dtype=int)
    test_metrics = compute_classification_metrics(
        test_labels, test_prob, threshold=threshold
    )

    test_pred_labels = (test_prob >= threshold).astype(int)
    confusion_matrix = _confusion_matrix(test_labels, test_pred_labels)

    record: dict[str, Any] = {
        "model": model_cfg.name,
        "hf_id": model_cfg.hf_id,
        "fold": fold_index,
        "seed": seed,
        "learning_rate": float(learning_rate),
        "threshold": float(threshold),
        "class_weights": [float(w) for w in weights],
        "n_train": int(len(train_ds)),
        "n_val": int(len(val_ds)),
        "n_test": int(len(test_ds)),
        "metrics": {"val": val_metrics, "test": test_metrics},
        "test_confusion_matrix": confusion_matrix.tolist(),
        "model_cfg": asdict(model_cfg),
        "log_history": _serialize_log_history(trainer.state.log_history),
    }
    with (output_dir / _RUN_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, default=_json_default)
    return record


def _build_training_arguments(
    *,
    model_cfg: ModelConfig,
    sweep_cfg: SweepConfig,
    learning_rate: float,
    seed: int,
    output_dir: Path,
    log_subdir: str,
) -> TrainingArguments:
    """Translate model + sweep config into HF ``TrainingArguments``."""

    metric_key = sweep_cfg.metric_for_best_model.removeprefix("val_")
    return TrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(output_dir / log_subdir),
        num_train_epochs=model_cfg.epochs,
        per_device_train_batch_size=model_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=model_cfg.per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=model_cfg.weight_decay,
        warmup_ratio=model_cfg.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=metric_key,
        greater_is_better=True,
        logging_steps=20,
        report_to=["none"],
        seed=seed,
        data_seed=seed,
        bf16=bool(model_cfg.bf16 and torch.cuda.is_available()),
        gradient_checkpointing=model_cfg.gradient_checkpointing,
        optim=model_cfg.optimizer,
        disable_tqdm=False,
        dataloader_num_workers=0,
    )


def _compute_metrics(eval_pred) -> dict[str, float]:  # noqa: ANN001
    """Trainer-side metric callback: returns AUROC for best-model selection.

    Returns ``nan`` when labels are single-class or when predictions are
    not finite. HF ``Trainer`` tolerates NaN here by falling back to the
    most recent finite value when picking the best checkpoint.
    """

    logits, labels = eval_pred
    prob = _positive_class_prob(logits)
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2 or not np.all(np.isfinite(prob)):
        return {"auroc": float("nan")}
    from sklearn.metrics import roc_auc_score

    return {"auroc": float(roc_auc_score(labels, prob))}


def _positive_class_prob(logits: np.ndarray | torch.Tensor) -> np.ndarray:
    """Softmax over the two-class logits and return the positive column."""

    arr = np.asarray(logits, dtype=np.float64)
    shifted = arr - arr.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    softmax = exp / exp.sum(axis=-1, keepdims=True)
    return softmax[:, 1]


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Two-class confusion matrix as ``[[TN, FP], [FN, TP]]``."""

    cm = np.zeros((2, 2), dtype=np.int64)
    for true, pred in zip(y_true.tolist(), y_pred.tolist()):
        cm[int(true), int(pred)] += 1
    return cm


def _serialize_log_history(log_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coerce HF ``Trainer`` log entries to JSON-safe types."""

    return [
        {k: _to_jsonable(v) for k, v in entry.items()} for entry in log_history
    ]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unserialisable type: {type(value)!r}")


def _seed_everything(seed: int) -> None:
    """Set Python, NumPy, and Torch seeds (CPU + CUDA)."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
