"""Classification metrics + threshold + bootstrap helpers.

All functions are pure NumPy + scikit-learn so they can be reused by the
trainer (per-run) and by ``aggregate_results`` / plots (post-hoc) without
importing torch.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


_DEFAULT_BOOTSTRAP_ITERS = 2000
_DEFAULT_CI_LEVEL = 0.95
_RECALL_TARGET = 0.50
_PRECISION_TARGET = 0.70


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    """Compute the canonical metric bundle reported per run.

    Args:
        y_true: Binary ground-truth labels.
        y_prob: Predicted positive-class probabilities.
        threshold: Decision threshold for the threshold-dependent metrics.

    Returns:
        Dict with keys ``auroc, auprc, accuracy, precision, recall, f1,
        log_loss, brier, ece, precision_at_recall_50,
        recall_at_precision_70, threshold``.
    """

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_prob = _sanitize_probabilities(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    auroc = _safe_roc_auc(y_true, y_prob)
    auprc = _safe_average_precision(y_true, y_prob)
    metrics: dict[str, float] = {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "log_loss": _safe_log_loss(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": _expected_calibration_error(y_true, y_prob),
        "precision_at_recall_50": _precision_at_recall(
            y_true, y_prob, target_recall=_RECALL_TARGET
        ),
        "recall_at_precision_70": _recall_at_precision(
            y_true, y_prob, target_precision=_PRECISION_TARGET
        ),
        "threshold": float(threshold),
    }
    return metrics


def select_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return the Youden-J optimal threshold from the ROC curve.

    Falls back to ``0.5`` when the labels are degenerate (single class)
    so downstream code never sees a NaN threshold.
    """

    y_true = np.asarray(y_true, dtype=int)
    y_prob = _sanitize_probabilities(np.asarray(y_prob, dtype=float))
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_index = int(np.argmax(j_scores))
    best_threshold = float(thresholds[best_index])
    if not np.isfinite(best_threshold):
        return 0.5
    return float(np.clip(best_threshold, 0.0, 1.0))


def paired_delta_bootstrap_ci(
    per_fold_deltas: np.ndarray,
    *,
    n_bootstrap: int = _DEFAULT_BOOTSTRAP_ITERS,
    ci_level: float = _DEFAULT_CI_LEVEL,
    seed: int = 20260504,
) -> tuple[float, float, float]:
    """Bootstrap CI for a paired per-fold delta vector.

    Args:
        per_fold_deltas: Length-n array of seed-averaged per-fold
            (model_A - model_B) deltas.
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Two-sided coverage, default 95%.
        seed: RNG seed for reproducibility.

    Returns:
        ``(mean_delta, ci_low, ci_high)``.
    """

    arr = np.asarray(per_fold_deltas, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(low=0, high=arr.size, size=(n_bootstrap, arr.size))
    sample_means = arr[indices].mean(axis=1)
    alpha = (1.0 - ci_level) / 2.0
    return (
        float(arr.mean()),
        float(np.quantile(sample_means, alpha)),
        float(np.quantile(sample_means, 1.0 - alpha)),
    )


def _sanitize_probabilities(y_prob: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf prob values with 0.5 so sklearn metrics do not crash.

    NaN logits can appear in degenerate small-batch / mixed-precision runs.
    Returning 0.5 keeps the sample in the score curves but contributes no
    discriminative signal; the AUROC value will reflect that loss.
    """

    out = np.where(np.isfinite(y_prob), y_prob, 0.5)
    return np.clip(out, 0.0, 1.0)


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return AUROC or NaN when only one class is present."""

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _safe_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return AUPRC or NaN when only one class is present."""

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def _safe_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return binary cross-entropy on clipped probabilities.

    sklearn's :func:`log_loss` infinity-guards via clipping, but it still
    raises when only one class is present in ``y_true`` (cannot infer the
    label set), so we short-circuit that case to NaN to match the rest of
    the safe-metric helpers in this module.
    """

    if len(np.unique(y_true)) < 2:
        return float("nan")
    eps = 1e-15
    clipped = np.clip(y_prob, eps, 1.0 - eps)
    return float(log_loss(y_true, clipped, labels=[0, 1]))


def _expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 10
) -> float:
    """Equal-width binning ECE over the predicted probability."""

    y_prob = _sanitize_probabilities(y_prob)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(
        np.digitize(y_prob, bin_edges, right=False) - 1, 0, n_bins - 1
    )
    ece = 0.0
    total = float(y_prob.size)
    for b in range(n_bins):
        mask = bin_indices == b
        if not np.any(mask):
            continue
        bin_size = float(mask.sum())
        bin_conf = float(y_prob[mask].mean())
        bin_acc = float(y_true[mask].mean())
        ece += (bin_size / total) * abs(bin_conf - bin_acc)
    return float(ece)


def _precision_at_recall(
    y_true: np.ndarray, y_prob: np.ndarray, *, target_recall: float
) -> float:
    """Return the highest precision attainable at ``recall >= target``."""

    if len(np.unique(y_true)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    mask = recall >= target_recall
    if not np.any(mask):
        return 0.0
    return float(precision[mask].max())


def _recall_at_precision(
    y_true: np.ndarray, y_prob: np.ndarray, *, target_precision: float
) -> float:
    """Return the highest recall attainable at ``precision >= target``."""

    if len(np.unique(y_true)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    mask = precision >= target_precision
    if not np.any(mask):
        return 0.0
    return float(recall[mask].max())
