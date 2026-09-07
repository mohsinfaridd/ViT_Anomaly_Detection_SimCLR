"""Evaluation metrics extracted from the final Protocol A notebooks."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def binary_metrics(y_true, y_pred, y_score) -> Dict[str, float]:
    """Exact notebook ``binary_metrics`` return keys."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        output_dict=True,
        zero_division=0,
    )

    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr_value = fp / (fp + tn) if (fp + tn) else float("nan")
    fnr_value = fn / (fn + tp) if (fn + tp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    fdr = fp / (fp + tp) if (fp + tp) else float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "normal_precision": float(precision[0]),
        "normal_recall_specificity": float(recall[0]),
        "normal_f1": float(f1[0]),
        "normal_support": int(support[0]),
        "anomaly_precision": float(precision[1]),
        "anomaly_recall": float(recall[1]),
        "anomaly_recall_sensitivity_tpr": float(recall[1]),
        "anomaly_f1": float(f1[1]),
        "anomaly_support": int(support[1]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "specificity_tnr": float(specificity),
        "sensitivity_tpr": float(sensitivity),
        "false_positive_rate": float(fpr_value),
        "false_negative_rate": float(fnr_value),
        "negative_predictive_value": float(npv),
        "false_discovery_rate": float(fdr),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else float("nan"),
        "average_precision": (
            float(average_precision_score(y_true, y_score))
            if len(np.unique(y_true)) == 2
            else float("nan")
        ),
        "average_precision_pr_auc": (
            float(average_precision_score(y_true, y_score))
            if len(np.unique(y_true)) == 2
            else float("nan")
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "matthews_correlation_coefficient_mcc": float(matthews_corrcoef(y_true, y_pred)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def pixel_roc_auc(score_maps, masks) -> float:
    flat_y = np.asarray(masks).reshape(-1).astype(np.uint8)
    flat_s = np.asarray(score_maps).reshape(-1).astype(np.float32)
    if len(np.unique(flat_y)) != 2:
        return float("nan")
    return float(roc_auc_score(flat_y, flat_s))


def pixel_average_precision(score_maps, masks) -> float:
    flat_y = np.asarray(masks).reshape(-1).astype(np.uint8)
    flat_s = np.asarray(score_maps).reshape(-1).astype(np.float32)
    if len(np.unique(flat_y)) != 2:
        return float("nan")
    return float(average_precision_score(flat_y, flat_s))


def compute_aupro(score_maps, masks, max_fpr: float = 0.30, num_thresholds: int = 100) -> float:
    """Exact notebook ``compute_aupro`` (AUPRO@max_fpr)."""
    score_maps = np.asarray(score_maps, dtype=np.float32)
    masks = np.asarray(masks, dtype=bool)
    normal_pixel_count = int((~masks).sum())
    if normal_pixel_count == 0 or masks.sum() == 0:
        return float("nan")

    finite_scores = score_maps[np.isfinite(score_maps)]
    if finite_scores.size == 0:
        return float("nan")

    qs = np.linspace(1.0, 0.0, num_thresholds)
    thresholds = np.unique(np.quantile(finite_scores, qs))[::-1]
    points = [(0.0, 0.0)]

    for threshold in thresholds:
        pred = score_maps >= threshold
        fpr = float((pred & (~masks)).sum() / normal_pixel_count)

        region_overlaps = []
        for mask, pred_mask in zip(masks, pred):
            labeled, n_regions = ndimage.label(mask)
            for region_id in range(1, n_regions + 1):
                region = labeled == region_id
                denom = int(region.sum())
                if denom:
                    region_overlaps.append(float((pred_mask & region).sum() / denom))

        if region_overlaps:
            points.append((fpr, float(np.mean(region_overlaps))))

        if fpr > max_fpr:
            break

    if len(points) < 2:
        return float("nan")

    points = sorted(points, key=lambda x: x[0])
    fprs_all = np.asarray([p[0] for p in points], dtype=float)
    pros_all = np.asarray([p[1] for p in points], dtype=float)

    below = fprs_all < max_fpr
    fprs = fprs_all[below]
    pros = pros_all[below]

    if fprs_all.min() <= max_fpr <= fprs_all.max():
        pro_at_max = float(np.interp(max_fpr, fprs_all, pros_all))
        fprs = np.append(fprs, max_fpr)
        pros = np.append(pros, pro_at_max)

    if len(fprs) < 2:
        return float("nan")

    return float(np.trapezoid(pros, fprs) / max_fpr)


def mean_and_sample_sd(values: Sequence[float]) -> Dict[str, float]:
    """
    Multi-seed mean and sample standard deviation.

    Uses ``ddof=1`` (notebook multi-seed summary).
    """
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if len(arr) == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"),
    }
