"""Evaluation metrics.

Reports both fixed-threshold (0.5) and optimal-threshold pixel F1 --
per the blueprint, optimal-only numbers are not comparable across
papers and reviewers know it. Also implements Expected Calibration
Error / Brier score (for the calibration claim) and a risk-coverage
curve (for the abstention claim) -- these four functions back the
"USP 2 -- calibrated abstention" and evaluation-protocol sections of
the blueprint directly.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef


def pixel_f1(pred: np.ndarray, gt: np.ndarray, mode: str = "fixed") -> float:
    """mode='fixed' uses threshold 0.5; mode='optimal' sweeps 0.05-0.95.
    Always report both in a results table -- see blueprint section 13.
    """
    gt_flat = (gt.ravel() > 0.5).astype(np.uint8)
    if gt_flat.sum() == 0 and (pred.ravel() > 0.5).sum() == 0:
        return 1.0  # correctly predicted an all-authentic image
    if mode == "fixed":
        pred_flat = (pred.ravel() > 0.5).astype(np.uint8)
        return float(f1_score(gt_flat, pred_flat, zero_division=0))
    best = 0.0
    for t in np.arange(0.05, 1.0, 0.05):
        pred_flat = (pred.ravel() > t).astype(np.uint8)
        best = max(best, f1_score(gt_flat, pred_flat, zero_division=0))
    return float(best)


def pixel_iou(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
    p = pred.ravel() > threshold
    g = gt.ravel() > 0.5
    intersection = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def pixel_mcc(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
    p = (pred.ravel() > threshold).astype(np.uint8)
    g = (gt.ravel() > 0.5).astype(np.uint8)
    if len(np.unique(g)) < 2:
        return float(p.sum() == 0)
    return float(matthews_corrcoef(g, p))


def image_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def precision_at_recall(probs: np.ndarray, labels: np.ndarray, target_recall: float = 0.9) -> float:
    order = np.argsort(-probs)
    probs_sorted = probs[order]
    labels_sorted = labels[order]
    n_pos = labels.sum()
    if n_pos == 0:
        return float("nan")
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1)
    idx = np.searchsorted(recall, target_recall)
    if idx >= len(precision):
        return float(precision[-1])
    return float(precision[idx])


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def reliability_diagram_data(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, confs, counts = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        centers.append((lo + hi) / 2)
        counts.append(int(mask.sum()))
        if mask.sum() == 0:
            accs.append(None)
            confs.append(None)
        else:
            accs.append(float(labels[mask].mean()))
            confs.append(float(probs[mask].mean()))
    return {"bin_center": centers, "accuracy": accs, "confidence": confs, "count": counts}


def risk_coverage_curve(probs: np.ndarray, labels: np.ndarray, preds: np.ndarray) -> dict:
    """Sorts by prediction confidence (distance from 0.5) descending and
    reports accuracy as a function of the fraction of images answered --
    the honest headline the blueprint asks for instead of one accuracy
    number: "94% precision at 70% coverage" style reporting.
    """
    confidence = np.abs(probs - 0.5)
    order = np.argsort(-confidence)
    correct = (preds[order] == labels[order]).astype(np.float64)
    cum_acc = np.cumsum(correct) / (np.arange(len(correct)) + 1)
    coverage = (np.arange(len(correct)) + 1) / len(correct)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    aurc = float(trapezoid(1 - cum_acc, coverage))
    return {"coverage": coverage.tolist(), "accuracy": cum_acc.tolist(), "aurc": aurc}
