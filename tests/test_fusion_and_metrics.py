"""Tests for fusion/calibration/metrics -- pure numpy/sklearn, no GPU."""
from __future__ import annotations
import numpy as np
import pytest

from forgelens.fusion.fuse import FusionModel, verdict_band, AbstentionGate, features_to_vector
from forgelens.eval.metrics import (
    pixel_f1, pixel_iou, expected_calibration_error, brier_score,
    risk_coverage_curve,
)


def test_verdict_bands_cover_full_range():
    for p in np.linspace(0, 1, 101):
        band = verdict_band(float(p))
        assert band["key"] in {
            "no_evidence", "weak_indicators", "inconclusive",
            "likely_manipulated", "strong_evidence",
        }


def test_fusion_model_fit_and_calibrate_logreg():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, len(features_to_vector({}))))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)

    model = FusionModel().fit(X, y, use_lgb=False)
    probs = model.raw_score(X)
    assert probs.shape == (200,)
    assert probs.min() >= 0.0 and probs.max() <= 1.0

    model.fit_temperature(X, y)
    calibrated = model.calibrated_probability(X)
    assert calibrated.min() >= 0.0 and calibrated.max() <= 1.0


def test_abstention_gate_low_quality():
    gate = AbstentionGate()
    abstain, reason = gate.should_abstain(
        fused_p=0.5, branch_scores=[0.5, 0.5], jpeg_quality=20.0, width=800, height=600
    )
    assert abstain is True
    assert "quality" in reason.lower()


def test_abstention_gate_low_resolution():
    gate = AbstentionGate()
    abstain, reason = gate.should_abstain(
        fused_p=0.1, branch_scores=[0.1], jpeg_quality=90.0, width=100, height=100
    )
    assert abstain is True


def test_abstention_gate_confident_case_does_not_abstain():
    gate = AbstentionGate()
    abstain, reason = gate.should_abstain(
        fused_p=0.92, branch_scores=[0.9, 0.88], jpeg_quality=90.0, width=1200, height=900
    )
    assert abstain is False


def test_pixel_f1_perfect_match():
    gt = np.zeros((10, 10), dtype=np.float32)
    gt[2:5, 2:5] = 1.0
    pred = gt.copy()
    assert pixel_f1(pred, gt, mode="fixed") == pytest.approx(1.0)


def test_pixel_f1_all_authentic_correct_prediction():
    gt = np.zeros((10, 10), dtype=np.float32)
    pred = np.zeros((10, 10), dtype=np.float32)
    assert pixel_f1(pred, gt) == 1.0


def test_pixel_iou_no_overlap():
    gt = np.zeros((10, 10), dtype=np.float32)
    gt[0:3, 0:3] = 1.0
    pred = np.zeros((10, 10), dtype=np.float32)
    pred[7:10, 7:10] = 1.0
    assert pixel_iou(pred, gt) == 0.0


def test_expected_calibration_error_perfect_calibration():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, 5000)
    labels = (rng.uniform(0, 1, 5000) < probs).astype(float)
    ece = expected_calibration_error(probs, labels, n_bins=20)
    assert ece < 0.05  # well-calibrated by construction


def test_brier_score_range():
    probs = np.array([0.0, 1.0, 0.5])
    labels = np.array([0.0, 1.0, 1.0])
    score = brier_score(probs, labels)
    assert 0.0 <= score <= 1.0


def test_risk_coverage_curve_monotonic_structure():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, 500)
    labels = (probs > 0.5).astype(float)
    preds = (probs > 0.5).astype(float)
    out = risk_coverage_curve(probs, labels, preds)
    assert len(out["coverage"]) == len(out["accuracy"]) == 500
    assert out["accuracy"][-1] == pytest.approx(1.0)  # perfect separability here


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
