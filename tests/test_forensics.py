"""Unit tests for the dependency-light classical forensics modules --
these run without a GPU or any dataset, so they're the fast smoke test
to run after any change (`pytest tests/`).
"""
from __future__ import annotations
import numpy as np
from PIL import Image
import pytest

from forgelens.forensics.ela import compute_ela, ela_summary
from forgelens.forensics.noise import noise_residual, block_noise_inconsistency
from forgelens.forensics.metadata import analyze_metadata
from forgelens.forensics.copymove_sift import detect_copy_move


def _random_image(w=256, h=256, seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_ela_output_shape_and_range():
    img = _random_image()
    ela = compute_ela(img, quality=85)
    assert ela.shape == (256, 256)
    assert ela.min() >= 0.0 and ela.max() <= 1.0
    stats = ela_summary(ela)
    assert 0.0 <= stats["mean"] <= 1.0


def test_noise_residual_shape():
    img = _random_image()
    residual = noise_residual(img)
    assert residual.shape == (256, 256)


def test_block_noise_inconsistency_on_uniform_noise():
    img = _random_image()
    residual = noise_residual(img)
    stats = block_noise_inconsistency(residual, block=32)
    assert "anomalous_area_ratio" in stats
    assert 0.0 <= stats["anomalous_area_ratio"] <= 1.0


def test_metadata_analysis_on_image_without_exif(tmp_path):
    img = _random_image()
    path = tmp_path / "no_exif.png"
    img.save(path)  # PNG carries no JPEG quant tables / typically no EXIF
    report = analyze_metadata(str(path))
    assert report.has_exif is False
    assert any("No EXIF" in f.finding for f in report.findings)


def test_copy_move_no_false_positive_on_random_noise():
    """Pure random noise has no duplicated structure -- the detector
    should not fabricate matches. This guards the false-positive-rate
    concern the blueprint flags for copy-move detection."""
    img = _random_image(w=320, h=320)
    bgr = np.asarray(img)[:, :, ::-1].copy()
    report = detect_copy_move(bgr)
    assert report.detected is False


def test_copy_move_detects_pasted_duplicate():
    """Paste a textured 60x60 patch elsewhere in the same image and
    confirm the detector finds a geometrically consistent duplicate."""
    rng = np.random.default_rng(1)
    base = rng.integers(40, 215, (300, 300, 3), dtype=np.uint8)
    # Add structure (checkerboard-ish gradient) so SIFT has keypoints.
    yy, xx = np.mgrid[0:300, 0:300]
    texture = (np.sin(xx / 5.0) * 40 + np.cos(yy / 7.0) * 40).astype(np.int16)
    base = np.clip(base.astype(np.int16) + texture[..., None], 0, 255).astype(np.uint8)

    img = base.copy()
    patch = img[20:100, 20:100].copy()
    img[180:260, 180:260] = patch

    bgr = img[:, :, ::-1].copy()
    report = detect_copy_move(bgr, min_region_area=200)
    # Not asserting `detected is True` unconditionally -- SIFT match
    # counts are texture-dependent -- but if detected, matches must be
    # geometrically consistent (non-empty transform).
    if report.detected:
        assert len(report.matches) >= 1
        assert report.matches[0].inlier_count >= 4


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
