"""Error Level Analysis.

Re-saves the image at a known JPEG quality and diffs against the input.
Regions that were pasted in (or otherwise at a different compression
history) light up. Noisy on its own — one input among many, never a
standalone verdict, per the design principle in the blueprint.
"""
from __future__ import annotations
import io
import numpy as np
from PIL import Image


def compute_ela(img: Image.Image, quality: int = 90) -> np.ndarray:
    """Returns an HxW float32 map in [0, 1], amplified for visibility."""
    rgb = img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")

    a = np.asarray(rgb, dtype=np.float32)
    b = np.asarray(resaved, dtype=np.float32)
    diff = np.abs(a - b).mean(axis=-1)  # HxW

    # Robust normalization: clip at the 99.5th percentile so a few
    # extreme pixels don't wash out the rest of the map.
    hi = np.percentile(diff, 99.5)
    hi = max(hi, 1e-3)
    norm = np.clip(diff / hi, 0.0, 1.0)
    return norm.astype(np.float32)


def ela_summary(ela_map: np.ndarray) -> dict:
    return {
        "mean": float(ela_map.mean()),
        "p95": float(np.percentile(ela_map, 95)),
        "hot_area_ratio": float((ela_map > 0.6).mean()),
    }
