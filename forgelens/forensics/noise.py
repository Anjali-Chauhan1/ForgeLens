"""Fast, dependency-light noise-residual estimate for the classical
pipeline (the deep Bayar/SRM stems in forgelens/models/stems.py are the
learned version used by Branch A)."""
from __future__ import annotations
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter


def noise_residual(img: Image.Image, size: int = 3) -> np.ndarray:
    """Denoise-and-subtract residual, averaged over channels."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    denoised = np.stack(
        [median_filter(arr[..., c], size=size) for c in range(3)], axis=-1
    )
    residual = arr - denoised
    return residual.mean(axis=-1)


def block_noise_inconsistency(residual: np.ndarray, block: int = 32) -> dict:
    """Splits the residual into blocks and measures how much block-level
    noise variance deviates from the image-wide median — a coarse,
    fast localization signal that complements the deep model and is
    cheap enough to run on every request."""
    h, w = residual.shape
    variances = []
    coords = []
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            patch = residual[y:y + block, x:x + block]
            variances.append(float(patch.var()))
            coords.append((x, y))
    if not variances:
        return {"anomalous_blocks": [], "global_variance": 0.0}

    variances_arr = np.array(variances)
    med = float(np.median(variances_arr))
    mad = float(np.median(np.abs(variances_arr - med))) + 1e-6
    z = np.abs(variances_arr - med) / (1.4826 * mad)

    anomalous = [
        {"x": x, "y": y, "w": block, "h": block, "z_score": float(zi)}
        for (x, y), zi in zip(coords, z) if zi > 3.5
    ]
    return {
        "anomalous_blocks": anomalous,
        "global_variance": med,
        "anomalous_area_ratio": len(anomalous) / max(len(variances), 1),
    }
