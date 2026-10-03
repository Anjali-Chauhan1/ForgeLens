"""DCT coefficient histogram analysis — double-JPEG-compression detection.

When a region is pasted in after an initial save, then the whole image is
re-saved, that region carries two overlapping quantization grids. Because
JPEG operates on 8x8 blocks, a non-grid-aligned paste produces periodic
peaks and gaps in the AC coefficient histograms that a single compression
never produces. This module works directly on JPEG DCT coefficients
(no decode-recompress round trip), which is what makes it evidence rather
than a heuristic derived from ELA.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

try:
    import jpegio  # type: ignore
    _HAS_JPEGIO = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_JPEGIO = False


@dataclass
class DCTReport:
    available: bool
    double_compression_score: float
    histogram_periodicity: float
    notes: str


def _periodicity_score(hist: np.ndarray) -> float:
    """Measures periodic peaks in a coefficient histogram via the
    autocorrelation of the histogram's FFT magnitude — a simple, fast
    proxy for the classic Benford/periodic-artifact double-JPEG tests.
    """
    hist = hist.astype(np.float64)
    if hist.sum() == 0:
        return 0.0
    hist = hist / hist.sum()
    spec = np.abs(np.fft.rfft(hist - hist.mean()))
    if spec.max() <= 1e-12:
        return 0.0
    # Energy concentrated in a few periodic bins (as opposed to spread
    # smoothly) indicates a second quantization pass.
    top = np.sort(spec)[-5:].sum()
    total = spec.sum() + 1e-12
    return float(top / total)


def analyze_dct(jpeg_path: str) -> DCTReport:
    if not _HAS_JPEGIO:
        return DCTReport(
            available=False,
            double_compression_score=0.0,
            histogram_periodicity=0.0,
            notes="jpegio not installed — DCT-level analysis skipped. "
                  "Install `jpegio` for direct coefficient access.",
        )
    try:
        jpg = jpegio.read(jpeg_path)
    except Exception as e:  # not a JPEG, or unreadable
        return DCTReport(
            available=False,
            double_compression_score=0.0,
            histogram_periodicity=0.0,
            notes=f"Could not parse JPEG DCT stream: {e}",
        )

    coef = jpg.coef_arrays[0]  # luminance channel, block-DCT coefficients
    # Low/mid frequency AC coefficients carry the strongest double-JPEG
    # signature (e.g. (0,1) and (1,0) in each 8x8 block).
    ac01 = coef[0::8, 1::8].ravel()
    ac10 = coef[1::8, 0::8].ravel()

    bins = np.arange(-30, 31)
    h01, _ = np.histogram(ac01, bins=bins)
    h10, _ = np.histogram(ac10, bins=bins)

    p01 = _periodicity_score(h01)
    p10 = _periodicity_score(h10)
    periodicity = float(max(p01, p10))

    # Heuristic mapping to a [0,1] "double compression likely" score.
    score = float(np.clip((periodicity - 0.15) / 0.35, 0.0, 1.0))

    return DCTReport(
        available=True,
        double_compression_score=score,
        histogram_periodicity=periodicity,
        notes="Periodicity computed from AC(0,1)/AC(1,0) coefficient "
              "histograms across 8x8 blocks.",
    )
