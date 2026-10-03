"""EXIF / container metadata analysis — zero-ML evidence.

The single highest-precision signal in the whole pipeline: a JPEG
quantization table that matches known editing software while EXIF claims
a camera is direct, explainable proof of re-saving.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import io
import struct

from PIL import Image
from PIL.ExifTags import TAGS

# A small reference table of known JPEG quantization-table fingerprints.
# Real deployments should grow this from a curated camera/software corpus
# (e.g. FotoForensics' public JPEG dimples dataset). Values are the
# luminance (0th) quant table flattened in zig-zag order, quality~ tag.
KNOWN_QTABLE_SIGNATURES: dict[tuple[int, ...], str] = {
    # Photoshop "Save for Web" quality 80, luminance table (approximate,
    # illustrative — replace with a verified corpus before production use).
    (6, 4, 4, 6, 10, 16, 20, 24): "Adobe Photoshop (Save for Web, ~Q80)",
    (3, 2, 2, 3, 4, 6, 10, 12): "Adobe Photoshop (Save for Web, ~Q90)",
    (8, 6, 5, 8, 12, 20, 26, 31): "libjpeg / generic encoder (~Q75)",
}

EDITING_SOFTWARE_MARKERS = (
    "adobe photoshop", "gimp", "lightroom", "affinity photo",
    "paint.net", "pixelmator", "snapseed", "picsart", "canva",
    "capture one", "luminar",
)


@dataclass
class MetadataFinding:
    branch: str = "metadata"
    finding: str = ""
    weight: str = "neutral"  # neutral | low | medium | high
    detail: dict = field(default_factory=dict)


@dataclass
class MetadataReport:
    has_exif: bool
    exif: dict
    software_tag: str | None
    camera_make: str | None
    camera_model: str | None
    jpeg_quant_tables: list[list[int]]
    jpeg_quality_estimate: float | None
    quant_table_match: str | None
    findings: list[MetadataFinding]


def _extract_exif(img: Image.Image) -> dict:
    raw = img.getexif()
    out = {}
    if not raw:
        return out
    for tag_id, value in raw.items():
        tag = TAGS.get(tag_id, tag_id)
        try:
            if isinstance(value, bytes):
                value = value.decode(errors="replace")
            out[str(tag)] = value
        except Exception:
            continue
    return out


def _extract_quant_tables(img: Image.Image) -> list[list[int]]:
    """PIL exposes JPEG quantization tables via `quantization`."""
    tables = getattr(img, "quantization", None)
    if not tables:
        return []
    return [list(t) for t in tables.values()]


def _estimate_quality(tables: list[list[int]]) -> float | None:
    """Rough JPEG quality estimate from the luminance quant table mean.

    Not a precise IJG-quality inverse — a monotonic proxy sufficient for
    the abstention gate and fusion covariate.
    """
    if not tables:
        return None
    lum = tables[0]
    mean_val = sum(lum) / len(lum)
    # Empirical monotonic mapping: low mean -> high quality.
    if mean_val <= 2:
        return 95.0
    if mean_val >= 60:
        return 20.0
    # Linear interpolation between anchor points.
    return max(20.0, min(95.0, 100.0 - mean_val * 1.3))


def _match_quant_signature(tables: list[list[int]]) -> str | None:
    if not tables:
        return None
    lum = tuple(tables[0][:8])
    for sig, label in KNOWN_QTABLE_SIGNATURES.items():
        if all(abs(a - b) <= 1 for a, b in zip(lum, sig)):
            return label
    return None


def analyze_metadata(image_path: str | Path) -> MetadataReport:
    img = Image.open(image_path)
    exif = _extract_exif(img)
    tables = _extract_quant_tables(img)
    quality = _estimate_quality(tables)
    qmatch = _match_quant_signature(tables)

    software = exif.get("Software")
    make = exif.get("Make")
    model = exif.get("Model")

    findings: list[MetadataFinding] = []

    if not exif:
        findings.append(MetadataFinding(
            finding="No EXIF metadata present. Common after social-media "
                    "re-upload or deliberate stripping; not itself proof of "
                    "manipulation.",
            weight="low",
        ))

    if software and any(m in str(software).lower() for m in EDITING_SOFTWARE_MARKERS):
        findings.append(MetadataFinding(
            finding=f"EXIF Software tag reports editing software: '{software}'.",
            weight="medium",
            detail={"software": software},
        ))

    if qmatch:
        contradiction = bool(make) and "adobe" not in str(make).lower()
        findings.append(MetadataFinding(
            finding=(
                f"JPEG quantization table matches known signature: {qmatch}."
                + (f" EXIF claims camera make '{make}', which is inconsistent "
                   f"with an editing-software save." if contradiction else "")
            ),
            weight="high" if contradiction else "medium",
            detail={"signature": qmatch, "camera_make": make},
        ))

    if make and model and not software:
        findings.append(MetadataFinding(
            finding=f"Camera EXIF present ({make} {model}) with no editing "
                    f"software tag — weak indicator of an unedited original.",
            weight="low",
        ))

    return MetadataReport(
        has_exif=bool(exif),
        exif=exif,
        software_tag=str(software) if software else None,
        camera_make=str(make) if make else None,
        camera_model=str(model) if model else None,
        jpeg_quant_tables=tables,
        jpeg_quality_estimate=quality,
        quant_table_match=qmatch,
        findings=findings,
    )
