"""Standalone quantization-table extraction, kept separate from
metadata.py so it can be reused by the DCT and fusion modules without
pulling in EXIF parsing."""
from __future__ import annotations
from PIL import Image


def extract_quant_tables(image_path: str) -> list[list[int]]:
    img = Image.open(image_path)
    tables = getattr(img, "quantization", None)
    if not tables:
        return []
    return [list(t) for t in tables.values()]


def quality_from_tables(tables: list[list[int]]) -> float | None:
    if not tables:
        return None
    lum = tables[0]
    mean_val = sum(lum) / len(lum)
    if mean_val <= 2:
        return 95.0
    if mean_val >= 60:
        return 20.0
    return max(20.0, min(95.0, 100.0 - mean_val * 1.3))
