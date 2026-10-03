"""The degradation-grid robustness evaluation: JPEG QF x scale, as
called for in the blueprint (section 17). Runs a scoring function over
every cell of the grid and returns a table -- one of the most useful
and rarely-reported figures in this field.
"""
from __future__ import annotations
from typing import Callable
import numpy as np

from forgelens.data.laundering import _jpeg_recompress, _resize_random_interp

JPEG_QFS = (90, 70, 50, 30)
SCALES = (1.0, 0.75, 0.5)


def degradation_grid(image: np.ndarray, score_fn: Callable[[np.ndarray], float]) -> dict:
    """score_fn takes a degraded uint8 HWC image and returns a metric
    (e.g. pixel F1 against a precomputed, correspondingly-degraded mask,
    or a bare detection score). Returns {(qf, scale): value}.
    """
    grid = {}
    for qf in JPEG_QFS:
        for scale in SCALES:
            img = image
            if scale != 1.0:
                img = _resize_random_interp(img, scale)
            img = _jpeg_recompress(img, qf)
            grid[(qf, scale)] = score_fn(img)
    return grid


def grid_to_markdown_table(grid: dict[tuple[int, float], float], metric_name: str = "F1") -> str:
    lines = [f"| JPEG QF \\ Scale | {' | '.join(str(s) for s in SCALES)} |",
             "|---" * (len(SCALES) + 1) + "|"]
    for qf in JPEG_QFS:
        row = [f"{qf}"]
        for scale in SCALES:
            val = grid.get((qf, scale))
            row.append(f"{val:.3f}" if val is not None else "-")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
