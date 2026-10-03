"""Synthetic forgery generator: builds (image, mask) training pairs from
a source-image + instance-mask corpus (e.g. COCO/OpenImages).

Implements splice, copy-move, and classical-inpainting removal directly.
Generative-fill (SD/SDXL) is deliberately factored into a pluggable
`GenerativeFiller` protocol rather than a hard dependency, since it
needs a diffusion model download the base pipeline shouldn't require --
wire in `diffusers.StableDiffusionInpaintPipeline` there when available.

The blueprint's guidance to vary blending quality deliberately --
including good forgeries, not just visible-seam ones -- is implemented
via `blend_quality` sampling across the full range.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import random

import numpy as np
import cv2


@dataclass
class ForgedPair:
    image: np.ndarray       # HWC uint8
    mask: np.ndarray        # HW float32 in {0,1}
    manipulation: str       # "splice" | "copy_move" | "removal" | "generative_fill"


def _poisson_blend(dst: np.ndarray, src_patch: np.ndarray, mask: np.ndarray,
                    center: tuple[int, int]) -> np.ndarray:
    mask_u8 = (mask * 255).astype(np.uint8)
    try:
        return cv2.seamlessClone(src_patch, dst, mask_u8, center, cv2.NORMAL_CLONE)
    except cv2.error:
        # Degenerate mask/geometry -- fall back to hard paste.
        out = dst.copy()
        y0, x0 = center[1] - src_patch.shape[0] // 2, center[0] - src_patch.shape[1] // 2
        region = out[y0:y0 + src_patch.shape[0], x0:x0 + src_patch.shape[1]]
        m = mask[..., None].astype(bool)
        region[:] = np.where(m, src_patch, region)
        return out


def _harmonize_color(patch: np.ndarray, dst_region: np.ndarray, strength: float) -> np.ndarray:
    """Shift the patch's mean color toward the destination region's mean,
    scaled by `strength` (0 = no harmonization -> obvious splice,
    1 = full match -> hard splice). Deliberately sampling the full range
    is what teaches the model to detect subtle forgeries, not just
    visible seams.
    """
    patch_mean = patch.reshape(-1, 3).mean(axis=0)
    dst_mean = dst_region.reshape(-1, 3).mean(axis=0)
    shift = (dst_mean - patch_mean) * strength
    out = np.clip(patch.astype(np.float32) + shift, 0, 255).astype(np.uint8)
    return out


def make_splice(host: np.ndarray, donor: np.ndarray, donor_mask: np.ndarray,
                 blend_quality: float | None = None) -> ForgedPair:
    """Pastes the donor's masked object into a random location in host."""
    blend_quality = random.uniform(0.0, 1.0) if blend_quality is None else blend_quality

    ys, xs = np.where(donor_mask > 0)
    if len(ys) == 0:
        return ForgedPair(host, np.zeros(host.shape[:2], np.float32), "splice")
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    patch = donor[y0:y1 + 1, x0:x1 + 1]
    pmask = donor_mask[y0:y1 + 1, x0:x1 + 1]

    scale = random.uniform(0.6, 1.3)
    ph, pw = patch.shape[:2]
    new_h, new_w = max(4, int(ph * scale)), max(4, int(pw * scale))
    patch = cv2.resize(patch, (new_w, new_h))
    pmask = cv2.resize(pmask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    if random.random() < 0.3:
        patch = np.fliplr(patch)
        pmask = np.fliplr(pmask)

    hh, hw = host.shape[:2]
    if new_h >= hh or new_w >= hw:
        return ForgedPair(host, np.zeros(host.shape[:2], np.float32), "splice")

    cy = random.randint(new_h // 2 + 1, hh - new_h // 2 - 1)
    cx = random.randint(new_w // 2 + 1, hw - new_w // 2 - 1)

    dst_region = host[cy - new_h // 2:cy - new_h // 2 + new_h,
                       cx - new_w // 2:cx - new_w // 2 + new_w]
    patch = _harmonize_color(patch, dst_region, strength=blend_quality)

    if blend_quality > 0.5:
        result = _poisson_blend(host, patch, pmask, (cx, cy))
    else:
        result = host.copy()
        y0d, x0d = cy - new_h // 2, cx - new_w // 2
        m = pmask[..., None].astype(bool)
        region = result[y0d:y0d + new_h, x0d:x0d + new_w]
        region[:] = np.where(m, patch, region)

    full_mask = np.zeros(host.shape[:2], dtype=np.float32)
    y0d, x0d = cy - new_h // 2, cx - new_w // 2
    full_mask[y0d:y0d + new_h, x0d:x0d + new_w] = pmask

    return ForgedPair(result, full_mask, "splice")


def make_copy_move(host: np.ndarray, obj_mask: np.ndarray) -> ForgedPair:
    """Duplicates a masked object elsewhere within the same image."""
    ys, xs = np.where(obj_mask > 0)
    if len(ys) == 0:
        return ForgedPair(host, np.zeros(host.shape[:2], np.float32), "copy_move")
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    patch = host[y0:y1 + 1, x0:x1 + 1].copy()
    pmask = obj_mask[y0:y1 + 1, x0:x1 + 1]
    ph, pw = patch.shape[:2]

    hh, hw = host.shape[:2]
    if ph >= hh or pw >= hw:
        return ForgedPair(host, np.zeros(host.shape[:2], np.float32), "copy_move")

    for _ in range(20):
        cy = random.randint(ph // 2 + 1, hh - ph // 2 - 1)
        cx = random.randint(pw // 2 + 1, hw - pw // 2 - 1)
        # Reject placements overlapping the source region.
        if not (y0 - ph <= cy <= y1 + ph and x0 - pw <= cx <= x1 + pw):
            break
    else:
        cy, cx = hh // 2, hw // 2

    result = host.copy()
    y0d, x0d = cy - ph // 2, cx - pw // 2
    m = pmask[..., None].astype(bool)
    region = result[y0d:y0d + ph, x0d:x0d + pw]
    region[:] = np.where(m, patch, region)

    full_mask = np.zeros(host.shape[:2], dtype=np.float32)
    full_mask[y0:y1 + 1, x0:x1 + 1] = np.maximum(full_mask[y0:y1 + 1, x0:x1 + 1], pmask)
    full_mask[y0d:y0d + ph, x0d:x0d + pw] = np.maximum(
        full_mask[y0d:y0d + ph, x0d:x0d + pw], pmask
    )
    return ForgedPair(result, full_mask, "copy_move")


def make_removal(host: np.ndarray, obj_mask: np.ndarray) -> ForgedPair:
    """Removes a masked object via classical inpainting (Telea).
    Swap `cv2.inpaint` for a `GenerativeFiller` for higher-realism
    removal training examples once one is wired in.
    """
    mask_u8 = (obj_mask * 255).astype(np.uint8)
    result = cv2.inpaint(host, mask_u8, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    return ForgedPair(result, obj_mask.astype(np.float32), "removal")


class GenerativeFiller(Protocol):
    """Pluggable interface for diffusion-based generative fill, so the
    base package has no hard dependency on a multi-GB model download."""

    def fill(self, image: np.ndarray, mask: np.ndarray, prompt: str) -> np.ndarray: ...


def make_generative_fill(host: np.ndarray, region_mask: np.ndarray,
                          filler: GenerativeFiller, prompt: str) -> ForgedPair:
    result = filler.fill(host, region_mask, prompt)
    return ForgedPair(result, region_mask.astype(np.float32), "generative_fill")


def sample_manipulation(host: np.ndarray, host_mask: np.ndarray | None,
                         donor: np.ndarray | None, donor_mask: np.ndarray | None,
                         weights: dict[str, float] | None = None) -> ForgedPair:
    """Randomly samples one manipulation type per the given class weights,
    falling back gracefully when the required inputs aren't available."""
    weights = weights or {"splice": 0.4, "copy_move": 0.3, "removal": 0.3}
    choices, probs = zip(*weights.items())
    kind = random.choices(choices, weights=probs, k=1)[0]

    if kind == "splice" and donor is not None and donor_mask is not None:
        return make_splice(host, donor, donor_mask)
    if kind == "copy_move" and host_mask is not None:
        return make_copy_move(host, host_mask)
    if kind == "removal" and host_mask is not None:
        return make_removal(host, host_mask)
    # Fallback: authentic pass-through (keeps the batch composition sane
    # if the sampled type's required inputs are missing this iteration).
    return ForgedPair(host, np.zeros(host.shape[:2], np.float32), "none")
