"""Dataset loading for Branch A.

Two design rules from the blueprint are enforced structurally here,
not left to caller discipline:

1. Never interpolate-resize. Crops are taken at native resolution and
   zero-padded to a fixed canonical size. Resizing destroys the
   high-frequency forensic traces the noise/DCT streams depend on and
   shifts the JPEG 8x8 grid.
2. Authentic images make up >=30% of every batch (`AuthenticMixSampler`),
   which is what keeps the false-positive rate usable — a model trained
   only on forged images learns to always predict "tampered".
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image


def pad_to_canonical(img: np.ndarray, canonical: int = 512,
                      pad_value: float = 0.0) -> tuple[np.ndarray, tuple[int, int]]:
    """Center-crops if larger than canonical, zero-pads if smaller.
    Never resamples pixels. Returns the array and the (top, left) pad
    offset so masks can be aligned identically.
    """
    h, w = img.shape[:2]
    out_shape = (canonical, canonical) + img.shape[2:]
    out = np.full(out_shape, pad_value, dtype=img.dtype)

    # Crop source if needed (random crop during training is applied by
    # the caller before this function; here we just center-fit).
    ch, cw = min(h, canonical), min(w, canonical)
    sy = (h - ch) // 2
    sx = (w - cw) // 2
    dy = (canonical - ch) // 2
    dx = (canonical - cw) // 2

    out[dy:dy + ch, dx:dx + cw, ...] = img[sy:sy + ch, sx:sx + cw, ...]
    return out, (dy, dx)


def random_crop_or_pad(img: np.ndarray, mask: np.ndarray | None,
                        canonical: int = 512) -> tuple[np.ndarray, np.ndarray | None]:
    h, w = img.shape[:2]
    if h > canonical or w > canonical:
        y0 = random.randint(0, max(h - canonical, 0))
        x0 = random.randint(0, max(w - canonical, 0))
        img = img[y0:y0 + canonical, x0:x0 + canonical, ...]
        if mask is not None:
            mask = mask[y0:y0 + canonical, x0:x0 + canonical, ...]
    img, (dy, dx) = pad_to_canonical(img, canonical)
    if mask is not None:
        mask, _ = pad_to_canonical(mask, canonical, pad_value=0.0)
    return img, mask


@dataclass
class Sample:
    image_path: str
    mask_path: str | None  # None => authentic image, all-zero mask
    is_forged: bool


class ManipulationDataset(Dataset):
    """Expects a manifest list of Sample. Building manifests per source
    dataset (CASIA v2, NIST16, IMD2020, ...) is left to
    scripts/build_manifest.py, since directory layouts differ.
    """

    def __init__(self, samples: list[Sample], canonical: int = 512,
                 augment=None, train: bool = True):
        self.samples = samples
        self.canonical = canonical
        self.augment = augment
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")
        img_np = np.asarray(img, dtype=np.uint8)

        if s.mask_path is not None:
            mask = Image.open(s.mask_path).convert("L")
            mask_np = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)
        else:
            mask_np = np.zeros(img_np.shape[:2], dtype=np.float32)

        if self.augment is not None and self.train:
            img_np, mask_np = self.augment(img_np, mask_np)

        img_np, mask_np = random_crop_or_pad(img_np, mask_np, self.canonical)

        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).float()
        image_label = torch.tensor(
            1.0 if (s.is_forged and mask_np.sum() > 0) else 0.0, dtype=torch.float32
        )

        return {
            "image": img_t,
            "mask": mask_t,
            "image_label": image_label,
            "path": s.image_path,
        }


class AuthenticMixBatchSampler(Sampler):
    """Yields batches guaranteed to contain >= `min_authentic_ratio`
    authentic (all-zero-mask) samples, per the blueprint's
    non-negotiable training rule.
    """

    def __init__(self, samples: list[Sample], batch_size: int = 12,
                 min_authentic_ratio: float = 0.3, seed: int = 0):
        self.batch_size = batch_size
        self.min_authentic = max(1, int(round(batch_size * min_authentic_ratio)))
        self.forged_idx = [i for i, s in enumerate(samples) if s.is_forged]
        self.authentic_idx = [i for i, s in enumerate(samples) if not s.is_forged]
        if not self.authentic_idx:
            raise ValueError(
                "No authentic (unforged) samples in manifest — the "
                "authentic-image constraint cannot be enforced. Add "
                "authentic images to the training manifest."
            )
        self.rng = random.Random(seed)
        self.n_batches = max(1, len(samples) // batch_size)

    def __iter__(self):
        for _ in range(self.n_batches):
            n_auth = self.min_authentic
            n_forged = self.batch_size - n_auth
            batch = (
                self.rng.sample(self.authentic_idx, k=min(n_auth, len(self.authentic_idx)))
                + self.rng.sample(self.forged_idx, k=min(n_forged, len(self.forged_idx)))
            )
            self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches
