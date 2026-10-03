"""Generates a labeled authentic/forged image benchmark without any
external dataset dependency, so the pipeline can be trained and
evaluated end-to-end in an environment with no internet dataset access.

Authentic images are procedurally rendered (gradient backgrounds +
random geometric shapes + photographic-style noise) rather than pure
random noise, so the task isn't trivially separable by noise statistics
alone. Forged images are produced by running the SAME splice/copy-move/
removal generator used for real training data
(forgelens/data/synth_generator.py) between pairs of these procedural
images, then BOTH authentic and forged images are passed through the
laundering pipeline identically -- per the blueprint's non-negotiable
rule that laundering must not be applied only to forged images.

This is explicitly an in-domain synthetic benchmark for validating that
the training pipeline works and measuring achievable image-level
accuracy in this environment -- not a substitute for the real
cross-dataset benchmarks (CASIA/NIST16/IMD2020) the blueprint specifies
for production evaluation claims.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from forgelens.data.synth_generator import sample_manipulation
from forgelens.data.laundering import LaunderingAugment


def _random_gradient(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    c1 = rng.integers(20, 235, 3)
    c2 = rng.integers(20, 235, 3)
    angle = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    proj = (xx * np.cos(angle) + yy * np.sin(angle))
    proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)
    grad = c1[None, None, :] * (1 - proj[..., None]) + c2[None, None, :] * proj[..., None]
    return grad.astype(np.uint8)


def _add_shapes(img: Image.Image, rng: np.random.Generator, n_shapes: int) -> Image.Image:
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(n_shapes):
        kind = rng.choice(["ellipse", "rectangle", "polygon", "line"])
        color = tuple(int(v) for v in rng.integers(0, 255, 3))
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        x1, y1 = x0 + rng.integers(10, w // 2), y0 + rng.integers(10, h // 2)
        try:
            if kind == "ellipse":
                draw.ellipse([x0, y0, x1, y1], fill=color)
            elif kind == "rectangle":
                draw.rectangle([x0, y0, x1, y1], fill=color)
            elif kind == "polygon":
                pts = [(int(rng.integers(0, w)), int(rng.integers(0, h))) for _ in range(5)]
                draw.polygon(pts, fill=color)
            else:
                draw.line([x0, y0, x1, y1], fill=color, width=int(rng.integers(2, 8)))
        except Exception:
            continue
    return img


def make_authentic_image(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = _random_gradient(size, size, rng)
    img = Image.fromarray(base)
    img = _add_shapes(img, rng, n_shapes=int(rng.integers(4, 12)))
    img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.3, 1.5))))
    arr = np.asarray(img).astype(np.float32)
    # Photographic-style sensor noise (not pure white noise -- correlated
    # slightly across channels, like real camera noise).
    noise = rng.normal(0, rng.uniform(2, 6), (size, size, 1)).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return arr


def build(args):
    out_images = Path(args.out_dir) / "images"
    out_masks = Path(args.out_dir) / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.out_dir) / "manifest.jsonl"

    augment = LaunderingAugment(p_apply=args.laundering_p)
    random.seed(args.seed)

    pool = [make_authentic_image(args.size, seed=args.seed * 10_000 + i)
            for i in range(args.n_base_images)]

    records = []
    n_authentic = args.n_total // 2
    n_forged = args.n_total - n_authentic

    for i in range(n_authentic):
        host = pool[i % len(pool)].copy()
        img, mask = augment(host, np.zeros(host.shape[:2], dtype=np.float32))
        stem = f"auth_{i:05d}"
        img_path = out_images / f"{stem}.jpg"
        mask_path = out_masks / f"{stem}.png"
        Image.fromarray(img).save(img_path, quality=random.randint(75, 96))
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
        records.append({"image_path": str(img_path), "mask_path": str(mask_path),
                         "is_forged": False, "manipulation": "none"})

    kinds_weights = {"splice": 0.4, "copy_move": 0.3, "removal": 0.3}
    made = 0
    attempts = 0
    while made < n_forged and attempts < n_forged * 5:
        attempts += 1
        host = pool[random.randrange(len(pool))].copy()
        donor = pool[random.randrange(len(pool))].copy()

        host_mask = np.zeros(host.shape[:2], dtype=np.float32)
        s = args.size
        bw, bh = random.randint(s // 6, s // 3), random.randint(s // 6, s // 3)
        x0 = random.randint(0, s - bw - 1)
        y0 = random.randint(0, s - bh - 1)
        host_mask[y0:y0 + bh, x0:x0 + bw] = 1.0

        donor_mask = np.zeros(donor.shape[:2], dtype=np.float32)
        dbw, dbh = random.randint(s // 6, s // 3), random.randint(s // 6, s // 3)
        dx0 = random.randint(0, s - dbw - 1)
        dy0 = random.randint(0, s - dbh - 1)
        donor_mask[dy0:dy0 + dbh, dx0:dx0 + dbw] = 1.0

        pair = sample_manipulation(host, host_mask, donor, donor_mask, weights=kinds_weights)
        if pair.manipulation == "none" or pair.mask.sum() < 30:
            continue

        img, mask = augment(pair.image, pair.mask)
        stem = f"forged_{made:05d}"
        img_path = out_images / f"{stem}.jpg"
        mask_path = out_masks / f"{stem}.png"
        Image.fromarray(img).save(img_path, quality=random.randint(75, 96))
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
        records.append({"image_path": str(img_path), "mask_path": str(mask_path),
                         "is_forged": True, "manipulation": pair.manipulation})
        made += 1

    random.shuffle(records)
    with open(manifest_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(records)} images ({n_authentic} authentic, {made} forged) "
          f"to {args.out_dir}")
    if made < n_forged:
        print(f"WARNING: only generated {made}/{n_forged} forged images "
              f"(some geometric placements were rejected); consider more "
              f"base images or a larger canvas.")


def build_argparser():
    p = argparse.ArgumentParser(description="Generate a procedural authentic/forged benchmark")
    p.add_argument("--out-dir", default="data/procedural")
    p.add_argument("--n-total", type=int, default=360)
    p.add_argument("--n-base-images", type=int, default=60)
    p.add_argument("--size", type=int, default=160)
    p.add_argument("--laundering-p", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    build(build_argparser().parse_args())
