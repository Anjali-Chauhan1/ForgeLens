"""Builds a unified manifest (list[Sample]) from heterogeneous dataset
layouts. Each supported dataset gets one small adapter function --
directory layouts differ enough across CASIA/NIST16/IMD2020/DEFACTO
that a single generic loader would be more fragile than explicit
per-dataset code, per the blueprint's data section.

Usage:
    python scripts/build_manifest.py \
        --casia2 /data/CASIA2 --authentic-extra /data/coco_authentic \
        --out manifests/train.jsonl
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from forgelens.data.datasets import Sample

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".bmp"}


def _find_mask_for(image_path: Path, masks_dir: Path) -> Path | None:
    """CASIA-style: mask filename shares the tampered image's stem."""
    for ext in (".png", ".jpg", ".bmp", ".tif"):
        candidate = masks_dir / (image_path.stem + ext)
        if candidate.exists():
            return candidate
        candidate = masks_dir / (image_path.stem + "_gt" + ext)
        if candidate.exists():
            return candidate
    return None


def adapt_casia2(root: str) -> list[Sample]:
    """Expected layout:
        root/Au/*.jpg              (authentic)
        root/Tp/*.jpg              (tampered)
        root/CASIA2_Groundtruth/*  (masks, matched by stem)
    Adjust paths to match whatever mirror you downloaded -- CASIA2
    ships with inconsistent folder naming across mirrors.
    """
    root = Path(root)
    samples: list[Sample] = []

    au_dir = root / "Au"
    if au_dir.exists():
        for p in au_dir.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                samples.append(Sample(image_path=str(p), mask_path=None, is_forged=False))

    tp_dir = root / "Tp"
    gt_dir = root / "CASIA2_Groundtruth"
    if tp_dir.exists():
        for p in tp_dir.iterdir():
            if p.suffix.lower() not in IMG_EXTS:
                continue
            mask = _find_mask_for(p, gt_dir) if gt_dir.exists() else None
            samples.append(Sample(image_path=str(p),
                                   mask_path=str(mask) if mask else None,
                                   is_forged=True))
    return samples


def adapt_generic_image_mask_dirs(images_dir: str, masks_dir: str | None,
                                   is_forged: bool) -> list[Sample]:
    """Fallback adapter for any dataset with a flat images/ + masks/
    layout and matching filenames (NIST16, IMD2020, COVERAGE, CoMoFoD,
    tampCOCO, AutoSplice all fit this shape once unpacked)."""
    images_dir = Path(images_dir)
    masks_path = Path(masks_dir) if masks_dir else None
    samples: list[Sample] = []
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        mask = _find_mask_for(p, masks_path) if masks_path else None
        samples.append(Sample(image_path=str(p),
                               mask_path=str(mask) if mask else None,
                               is_forged=is_forged and mask is not None))
    return samples


def adapt_authentic_pool(images_dir: str) -> list[Sample]:
    """Plain authentic image pool (e.g. COCO/Flickr) with no masks --
    used to satisfy the >=30%-authentic-per-batch training rule and the
    dedicated 10k-image false-positive-rate evaluation set."""
    images_dir = Path(images_dir)
    return [Sample(image_path=str(p), mask_path=None, is_forged=False)
            for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS]


_SAMPLE_FIELDS = {"image_path", "mask_path", "is_forged"}


def load_manifest(path: str) -> list[Sample]:
    """Tolerates extra manifest keys (e.g. `manipulation`, written by
    scripts/make_procedural_dataset.py and scripts/build_synth.py for
    ablation bookkeeping) that aren't Sample fields."""
    samples = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            filtered = {k: v for k, v in rec.items() if k in _SAMPLE_FIELDS}
            samples.append(Sample(**filtered))
    return samples


def load_synthetic_manifest(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_manifest(samples: list[Sample], out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s.__dict__) + "\n")
    n_forged = sum(1 for s in samples if s.is_forged)
    print(f"Wrote {len(samples)} samples ({n_forged} forged, "
          f"{len(samples) - n_forged} authentic) to {out_path}")


def main():
    p = argparse.ArgumentParser(description="Build a training/eval manifest")
    p.add_argument("--casia2", help="Path to a CASIA v2.0 root")
    p.add_argument("--images-dir", help="Generic images/ dir for --masks-dir adapter")
    p.add_argument("--masks-dir")
    p.add_argument("--forged", action="store_true", help="Treat --images-dir as forged samples")
    p.add_argument("--authentic-extra", help="Plain authentic image pool, no masks")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    samples: list[Sample] = []
    if args.casia2:
        samples.extend(adapt_casia2(args.casia2))
    if args.images_dir:
        samples.extend(adapt_generic_image_mask_dirs(args.images_dir, args.masks_dir, args.forged))
    if args.authentic_extra:
        samples.extend(adapt_authentic_pool(args.authentic_extra))

    if not samples:
        raise SystemExit("No samples collected -- pass at least one dataset source flag.")

    save_manifest(samples, args.out)


if __name__ == "__main__":
    main()
