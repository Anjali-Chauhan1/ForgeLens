"""Generates the large-scale synthetic training set (target ~200k pairs,
per the blueprint) from a COCO-style images+instance-masks source.
Applies the laundering pipeline to every generated pair (forged AND a
matched authentic pass-through) so the training distribution matches
what the deployed system will actually see.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask_utils

from forgelens.data.synth_generator import sample_manipulation
from forgelens.data.laundering import LaunderingAugment


def load_coco_instances(ann_file: str, images_dir: str):
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    random.shuffle(img_ids)
    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        path = Path(images_dir) / info["file_name"]
        if not path.exists():
            continue
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        if not anns:
            continue
        ann = random.choice(anns)
        rle = coco.annToRLE(ann)
        obj_mask = coco_mask_utils.decode(rle).astype(np.float32)
        yield str(path), obj_mask


def main(args):
    out_images = Path(args.out_dir) / "images"
    out_masks = Path(args.out_dir) / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.out_dir) / "manifest.jsonl"
    augment = LaunderingAugment(p_apply=1.0)  # always launder the synth set

    source = list(load_coco_instances(args.ann_file, args.images_dir))
    if len(source) < 2:
        raise SystemExit("Need at least 2 annotated images to sample donor pairs.")

    manifest_lines = []
    n_written = 0
    attempts = 0
    max_attempts = args.n_pairs * 8
    while n_written < args.n_pairs and attempts < max_attempts:
        attempts += 1
        host_path, host_mask = random.choice(source)
        donor_path, donor_mask = random.choice(source)
        try:
            host = np.asarray(Image.open(host_path).convert("RGB"))
            donor = np.asarray(Image.open(donor_path).convert("RGB"))
        except Exception:
            continue

        if host_mask.shape != host.shape[:2]:
            continue
        if donor_mask.shape != donor.shape[:2]:
            continue

        pair = sample_manipulation(host, host_mask, donor, donor_mask,
                                    weights={"splice": 0.4, "copy_move": 0.3, "removal": 0.3})
        if pair.manipulation == "none" or pair.mask.sum() < 20:
            continue

        img, mask = augment(pair.image, pair.mask)

        stem = f"forged_{n_written:07d}"
        img_out = out_images / f"{stem}.jpg"
        mask_out = out_masks / f"{stem}.png"
        Image.fromarray(img).save(img_out, quality=92)
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_out)

        manifest_lines.append({
            "image_path": str(img_out), "mask_path": str(mask_out),
            "is_forged": True, "manipulation": pair.manipulation,
        })
        n_written += 1

        if n_written % 500 == 0:
            print(f"{n_written}/{args.n_pairs} forged pairs generated")

    if n_written < args.n_pairs:
        print(f"WARNING: only generated {n_written}/{args.n_pairs} forged pairs "
              f"after {attempts} attempts (source images/masks too small or "
              f"incompatible). Continuing with what was generated.")

    # Authentic counterpart set -- REQUIRED by AuthenticMixBatchSampler and
    # by the blueprint's non-negotiable "authentic images in every batch"
    # training rule. Laundered identically to the forged set, sampled from
    # source images NOT reused as a forgery host/donor pair here to avoid
    # near-duplicate authentic/forged images of the same underlying photo
    # dominating a single training batch.
    n_authentic = args.n_pairs if args.n_authentic is None else args.n_authentic
    n_auth_written = 0
    auth_attempts = 0
    while n_auth_written < n_authentic and auth_attempts < n_authentic * 4:
        auth_attempts += 1
        path, _ = random.choice(source)
        try:
            img = np.asarray(Image.open(path).convert("RGB"))
        except Exception:
            continue
        img, _ = augment(img, np.zeros(img.shape[:2], dtype=np.float32))

        stem = f"auth_{n_auth_written:07d}"
        img_out = out_images / f"{stem}.jpg"
        mask_out = out_masks / f"{stem}.png"
        Image.fromarray(img).save(img_out, quality=92)
        Image.fromarray(np.zeros(img.shape[:2], dtype=np.uint8)).save(mask_out)

        manifest_lines.append({
            "image_path": str(img_out), "mask_path": str(mask_out),
            "is_forged": False, "manipulation": "none",
        })
        n_auth_written += 1

        if n_auth_written % 500 == 0:
            print(f"{n_auth_written}/{n_authentic} authentic images generated")

    random.shuffle(manifest_lines)
    with open(manifest_path, "w") as manifest:
        for rec in manifest_lines:
            manifest.write(json.dumps(rec) + "\n")

    print(f"Done. {n_written} forged + {n_auth_written} authentic = "
          f"{len(manifest_lines)} total written to {args.out_dir}")


def build_argparser():
    p = argparse.ArgumentParser(description="Generate synthetic forgery training pairs")
    p.add_argument("--ann-file", required=True, help="COCO instances_*.json")
    p.add_argument("--images-dir", required=True)
    p.add_argument("--out-dir", default="data/synth")
    p.add_argument("--n-pairs", type=int, default=200_000)
    p.add_argument("--n-authentic", type=int, default=None,
                    help="Authentic images to generate. Defaults to n-pairs "
                         "(1:1 ratio, satisfying the authentic-mix training rule).")
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
