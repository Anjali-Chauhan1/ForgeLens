"""Dumps per-image branch features + labels to JSONL, the input format
`forgelens/train/calibrate.py` expects for fitting the fusion meta-model
and temperature scaling (training stage S5 in the blueprint).

Runs the full ForgeLensPipeline (with the trained localization checkpoint
wired in) over a manifest and records `raw_features` + the true label
for each image -- this is exactly what the pipeline computed right
before fusion, so the meta-model trains on the same feature space used
at inference.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

from scripts.build_manifest import load_manifest
from forgelens.models.backbone import LocalizationNet
from forgelens.inference.pipeline import ForgeLensPipeline


def build_pipeline(checkpoint: str | None) -> ForgeLensPipeline:
    localization_model = None
    if checkpoint:
        model = LocalizationNet(backbone_name="convnext_tiny", pretrained=False)
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])  # raw weights -- see evaluate_test_set.py
        model.eval()
        localization_model = model
    return ForgeLensPipeline(localization_model=localization_model, run_deep_branches=True)


def main(args):
    samples = load_manifest(args.manifest)
    pipeline = build_pipeline(args.checkpoint)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for i, s in enumerate(samples):
            report = pipeline.analyze(s.image_path)
            f.write(json.dumps({
                "features": report.raw_features,
                "label": 1.0 if s.is_forged else 0.0,
            }) + "\n")
            if (i + 1) % 25 == 0:
                print(f"{i+1}/{len(samples)}")

    print(f"Wrote {len(samples)} feature rows to {args.out}")


def build_argparser():
    p = argparse.ArgumentParser(description="Dump branch features for fusion training")
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", default="runs/localize/best.pt")
    p.add_argument("--out", required=True)
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
