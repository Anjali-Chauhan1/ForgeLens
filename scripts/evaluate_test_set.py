"""Evaluates the trained pipeline on a held-out test manifest and
reports REAL, measured image-level accuracy/precision/recall/F1/AUC --
not an estimate. Runs the full ForgeLensPipeline (trained localization
model + classical branches + fallback fusion) exactly as a deployment
would, so the numbers reflect the actual end-to-end system, not just
the localization model in isolation.

Per the blueprint's evaluation-protocol section, this is explicitly an
IN-DOMAIN benchmark (same procedural-image distribution used for
training) -- it measures whether the pipeline correctly learned the
task, not cross-dataset real-world generalization. See MODEL_CARD.md
and README.md for how this differs from the CASIA/NIST16/IMD2020
cross-dataset numbers a production claim would require.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

from scripts.build_manifest import load_manifest
from forgelens.models.backbone import LocalizationNet
from forgelens.inference.pipeline import ForgeLensPipeline
from forgelens.fusion.fuse import FusionModel
from forgelens.eval.metrics import pixel_f1, pixel_iou, expected_calibration_error, brier_score


def load_trained_pipeline(checkpoint_path: str, use_ema: bool = False,
                           fusion_model_path: str | None = None) -> ForgeLensPipeline:
    model = LocalizationNet(backbone_name="convnext_tiny", pretrained=False)
    state = torch.load(checkpoint_path, map_location="cpu")
    # Default to the RAW weights, not EMA. The blueprint's "EMA is worth
    # ~1 F1 point" guidance assumes a full-length run; on a short run
    # (few hundred steps) EMA's shadow weights are still dominated by the
    # near-random initialization (verified: an EMA checkpoint from a
    # 312-step run predicted nothing on every test image, while the raw
    # weights -- the ones `evaluate()` actually validated as "best" during
    # training -- scored val pixel_f1=0.55). Pass use_ema=True once
    # training runs are long enough for EMA's warmup schedule (see
    # train/trainer_localize.py's EMA class) to have converged.
    key = "ema" if use_ema else "model"
    model.load_state_dict(state[key])
    model.eval()

    fusion_model = None
    if fusion_model_path and Path(fusion_model_path + ".json").exists():
        fusion_model = FusionModel.load(fusion_model_path)
        print(f"Loaded trained fusion model ({fusion_model.backend}, "
              f"temperature={fusion_model.temperature:.3f}) from {fusion_model_path}")
    else:
        print("No trained fusion model found -- using the documented linear "
              "fallback weighting (see pipeline.py).")

    return ForgeLensPipeline(localization_model=model, fusion_model=fusion_model,
                             run_deep_branches=True)


def main(args):
    samples = load_manifest(args.test_manifest)
    # Ground-truth masks for pixel-level metrics, where available.
    masks_by_path = {}
    with open(args.test_manifest) as f:
        for line in f:
            rec = json.loads(line)
            masks_by_path[rec["image_path"]] = rec.get("mask_path")

    pipeline = load_trained_pipeline(args.checkpoint, fusion_model_path=args.fusion_model)

    y_true, y_prob, y_pred = [], [], []
    # Pixel metrics are kept separate for forged vs authentic images:
    # an authentic image with an all-zero ground-truth mask trivially
    # scores F1=1.0 whenever the model predicts no region, which would
    # silently inflate a pooled "pixel F1" average without saying so.
    # Only forged-image pixel F1/IoU is the number the blueprint's
    # localization benchmarks (CASIA/NIST16/IMD2020) actually report.
    pixel_f1s_forged, pixel_ious_forged = [], []
    per_sample = []

    for s in samples:
        report = pipeline.analyze(s.image_path)
        label = 1.0 if s.is_forged else 0.0
        prob = report.probability
        # Abstained predictions still get a threshold-based pred here so
        # accuracy/F1/AUC are computable; `abstained` is tracked per-
        # sample so the caller can see how much of the reported accuracy
        # came from verdicts the system would actually have refused to
        # give in a real deployment (see the abstained count printed
        # below -- a high count means the point-accuracy numbers are
        # less representative of real behavior than the risk-coverage
        # framing the blueprint recommends).
        pred = 1.0 if prob >= 0.5 else 0.0

        y_true.append(label)
        y_prob.append(prob)
        y_pred.append(pred)
        per_sample.append({
            "path": s.image_path, "label": label, "prob": prob,
            "pred": pred, "abstained": report.abstained,
        })

        if s.is_forged:
            gt_mask_path = masks_by_path.get(s.image_path)
            if gt_mask_path and report.mask is not None:
                from PIL import Image
                gt = np.asarray(Image.open(gt_mask_path).convert("L")).astype(np.float32) / 255.0
                pred_mask = report.mask
                if pred_mask.shape == gt.shape:
                    pixel_f1s_forged.append(pixel_f1(pred_mask, gt, mode="fixed"))
                    pixel_ious_forged.append(pixel_iou(pred_mask, gt))

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    ece = expected_calibration_error(y_prob, y_true)
    brier = brier_score(y_prob, y_true)
    cm = confusion_matrix(y_true, y_pred)

    n_abstained = sum(1 for r in per_sample if r["abstained"])

    print("=" * 70)
    print(f"IMAGE-LEVEL RESULTS on {args.test_manifest} (n={len(samples)})")
    print("=" * 70)
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1:         {f1:.4f}")
    print(f"AUC:        {auc:.4f}")
    print(f"ECE:        {ece:.4f}")
    print(f"Brier:      {brier:.4f}")
    print(f"Abstained:  {n_abstained}/{len(samples)}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]:\n{cm}")

    if pixel_f1s_forged:
        print("-" * 70)
        print(f"PIXEL-LEVEL RESULTS (n={len(pixel_f1s_forged)} FORGED images only -- "
              f"authentic images excluded, since their trivial all-zero mask "
              f"would inflate this average)")
        print(f"Pixel F1 (fixed thresh):  {np.mean(pixel_f1s_forged):.4f}")
        print(f"Pixel IoU:                {np.mean(pixel_ious_forged):.4f}")
    print("=" * 70)
    print("NOTE: this is an in-domain synthetic benchmark (see MODEL_CARD.md) -- "
          "these numbers demonstrate the pipeline trains and evaluates correctly "
          "end-to-end, not cross-dataset real-world generalization.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                "auc": auc, "ece": ece, "brier": brier,
                "n_abstained": n_abstained, "n_total": len(samples),
                "confusion_matrix": cm.tolist(),
                "pixel_f1_forged_mean": float(np.mean(pixel_f1s_forged)) if pixel_f1s_forged else None,
                "pixel_iou_forged_mean": float(np.mean(pixel_ious_forged)) if pixel_ious_forged else None,
                "per_sample": per_sample,
            }, f, indent=2)
        print(f"Saved detailed results to {args.out}")


def build_argparser():
    p = argparse.ArgumentParser(description="Evaluate the trained pipeline on a test manifest")
    p.add_argument("--test-manifest", default="data/procedural/test.jsonl")
    p.add_argument("--checkpoint", default="runs/localize/best.pt")
    p.add_argument("--fusion-model", default="runs/fusion/fusion_model",
                    help="Path prefix for a trained FusionModel (from train/calibrate.py). "
                         "Falls back to the documented linear weighting if not found.")
    p.add_argument("--out", default="runs/eval_results.json")
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
