"""Stage S5 of the training curriculum: freeze all branches, train the
fusion meta-model and temperature scaling on a DEDICATED held-out split
that was never used for any branch's validation. Reusing a branch's own
val split here leaks and inflates the calibration numbers -- see the
explicit warning in the blueprint's training-recipe section.
"""
from __future__ import annotations
import argparse
import json
import numpy as np

from forgelens.fusion.fuse import FusionModel, features_to_vector
from forgelens.eval.metrics import expected_calibration_error, brier_score, reliability_diagram_data


def load_feature_rows(jsonl_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Each line: {"features": {...}, "label": 0|1}. Produced by running
    the full branch pipeline over a labeled dataset and dumping branch
    outputs -- see scripts/dump_branch_features.py.
    """
    X, y = [], []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            X.append(features_to_vector(rec["features"]))
            y.append(float(rec["label"]))
    return np.stack(X), np.array(y)


def main(args):
    X_train, y_train = load_feature_rows(args.fusion_train)
    X_cal, y_cal = load_feature_rows(args.calibration_split)

    fusion = FusionModel().fit(X_train, y_train, use_lgb=not args.force_logreg)
    fusion.fit_temperature(X_cal, y_cal)

    probs = fusion.calibrated_probability(X_cal)
    ece = expected_calibration_error(probs, y_cal)
    brier = brier_score(probs, y_cal)
    reliability = reliability_diagram_data(probs, y_cal)

    print(f"backend={fusion.backend} temperature={fusion.temperature:.3f}")
    print(f"calibration split: ECE={ece:.4f} Brier={brier:.4f}")
    if ece > 0.05:
        print("WARNING: ECE above the 0.05 target in the blueprint -- "
              "consider isotonic regression or more calibration data.")

    fusion.save(args.out)
    with open(args.out + ".reliability.json", "w") as f:
        json.dump({"ece": ece, "brier": brier, "reliability": reliability}, f, indent=2)
    print(f"Saved fusion model to {args.out}.*")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train fusion + calibration (stage S5)")
    p.add_argument("--fusion-train", required=True,
                    help="JSONL of {features, label} for fitting the meta-model")
    p.add_argument("--calibration-split", required=True,
                    help="A DISTINCT JSONL, never seen by any branch's validation")
    p.add_argument("--out", default="runs/fusion/fusion_model")
    p.add_argument("--force-logreg", action="store_true")
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
