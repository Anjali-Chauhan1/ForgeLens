"""Fusion, calibration, and abstention.

Per the blueprint: do NOT average branch scores. A small meta-model
(here, gradient-boosted trees via LightGBM with a logistic-regression
fallback) is trained on held-out branch outputs plus quality covariates
(JPEG QF, resolution, blur). That is what lets the fusion layer learn
"discount the noise branch when JPEG QF is low" automatically instead
of a hand-tuned rule -- the single largest source of unforced accuracy
loss in naive multi-branch systems.

Calibration (temperature scaling) is applied on top of the fusion
output so the reported probability is a real probability, not a score.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:  # pragma: no cover
    _HAS_LGB = False

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


FEATURE_NAMES = [
    "seg_max", "seg_mean", "seg_area_ratio", "seg_connected_components",
    "seg_boundary_sharpness",
    "copymove_confidence", "copymove_num_pairs", "copymove_transform_inliers",
    "synthetic_global_score", "synthetic_patch_max", "synthetic_patch_area",
    "meta_qtable_mismatch", "meta_exif_missing", "meta_software_tag_present",
    "c2pa_status",              # 0=absent, 1=invalid, 2=valid
    "jpeg_quality_estimate", "image_resolution_log", "blur_estimate",
    "classical_noise_anomaly",  # block_noise_inconsistency ratio -- the
                                 # only branch signal available before
                                 # Branch A has been trained, so it must
                                 # be in the feature set and the fallback
                                 # weighting, not just the evidence list.
]


def features_to_vector(features: dict) -> np.ndarray:
    return np.array([float(features.get(name, 0.0)) for name in FEATURE_NAMES],
                     dtype=np.float64)


@dataclass
class FusionModel:
    model: object | None = None
    temperature: float = 1.0
    isotonic: IsotonicRegression | None = None
    backend: str = "logreg"
    feature_names: list = field(default_factory=lambda: list(FEATURE_NAMES))

    def fit(self, X: np.ndarray, y: np.ndarray, use_lgb: bool = True):
        if use_lgb and _HAS_LGB:
            self.backend = "lgbm"
            train_set = lgb.Dataset(X, label=y)
            params = dict(
                objective="binary", metric="binary_logloss",
                num_leaves=15, max_depth=4, learning_rate=0.05,
                min_data_in_leaf=10, verbose=-1,
            )
            self.model = lgb.train(params, train_set, num_boost_round=200)
        else:
            self.backend = "logreg"
            self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
            self.model.fit(X, y)
        return self

    def raw_score(self, X: np.ndarray) -> np.ndarray:
        if self.backend == "lgbm":
            return self.model.predict(X)
        return self.model.predict_proba(X)[:, 1]

    def fit_temperature(self, X_val: np.ndarray, y_val: np.ndarray, steps: int = 200):
        """1-parameter temperature scaling on the logit of the fusion
        score, minimizing NLL on a held-out split that was NOT used to
        train the fusion model (the blueprint explicitly warns against
        reusing the branch validation split here -- it leaks)."""
        raw = self.raw_score(X_val)
        raw = np.clip(raw, 1e-6, 1 - 1e-6)
        logit = np.log(raw / (1 - raw))

        best_t, best_nll = 1.0, float("inf")
        for t in np.linspace(0.3, 3.0, steps):
            p = 1.0 / (1.0 + np.exp(-logit / t))
            p = np.clip(p, 1e-6, 1 - 1e-6)
            nll = -np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))
            if nll < best_nll:
                best_nll, best_t = nll, t
        self.temperature = float(best_t)
        return self

    def calibrated_probability(self, X: np.ndarray) -> np.ndarray:
        raw = np.clip(self.raw_score(X), 1e-6, 1 - 1e-6)
        logit = np.log(raw / (1 - raw))
        return 1.0 / (1.0 + np.exp(-logit / self.temperature))

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {"temperature": self.temperature, "backend": self.backend,
                   "feature_names": self.feature_names}
        if self.backend == "lgbm":
            self.model.save_model(path + ".lgb")
        else:
            import joblib
            joblib.dump(self.model, path + ".joblib")
        with open(path + ".json", "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FusionModel":
        with open(path + ".json") as f:
            payload = json.load(f)
        obj = cls(temperature=payload["temperature"], backend=payload["backend"],
                   feature_names=payload["feature_names"])
        if obj.backend == "lgbm":
            obj.model = lgb.Booster(model_file=path + ".lgb")
        else:
            import joblib
            obj.model = joblib.load(path + ".joblib")
        return obj


VERDICT_BANDS = [
    (0.00, 0.15, "no_evidence", "No manipulation evidence found"),
    (0.15, 0.35, "weak_indicators", "Weak indicators"),
    (0.35, 0.65, "inconclusive", "Inconclusive - insufficient evidence"),
    (0.65, 0.85, "likely_manipulated", "Likely manipulated"),
    (0.85, 1.01, "strong_evidence", "Strong evidence of manipulation"),
]


def verdict_band(p: float) -> dict:
    for lo, hi, key, label in VERDICT_BANDS:
        if lo <= p < hi:
            return {"key": key, "label": label}
    return {"key": "strong_evidence", "label": "Strong evidence of manipulation"}


@dataclass
class AbstentionGate:
    """Refuses to answer when the image's forensic traces are unlikely
    to have survived, or when branches disagree strongly, or when the
    fused probability sits in a low-margin band. Calibrated abstention
    is USP 2 in the blueprint -- treat it as a first-class output, not
    an afterthought."""
    min_jpeg_quality: float = 45.0
    min_dimension: int = 256
    disagreement_threshold: float = 0.45
    low_margin_band: tuple[float, float] = (0.35, 0.65)

    def should_abstain(self, fused_p: float, branch_scores: list[float],
                        jpeg_quality: float | None, width: int, height: int) -> tuple[bool, str]:
        if jpeg_quality is not None and jpeg_quality < self.min_jpeg_quality:
            return True, ("Image JPEG quality is too low for forensic traces "
                           "to reliably survive re-compression.")
        if min(width, height) < self.min_dimension:
            return True, "Image resolution is too low for reliable analysis."
        if len(branch_scores) >= 2:
            disagreement = float(np.std(branch_scores))
            if disagreement > self.disagreement_threshold:
                return True, "Forensic branches produced strongly conflicting evidence."
        lo, hi = self.low_margin_band
        if lo <= fused_p <= hi:
            return True, "Fused evidence falls in the low-margin, inconclusive range."
        return False, ""
