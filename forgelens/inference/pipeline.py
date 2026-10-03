"""The full cascaded inference pipeline: hash -> metadata -> classical
forensics -> deep branches -> fusion -> calibration -> abstention ->
evidence report. Mirrors the nine-step pipeline diagram in the
blueprint (section 04) and the /v1/analyze result schema (section 15).

Cascading for cost: cheap stages (hash, metadata, quick noise-block
check) always run; the deep branches run unconditionally in this
reference implementation for correctness, but are structured so a
production deployment can short-circuit once metadata/retrieval already
produces a confident verdict (see `run_deep_branches` flag).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import math
import time

import numpy as np
import cv2
import torch
from PIL import Image

from forgelens.utils.hashing import sha256_file, perceptual_hashes
from forgelens.forensics.metadata import analyze_metadata
from forgelens.forensics.ela import compute_ela, ela_summary
from forgelens.forensics.noise import noise_residual, block_noise_inconsistency
from forgelens.forensics.copymove_sift import detect_copy_move
from forgelens.fusion.fuse import FusionModel, features_to_vector, verdict_band, AbstentionGate
from forgelens.data.datasets import pad_to_canonical
from forgelens.provenance.c2pa import verify_manifest, to_evidence_finding, STATUS_TO_FEATURE
from forgelens.provenance.retrieval import PerceptualHashIndex, diff_against_match


@dataclass
class Evidence:
    branch: str
    finding: str
    weight: str  # neutral | low | medium | high
    detail: dict = field(default_factory=dict)


@dataclass
class AnalysisReport:
    sha256: str
    verdict: str
    verdict_label: str
    probability: float
    abstained: bool
    abstain_reason: str
    regions: list[dict]
    mask: np.ndarray | None
    evidence: list[Evidence]
    quality: dict
    limitations: str
    timing_ms: dict
    raw_features: dict


LIMITATIONS_TEXT = (
    "This analysis cannot detect manipulations that preserve compression "
    "and noise statistics (e.g. a forgery performed and re-saved entirely "
    "within the original camera's noise/compression envelope), cannot "
    "verify whether an image's caption, date, or location claim is "
    "accurate, and its confidence degrades on images that have been "
    "heavily compressed or resized by prior sharing. A positive verdict "
    "indicates evidence consistent with manipulation, not certainty."
)


class ForgeLensPipeline:
    def __init__(self, localization_model=None, synthetic_model=None,
                 fusion_model: FusionModel | None = None,
                 retrieval_index: PerceptualHashIndex | None = None,
                 device: str | None = None, run_deep_branches: bool = True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.localization_model = localization_model
        self.synthetic_model = synthetic_model
        self.fusion_model = fusion_model or FusionModel()
        # Branch D's retrieval half. None by default -- an empty/no-op
        # index would silently never match anything, so the pipeline
        # explicitly skips the retrieval step (and its evidence card)
        # rather than pretending to have searched an empty corpus.
        self.retrieval_index = retrieval_index
        self.run_deep_branches = run_deep_branches
        self.abstention_gate = AbstentionGate()

    # -- individual branch runners -------------------------------------

    def _run_localization(self, img_rgb: np.ndarray) -> dict:
        if self.localization_model is None:
            return {"available": False}
        h, w = img_rgb.shape[:2]
        # Pad to the smallest multiple of the encoder's total stride (32
        # for the 4-stage ConvNeXt backbone) that fits the image, NOT a
        # fixed large canvas. A fixed canonical (e.g. 512) pads a small
        # test image into mostly black border -- a massive train/inference
        # distribution mismatch versus the near-full-frame crops the model
        # was actually trained on (verified: this was silently causing
        # every prediction to default to "authentic" on sub-512 images).
        # The backbone is fully convolutional, so it accepts any
        # stride-aligned size; only training batching needed a fixed crop.
        stride = 32
        canonical = int(np.ceil(max(h, w, 1) / stride) * stride)
        canonical = max(canonical, stride)
        padded, (dy, dx) = pad_to_canonical(img_rgb, canonical)
        tensor = torch.from_numpy(padded).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)
        with torch.no_grad():
            out = self.localization_model(tensor)
            prob = torch.sigmoid(out["region_logit"])[0, 0].cpu().numpy()
        # Crop back to the original (unpadded) region.
        crop = prob[dy:dy + min(h, canonical), dx:dx + min(w, canonical)]
        if crop.shape != (h, w):
            crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

        # Connected-component stats for the region feature set.
        binary = (crop > 0.5).astype(np.uint8)
        n_components, labels = cv2.connectedComponents(binary)
        area_ratio = float(binary.mean())

        # Boundary sharpness: gradient magnitude concentration along the
        # predicted mask edge -- a crude proxy consistent with the
        # edge-supervision idea in Branch A's design.
        gy, gx = np.gradient(crop.astype(np.float32))
        boundary_sharpness = float(np.sqrt(gx ** 2 + gy ** 2).mean())

        return {
            "available": True,
            "mask": crop,
            "max": float(crop.max()),
            "mean": float(crop.mean()),
            "area_ratio": area_ratio,
            "connected_components": max(0, n_components - 1),
            "boundary_sharpness": boundary_sharpness,
        }

    def _run_synthetic(self, img_rgb: np.ndarray) -> dict:
        if self.synthetic_model is None:
            return {"available": False}
        img224 = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_CUBIC)
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        norm = (img224.astype(np.float32) / 255.0 - mean) / std
        t224 = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).to(self.device)

        full = img_rgb.astype(np.float32) / 255.0
        tfull = torch.from_numpy(full).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.synthetic_model(t224, tfull)
            global_score = float(torch.sigmoid(out["global_logit"])[0])
            patch_max = float(torch.sigmoid(out["patch_max"])[0]) if "patch_max" in out else 0.0
            patch_area = float(out.get("patch_area_ratio", torch.zeros(1))[0])

        return {"available": True, "global_score": global_score,
                "patch_max": patch_max, "patch_area_ratio": patch_area}

    def _run_copy_move(self, img_bgr: np.ndarray) -> dict:
        report = detect_copy_move(img_bgr)
        return {
            "available": True,
            "detected": report.detected,
            "confidence": report.confidence,
            "num_pairs": len(report.matches),
            "inliers": sum(m.inlier_count for m in report.matches),
            "matches": report.matches,
        }

    # -- orchestration ----------------------------------------------------

    def analyze(self, image_path: str) -> AnalysisReport:
        t = {}
        t0 = time.time()

        digest = sha256_file(image_path)
        t["hash_ms"] = (time.time() - t0) * 1000

        t1 = time.time()
        meta = analyze_metadata(image_path)
        t["metadata_ms"] = (time.time() - t1) * 1000

        pil_img = Image.open(image_path).convert("RGB")
        img_rgb = np.asarray(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h, w = img_rgb.shape[:2]

        t2 = time.time()
        ela_map = compute_ela(pil_img)
        ela_stats = ela_summary(ela_map)
        residual = noise_residual(pil_img)
        noise_stats = block_noise_inconsistency(residual)
        t["classical_ms"] = (time.time() - t2) * 1000

        t2b = time.time()
        c2pa_report = verify_manifest(image_path)
        t["c2pa_ms"] = (time.time() - t2b) * 1000

        retrieval_diff = None
        if self.retrieval_index is not None:
            t2c = time.time()
            matches = self.retrieval_index.query(image_path, max_hamming=10, top_k=1)
            if matches:
                retrieval_diff = diff_against_match(image_path, matches[0].entry.path)
                retrieval_diff["matched_label"] = matches[0].entry.label
                retrieval_diff["hamming_distance"] = matches[0].hamming_distance
            t["retrieval_ms"] = (time.time() - t2c) * 1000

        localization = {"available": False}
        synthetic = {"available": False}
        copy_move = {"available": False}

        if self.run_deep_branches:
            t3 = time.time()
            localization = self._run_localization(img_rgb)
            synthetic = self._run_synthetic(img_rgb)
            copy_move = self._run_copy_move(img_bgr)
            t["deep_branches_ms"] = (time.time() - t3) * 1000

        evidence: list[Evidence] = []
        for f in meta.findings:
            evidence.append(Evidence(branch="metadata", finding=f.finding,
                                      weight=f.weight, detail=f.detail))

        c2pa_finding = to_evidence_finding(c2pa_report)
        evidence.append(Evidence(**c2pa_finding))

        if retrieval_diff is not None:
            if retrieval_diff.get("aligned"):
                area = retrieval_diff["diff_area_ratio"]
                if area > 0.02:
                    evidence.append(Evidence(
                        branch="retrieval",
                        finding=f"A near-duplicate of this image was found in the "
                                f"reference index ({retrieval_diff['matched_label']}). "
                                f"Pixel-level differencing after alignment shows "
                                f"{area*100:.1f}% of the image differs from the "
                                f"matched reference -- this is a ground-truth diff, "
                                f"not an inferred estimate.",
                        weight="high",
                        detail={"diff_area_ratio": area,
                                "matched_label": retrieval_diff["matched_label"]},
                    ))
                else:
                    evidence.append(Evidence(
                        branch="retrieval",
                        finding=f"A near-duplicate of this image was found in the "
                                f"reference index ({retrieval_diff['matched_label']}) "
                                f"with negligible pixel differences after alignment.",
                        weight="neutral",
                    ))
            else:
                evidence.append(Evidence(
                    branch="retrieval",
                    finding=f"No reliably alignable near-duplicate found in the "
                            f"reference index ({retrieval_diff.get('reason', '')}).",
                    weight="neutral",
                ))

        if noise_stats["anomalous_area_ratio"] > 0.05:
            evidence.append(Evidence(
                branch="noise_block",
                finding=f"{noise_stats['anomalous_area_ratio']*100:.1f}% of "
                        f"32px blocks show statistically anomalous noise "
                        f"variance relative to the image median.",
                weight="medium",
                detail={"anomalous_blocks": len(noise_stats["anomalous_blocks"])},
            ))

        if localization.get("available") and localization["area_ratio"] > 0.005:
            evidence.append(Evidence(
                branch="localization",
                finding=f"Contiguous region covering "
                        f"{localization['area_ratio']*100:.1f}% of the image "
                        f"flagged by the manipulation-localization model "
                        f"(max confidence {localization['max']:.2f}).",
                weight="high" if localization["max"] > 0.8 else "medium",
                detail={"connected_components": localization["connected_components"]},
            ))
        elif localization.get("available"):
            evidence.append(Evidence(branch="localization",
                                      finding="No contiguous tampered region detected.",
                                      weight="neutral"))

        if copy_move.get("detected"):
            evidence.append(Evidence(
                branch="copy_move",
                finding=f"{copy_move['num_pairs']} duplicated region(s) found "
                        f"with a geometrically consistent transform "
                        f"({copy_move['inliers']} inlier keypoint matches).",
                weight="high",
                detail={"num_pairs": copy_move["num_pairs"]},
            ))
        elif copy_move.get("available"):
            evidence.append(Evidence(branch="copy_move",
                                      finding="No duplicated regions detected.",
                                      weight="neutral"))

        if synthetic.get("available"):
            gscore = synthetic["global_score"]
            if gscore > 0.7:
                evidence.append(Evidence(
                    branch="synthetic",
                    finding=f"Global image statistics are consistent with "
                            f"AI-generated content (score {gscore:.2f}).",
                    weight="high",
                ))
            elif synthetic["patch_max"] > 0.7:
                evidence.append(Evidence(
                    branch="synthetic",
                    finding=f"A localized region shows statistics consistent "
                            f"with AI-generated/generative-fill content "
                            f"(patch score {synthetic['patch_max']:.2f}), while "
                            f"the rest of the image appears camera-captured.",
                    weight="medium",
                ))
            else:
                evidence.append(Evidence(branch="synthetic",
                                          finding="No evidence of synthetic generation.",
                                          weight="neutral"))

        features = {
            "seg_max": localization.get("max", 0.0),
            "seg_mean": localization.get("mean", 0.0),
            "seg_area_ratio": localization.get("area_ratio", 0.0),
            "seg_connected_components": localization.get("connected_components", 0),
            "seg_boundary_sharpness": localization.get("boundary_sharpness", 0.0),
            "copymove_confidence": copy_move.get("confidence", 0.0),
            "copymove_num_pairs": copy_move.get("num_pairs", 0),
            "copymove_transform_inliers": copy_move.get("inliers", 0),
            "synthetic_global_score": synthetic.get("global_score", 0.0),
            "synthetic_patch_max": synthetic.get("patch_max", 0.0),
            "synthetic_patch_area": synthetic.get("patch_area_ratio", 0.0),
            "meta_qtable_mismatch": 1.0 if meta.quant_table_match else 0.0,
            "meta_exif_missing": 0.0 if meta.has_exif else 1.0,
            "meta_software_tag_present": 1.0 if meta.software_tag else 0.0,
            "c2pa_status": STATUS_TO_FEATURE[c2pa_report.status],
            "jpeg_quality_estimate": meta.jpeg_quality_estimate or 70.0,
            "image_resolution_log": float(math.log(max(h * w, 1))),
            "blur_estimate": float(ela_stats["mean"]),
            "classical_noise_anomaly": float(noise_stats["anomalous_area_ratio"]),
        }
        retrieval_diff_area = (
            retrieval_diff["diff_area_ratio"]
            if retrieval_diff is not None and retrieval_diff.get("aligned") else 0.0
        )

        if self.fusion_model.model is not None:
            X = features_to_vector(features).reshape(1, -1)
            fused_p = float(self.fusion_model.calibrated_probability(X)[0])
        else:
            # No trained fusion model available yet: a documented,
            # transparent fallback (weighted average) so the pipeline
            # still runs end-to-end during development. Includes the
            # classical noise-block signal so the fallback is not blind
            # whenever the deep branches (localization_model /
            # synthetic_model) haven't been trained yet -- without this
            # the classical-only pipeline always scored 0.0 regardless
            # of what the noise-anomaly evidence card actually found.
            #
            # A ground-truth retrieval diff is the highest-precision
            # signal available (per the blueprint) and dominates the
            # fallback when present. A valid C2PA credential pulls the
            # score down; an invalid one is treated as strong positive
            # evidence, since a forged/broken signature is unusual and
            # suspicious on a file that claims to carry one.
            base_score = float(np.clip(
                0.30 * localization.get("max", 0.0)
                + 0.20 * copy_move.get("confidence", 0.0)
                + 0.20 * synthetic.get("global_score", 0.0)
                + 0.15 * (1.0 if meta.quant_table_match else 0.0)
                + 0.15 * min(1.0, features["classical_noise_anomaly"] * 4.0),
                0.0, 1.0,
            ))
            if retrieval_diff_area > 0.02:
                base_score = max(base_score, min(1.0, 0.5 + retrieval_diff_area * 2.0))
            if c2pa_report.status.value == "valid":
                base_score *= 0.5
            elif c2pa_report.status.value == "invalid":
                base_score = max(base_score, 0.75)
            fused_p = float(np.clip(base_score, 0.0, 1.0))

        branch_scores = [
            v for v in [
                localization.get("max"), copy_move.get("confidence"),
                synthetic.get("global_score"), features["classical_noise_anomaly"],
            ] if v is not None
        ]
        abstain, reason = self.abstention_gate.should_abstain(
            fused_p, branch_scores, meta.jpeg_quality_estimate, w, h
        )

        band = verdict_band(fused_p)

        regions = []
        if localization.get("available") and localization["area_ratio"] > 0.005:
            mask_bin = (localization["mask"] > 0.5).astype(np.uint8)
            n, labels_im, stats, _ = cv2.connectedComponentsWithStats(mask_bin)
            for i in range(1, n):
                x, y, bw, bh, area = stats[i]
                if area < 50:
                    continue
                region_conf = float(localization["mask"][y:y + bh, x:x + bw].mean())
                regions.append({
                    "bbox": [int(x), int(y), int(bw), int(bh)],
                    "area_pct": round(100.0 * area / (w * h), 2),
                    "score": round(region_conf, 3),
                })

        t["total_ms"] = sum(t.values())

        return AnalysisReport(
            sha256=digest,
            verdict=band["key"] if not abstain else "abstained",
            verdict_label=band["label"] if not abstain else "Inconclusive - insufficient evidence",
            probability=round(fused_p, 4),
            abstained=abstain,
            abstain_reason=reason,
            regions=regions,
            mask=localization.get("mask"),
            evidence=evidence,
            quality={
                "jpeg_qf_est": meta.jpeg_quality_estimate,
                "resolution": [w, h],
                "has_exif": meta.has_exif,
            },
            limitations=LIMITATIONS_TEXT,
            timing_ms={k: round(v, 1) for k, v in t.items()},
            raw_features=features,
        )
