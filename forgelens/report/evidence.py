"""Serializes an AnalysisReport into the JSON schema described in the
blueprint (section 15) and into a plain-language narrative. The
narrative function is explicitly restricted to DESCRIBING the pipeline's
already-computed findings -- it never makes or changes the determination,
per the correctness point flagged in the blueprint's explainability
section (a VLM or template narrates; it does not decide).
"""
from __future__ import annotations
from datetime import datetime, timezone

from forgelens.inference.pipeline import AnalysisReport


def to_result_dict(report: AnalysisReport) -> dict:
    return {
        "verdict": report.verdict,
        "verdict_label": report.verdict_label,
        "probability": report.probability,
        "abstained": report.abstained,
        "abstain_reason": report.abstain_reason or None,
        "regions": report.regions,
        "evidence": [
            {"branch": e.branch, "finding": e.finding, "weight": e.weight,
             "detail": e.detail}
            for e in report.evidence
        ],
        "quality": report.quality,
        "limitations": report.limitations,
        "sha256": report.sha256,
        "timing_ms": report.timing_ms,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }


def narrative_summary(report: AnalysisReport) -> str:
    """Template-based plain-language summary (swap for a VLM call that
    is fed this same evidence list, per the blueprint's XAI section --
    the VLM's role is strictly to phrase these findings in prose, never
    to introduce new findings or override the verdict).
    """
    if report.abstained:
        return (f"This image could not be conclusively analyzed: "
                f"{report.abstain_reason} No verdict is given; treat the "
                f"image as unverified rather than as cleared.")

    high = [e for e in report.evidence if e.weight == "high"]
    medium = [e for e in report.evidence if e.weight == "medium"]

    if not high and not medium:
        return (f"No significant evidence of manipulation was found "
                f"(probability {report.probability:.2f}). This does not "
                f"guarantee authenticity -- see the limitations statement.")

    parts = [f"The combined evidence yields a manipulation probability of "
             f"{report.probability:.2f} ({report.verdict_label})."]
    for e in high:
        parts.append(f"Strong evidence: {e.finding}")
    for e in medium:
        parts.append(f"Supporting evidence: {e.finding}")
    if report.regions:
        parts.append(f"{len(report.regions)} region(s) were flagged, "
                      f"covering up to {max(r['area_pct'] for r in report.regions):.1f}% "
                      f"of the image.")
    return " ".join(parts)
