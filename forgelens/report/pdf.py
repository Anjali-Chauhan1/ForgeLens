"""Signed PDF export for chain-of-custody use. Uses reportlab (pure
Python, no system dependency) so the service stays easy to deploy.
Signing here means embedding the SHA-256 of the source image and a
SHA-256 of the report body itself, which a verifier can recompute --
full cryptographic signing (e.g. with a service keypair) is a drop-in
extension via `sign_bytes`.
"""
from __future__ import annotations
import hashlib
from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from forgelens.inference.pipeline import AnalysisReport
from forgelens.report.evidence import narrative_summary


def _wrap(text: str, width: int = 95) -> list[str]:
    import textwrap
    return textwrap.wrap(text, width=width)


def render_report_pdf(report: AnalysisReport, image_path: str,
                       sign_bytes=None) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter
    margin = 0.75 * inch
    y = h - margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "ForgeLens -- Forensic Analysis Report")
    y -= 0.35 * inch

    c.setFont("Helvetica", 9)
    c.setFillGray(0.4)
    c.drawString(margin, y, f"SHA-256: {report.sha256}")
    y -= 0.2 * inch
    c.drawString(margin, y, f"Source: {image_path}")
    c.setFillGray(0.0)
    y -= 0.4 * inch

    c.setFont("Helvetica-Bold", 13)
    verdict_text = f"Verdict: {report.verdict_label}"
    if not report.abstained:
        verdict_text += f"   (p = {report.probability:.2f})"
    c.drawString(margin, y, verdict_text)
    y -= 0.35 * inch

    c.setFont("Helvetica", 10)
    for line in _wrap(narrative_summary(report)):
        c.drawString(margin, y, line)
        y -= 0.18 * inch
    y -= 0.2 * inch

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Evidence")
    y -= 0.25 * inch
    c.setFont("Helvetica", 9)
    for e in report.evidence:
        prefix = f"[{e.weight.upper():8s}] {e.branch}: "
        for i, line in enumerate(_wrap(prefix + e.finding, width=100)):
            c.drawString(margin, y, line)
            y -= 0.16 * inch
        if y < margin + 1.5 * inch:
            c.showPage()
            y = h - margin
            c.setFont("Helvetica", 9)

    y -= 0.2 * inch
    if report.regions:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Flagged regions")
        y -= 0.25 * inch
        c.setFont("Helvetica", 9)
        for r in report.regions:
            c.drawString(margin, y,
                         f"bbox={r['bbox']}  area={r['area_pct']}%  score={r['score']}")
            y -= 0.16 * inch

    y -= 0.25 * inch
    c.setFont("Helvetica-Oblique", 8)
    for line in _wrap(f"Limitations: {report.limitations}", width=110):
        c.drawString(margin, y, line)
        y -= 0.14 * inch

    # Integrity/signature page. The body hash is computed over every
    # field that determines the report's content (not the final PDF
    # bytes, which don't exist yet) so it can be recomputed by a
    # verifier from the same AnalysisReport data.
    body_payload = (
        f"{report.sha256}|{report.verdict}|{report.probability}|"
        f"{report.abstained}|{[e.finding for e in report.evidence]}"
    ).encode()
    body_hash = hashlib.sha256(body_payload).hexdigest()
    signature = sign_bytes(body_payload) if sign_bytes is not None else None

    c.showPage()
    c.setFont("Helvetica", 9)
    c.drawString(margin, h - margin, "Report integrity")
    c.setFont("Helvetica", 8)
    c.drawString(margin, h - margin - 0.3 * inch, f"Report content SHA-256: {body_hash}")
    if signature:
        c.drawString(margin, h - margin - 0.5 * inch, f"Signature: {signature}")
    c.showPage()
    c.save()

    return buf.getvalue()
