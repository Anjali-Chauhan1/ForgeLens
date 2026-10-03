from __future__ import annotations
from pydantic import BaseModel


class EvidenceItem(BaseModel):
    branch: str
    finding: str
    weight: str
    detail: dict = {}


class RegionItem(BaseModel):
    bbox: list[int]
    area_pct: float
    score: float


class QualityInfo(BaseModel):
    jpeg_qf_est: float | None
    resolution: list[int]
    has_exif: bool


class AnalyzeResult(BaseModel):
    verdict: str
    verdict_label: str
    probability: float
    abstained: bool
    abstain_reason: str | None
    regions: list[RegionItem]
    evidence: list[EvidenceItem]
    quality: QualityInfo
    limitations: str
    sha256: str
    timing_ms: dict
    analyzed_at: str
    narrative: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | running | done | failed
    result: AnalyzeResult | None = None
    error: str | None = None
