// Mirrors service/schemas.py on the backend (forgelens FastAPI service).

export type EvidenceWeight = "neutral" | "low" | "medium" | "high";

export interface EvidenceItem {
  branch: string;
  finding: string;
  weight: EvidenceWeight;
  detail?: Record<string, unknown>;
}

export interface RegionItem {
  bbox: [number, number, number, number];
  area_pct: number;
  score: number;
}

export interface QualityInfo {
  jpeg_qf_est: number | null;
  resolution: [number, number];
  has_exif: boolean;
}

export type VerdictKey =
  | "no_evidence"
  | "weak_indicators"
  | "inconclusive"
  | "likely_manipulated"
  | "strong_evidence"
  | "abstained";

export interface AnalyzeResult {
  verdict: VerdictKey;
  verdict_label: string;
  probability: number;
  abstained: boolean;
  abstain_reason: string | null;
  regions: RegionItem[];
  evidence: EvidenceItem[];
  quality: QualityInfo;
  limitations: string;
  sha256: string;
  timing_ms: Record<string, number>;
  analyzed_at: string;
  narrative: string;
}
