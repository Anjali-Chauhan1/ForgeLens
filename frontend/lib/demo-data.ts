import type { AnalyzeResult, EvidenceItem, VerdictKey } from "./types";

// Deterministic, file-derived "analysis" used only when the live API
// (NEXT_PUBLIC_FORGELENS_API) isn't reachable, so the demo section stays
// interactive without a running backend. Always surfaced to the user as
// simulated — see the `DEMO MODE` banner in demo-section.tsx — never
// presented as a real forensic result.

async function hashFile(file: File): Promise<number> {
  const buf = await file.arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256", buf);
  const bytes = new Uint8Array(digest);
  let seed = 0;
  for (let i = 0; i < 8; i++) seed = (seed << 4) | (bytes[i] & 0xf);
  return Math.abs(seed);
}

function mulberry32(seed: number) {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const VERDICT_BANDS: { max: number; key: VerdictKey; label: string }[] = [
  { max: 0.15, key: "no_evidence", label: "No manipulation evidence found" },
  { max: 0.35, key: "weak_indicators", label: "Weak indicators" },
  { max: 0.65, key: "inconclusive", label: "Inconclusive — insufficient evidence" },
  { max: 0.85, key: "likely_manipulated", label: "Likely manipulated" },
  { max: 1.01, key: "strong_evidence", label: "Strong evidence of manipulation" },
];

export async function simulateAnalysis(file: File): Promise<AnalyzeResult> {
  const seed = await hashFile(file);
  const rand = mulberry32(seed);

  const probability = Math.round(rand() * 1000) / 1000;
  const band = VERDICT_BANDS.find((b) => probability < b.max)!;
  const abstained = probability > 0.35 && probability < 0.65 && rand() < 0.4;

  const candidateEvidence: EvidenceItem[] = [
    {
      branch: "metadata",
      finding: rand() > 0.5
        ? "No EXIF metadata present. Common after social-media re-upload; not itself proof of manipulation."
        : "Camera EXIF present with no editing-software tag — weak indicator of an unedited original.",
      weight: "low",
    },
    {
      branch: "localization",
      finding: probability > 0.5
        ? `Contiguous region covering ${(8 + rand() * 14).toFixed(1)}% of the image flagged by the manipulation-localization model.`
        : "No contiguous tampered region detected.",
      weight: probability > 0.75 ? "high" : probability > 0.5 ? "medium" : "neutral",
    },
    (() => {
      const copyMoveDetected = rand() < 0.15;
      return {
        branch: "copy_move",
        finding: copyMoveDetected
          ? "2 duplicated region(s) found with a geometrically consistent transform."
          : "No duplicated regions detected.",
        weight: copyMoveDetected ? "high" : "neutral",
      } as const;
    })(),
    (() => {
      const syntheticDetected = rand() < 0.1;
      return {
        branch: "synthetic",
        finding: syntheticDetected
          ? "Global image statistics are consistent with AI-generated content."
          : "No evidence of synthetic generation.",
        weight: syntheticDetected ? "high" : "neutral",
      } as const;
    })(),
  ];

  const regions =
    probability > 0.5
      ? [
          {
            bbox: [
              Math.round(40 + rand() * 120),
              Math.round(40 + rand() * 120),
              Math.round(80 + rand() * 100),
              Math.round(80 + rand() * 100),
            ] as [number, number, number, number],
            area_pct: Math.round((8 + rand() * 14) * 10) / 10,
            score: Math.round((0.5 + rand() * 0.45) * 100) / 100,
          },
        ]
      : [];

  return {
    verdict: abstained ? "abstained" : band.key,
    verdict_label: abstained ? "Inconclusive — insufficient evidence" : band.label,
    probability,
    abstained,
    abstain_reason: abstained
      ? "Fused evidence falls in the low-margin, inconclusive range."
      : null,
    regions,
    evidence: candidateEvidence,
    quality: {
      jpeg_qf_est: Math.round(55 + rand() * 35),
      resolution: [1200 + Math.round(rand() * 800), 900 + Math.round(rand() * 600)],
      has_exif: rand() > 0.5,
    },
    limitations:
      "This analysis cannot detect manipulations that preserve compression and noise statistics, cannot verify caption/date/location claims, and its confidence degrades on heavily compressed or resized images. A positive verdict indicates evidence consistent with manipulation, not certainty.",
    sha256: Array.from({ length: 64 }, () => Math.floor(rand() * 16).toString(16)).join(""),
    timing_ms: { hash_ms: 2.1, metadata_ms: 8.4, classical_ms: 61.2, deep_branches_ms: 340.5 },
    analyzed_at: new Date().toISOString(),
    narrative: abstained
      ? "This image could not be conclusively analyzed. No verdict is given; treat the image as unverified rather than as cleared."
      : `The combined evidence yields a manipulation probability of ${probability.toFixed(2)} (${band.label}).`,
  };
}
