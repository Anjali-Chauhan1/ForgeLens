import type { AnalyzeResult } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_FORGELENS_API ?? "http://localhost:8000";

export class ApiUnavailableError extends Error {}

export async function analyzeImage(file: File): Promise<AnalyzeResult> {
  const form = new FormData();
  form.append("image", file);

  let res: Response;
  try {
    res = await fetch(`${API_BASE}/v1/analyze`, {
      method: "POST",
      body: form,
      signal: AbortSignal.timeout(20_000),
    });
  } catch {
    throw new ApiUnavailableError("Could not reach the ForgeLens API.");
  }

  if (!res.ok) {
    throw new Error(`Analysis failed (${res.status})`);
  }

  const data = await res.json();
  if ("job_id" in data) {
    return pollJob(data.job_id);
  }
  return data as AnalyzeResult;
}

async function pollJob(jobId: string): Promise<AnalyzeResult> {
  for (let i = 0; i < 30; i++) {
    await new Promise((r) => setTimeout(r, 2000));
    const res = await fetch(`${API_BASE}/v1/analyze/${jobId}`);
    const data = await res.json();
    if (data.status === "done") return data.result as AnalyzeResult;
    if (data.status === "failed") throw new Error(data.error ?? "Analysis failed");
  }
  throw new Error("Analysis timed out");
}
