"use client";

import { useCallback, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCcw, ShieldAlert, UploadCloud } from "lucide-react";
import { SectionHeading } from "./branches-section";
import { VerdictRing } from "./verdict-ring";
import { EvidenceList } from "./evidence-list";
import { Magnetic } from "./magnetic";
import { analyzeImage, ApiUnavailableError } from "@/lib/api";
import { simulateAnalysis } from "@/lib/demo-data";
import type { AnalyzeResult } from "@/lib/types";
import { cn } from "@/lib/utils";

type Status = "idle" | "analyzing" | "result" | "error";

const ANALYZE_STEPS = [
  "Hashing & reading metadata…",
  "Running classical forensics…",
  "Scoring localization branch…",
  "Fusing evidence…",
];

export function DemoSection() {
  const [status, setStatus] = useState<Status>("idle");
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [isDemoMode, setIsDemoMode] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stepIndex, setStepIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  const runAnalysis = useCallback(async (file: File) => {
    setPreview(URL.createObjectURL(file));
    setStatus("analyzing");
    setError(null);
    setStepIndex(0);

    const stepTimer = setInterval(() => {
      setStepIndex((i) => Math.min(i + 1, ANALYZE_STEPS.length - 1));
    }, 550);

    try {
      const data = await analyzeImage(file);
      setIsDemoMode(false);
      setResult(data);
      setStatus("result");
    } catch (err) {
      if (err instanceof ApiUnavailableError) {
        // Backend not reachable — fall back to a clearly labeled simulation
        // so the demo stays interactive.
        const data = await simulateAnalysis(file);
        await new Promise((r) => setTimeout(r, 900));
        setIsDemoMode(true);
        setResult(data);
        setStatus("result");
      } else {
        setError(err instanceof Error ? err.message : "Analysis failed.");
        setStatus("error");
      }
    } finally {
      clearInterval(stepTimer);
    }
  }, []);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0];
      if (!file || !file.type.startsWith("image/")) return;
      void runAnalysis(file);
    },
    [runAnalysis],
  );

  const reset = () => {
    setStatus("idle");
    setResult(null);
    setPreview(null);
    setError(null);
  };

  return (
    <section id="demo" className="relative py-24 sm:py-32">
      <div className="mx-auto max-w-5xl px-6">
        <SectionHeading
          eyebrow="Live demo"
          title="Drop in an image, get an evidence report."
          desc="Runs against the ForgeLens API when it's reachable, otherwise falls back to a clearly labeled client-side simulation."
        />

        <div className="mt-14 overflow-hidden rounded-4xl border border-border-strong bg-surface shadow-[0_30px_80px_-30px_rgba(20,21,26,0.15)]">
          <AnimatePresence mode="wait">
            {status === "idle" && (
              <motion.div
                key="idle"
                exit={{ opacity: 0 }}
                className={cn(
                  "flex min-h-95 flex-col items-center justify-center gap-4 p-12 text-center transition-colors",
                  dragActive && "bg-primary-soft",
                )}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragActive(true);
                }}
                onDragLeave={() => setDragActive(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragActive(false);
                  handleFiles(e.dataTransfer.files);
                }}
              >
                <motion.span
                  animate={{ y: [0, -6, 0] }}
                  transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
                  className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary-soft text-primary"
                >
                  <UploadCloud className="h-7 w-7" strokeWidth={1.75} />
                </motion.span>
                <div>
                  <p className="text-lg font-semibold">Drag an image here</p>
                  <p className="mt-1 text-sm text-foreground-muted">
                    or click to browse — JPEG or PNG, analyzed on the spot
                  </p>
                </div>
                <Magnetic strength={0.3}>
                  <button
                    onClick={() => inputRef.current?.click()}
                    className="mt-2 rounded-full bg-foreground px-5 py-2.5 text-sm font-semibold text-background transition-shadow hover:shadow-md active:scale-[0.98]"
                  >
                    Choose a file
                  </button>
                </Magnetic>
                <input
                  ref={inputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => handleFiles(e.target.files)}
                />
              </motion.div>
            )}

            {status === "analyzing" && preview && (
              <motion.div
                key="analyzing"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="grid grid-cols-1 items-center gap-8 p-8 sm:grid-cols-[1fr_1.1fr] sm:p-12"
              >
                <div className="relative overflow-hidden rounded-2xl border border-border">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={preview} alt="Uploaded" className="aspect-square w-full object-cover" />
                  <motion.div
                    className="absolute inset-x-0 h-20"
                    style={{
                      background:
                        "linear-gradient(180deg, transparent, color-mix(in srgb, var(--primary) 30%, transparent) 50%, transparent)",
                    }}
                    animate={{ top: ["-10%", "105%"] }}
                    transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
                  />
                  <div className="absolute inset-0 bg-grid opacity-20 mix-blend-multiply" />
                </div>
                <div>
                  <div className="flex items-center gap-3">
                    <span className="relative flex h-2.5 w-2.5">
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-60" />
                      <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-primary" />
                    </span>
                    <p className="text-sm font-semibold">Running the pipeline</p>
                  </div>
                  <div className="mt-5 space-y-3">
                    {ANALYZE_STEPS.map((s, i) => (
                      <div
                        key={s}
                        className={cn(
                          "flex items-center gap-2.5 text-sm transition-colors",
                          i <= stepIndex ? "text-foreground" : "text-foreground-muted/50",
                        )}
                      >
                        <span
                          className={cn(
                            "h-1.5 w-1.5 rounded-full transition-colors",
                            i <= stepIndex ? "bg-primary" : "bg-border-strong",
                          )}
                        />
                        {s}
                      </div>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}

            {status === "result" && result && preview && (
              <motion.div
                key="result"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="p-8 sm:p-10"
              >
                {isDemoMode && (
                  <div className="mb-6 flex items-center gap-2 rounded-full border border-caution/30 bg-caution-soft px-4 py-2 text-xs font-medium text-caution">
                    <ShieldAlert className="h-3.5 w-3.5" />
                    Demo mode — API not reachable, showing a simulated result
                  </div>
                )}

                <div className="grid grid-cols-1 gap-10 lg:grid-cols-[1fr_1.15fr]">
                  <div>
                    <div className="relative overflow-hidden rounded-2xl border border-border">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={preview} alt="Analyzed" className="aspect-square w-full object-cover" />
                      {result.regions.map((r, i) => {
                        // Region bboxes are in original-image pixel space
                        // (see AnalysisReport.regions in the backend
                        // pipeline), so they must be normalized against
                        // the image's actual reported resolution, not a
                        // fixed assumed size — the preview is displayed
                        // at `object-cover`, so this is only exact for
                        // square source images, but it's correct for the
                        // common case and far closer than a hardcoded
                        // constant for every other resolution.
                        const [imgW, imgH] = result.quality.resolution;
                        return (
                          <motion.div
                            key={i}
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: 0.5 + i * 0.1 }}
                            className="absolute rounded-md border-2 border-alert"
                            style={{
                              left: `${(r.bbox[0] / imgW) * 100}%`,
                              top: `${(r.bbox[1] / imgH) * 100}%`,
                              width: `${(r.bbox[2] / imgW) * 100}%`,
                              height: `${(r.bbox[3] / imgH) * 100}%`,
                            }}
                          >
                            <span className="absolute -top-6 left-0 rounded-md bg-alert px-1.5 py-0.5 text-[10px] font-semibold text-white">
                              {(r.score * 100).toFixed(0)}%
                            </span>
                          </motion.div>
                        );
                      })}
                    </div>

                    <div className="mt-5 flex items-center justify-between">
                      <VerdictRing probability={result.probability} verdict={result.verdict} size={116} />
                      <div className="max-w-45 text-right">
                        <p className="text-sm font-semibold">{result.verdict_label}</p>
                        <button
                          onClick={reset}
                          className="mt-3 inline-flex items-center gap-1.5 text-xs font-medium text-foreground-muted transition-colors hover:text-foreground"
                        >
                          <RotateCcw className="h-3.5 w-3.5" />
                          Analyze another
                        </button>
                      </div>
                    </div>
                  </div>

                  <div>
                    <p className="text-sm leading-relaxed text-foreground-muted">{result.narrative}</p>
                    <div className="mt-6">
                      <p className="mono-tag mb-3 text-xs font-medium uppercase tracking-wider text-foreground-muted">
                        Evidence
                      </p>
                      <EvidenceList items={result.evidence} />
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {status === "error" && (
              <motion.div
                key="error"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex min-h-80 flex-col items-center justify-center gap-4 p-12 text-center"
              >
                <ShieldAlert className="h-10 w-10 text-alert" />
                <p className="font-semibold">{error}</p>
                <button
                  onClick={reset}
                  className="rounded-full border border-border-strong px-5 py-2.5 text-sm font-semibold transition-colors hover:bg-surface-muted"
                >
                  Try again
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </section>
  );
}
