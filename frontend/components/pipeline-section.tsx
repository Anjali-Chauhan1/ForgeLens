"use client";

import { useRef } from "react";
import { motion, useScroll, useSpring, useTransform } from "motion/react";
import {
  Hash,
  FileSearch,
  Microscope,
  BrainCircuit,
  GitMerge,
  Gauge,
  OctagonPause,
  FileCheck2,
} from "lucide-react";
import { SectionHeading } from "./branches-section";
import { EASE_OUT } from "@/lib/motion";

const steps = [
  { icon: Hash, title: "Hash & fingerprint", desc: "SHA-256 + perceptual hashes for chain-of-custody and retrieval." },
  { icon: FileSearch, title: "Metadata", desc: "EXIF, JPEG quant-table fingerprints, C2PA credential check." },
  { icon: Microscope, title: "Classical forensics", desc: "ELA, noise-block variance, SIFT copy-move — cheap, explainable." },
  { icon: BrainCircuit, title: "Deep branches", desc: "Localization U-Net and frozen-CLIP synthetic-content detector." },
  { icon: GitMerge, title: "Fusion", desc: "A trained meta-model — not an average — learns how to weight each branch." },
  { icon: Gauge, title: "Calibration", desc: "Temperature scaling so the reported probability is a real probability." },
  { icon: OctagonPause, title: "Abstention gate", desc: "Refuses a verdict on low-quality or disagreeing evidence, on purpose." },
  { icon: FileCheck2, title: "Evidence report", desc: "Verdict, region map, and a plain-language list of what was found." },
];

function StepCard({ s, i }: { s: (typeof steps)[number]; i: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start 90%", "start 55%"] });
  const scale = useTransform(scrollYProgress, [0, 1], [0.85, 1]);
  const iconColor = useTransform(scrollYProgress, [0, 1], ["var(--foreground-muted)", "var(--primary)"]);

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: 0.5, delay: (i % 4) * 0.1, ease: EASE_OUT }}
      className="relative"
    >
      <motion.div
        style={{ scale }}
        className="relative z-10 flex h-12 w-12 items-center justify-center rounded-2xl border border-border-strong bg-surface shadow-sm"
      >
        <motion.div style={{ color: iconColor }}>
          <s.icon className="h-5 w-5" strokeWidth={2} />
        </motion.div>
        <span className="absolute -right-1.5 -top-1.5 flex h-5 w-5 items-center justify-center rounded-full bg-foreground text-[10px] font-semibold text-background">
          {i + 1}
        </span>
      </motion.div>
      <h3 className="mt-4 text-[15px] font-semibold tracking-tight">{s.title}</h3>
      <p className="mt-1.5 text-sm leading-relaxed text-foreground-muted">{s.desc}</p>
    </motion.div>
  );
}

export function PipelineSection() {
  const containerRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start 75%", "end 40%"],
  });
  const lineScale = useSpring(scrollYProgress, { stiffness: 200, damping: 34 });

  return (
    <section id="pipeline" className="relative py-24 sm:py-32">
      <div className="mx-auto max-w-6xl px-6">
        <SectionHeading
          eyebrow="Pipeline"
          title="Cheap checks first, deep models when it matters."
          desc="Every stage produces evidence, not just a score — and the whole chain can end in a deliberate refusal to answer."
        />

        <div ref={containerRef} className="relative mt-16">
          {/* connecting line — draws in as you scroll through the grid,
              not a one-shot reveal, so it reads as "tracking progress"
              through the pipeline rather than just a decorative line */}
          <div className="absolute left-0 right-0 top-6 hidden h-px bg-border lg:block" />
          <motion.div
            style={{ scaleX: lineScale, transformOrigin: "left" }}
            className="absolute left-0 right-0 top-6 hidden h-px bg-primary lg:block"
          />

          <div className="grid grid-cols-1 gap-x-6 gap-y-10 sm:grid-cols-2 lg:grid-cols-4">
            {steps.map((s, i) => (
              <StepCard key={s.title} s={s} i={i} />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
