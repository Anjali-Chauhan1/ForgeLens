"use client";

import { motion } from "motion/react";
import { Copy, ScanSearch, Sparkles, Stamp } from "lucide-react";
import { EASE_OUT } from "@/lib/motion";
import { TiltCard } from "./tilt-card";
import { TextReveal } from "./text-reveal";

const branches = [
  {
    letter: "A",
    icon: ScanSearch,
    title: "Localization",
    tagline: "Where, not just whether",
    desc: "A Bayar/SRM forensic stem feeds a ConvNeXt U-Net that segments tampered pixels and their boundary, so a splice or removal is highlighted, not just flagged.",
    accent: "var(--primary)",
    accentSoft: "var(--primary-soft)",
  },
  {
    letter: "B",
    icon: Copy,
    title: "Copy-move",
    tagline: "Self-correlation, not anomaly",
    desc: "SIFT self-matching plus RANSAC geometric verification catches a region duplicated within the same image — invisible to noise-based detectors since it shares the source's exact sensor statistics.",
    accent: "var(--verified)",
    accentSoft: "var(--verified-soft)",
  },
  {
    letter: "C",
    icon: Sparkles,
    title: "Synthetic content",
    tagline: "Generalizes across generators",
    desc: "A frozen CLIP encoder with a light trainable head — a from-scratch CNN nails one generator and collapses on the next. Freezing is what keeps this branch honest.",
    accent: "var(--caution)",
    accentSoft: "var(--caution-soft)",
  },
  {
    letter: "D",
    icon: Stamp,
    title: "Provenance",
    tagline: "Ground truth when it hits",
    desc: "C2PA credential verification plus perceptual-hash retrieval against known originals — when a near-duplicate is found, the pixel diff is stronger evidence than any inference.",
    accent: "var(--alert)",
    accentSoft: "var(--alert-soft)",
  },
];

export function BranchesSection() {
  return (
    <section id="branches" className="relative py-24 sm:py-32">
      <div className="mx-auto max-w-6xl px-6">
        <SectionHeading
          eyebrow="Architecture"
          title="Four specialists. One verdict."
          desc="Each branch is built for an attack class the others structurally can't see. A fusion model learns how much to trust each one, per image."
        />

        <div className="mt-16 grid grid-cols-1 gap-5 sm:grid-cols-2">
          {branches.map((b, i) => (
            <motion.div
              key={b.letter}
              initial={{ opacity: 0, y: 28 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-80px" }}
              transition={{ duration: 0.6, delay: i * 0.08, ease: EASE_OUT }}
            >
              <TiltCard className="group relative overflow-hidden rounded-3xl border border-border bg-surface p-8 shadow-[0_1px_2px_rgba(20,21,26,0.04)] transition-[border-color,box-shadow] duration-300 hover:border-border-strong hover:shadow-[0_24px_48px_-24px_rgba(20,21,26,0.18)]">
                <div
                  className="pointer-events-none absolute -right-16 -top-16 h-48 w-48 rounded-full opacity-0 blur-3xl transition-opacity duration-500 group-hover:opacity-100"
                  style={{ background: b.accent }}
                />

                <div className="flex items-start justify-between" style={{ transform: "translateZ(40px)" }}>
                  <span
                    className="flex h-11 w-11 items-center justify-center rounded-2xl"
                    style={{ background: b.accentSoft, color: b.accent }}
                  >
                    <b.icon className="h-5 w-5" strokeWidth={2.2} />
                  </span>
                  <span className="mono-tag text-3xl font-semibold text-border-strong">
                    {b.letter}
                  </span>
                </div>

                <div style={{ transform: "translateZ(30px)" }}>
                  <h3 className="mt-6 text-xl font-semibold tracking-tight">{b.title}</h3>
                  <p className="mt-1 text-sm font-medium" style={{ color: b.accent }}>
                    {b.tagline}
                  </p>
                  <p className="mt-3 text-[15px] leading-relaxed text-foreground-muted">
                    {b.desc}
                  </p>
                </div>
              </TiltCard>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

export function SectionHeading({
  eyebrow,
  title,
  desc,
}: {
  eyebrow: string;
  title: string;
  desc: string;
}) {
  return (
    <div className="max-w-2xl">
      <motion.span
        initial={{ opacity: 0, y: 10 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-100px" }}
        transition={{ duration: 0.5, ease: EASE_OUT }}
        className="mono-tag inline-block text-xs font-medium uppercase tracking-widest text-primary"
      >
        {eyebrow}
      </motion.span>
      <h2 className="mt-3 text-balance text-3xl font-semibold tracking-tight sm:text-4xl">
        <TextReveal text={title} stagger={0.03} trigger="scroll" />
      </h2>
      <motion.p
        initial={{ opacity: 0, y: 12 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-100px" }}
        transition={{ duration: 0.6, delay: 0.15, ease: EASE_OUT }}
        className="mt-4 text-lg leading-relaxed text-foreground-muted"
      >
        {desc}
      </motion.p>
    </div>
  );
}
