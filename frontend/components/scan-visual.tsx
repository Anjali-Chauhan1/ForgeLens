"use client";

import { motion } from "motion/react";
import { Waves, Layers, ScanLine, ShieldCheck } from "lucide-react";
import { EASE_OUT } from "@/lib/motion";

const chips = [
  { icon: Waves, label: "Noise residual", top: "8%", left: "-8%", delay: 0.6 },
  { icon: Layers, label: "DCT double-compress", top: "62%", left: "-14%", delay: 0.9 },
  { icon: ScanLine, label: "Copy-move match", top: "18%", left: "88%", delay: 1.2 },
  { icon: ShieldCheck, label: "C2PA: unsigned", top: "70%", left: "84%", delay: 1.5 },
];

export function ScanVisual() {
  return (
    <div className="relative mx-auto aspect-[4/3] w-full max-w-lg select-none">
      {/* ambient glow */}
      <div className="absolute inset-0 -z-10 scale-125 rounded-[3rem] bg-glow" style={{ ["--x" as string]: "50%", ["--y" as string]: "40%" }} />

      {/* the "photo" frame */}
      <motion.div
        initial={{ opacity: 0, y: 24, scale: 0.96 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.8, ease: EASE_OUT }}
        className="relative h-full w-full overflow-hidden rounded-[1.75rem] border border-border-strong bg-surface shadow-[0_20px_60px_-15px_rgba(20,21,26,0.18)]"
      >
        {/* abstract crafted "photo" content — layered gradient mesh, not a stock image */}
        <div
          className="absolute inset-0"
          style={{
            background:
              "radial-gradient(120% 90% at 15% 15%, #ffd9c2 0%, transparent 55%), radial-gradient(90% 70% at 85% 25%, #c7d2ff 0%, transparent 50%), radial-gradient(100% 100% at 50% 100%, #b9f3d6 0%, transparent 60%), linear-gradient(180deg, #f4f1ea 0%, #ece8de 100%)",
          }}
        />
        {/* subtle "mountain" silhouette shapes for visual interest */}
        <svg
          className="absolute inset-x-0 bottom-0 h-2/3 w-full opacity-70"
          viewBox="0 0 400 200"
          preserveAspectRatio="none"
        >
          <polygon points="0,200 0,120 90,50 170,110 240,60 330,130 400,90 400,200" fill="#ffffff" fillOpacity="0.55" />
          <polygon points="0,200 0,160 120,100 220,150 300,110 400,150 400,200" fill="#ffffff" fillOpacity="0.75" />
        </svg>

        {/* grid overlay, fades in with the scan */}
        <motion.div
          className="absolute inset-0 bg-grid opacity-0 mix-blend-multiply"
          animate={{ opacity: [0, 0.35, 0.35, 0] }}
          transition={{ duration: 4, repeat: Infinity, ease: "easeInOut", times: [0, 0.15, 0.85, 1] }}
        />

        {/* corner focus brackets */}
        {[
          "top-3 left-3 border-t-2 border-l-2 rounded-tl-lg",
          "top-3 right-3 border-t-2 border-r-2 rounded-tr-lg",
          "bottom-3 left-3 border-b-2 border-l-2 rounded-bl-lg",
          "bottom-3 right-3 border-b-2 border-r-2 rounded-br-lg",
        ].map((cls, i) => (
          <motion.div
            key={cls}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 + i * 0.08, duration: 0.5 }}
            className={`absolute h-7 w-7 border-primary/70 ${cls}`}
          />
        ))}

        {/* sweeping scan line */}
        <motion.div
          className="absolute inset-x-0 h-24"
          style={{
            background:
              "linear-gradient(180deg, transparent, color-mix(in srgb, var(--primary) 22%, transparent) 45%, color-mix(in srgb, var(--primary) 55%, transparent) 50%, color-mix(in srgb, var(--primary) 22%, transparent) 55%, transparent)",
          }}
          animate={{ top: ["-15%", "105%"] }}
          transition={{ duration: 2.6, repeat: Infinity, ease: "easeInOut", repeatDelay: 0.6 }}
        />

        {/* live "analyzing" pill */}
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.9 }}
          className="absolute left-4 top-4 flex items-center gap-1.5 rounded-full border border-border-strong bg-surface/90 px-3 py-1.5 text-xs font-medium text-foreground shadow-sm backdrop-blur"
        >
          <span className="relative flex h-1.5 w-1.5">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-60" />
            <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-primary" />
          </span>
          Analyzing
        </motion.div>

        {/* verdict badge, ticks up */}
        <motion.div
          initial={{ opacity: 0, scale: 0.85, y: 8 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ delay: 1.4, duration: 0.5, ease: EASE_OUT }}
          className="absolute bottom-4 right-4 rounded-2xl border border-border-strong bg-surface/95 px-4 py-3 shadow-lg backdrop-blur"
        >
          <p className="mono-tag text-[10px] uppercase tracking-wider text-foreground-muted">
            Fused probability
          </p>
          <p className="mt-0.5 text-2xl font-semibold tabular-nums text-alert">0.87</p>
        </motion.div>
      </motion.div>

      {/* floating evidence chips */}
      {chips.map(({ icon: Icon, label, top, left, delay }) => (
        <motion.div
          key={label}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: [0, -6, 0] }}
          transition={{
            opacity: { delay, duration: 0.5 },
            y: { delay: delay + 0.5, duration: 3.4, repeat: Infinity, ease: "easeInOut" },
          }}
          style={{ top, left }}
          className="absolute hidden items-center gap-1.5 rounded-full border border-border-strong bg-surface px-3 py-1.5 text-xs font-medium text-foreground shadow-md sm:flex"
        >
          <Icon className="h-3.5 w-3.5 text-primary" strokeWidth={2} />
          {label}
        </motion.div>
      ))}
    </div>
  );
}
