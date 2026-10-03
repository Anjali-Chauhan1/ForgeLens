"use client";

import { motion } from "motion/react";
import { Layers, ScanLine, ShieldCheck, Waves } from "lucide-react";
import { Scene3D } from "./three/scene-loader";
import { EASE_OUT } from "@/lib/motion";

const chips = [
  { icon: Waves, label: "Noise residual", top: "6%", left: "-10%", delay: 1.1 },
  { icon: Layers, label: "DCT double-compress", top: "64%", left: "-16%", delay: 1.4 },
  { icon: ScanLine, label: "Copy-move match", top: "14%", left: "90%", delay: 1.7 },
  { icon: ShieldCheck, label: "C2PA: unsigned", top: "72%", left: "86%", delay: 2.0 },
];

export function HeroVisual() {
  return (
    <div className="relative mx-auto aspect-square w-full max-w-lg select-none">
      {/* ambient glow behind the 3D canvas */}
      <motion.div
        className="absolute inset-0 -z-10 scale-125 rounded-[3rem] bg-glow"
        style={{ ["--x" as string]: "50%", ["--y" as string]: "45%" }}
        animate={{ opacity: [0.7, 1, 0.7] }}
        transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
      />

      <motion.div
        initial={{ opacity: 0, scale: 0.92 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 1, ease: EASE_OUT }}
        className="relative h-full w-full"
      >
        <Scene3D className="h-full w-full" />
      </motion.div>

      {/* live "analyzing" pill */}
      <motion.div
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.9 }}
        className="pointer-events-none absolute left-2 top-2 flex items-center gap-1.5 rounded-full border border-border-strong bg-surface/90 px-3 py-1.5 text-xs font-medium text-foreground shadow-sm backdrop-blur"
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
        className="pointer-events-none absolute bottom-2 right-2 rounded-2xl border border-border-strong bg-surface/95 px-4 py-3 shadow-lg backdrop-blur"
      >
        <p className="mono-tag text-[10px] uppercase tracking-wider text-foreground-muted">
          Fused probability
        </p>
        <p className="mt-0.5 text-2xl font-semibold tabular-nums text-alert">0.87</p>
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
          className="pointer-events-none absolute hidden items-center gap-1.5 rounded-full border border-border-strong bg-surface px-3 py-1.5 text-xs font-medium text-foreground shadow-md sm:flex"
        >
          <Icon className="h-3.5 w-3.5 text-primary" strokeWidth={2} />
          {label}
        </motion.div>
      ))}
    </div>
  );
}
