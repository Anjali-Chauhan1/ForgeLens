"use client";

import { motion, useMotionValue, useTransform, animate } from "motion/react";
import { useEffect, useState } from "react";
import type { VerdictKey } from "@/lib/types";
import { EASE_OUT } from "@/lib/motion";

const VERDICT_COLOR: Record<VerdictKey, string> = {
  no_evidence: "var(--verified)",
  weak_indicators: "var(--verified)",
  inconclusive: "var(--caution)",
  abstained: "var(--caution)",
  likely_manipulated: "var(--alert)",
  strong_evidence: "var(--alert)",
};

export function VerdictRing({
  probability,
  verdict,
  size = 148,
}: {
  probability: number;
  verdict: VerdictKey;
  size?: number;
}) {
  const color = VERDICT_COLOR[verdict];
  const radius = (size - 14) / 2;
  const circumference = 2 * Math.PI * radius;

  const progress = useMotionValue(0);
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    const controls = animate(progress, probability, {
      duration: 1.1,
      delay: 0.15,
      ease: EASE_OUT,
      onUpdate: (v) => setDisplay(v),
    });
    return () => controls.stop();
  }, [probability, progress]);

  const dashoffset = useTransform(progress, (v) => circumference * (1 - v));

  return (
    <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="var(--surface-muted)"
          strokeWidth={10}
        />
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={10}
          strokeLinecap="round"
          strokeDasharray={circumference}
          style={{ strokeDashoffset: dashoffset }}
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className="mono-tag text-3xl font-semibold tabular-nums" style={{ color }}>
          {display.toFixed(2)}
        </span>
        <span className="text-[11px] font-medium text-foreground-muted">probability</span>
      </div>
    </div>
  );
}
