"use client";

import { motion } from "motion/react";
import { AlertTriangle, CircleAlert, Info, Minus } from "lucide-react";
import type { EvidenceItem, EvidenceWeight } from "@/lib/types";
import { EASE_OUT } from "@/lib/motion";

const WEIGHT_STYLE: Record<
  EvidenceWeight,
  { icon: typeof Info; color: string; bg: string; label: string }
> = {
  high: { icon: AlertTriangle, color: "var(--alert)", bg: "var(--alert-soft)", label: "Strong" },
  medium: { icon: CircleAlert, color: "var(--caution)", bg: "var(--caution-soft)", label: "Supporting" },
  low: { icon: Info, color: "var(--foreground-muted)", bg: "var(--surface-muted)", label: "Weak" },
  neutral: { icon: Minus, color: "var(--foreground-muted)", bg: "var(--surface-muted)", label: "Neutral" },
};

const BRANCH_LABEL: Record<string, string> = {
  metadata: "Metadata",
  localization: "Localization",
  copy_move: "Copy-move",
  synthetic: "Synthetic content",
  noise_block: "Noise analysis",
  retrieval: "Retrieval",
  c2pa: "Provenance (C2PA)",
};

export function EvidenceList({ items }: { items: EvidenceItem[] }) {
  return (
    <ul className="space-y-2.5">
      {items.map((item, i) => {
        const style = WEIGHT_STYLE[item.weight];
        return (
          <motion.li
            key={`${item.branch}-${i}`}
            initial={{ opacity: 0, x: -12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 + i * 0.07, duration: 0.4, ease: EASE_OUT }}
            className="flex items-start gap-3 rounded-2xl border border-border p-3.5"
          >
            <span
              className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full"
              style={{ background: style.bg, color: style.color }}
            >
              <style.icon className="h-3.5 w-3.5" strokeWidth={2.25} />
            </span>
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold uppercase tracking-wide text-foreground-muted">
                  {BRANCH_LABEL[item.branch] ?? item.branch}
                </span>
                <span
                  className="rounded-full px-2 py-0.5 text-[10px] font-medium"
                  style={{ background: style.bg, color: style.color }}
                >
                  {style.label}
                </span>
              </div>
              <p className="mt-1 text-sm leading-relaxed text-foreground">{item.finding}</p>
            </div>
          </motion.li>
        );
      })}
    </ul>
  );
}
