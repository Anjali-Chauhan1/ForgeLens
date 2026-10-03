"use client";

import { motion } from "motion/react";
import { Fragment } from "react";

export function Marquee({
  items,
  speed = 32,
  className,
}: {
  items: string[];
  speed?: number;
  className?: string;
}) {
  return (
    <div className={`group relative overflow-hidden border-y border-border py-5 ${className ?? ""}`}>
      <div className="pointer-events-none absolute inset-y-0 left-0 z-10 w-24 bg-gradient-to-r from-background to-transparent" />
      <div className="pointer-events-none absolute inset-y-0 right-0 z-10 w-24 bg-gradient-to-l from-background to-transparent" />
      <motion.div
        className="flex w-max items-center gap-10 group-hover:[animation-play-state:paused]"
        animate={{ x: ["0%", "-50%"] }}
        transition={{ duration: speed, repeat: Infinity, ease: "linear" }}
      >
        {[0, 1].map((rep) => (
          <Fragment key={rep}>
            {items.map((item, i) => (
              <div key={`${rep}-${i}`} className="flex items-center gap-10">
                <span className="mono-tag whitespace-nowrap text-sm font-medium tracking-wide text-foreground-muted">
                  {item}
                </span>
                <span className="h-1.5 w-1.5 rounded-full bg-border-strong" />
              </div>
            ))}
          </Fragment>
        ))}
      </motion.div>
    </div>
  );
}
