"use client";

import { motion } from "motion/react";
import { EASE_OUT } from "@/lib/motion";

export function TextReveal({
  text,
  className,
  delay = 0,
  stagger = 0.035,
  as: Tag = "span",
  trigger = "mount",
}: {
  text: string;
  className?: string;
  delay?: number;
  stagger?: number;
  as?: "span" | "div";
  /** "mount" plays immediately (hero text, visible on first paint).
   * "scroll" waits until the element enters the viewport — required for
   * any heading below the fold, otherwise it finishes animating before
   * the user ever scrolls to it. */
  trigger?: "mount" | "scroll";
}) {
  const words = text.split(" ");
  const animProps =
    trigger === "scroll"
      ? { initial: "hidden" as const, whileInView: "show" as const, viewport: { once: true, margin: "-80px" } }
      : { initial: "hidden" as const, animate: "show" as const };

  return (
    <Tag className={className} aria-label={text}>
      {words.map((word, i) => (
        <span key={i} className="inline-block overflow-hidden pb-1 align-bottom">
          <motion.span
            className="inline-block"
            {...animProps}
            variants={{
              hidden: { y: "110%", rotate: 4 },
              show: { y: "0%", rotate: 0 },
            }}
            transition={{
              duration: 0.7,
              delay: delay + i * stagger,
              ease: EASE_OUT,
            }}
          >
            {word}
            {i < words.length - 1 ? " " : ""}
          </motion.span>
        </span>
      ))}
    </Tag>
  );
}
