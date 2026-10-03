"use client";

import { useRef } from "react";
import { motion, useScroll, useTransform } from "motion/react";
import { ArrowRight, Sparkles } from "lucide-react";
import { HeroVisual } from "./hero-visual";
import { TextReveal } from "./text-reveal";
import { Magnetic } from "./magnetic";
import { Counter } from "./counter";
import { EASE_OUT } from "@/lib/motion";

const fadeUp = {
  hidden: { opacity: 0, y: 18 },
  show: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: 0.1 + i * 0.08, duration: 0.65, ease: EASE_OUT },
  }),
};

export function Hero() {
  const sectionRef = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ["start start", "end start"],
  });

  const visualY = useTransform(scrollYProgress, [0, 1], [0, 90]);
  const visualOpacity = useTransform(scrollYProgress, [0, 0.8], [1, 0]);
  const gridY = useTransform(scrollYProgress, [0, 1], [0, -60]);

  return (
    <section
      id="top"
      ref={sectionRef}
      className="relative overflow-hidden pt-16 pb-20 sm:pt-24 sm:pb-28"
    >
      <motion.div
        style={{ y: gridY }}
        className="pointer-events-none absolute inset-0 -z-10 bg-grid opacity-[0.35] mask-[radial-gradient(ellipse_60%_50%_at_50%_0%,black,transparent)]"
      />

      <div className="mx-auto grid max-w-6xl grid-cols-1 items-center gap-16 px-6 lg:grid-cols-[1.05fr_0.95fr]">
        <div>
          <motion.div
            custom={0}
            initial="hidden"
            animate="show"
            variants={fadeUp}
            className="mb-6 inline-flex items-center gap-1.5 rounded-full border border-border-strong bg-surface px-3 py-1.5 text-xs font-medium text-foreground-muted"
          >
            <motion.span
              animate={{ rotate: [0, 15, -10, 0] }}
              transition={{ duration: 2.5, repeat: Infinity, repeatDelay: 1.5, ease: "easeInOut" }}
            >
              <Sparkles className="h-3.5 w-3.5 text-primary" />
            </motion.span>
            Four specialist branches, one calibrated verdict
          </motion.div>

          <h1 className="text-balance text-5xl font-semibold tracking-tight text-foreground sm:text-6xl">
            <TextReveal text="See what was" delay={0.15} />
            <span className="relative mx-3 inline-block">
              <span className="relative z-10">
                <TextReveal text="changed" delay={0.45} />
              </span>
              <motion.span
                initial={{ scaleX: 0 }}
                animate={{ scaleX: 1 }}
                transition={{ delay: 0.85, duration: 0.6, ease: EASE_OUT }}
                style={{ transformOrigin: "left" }}
                className="absolute inset-x-0 bottom-1.5 -z-0 h-4 origin-left bg-primary-soft sm:bottom-2 sm:h-5"
              />
            </span>
            <TextReveal text="in an image — and why." delay={0.55} />
          </h1>

          <motion.p
            custom={2}
            initial="hidden"
            animate="show"
            variants={fadeUp}
            className="mt-6 max-w-xl text-lg leading-relaxed text-foreground-muted"
          >
            ForgeLens localizes splices, catches copy-move duplication,
            flags AI-generated content, and checks provenance — then fuses
            all four into one calibrated probability with a pixel-level
            evidence map. No verdict without evidence.
          </motion.p>

          <motion.div
            custom={3}
            initial="hidden"
            animate="show"
            variants={fadeUp}
            className="mt-9 flex flex-wrap items-center gap-4"
          >
            <Magnetic>
              <a
                href="#demo"
                className="group inline-flex items-center gap-2 rounded-full bg-foreground px-6 py-3.5 text-sm font-semibold text-background shadow-[0_1px_0_rgba(255,255,255,0.15)_inset] transition-shadow hover:shadow-lg active:scale-[0.98]"
              >
                Try the live demo
                <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
              </a>
            </Magnetic>
            <Magnetic strength={0.25}>
              <a
                href="#pipeline"
                className="inline-flex items-center gap-2 rounded-full border border-border-strong px-6 py-3.5 text-sm font-semibold text-foreground transition-colors hover:bg-surface-muted"
              >
                How it works
              </a>
            </Magnetic>
          </motion.div>

          <motion.div
            custom={4}
            initial="hidden"
            animate="show"
            variants={fadeUp}
            className="mt-12 flex flex-wrap items-center gap-x-8 gap-y-3 text-sm text-foreground-muted"
          >
            <div className="flex items-baseline gap-1.5">
              <Counter to={4} className="mono-tag text-lg font-semibold text-foreground" />
              <span>specialist branches</span>
            </div>
            <div className="flex items-baseline gap-1.5">
              <span className="mono-tag text-lg font-semibold text-foreground">px</span>
              <span>level localization</span>
            </div>
            <div className="flex items-baseline gap-1.5">
              <Counter to={1} className="mono-tag text-lg font-semibold text-foreground" />
              <span>calibrated probability</span>
            </div>
          </motion.div>
        </div>

        <motion.div
          style={{ y: visualY, opacity: visualOpacity }}
          initial={{ opacity: 0, x: 24 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.9, delay: 0.2, ease: EASE_OUT }}
        >
          <HeroVisual />
        </motion.div>
      </div>
    </section>
  );
}
