"use client";

import { motion } from "motion/react";
import { ArrowRight, Code2 } from "lucide-react";
import { EASE_OUT } from "@/lib/motion";
import { TextReveal } from "./text-reveal";
import { Magnetic } from "./magnetic";

export function CtaSection() {
  return (
    <section className="relative py-24 sm:py-32">
      <div className="mx-auto max-w-5xl px-6">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.7, ease: EASE_OUT }}
          className="relative overflow-hidden rounded-[2.5rem] border border-border-strong bg-foreground px-8 py-16 text-center sm:px-16"
        >
          <div className="pointer-events-none absolute inset-0 bg-grid opacity-10 mask-[radial-gradient(ellipse_70%_70%_at_50%_50%,black,transparent)]" />
          <motion.div
            className="pointer-events-none absolute -top-32 left-1/2 h-64 w-[36rem] -translate-x-1/2 rounded-full opacity-25 blur-3xl"
            style={{ background: "var(--primary)" }}
            animate={{ opacity: [0.15, 0.3, 0.15] }}
            transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }}
          />

          <h2 className="relative text-balance text-3xl font-semibold tracking-tight text-background sm:text-4xl">
            <TextReveal text="Every verdict comes with its evidence." trigger="scroll" />
          </h2>
          <p className="relative mx-auto mt-4 max-w-xl text-lg leading-relaxed text-background/70">
            No black-box scores. Pixel maps, plain-language findings, and a
            documented limitations statement on every single report.
          </p>

          <div className="relative mt-9 flex flex-wrap items-center justify-center gap-4">
            <Magnetic>
              <a
                href="#demo"
                className="group inline-flex items-center gap-2 rounded-full bg-background px-6 py-3.5 text-sm font-semibold text-foreground transition-shadow hover:shadow-lg active:scale-[0.98]"
              >
                Try the live demo
                <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
              </a>
            </Magnetic>
            <Magnetic strength={0.25}>
              <a
                href="https://github.com"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 rounded-full border border-background/25 px-6 py-3.5 text-sm font-semibold text-background transition-colors hover:bg-background/10"
              >
                <Code2 className="h-4 w-4" />
                View the repo
              </a>
            </Magnetic>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
