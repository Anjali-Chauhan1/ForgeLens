"use client";

import { motion } from "motion/react";
import { ShieldCheck } from "lucide-react";
import { EASE_OUT } from "@/lib/motion";
import { Magnetic } from "./magnetic";

const links = [
  { href: "#branches", label: "Branches" },
  { href: "#pipeline", label: "Pipeline" },
  { href: "#demo", label: "Try it" },
];

export function Navbar() {
  return (
    <motion.header
      initial={{ opacity: 0, y: -16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: EASE_OUT }}
      className="sticky top-0 z-50 border-b border-border/80 bg-background/80 backdrop-blur-lg"
    >
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <a href="#top" className="group flex items-center gap-2">
          <motion.span
            whileHover={{ rotate: -8, scale: 1.05 }}
            transition={{ type: "spring", stiffness: 300, damping: 15 }}
            className="flex h-8 w-8 items-center justify-center rounded-xl bg-foreground text-background"
          >
            <ShieldCheck className="h-4.5 w-4.5" strokeWidth={2.25} />
          </motion.span>
          <span className="text-[15px] font-semibold tracking-tight">ForgeLens</span>
        </a>

        <nav className="hidden items-center gap-8 text-sm text-foreground-muted md:flex">
          {links.map((l) => (
            <a key={l.href} href={l.href} className="group relative py-1 transition-colors hover:text-foreground">
              {l.label}
              <span className="absolute inset-x-0 -bottom-0.5 h-px origin-left scale-x-0 bg-foreground transition-transform duration-300 group-hover:scale-x-100" />
            </a>
          ))}
        </nav>

        <Magnetic strength={0.3}>
          <a
            href="#demo"
            className="rounded-full bg-foreground px-4 py-2 text-sm font-medium text-background transition-shadow hover:shadow-md active:scale-[0.98]"
          >
            Analyze an image
          </a>
        </Magnetic>
      </div>
    </motion.header>
  );
}
