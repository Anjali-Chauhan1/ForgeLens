"use client";

import dynamic from "next/dynamic";
import { motion } from "motion/react";

// WebGL only exists in the browser -- ssr:false keeps three.js out of the
// server render entirely, and the fallback below covers both the SSR gap
// and the moment before the client bundle finishes loading.
const Scene = dynamic(() => import("./forgelens-scene"), {
  ssr: false,
  loading: () => <SceneFallback />,
});

function SceneFallback() {
  return (
    <div className="flex h-full w-full items-center justify-center">
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 2.4, repeat: Infinity, ease: "linear" }}
        className="h-16 w-16 rounded-full border-2 border-primary-soft border-t-primary"
      />
    </div>
  );
}

export function Scene3D({ className }: { className?: string }) {
  return (
    <div className={className}>
      <Scene />
    </div>
  );
}
