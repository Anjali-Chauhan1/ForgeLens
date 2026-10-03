import { Navbar } from "@/components/navbar";
import { Hero } from "@/components/hero";
import { Marquee } from "@/components/marquee";
import { BranchesSection } from "@/components/branches-section";
import { PipelineSection } from "@/components/pipeline-section";
import { DemoSection } from "@/components/demo-section";
import { CtaSection } from "@/components/cta-section";
import { Footer } from "@/components/footer";

const MARQUEE_ITEMS = [
  "SPLICE DETECTION",
  "COPY-MOVE ANALYSIS",
  "AI-GENERATED CONTENT",
  "PROVENANCE VERIFICATION",
  "PIXEL-LEVEL LOCALIZATION",
  "CALIBRATED CONFIDENCE",
];

export default function Home() {
  return (
    <>
      <Navbar />
      <main className="flex-1">
        <Hero />
        <Marquee items={MARQUEE_ITEMS} />
        <BranchesSection />
        <PipelineSection />
        <DemoSection />
        <CtaSection />
      </main>
      <Footer />
    </>
  );
}
