# ForgeLens — Frontend

A Next.js (App Router) + TypeScript UI for the ForgeLens forensics
pipeline. Light mode only, animated with [Motion](https://motion.dev/),
styled with Tailwind CSS v4.

## Structure

- `app/page.tsx` — landing page: hero, branch showcase, pipeline diagram,
  live demo, CTA.
- `components/scan-visual.tsx` — the animated hero graphic (crafted from
  CSS gradients + SVG + Motion, not a stock image).
- `components/demo-section.tsx` — the interactive upload → analyze →
  evidence-report flow. Calls `NEXT_PUBLIC_FORGELENS_API`'s `/v1/analyze`
  endpoint; if unreachable, falls back to a clearly labeled client-side
  simulation (`lib/demo-data.ts`) deterministically derived from the
  uploaded file's hash, so the demo stays interactive without a running
  backend.
- `lib/types.ts` — mirrors `service/schemas.py` on the backend.
- `lib/motion.ts` — shared, properly-typed easing curve (Motion's
  `Easing` type rejects a plain `number[]`, so this is defined once as
  a `const` tuple and reused everywhere).

## Running

```bash
npm install
cp .env.example .env.local   # point at your running backend, or leave default
npm run dev
```

Then start the backend separately (`uvicorn service.api:app --reload`
from the repository root) for real analysis — the demo works without it
too, in simulation mode.

## Design notes

- **Light mode only**, per the project brief — `color-scheme: light` is
  set explicitly in `app/layout.tsx` and no dark-mode media query exists
  in `globals.css`, so the UI doesn't shift under a dark OS preference.
- Color tokens (`--primary`, `--verified`, `--alert`, `--caution`) map
  directly to the verdict semantics used throughout `lib/types.ts` — the
  same accent that marks a "Copy-move" branch card is the one used for
  a "strong evidence" verdict ring, so the visual language stays
  consistent from the marketing sections into the actual report.
