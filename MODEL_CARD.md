# Model Card — ForgeLens

## Current checkpoint

`runs/localize/best.pt` (raw weights, not EMA — see README.md) was
trained for 8 epochs on a 234-image procedurally generated in-domain
benchmark (no external dataset was available in this environment).
Measured on a 72-image held-out split with the trained fusion model:
**87.5% accuracy, 0.866 F1, 0.902 AUC, 0.533 pixel F1 on forged
regions.** This is a real, in-domain result, not a cross-dataset
(CASIA/NIST16/IMD2020) result — see README.md's "Real training results"
for the full breakdown and what would be needed to close the gap to
production-scale numbers.

## Overview

ForgeLens is a multi-branch image forensics system: a manipulation-
localization model (Branch A), a classical copy-move detector (Branch B),
an AI-generated-content detector (Branch C), and metadata/provenance
checks (Branch D), combined by a calibrated fusion model with an
abstention gate. See `README.md` for the architecture diagram and
`scripts/download_datasets.md` for training data sources.

## Intended use

Forensic triage and evidence-gathering support for insurance, legal,
journalism, and content-moderation workflows. Output is a probability
plus a pixel-level evidence map and a plain-language evidence list, NOT
a certainty. Every report carries an explicit limitations statement and
the system is designed to abstain rather than force a verdict when its
own quality/disagreement checks indicate low reliability.

## Out-of-scope use

- **Not a lie detector for captions/context.** Branch D's retrieval
  module can flag known-duplicate reuse, but the system cannot verify
  whether an unedited, genuine photograph's caption, date, or location
  claim is accurate.
- **Not admissible on its own.** Any evidentiary use requires the full
  chain-of-custody record (hash, timestamp, pinned model version) and
  independent corroboration.
- **Not a defense against a white-box adversary.** An attacker with
  model-weight access can defeat any of the branches with adversarial
  perturbation; see the blueprint's robustness section.

## Training data (once populated per `scripts/download_datasets.md`)

Branch A: CASIA v2.0, DEFACTO, tampCOCO, CocoGlide, AutoSplice, plus a
locally generated synthetic set (`scripts/build_synth.py`), all passed
through the laundering augmentation (`forgelens/data/laundering.py`).

Branch C: GenImage / UniversalFakeDetect, evaluated with a strict
leave-one-generator-out split.

## Evaluation protocol

Cross-dataset (train on A, test on B with zero finetuning) is the
reported headline, per `forgelens/eval/protocols.py`. In-dataset numbers,
if reported, are clearly labeled as such. See the blueprint's realistic
target-numbers table for what "good" means on each benchmark — IMD2020
and Coverage scores will legitimately be much lower than Columbia.

## Known limitations

- Localization confidence degrades sharply below JPEG quality ~45 and
  below 256px on the shortest side (the abstention gate encodes this).
- Copy-move detection can under-fire on smooth/textureless duplicated
  regions (SIFT finds no keypoints there); the deep self-correlation
  path in the blueprint is the planned mitigation, not yet implemented
  in this reference codebase.
- Branch C's frozen-CLIP approach generalizes across generator families
  better than an end-to-end CNN, but is not guaranteed against
  generators released after the encoder's pretraining cutoff or after
  the head's last training run — plan a versioned refresh cadence.
- The `KNOWN_QTABLE_SIGNATURES` table in `forensics/metadata.py` is a
  small illustrative seed, not a validated forensic corpus. Replace it
  with a curated dataset (e.g. derived from FotoForensics' public JPEG
  dimples data) before relying on quantization-table evidence in a
  real report.

## Fairness

Per the blueprint's ethics section, any deployment MUST run a
per-subgroup accuracy evaluation (skin tone, gender presentation) on
the face-manipulation path before production use, and report the
disparity alongside the aggregate numbers. This reference codebase does
not yet include a specialized face-manipulation branch or that
evaluation; both are flagged as required before broadening scope beyond
the current four branches.

## Versioning

Every `/v1/analyze` response should be traceable to a specific set of
branch checkpoint hashes and the fusion-model version used to produce
it. Wire this into `service/api.py`'s response before any deployment
beyond local development.
