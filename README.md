# ForgeLens

A multi-branch digital image forgery detection and localization system.
This is the working implementation of the "ForgeLens" project
blueprint (published separately as an artifact) — see `MODEL_CARD.md`
for scope/limitations and `scripts/download_datasets.md` for how to get
training data.

Instead of one classifier trying to cover every attack type, four
specialist branches share a forensic front end and get combined by a
calibrated fusion model with an abstention gate:

- **Branch A — Localization**: a ConvNeXt-Tiny U-Net fed by a
  Bayar-constrained-convolution + SRM-filter-bank stem (`forgelens/models/`),
  trained with focal+Dice region loss and an edge-boundary auxiliary loss
  (`forgelens/train/losses.py`).
- **Branch B — Copy-move**: classical SIFT self-matching + RANSAC affine
  fitting (`forgelens/forensics/copymove_sift.py`) — explainable, and the
  only branch that works when noise statistics are identical (a region
  duplicated within the same image).
- **Branch C — Synthetic content**: a frozen CLIP encoder + trainable head,
  global and patch-grid (`forgelens/models/synthetic.py`) — generalizes
  across unseen generators far better than a CNN trained from scratch.
- **Branch D — Metadata/provenance**: EXIF, JPEG quantization-table
  fingerprinting, DCT double-compression analysis, ELA, perceptual-hash
  retrieval (`forgelens/forensics/`, `forgelens/utils/hashing.py`).

All four feed a small gradient-boosted fusion model with temperature
calibration and a documented abstention policy (`forgelens/fusion/fuse.py`).

## Status

This is a complete, tested reference implementation of the pipeline
architecture, loss functions, training loops, evaluation protocol, and
serving layer, and it has now been **trained and evaluated end-to-end**
on a real (procedurally generated, in-domain) benchmark in this
environment — no external dataset or GPU was available, so this is not
the CASIA/NIST16/IMD2020 cross-dataset result a production claim would
need, but it is a genuine, measured result on real trained weights, not
an estimate. See "Real training results" below for the numbers, and
`scripts/download_datasets.md` + the blueprint's roadmap for scaling
this to production datasets.

**Headline, measured on a 72-image held-out test split:** 87.5%
image-level accuracy, 0.866 F1, 0.902 AUC, 0.533 pixel F1 on forged
regions. This is real but it is **not** the 95%+ that was asked for —
see "Real training results" for why, and what closing that gap would
take.

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Run the test suite (no GPU, no dataset needed)
pytest

# Start the API with classical-only branches (no trained models yet --
# the service runs end-to-end using metadata/ELA/noise/copy-move only)
uvicorn service.api:app --reload
# then: curl -F "image=@photo.jpg" http://localhost:8000/v1/analyze
```

## Training, once you have data

```bash
# 1. Build manifests from whatever datasets you've downloaded
python scripts/build_manifest.py --casia2 /data/CASIA2 \
    --authentic-extra /data/coco_authentic --out manifests/train.jsonl
python scripts/build_manifest.py --images-dir /data/NIST16/images \
    --masks-dir /data/NIST16/masks --forged --out manifests/val_nist16.jsonl

# 2. (Optional but recommended) generate the large synthetic pretraining set
python scripts/build_synth.py --ann-file /data/coco/annotations/instances_train2017.json \
    --images-dir /data/coco/train2017 --out-dir data/synth --n-pairs 200000

# 3. Train Branch A (localization)
python -m forgelens.train.trainer_localize \
    --train-manifest manifests/train.jsonl --val-manifest manifests/val_nist16.jsonl \
    --out-dir runs/localize --epochs 30

# 4. Train Branch C (synthetic detection) -- separate manifest format, see
#    trainer_synthetic.py's SyntheticImageDataset docstring
python -m forgelens.train.trainer_synthetic \
    --train-manifest manifests/genimage_train.jsonl \
    --val-manifest manifests/genimage_val_logo.jsonl --out-dir runs/synthetic

# 5. Dump branch features over a labeled held-out split, then fit fusion+calibration
python -m forgelens.train.calibrate \
    --fusion-train manifests/fusion_features_train.jsonl \
    --calibration-split manifests/fusion_features_cal.jsonl \
    --out runs/fusion/fusion_model

# API automatically picks up runs/localize/best.pt and runs/fusion/fusion_model.*
# once they exist (FORGELENS_MODEL_DIR env var to point elsewhere).
```

## Real training results

Run entirely on CPU (no GPU), using a procedurally generated in-domain
benchmark (`scripts/make_procedural_dataset.py`) since no external
forensics dataset was available in this environment: 360 256×256 images
(180 authentic / 180 forged, roughly even splice/copy-move/removal mix,
laundering augmentation applied identically to both classes per the
blueprint's rule), split 234 train / 54 val / 72 test.

Branch A (localization) trained 8 epochs, ConvNeXt-Tiny backbone,
ImageNet-pretrained, Bayar+SRM stem, focal+Dice+edge+image losses:

| Metric | Value |
|---|---|
| Val pixel F1 (best epoch, raw weights) | 0.563 |
| **Test pixel F1, forged images only** (fixed 0.5 threshold) | **0.533** |
| Test pixel IoU, forged images only | 0.396 |

Fusion (LightGBM meta-model + temperature calibration, trained on the
234-image train split, calibrated on the 54-image val split — a real
run of training stage S5) + the full pipeline (localization + classical
metadata/ELA/noise/copy-move branches), evaluated on the 72-image held
-out test split:

| Metric | Value |
|---|---|
| **Accuracy** | **0.875** |
| Precision | 0.879 |
| Recall | 0.853 |
| **F1** | **0.866** |
| **AUC** | **0.902** |
| ECE (ended up worse than target — see below) | 0.094 |
| Abstained | 21/72 (29%) |

**This is real, measured, and not 95%+.** Two honest reasons why, both
consistent with the blueprint's own target tables (which put even
near-SOTA cross-dataset pixel F1 in the 0.55–0.85 range, never 0.95+):
the localization model saw only 234 training images for 8 epochs (a
production run per the blueprint's recipe uses tens of thousands of
images and 30+ epochs), and the fusion meta-model was fit on the same
small scale, which is also why its calibration (ECE=0.094, and 0.230
before temperature scaling) misses the blueprint's 0.05 target — not
enough data to calibrate tightly. Closing the gap toward 95%+ needs the
real path the blueprint lays out: more data (start with
`scripts/download_datasets.md`), longer training, and activating
Branches B/C fully (only classical SIFT copy-move ran here; the CLIP-
based synthetic-content branch was never trained/wired into this run).

### Real bugs found and fixed by actually running this code

Four genuine bugs surfaced by training and evaluating end-to-end,
not by inspection — each is fixed in the current code:

1. **Bayar convolution numerical explosion.** The original
   `weight / weight.sum()` normalization blew up (kernel magnitudes in
   the tens of thousands) whenever the random-initialized weight sum
   passed near zero. Fixed with softmax normalization.
2. **CPU bf16 autocast was 35× slower, not faster.** The trainer wrapped
   every step in `torch.autocast(..., dtype=torch.bfloat16)` per the
   blueprint's "bf16 AMP" recommendation, which assumes GPU hardware.
   Measured on this machine: 56s/step with autocast vs. 1.6s/step
   without, for identical work — autocast is now gated to CUDA only.
   The first training attempt ran for ~20 minutes and completed 10 of
   290 steps before this was caught and fixed.
3. **Inference padded to a fixed 512px canvas regardless of image size.**
   A 160×160 test image was padded into ~90% black border — a massive
   train/inference distribution mismatch versus the near-full-frame
   crops the model actually trained on. Every prediction silently
   defaulted to "authentic" as a result. Fixed to pad to the smallest
   stride-aligned size that fits the actual image.
4. **EMA weights were unusable on a short run.** `decay=0.999` has a
   ~1000-step time constant; after this run's 312 total steps the EMA
   shadow weights were still ~73% dominated by the random initialization
   (`0.999^312 ≈ 0.73`), and a checkpoint loaded from EMA predicted
   nothing on every test image despite the raw weights scoring
   val pixel F1=0.55. Added an EMA warmup schedule for future runs, and
   changed both the eval script and the API to default to raw weights
   (matching what `evaluate()` actually validated as "best" during
   training) rather than blindly preferring EMA.

Two smaller bugs also surfaced: `FusionModel.save()` didn't create its
output directory, and `load_manifest()` errored on the `manipulation`
bookkeeping field written by the dataset generators.

## What's been verified in this environment

- `pytest tests/` — 17/17 passing (classical forensics, fusion +
  calibration, abstention gate, every metric in `eval/metrics.py`).
- The full training pipeline end to end: dataset generation, laundering
  augmentation, Branch A training (8 epochs, real convergence), fusion
  meta-model + temperature-scaling training, and evaluation against a
  held-out test split — see "Real training results" above.
- **Full inference pipeline** (`ForgeLensPipeline.analyze`) and the
  **FastAPI service** (including batch upload, API-key auth + rate
  limiting, chain-of-custody audit log): exercised end-to-end via
  `TestClient`, including `/v1/report/{sha}.pdf` producing a valid PDF.

Not exercised (blocked by the lack of a real forensics dataset and GPU
in this environment, but the code paths are complete): training on
CASIA/NIST16/IMD2020 or any cross-dataset protocol, the CLIP-based
Branch C forward pass (needs `open_clip_torch` + a model download), the
deep copy-move self-correlation path (only classical SIFT ran), ONNX
export, and ForensicHub/IMDL-BenCo integration.

## Repository layout

See the blueprint's "Repository layout" section — this implementation
follows it directly. Key entry points:

- `forgelens/inference/pipeline.py` — the orchestrator, start here to
  understand how the branches compose.
- `forgelens/models/stems.py` — the Bayar/SRM forensic front end.
- `forgelens/data/laundering.py` — the social-media degradation simulator,
  the single highest-leverage training detail per the blueprint.
- `forgelens/fusion/fuse.py` — fusion, calibration, verdict bands,
  abstention.
- `service/api.py` — the FastAPI service.
