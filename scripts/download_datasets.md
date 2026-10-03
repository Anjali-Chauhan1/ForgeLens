# Dataset acquisition

No public mirror allows unattended bulk download of every benchmark
used in the blueprint, and several require an email request to the
original authors. This is a checklist, not a script -- run it in week 1
per the roadmap, since link rot and access delays are the top schedule
risk.

| Dataset | Access | Notes |
|---|---|---|
| CASIA v2.0 | Request via the CASIA forensics group listing (search "CASIA Image Tampering Detection Evaluation Database"); several long-standing GitHub mirrors exist -- verify checksums before trusting one | ~5.1k images, splice/copy-move/enhance |
| CASIA v1.0 | Same source as v2.0 | Held out for test only |
| Columbia | Columbia DVMM Research Lab / Columbia Image Splicing dataset page | 363 uncompressed images |
| COVERAGE | github.com/wenbihan/coverage | 100 images, hardest copy-move set |
| NIST16 / MFC | NIST Media Forensics Challenge datasets (nist.gov, requires registration) | Highest-quality masks |
| DEFACTO | Kaggle/academic mirrors; search "DEFACTO forgery dataset" | ~149k, auto-generated from MS-COCO |
| IMD2020 | University of Trento CVLab page | ~35k real-world manipulations |
| CoMoFoD | comofod page (Univ. of Zagreb) | Copy-move + 6 post-processes |
| tampCOCO / CocoGlide | Referenced in CAT-Net / TruFor repos; follow their data-prep scripts | Diffusion-era local edits |
| AutoSplice | github.com/shanface33/AutoSplice_Dataset | Text-prompt manipulated images |
| UniversalFakeDetect | github.com/Yuheng-Li/UniversalFakeDetect | 720k, real vs many generators |
| GenImage | github.com/GenImage-Dataset/GenImage | 2.7M, 8 generators, ImageNet-aligned |
| COCO (for synth generation + authentic pool) | cocodataset.org | Source images + instance masks for scripts/build_synth.py |

## Recommended order

1. Download COCO train2017 + instances annotations first -- unblocks
   `scripts/build_synth.py` and the authentic-image pool immediately.
2. Request CASIA v2.0 -- the primary train/finetune set.
3. Download the small test-only sets (Columbia, COVERAGE) -- fast wins
   for the cross-dataset eval table.
4. Register for NIST16 and IMD2020 in parallel, since institutional
   approval can take days.
5. GenImage / UniversalFakeDetect for Branch C once Branch A is stable.

## Verify masks after download

Before training on any downloaded dataset, run:

```bash
python -c "
from scripts.build_manifest import adapt_casia2
from PIL import Image
samples = adapt_casia2('/path/to/CASIA2')
forged_with_mask = [s for s in samples if s.is_forged and s.mask_path]
print(f'{len(forged_with_mask)}/{sum(s.is_forged for s in samples)} forged images have a mask')
Image.open(forged_with_mask[0].mask_path).convert('L').save('/tmp/mask_check.png')
"
```

If the matched-mask ratio is low, the filename-matching heuristic in
`_find_mask_for` likely doesn't fit that mirror's naming convention --
inspect a few filenames and adjust the adapter rather than training on
a silently mismatched majority.
