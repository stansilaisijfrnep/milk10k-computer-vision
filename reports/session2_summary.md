# Session 2 — generated summary

_Written by `scripts/run_session2.py`; every number below is computed in that run._

## Part 1 — metadata vs diagnosis
- Unit of analysis: 5,240 lesions (10,480 images). Only `image_manipulation` (327 lesions), `image_type` (5,240 lesions) differ between a lesion's two images; those two are tested on images, everything else on lesions.
- `concomitant_biopsy` vs `diagnosis_confirm_type`: identical (1:1).
- `melanocytic` is recorded only for lesions with 11-class codes ['MEL', 'NV'].
- Ranking by effect size (Cramér's V, or η for age):
  - `diagnosis_2` [leakage, lesion] 1.000 (large)
  - `diagnosis_3` [leakage, lesion] 0.998 (large)
  - `melanocytic` [leakage, lesion] 0.414 (medium)
  - `diagnosis_4` [leakage, lesion] 0.403 (medium)
  - `age_approx` [usable, lesion] 0.390 (medium)
  - `concomitant_biopsy` [leakage, lesion] 0.311 (medium)
  - `diagnosis_confirm_type` [leakage, lesion] 0.311 (medium)
  - `anatom_site_general` [usable, lesion] 0.157 (small)
  - `image_manipulation` [acquisition, image] 0.139 (small)
  - `sex` [usable, lesion] 0.093 (negligible)
  - `anatom_site_special` [usable, lesion] 0.065 (negligible)
  - `image_type` [acquisition, image] 0.000 (negligible)
- Most associated *usable* fields: `age_approx` (0.39), `anatom_site_general` (0.16), `sex` (0.09)

## Part 2 — colour
- Sample: 120 lesions per class x 2 images = 720 images.
- η² of modality (dermoscopic vs clinical) on mean colour: mean_r 0.01, mean_g 0.04, mean_b 0.30, mean_gray 0.03
- η² of class, within a modality, largest: `mean_g` 0.148 (14.8% of variance).
- Colour-only logistic regression, dermoscopic: balanced accuracy 0.472 ± 0.026 (chance 0.333).
- Colour-only logistic regression, clinical: close-up: balanced accuracy 0.556 ± 0.018 (chance 0.333).

## Parts 3-5 — pipeline
- BatchLoader: 720 images on disk (0 metadata rows without a file), batch_size=16, 45 batches/epoch, shuffle=True, classes={'Benign': 240, 'Indeterminate': 240, 'Malignant': 240}, preprocessing={'channels_first': True}
- First batch: images (16, 3, 224, 224) float32, range [0.012, 1.000]; z-scored batch mean -0.193, std 1.054.
- `process_image` steps: resize 600x450 -> 224x224 (bilinear) → colour: rgb → minmax: x / channel max (fixed bounds) -> [0, 1] → layout: HWC -> CHW
