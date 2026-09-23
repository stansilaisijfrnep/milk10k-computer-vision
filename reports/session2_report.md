# Session 2 homework — Extended EDA & data pipeline (MILK10k)

**Stanislaus Lattorff** · Computer Vision & Speech Recognition, EADA · September 2026
**Code:** <https://github.com/stansilaisijfrnep/milk10k-computer-vision> — `src/milk10k/eda2.py` (Parts 1–2), `pipeline.py` (3–4), `viz.py` (5); `python scripts/run_session2.py` regenerates every number and figure below.

## Part 1 — Which metadata fields relate to the diagnosis?

- **Unit of analysis = the lesion.** `metadata.csv` has 10,480 rows but 5,240 lesions, each photographed twice. All fields except `image_type` and `image_manipulation` are identical on a lesion's two images, so testing image rows would count every lesion twice and double every χ². Lesion-constant fields are tested on 5,240 lesions, the two image-level fields on 10,480 images.
- **Statistics.** Categorical vs. the 3-class target: **χ² test of independence + Cramér's V** (V rescales χ² to 0–1 whatever the number of categories, so fields are comparable). Numeric `age_approx`: **one-way ANOVA + η** (share of variance explained by class; η is on the same 0–1 scale as V), with **Kruskal–Wallis** as a rank-based check because age is binned and skewed. At n = 5,240 every p-value is ≈ 0, so fields are **ranked by effect size**, not by significance. Missing values are kept as their own category, because *whether* a field was recorded turns out to be informative.

<img src="figures/fig16_association_ranking.png">

| Field | Role | Effect | Reading |
|---|---|---|---|
| `diagnosis_2`, `diagnosis_3` | leakage | V = 1.00 | finer levels of the diagnosis — they *are* the label |
| `melanocytic` | leakage | V = 0.41 | only ever filled in for nevi and melanomas; its presence names the lesion family |
| **`age_approx`** | **usable** | **η = 0.39** (η² = 0.15) | benign lesions: mean 52 y; malignant 65 y; indeterminate 68 y |
| `diagnosis_confirm_type` = `concomitant_biopsy` | leakage | V = 0.31 | identical variables; non-biopsied lesions are 2 % malignant (5 of 224) |
| **`anatom_site_general`** | **usable** | **V = 0.16** | head/neck carries most Indeterminate lesions; 37 % missing |
| `image_manipulation` | acquisition | V = 0.14 | "altered" photos are 13 % Indeterminate vs. 2 % overall |
| `sex`, `anatom_site_special` | usable | V = 0.09 / 0.06 | negligible |
| `image_type` | acquisition | V = 0.00 | every lesion has one of each, so independent by construction |
| `attribution`, `copyright_license` | constant | — | a single value for all rows |

<img src="figures/fig17_class_mix_by_field.png">

<img src="figures/fig18_age_by_class.png">

- **Most associated usable fields:** age (medium effect), then anatomic site (small). **Least useful:** `sex` and `anatom_site_special` (negligible), `image_type` (zero by design) and the two constant columns.
- **Leakage — never a model input:** `diagnosis_2/3/4` contain the answer; `diagnosis_confirm_type`/`concomitant_biopsy` record *that a biopsy was done*, a decision made because a clinician was already worried, and the rare lesions that were not biopsied are almost all benign; `melanocytic` leaks through its missingness.
- **Bias / shortcut risk:** `image_manipulation` describes how the photograph was produced, not the lesion, yet shifts the class mix; a model could learn "edited photo → indeterminate". Age and site are legitimate but also mark the *population* (older, sun-exposed patients), so a model must not reduce to "old + face → malignant".

## Part 2 — Colour and histograms across classes

- **Sample.** 120 lesions per class (Indeterminate has only 123), **both images of each** → 720 images at full 600×450 resolution, balanced so class imbalance cannot drive any difference. Histograms are normalised per image before averaging, so every image weighs the same.
- **Split by modality.** Dermoscopy (contact lens, own light source) and clinical close-ups are shown separately; pooling them blurs two different pictures: for all nine colour features the class effect (η²) is smaller pooled than within clinical images.

<img src="figures/fig19_gray_hist_by_class.png">

<img src="figures/fig20_rgb_hist_by_class.png">

<img src="figures/fig21_color_stats_by_class.png">

| Per image (mean ± sd) | Dermoscopic: Benign | Indet. | Malignant | Clinical: Benign | Indet. | Malignant |
|---|---|---|---|---|---|---|
| mean gray (brightness) | 148 ± 22 | 140 ± 22 | 148 ± 20 | 145 ± 21 | **127 ± 22** | 140 ± 19 |
| mean G | 141 ± 22 | 128 ± 20 | 139 ± 20 | 136 ± 23 | **114 ± 23** | 129 ± 20 |
| R / G (redness) | 1.2 ± 0.1 | 1.3 ± 0.1 | 1.2 ± 0.1 | 1.3 ± 0.2 | **1.5 ± 0.2** | 1.4 ± 0.2 |
| std gray (contrast) | 28 ± 15 | 23 ± 9 | 21 ± 11 | 21 ± 10 | 26 ± 12 | 19 ± 6 |

- **What differs:** Indeterminate images are darker and redder, clearly so in clinical photos. Benign dermoscopic images show a tail of bright pixels (190–230) and higher contrast. Benign and Malignant are close to each other in every channel.
- **Alternative explanations — the difference is not necessarily the lesion.** In this very sample, Indeterminate lesions are **80 % head/neck** (Benign 20 %, Malignant 28 %), **15 % Fitzpatrick type I** skin (others ≤ 1 %) and **17.5 % "altered"** photos (others 2–5 %): sun-damaged, fair facial skin photographed under different conditions. Five of the 120 Indeterminate clinical photos contain large dark backgrounds, which is where the spike at 0 in the clinical grayscale histogram comes from. Body site, skin type and photo production all change colour before any diagnosis does. The clinics and cameras cannot be checked directly — `attribution` is identical for every image — which is itself a limitation.
- **Is colour alone a useful signal? Weak, and partly confounded.** No colour feature explains more than 15 % of its variance by class (η² ≤ 0.15; ≤ 0.12 in dermoscopy). A logistic regression on all nine colour statistics reaches a **balanced accuracy of 0.47 on dermoscopic and 0.56 on clinical images** (5-fold CV, chance 0.33) — better than chance, far from useful, and strongest exactly where the site/skin-type confounding is strongest. The classes overlap heavily; the project needs a model that learns **spatial structure** (borders, texture, pigment networks), i.e. a CNN, with metadata such as age and site added carefully rather than colour thresholds.

## Parts 3–5 — Reusable pipeline

<img src="figures/fig22_raw_vs_processed.png">

- **`process_image(source, size=224, color_space="rgb"|"gray"|"hsv"|"lab", normalize="minmax"|"zscore"|"none", channels_first=False)`** accepts a path, an `isic_id`, a PIL image or an array and returns a `ProcessedImage`: the array plus `shape`, `dtype`, `value_range`, and `steps` (e.g. *resize 600×450 → 224×224 → rgb → minmax → HWC→CHW*). **Default normalisation is min–max with the encoding's fixed bounds (x / 255)**: it needs no fitted statistics, so it cannot leak across splits, and unlike per-image min–max it keeps the brightness differences Part 2 measured. Z-score uses the **training-split** mean/std from `artifacts/norm_stats.json` by default (or ImageNet's for a pretrained backbone).
- **`process_batch(paths | ids | DataFrame, **same options)`** never stops on one bad file: a truncated JPEG, a non-image and a missing id are each returned in `skipped` as `(id, "ExceptionType: message")` while the good images are stacked (tested in `tests/test_session2.py`).

<img src="figures/fig23_loader_batch.png">

```text
BatchLoader(metadata, img_dir=None, *, batch_size=32, label_col="diagnosis_1",
            classes=("Benign","Indeterminate","Malignant"), shuffle=False, seed=42,
            drop_last=False, channels_first=True, **process_image options)
len(loader)        -> batches per epoch        loader.n_missing / .missing_ids -> rows with no file on disk
for batch in loader:   # LoaderBatch, read and processed on the fly (memory = one batch)
    batch.images       # float32 (B, 3, 224, 224)       batch.labels      # int64 (B,) class indices
    batch.label_names  # ["Malignant", ...]             batch.ids         # isic_id per image
    batch.skipped      # [(isic_id, reason), ...] for files that failed in this batch
```

<img src="figures/fig24_batch_summary.png">

<img src="figures/fig25_class_balance.png">

- **Loader and visualiser in use.** The grid above is a real batch, drawn through `viz.to_displayable`, which undoes z-scoring and reorders CHW→HWC before display. The batch summary confirms z-scoring worked (mean −0.19, std 1.05) and exposes red pixels piling up at 255 (z ≈ 2.46): saturation worth knowing about before augmentation. **Class balance is severe (30:1 at image and lesion level)**, so later training needs lesion-grouped stratified splits, class weights or a weighted sampler, and evaluation by balanced accuracy / per-class recall rather than accuracy, which a model predicting "Malignant" for everything would already score at 69 %.
- **Design choices.** Each part is one layer that calls the one below instead of copying it. `process_image` reuses the project's existing resize (`preprocessing.preprocess_image`, pixel-identical — a test enforces it) so analysis and training can never preprocess differently; `process_batch` only adds error handling around it; `BatchLoader` only adds the availability filter (one directory listing, not a file check per row), labels, shuffling and batching around `process_batch`; and every function in `viz` accepts exactly what the loader yields. Every setting is a keyword argument with a documented default, and every result carries a record of what was done, so later milestones can call these functions rather than re-derive them. The torch `DataLoader` in `datasets.py` stays the training path (augmentation, workers); this numpy loader is for analysis and debugging.
