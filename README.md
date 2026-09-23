# MILK10k — Skin Lesion Classification

**Computer Vision and Speech Recognition · EADA Business School**
Individual course project — Stanislaus Lattorff

Sessions 1–3 homework (**Part A**) and the course project **Milestone 1: data pipeline**
(**Part B**), on the real [MILK10k](https://api.isic-archive.com/doi/milk10k/) dataset.

> **Submission index: [`SUBMISSION.md`](SUBMISSION.md)** — direct links to every deliverable.

---

## 1. What this project is

### The task

Given photographs of a skin lesion, predict **`diagnosis_1` — Benign / Indeterminate /
Malignant**. The stretch target is the finer **11-class** diagnosis (BCC, NV, BKL, SCCKA,
MEL, AKIEC, DF, INF, VASC, BEN_OTH, MAL_OTH).

**Input:** the two photographs of one lesion — a **dermoscopic** image (polarised light,
magnified, sub-surface structures visible) and a **clinical close-up** (what the naked eye
sees). Metadata (age, sex, body site) is available but is *not* used as a model input in
Milestone 1; the columns that would leak the label are listed in §5 and excluded by name.

**Output:** one prediction **per lesion**, not per image. The two images are two views of
the same object, so the lesion is the unit of prediction *and* the unit of evaluation.

**Who uses it, and when:** a triage aid at the point where a clinician is already looking
at a suspicious lesion and deciding whether to biopsy it. It does not screen the general
population — see the honest limits in §6.

### The dataset

| | |
|---|---|
| Source | [ISIC Archive — MILK10k](https://api.isic-archive.com/doi/milk10k/) |
| Paper | <https://doi.org/10.1016/j.jid.2025.06.1594> |
| Licence | **CC BY-NC** — attribution: *MILK study team* |
| Size | **10,480 images = 5,240 lesions × 2 images** (~345 MB) |
| Resolution | every image is exactly **600 × 450**, RGB JPEG |
| Labels | `diagnosis_1` (3 classes) per image; 11-class one-hot per lesion |

The dataset is **not redistributed here** — `scripts/download_data.sh` fetches it from the
official source. See [`data/README.md`](data/README.md).

### The goal of Milestone 1

Turn the Session 1–2 exploratory code into a **leak-free, reproducible data pipeline**:
verified integrity, a documented label strategy, grouped/stratified splits with a fixed
seed, train-only statistics, a justified augmentation pipeline, and a Dataset/DataLoader
that later milestones train against. **No model is trained in Milestone 1.** The
deliverable is a pipeline you can trust.

---

## 2. Setup and reproduction

**Python 3.14** (any 3.11+ works). Verified with pandas 3.0.5, numpy 2.5.3, torch 2.14.0,
torchvision 0.29.0, scikit-learn 1.9.1, Pillow 12.3.

```bash
# 1. Environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Data (~345 MB, CC-BY-NC, not committed)
bash scripts/download_data.sh
```

### Where the data lives, and how the code finds it

**There are no absolute paths anywhere in this repo.** Paths come from exactly one place,
[`src/milk10k/config.py`](src/milk10k/config.py):

- By default the code looks in `data/milk10k/` inside the repo.
- To point it somewhere else, set **one environment variable** — no code change:

```bash
export MILK10K_IMAGES_DIR=/path/to/your/milk10k/images
```

`config.py` is also the single place holding the **seed (42)**, the **input resolution
(224)**, the split proportions, the normalisation statistics and every output path.

### Commands that reproduce everything

```bash
source .venv/bin/activate

# --- Part B: the pipeline (run in this order; each step is independent and re-runnable)
python scripts/run_integrity_check.py     # B1  full 10,480-image verification
python scripts/run_quality_report.py      # B4  data-quality report
python scripts/make_splits.py             # B5  train/val/test, seed 42 -> artifacts/splits/
python scripts/verify_findings.py         # B2  re-check earlier EDA findings on the full data
python scripts/make_figures.py            # B2+B7+B8 all Milestone 1 figures
python scripts/run_pipeline.py            # B8  DataLoaders + sanity checks
python scripts/build_report_pdf.py        # B9  render the report, enforce the 2-page limit

# --- Tests
PYTHONPATH=src python -m pytest tests/ -v

# --- Part A: the homework notebook
python scripts/build_part_a.py                                  # assemble the notebook
python scripts/export_notebook_pdf.py notebooks/homework_part_a.ipynb   # run + export PDF
```

`make_splits.py` is the only script that writes the committed split files. Re-running it
with the same seed reproduces them byte for byte.

---

## 3. Repository structure

```
Computer_Vision/
├── README.md                          # this file
├── SUBMISSION.md                      # index of every deliverable, with links
├── requirements.txt                   # pinned dependencies
├── .gitignore                         # excludes data/ (345 MB) and .venv/
│
├── src/milk10k/                       # ← ALL reusable logic. Imported, never copy-pasted.
│   ├── config.py                      # THE config point: paths, seed, image size, label schemes
│   ├── data.py                        # loaders, table joins, lesion table, integrity checks
│   ├── labels.py                      # label strategy, label_map.json, encoding
│   ├── preprocessing.py               # resize/normalise (single + batch), fail-loud image loading
│   ├── quality.py                     # full-dataset verification + data-quality report
│   ├── splits.py                      # split_lesions(), verification, strategy comparison
│   ├── transforms.py                  # train_transform / eval_transform + augmentation table
│   ├── imbalance.py                   # class weights + WeightedRandomSampler
│   ├── datasets.py                    # MilkImageDataset, LesionDataset, DataLoaders
│   └── plots.py                       # every figure
│
├── scripts/                           # thin CLI entry points — orchestrate, contain no logic
│   ├── download_data.sh               # fetch + unzip the dataset
│   ├── run_eda.py                     # Session 1 figures/tables
│   ├── run_integrity_check.py         # B1
│   ├── run_quality_report.py          # B4
│   ├── make_splits.py                 # B5 — writes the committed splits
│   ├── verify_findings.py             # B2 — earlier findings re-checked on the full dataset
│   ├── make_figures.py                # B2 + B7 + B8 figures
│   ├── run_pipeline.py                # B8 DataLoader sanity checks
│   ├── build_report_pdf.py            # B9 report -> 2-page PDF (checks the page limit)
│   ├── build_notebook.py              # generates the Session 1 notebook
│   ├── build_part_a.py                # assembles the Part A notebook from notebooks/parts/
│   ├── run_part.py                    # runs ONE Part A section (fast dev loop)
│   └── export_notebook_pdf.py         # execute a notebook -> HTML -> PDF
│
├── tests/                             # the pipeline's guard rails
│   ├── test_splits.py                 # A3.2 property test: disjoint, sized, reproducible
│   ├── test_transforms.py             # B7: eval_transform is deterministic
│   ├── test_datasets.py               # A3.6: batch shapes, ids, class counts
│   └── test_integration.py            # end-to-end: tables -> splits -> loaders
│
├── notebooks/
│   ├── 01_eda_milk10k.ipynb           # Session 1 EDA (executed, with outputs)
│   ├── homework_part_a.ipynb          # ← PART A DELIVERABLE (executed, with outputs)
│   └── parts/                         # one source file per exercise; build_part_a.py assembles
│
├── artifacts/                         # GENERATED, machine-readable, COMMITTED (small)
│   ├── splits/{train,val,test}.csv    # the splits — seed 42 (leaky columns stripped)
│   ├── splits/split_manifest.json     # seed, date, sizes, which columns were dropped
│   ├── label_map.json                 # B3 label strategy
│   ├── class_weights.json             # B8 class weights (TRAIN only)
│   ├── norm_stats.json                # channel statistics (TRAIN only)
│   └── tables/                        # lesions.csv, image_size_summary.csv, quality + findings tables
│
├── reports/                           # GENERATED, human-readable
│   ├── figures/                       # Session 1 figures
│   │   └── milestone1/                # Milestone 1 figures (B2, B7)
│   ├── data_quality_report.md         # B4
│   ├── findings_check.md              # B2 — "finding -> still true? -> consequence"
│   ├── milestone1_report.md / .pdf    # B9
│   └── homework_part_a.pdf            # ← PART A PDF DELIVERABLE
│
├── docs/
│   ├── clinical_task.md               # what we are solving and why
│   ├── label_strategy.md              # B3 decision, in full
│   └── preprocessing_decisions.md     # B6 resolution + view handling, with evidence
│
└── data/                              # NOT COMMITTED (see data/README.md)
    └── milk10k/{metadata.csv, images/, supplements/}
```

### Why it is organised this way

Four directories, four different lifetimes, and that is the whole rule:

1. **`src/milk10k/` is the only place logic lives.** If a function is used twice, it goes
   here and gets imported. The notebooks import the *same* `split_lesions` the pipeline
   scripts use, so the homework and the project cannot silently disagree. This is why
   there is a package and not a pile of notebooks: a notebook cell copied into a second
   notebook is a bug waiting to happen, and the brief grades exactly that.
2. **`scripts/` is thin.** Every script is an entry point that parses arguments, calls the
   package and prints. No script contains analysis logic — move a script to a different
   repo and it breaks; move the package and everything still works.
3. **`artifacts/` vs `reports/` split on *who reads it*.** `artifacts/` holds what **later
   code loads** (splits, `label_map.json`, `class_weights.json`) — stable file names, no
   prose. `reports/` holds what a **human reads** (figures, the quality report, the
   milestone report). Both are generated and both are committed, but conflating them is
   how a project ends up with a figure that some training script secretly depends on.
4. **`tests/` exists because the dangerous failures here are silent.** A leaked split or a
   non-deterministic eval transform does not raise — it just quietly inflates a number.
   The tests assert the properties that have no visible symptom.

`notebooks/parts/` deserves a note: the Part A notebook is **generated** from one source
file per exercise rather than hand-edited. That keeps each exercise independently runnable
(`python scripts/run_part.py a3_1`), keeps the diffs readable, and means the notebook can
always be rebuilt from source instead of drifting into an unreproducible state.

---

## 4. Data handling rules

| | |
|---|---|
| **Committed** | all source code, the three split CSVs, the JSON artifacts, the figures, the reports, the notebooks with their outputs |
| **Not committed** | `data/` (the 345 MB image set and the CSVs — CC-BY-NC, fetched by script), `.venv/`, `__pycache__/`, generated HTML |
| **Generated files go to** | `artifacts/` (machine-readable) and `reports/` (human-readable). Nothing is generated into `src/`. |
| **Seed** | **42**, from `config.SEED`. Used by every split, sampler and torch generator. |
| **Splits created** | **2026-09-22** (`config.SPLIT_DATE`), by `scripts/make_splits.py` |

The raw images are excluded via `data/*` in `.gitignore`. The split CSVs are small and are
**deliberately committed** — they are the thing that makes a later result reproducible.

The split CSVs carry **no label-leaking columns**: `splits.write_splits()` strips everything in
`config.LEAKY_COLUMNS` (keeping only `lesion_id` as the grouping key and the two targets) and
records what it removed in `split_manifest.json`. The exclusion is enforced in code, not left
to whoever writes the training loop.

---

## 5. Key decisions so far

Full reasoning is in [`docs/label_strategy.md`](docs/label_strategy.md) and the
[Milestone 1 report](reports/milestone1_report.md). The short version:

| Decision | Choice | Why |
|---|---|---|
| **Label strategy** | Keep all **3** `diagnosis_1` classes; keep all **11** stretch classes | "Indeterminate" (123 lesions) means *the pathologist could not rule out malignancy* — merging it into Benign teaches the model to reassure on exactly the hard cases. Merging the rare 11-class tail into "other" would mix malignant MAL_OTH with benign VASC/DF and destroy the confusion matrix. |
| **A trap worth knowing** | `diagnosis_1` is **not** derivable from the 11-class code | **AKIEC straddles** Indeterminate (123) and Malignant (180). All 123 Indeterminate lesions are AKIEC. The two schemes are separate targets. |
| **Split design** | `StratifiedGroupKFold`, grouped on `lesion_id`, stratified on the 11-class label, **70/15/15**, seed 42 | A naive image-level split puts **89% of test lesions in train** (measured). Splitting at lesion level makes that structurally impossible. Stratifying on the 11-class label also balances `diagnosis_1`, since 10 of 11 codes map cleanly. |
| **Preprocessing** | **224 × 224** | Every image is 600 × 450, so **100% are ≥ 256 on both sides** — 224 downscales uniformly with no upsampling anywhere, and matches the pretrained backbones Milestone 2 starts from. |
| **View handling** | **Train** on each view as its own sample; **evaluate** per lesion by averaging the two views | Training per image doubles the effective training set and is safe *only because* the split is grouped on `lesion_id`. Evaluation is per lesion because the lesion is the clinical unit — scoring per image double-counts every lesion and gives half credit on lesions the model is inconsistent about. See [`docs/preprocessing_decisions.md`](docs/preprocessing_decisions.md). |
| **Normalisation** | ImageNet statistics; train-only channel statistics also computed and stored | Milestone 2 fine-tunes a pretrained backbone. Computing statistics over the full dataset is preprocessing leakage, so `norm_stats.json` is train-only. |
| **Imbalance** | Class weights **and** a `WeightedRandomSampler`, both from **train only** | On **train**: **252.3:1** on the 11-class scheme, **28.9:1** on `diagnosis_1` (280:1 and 29.5:1 on the full dataset). Both mechanisms are implemented; the trade-off is documented rather than assumed. |
| **Excluded inputs** | `diagnosis_1/2/3/4`, `diagnosis_confirm_type`, `diagnosis_full`, `invasion_thickness_interval`, `melanocytic`, `concomitant_biopsy`, `lesion_id` | They encode the label or are only knowable *after* the biopsy that produced it. `diagnosis_confirm_type` is the loudest: histopathology rows are 72% Malignant, clinical-assessment rows 2%. |

---

## 6. The limit a clinician must know

MILK10k is **biopsy-enriched**, not population-representative: **69.4% of its lesions are
malignant**, because the dataset is made of lesions a specialist was already worried enough
to biopsy. In a real skin clinic the malignant rate is a small fraction of that.

So a model trained here answers a **narrow** question — *"given that a dermatologist has
already decided this lesion needs a biopsy, is it cancer?"* — and **not** *"is this mole on
your arm dangerous?"*. Accuracy measured on this dataset would collapse if the model were
deployed as a general screening tool, because the prior it learned is wrong for that
population. Any performance number from this project must be quoted together with the
population it was measured on.

---

## 7. Course roadmap

- [x] **Session 1** — EDA, project structure, clinical task definition
- [x] **Sessions 1–3 homework (Part A)** — integrity checks, pixels, leakage, splits, augmentation audit
- [x] **Milestone 1 (Part B)** — leak-free, reproducible data pipeline
- [ ] Milestone 2 — baselines and transfer learning
- [ ] Milestone 3 — evaluation, calibration, error analysis
- [ ] Final — shareable application + MILK10k leaderboard submission

---

## 8. Licence and attribution

MILK10k is published by the **MILK study team** via the ISIC Archive under **CC BY-NC**.
It is not redistributed in this repository.

- Dataset: <https://api.isic-archive.com/doi/milk10k/>
- Paper: <https://doi.org/10.1016/j.jid.2025.06.1594>
