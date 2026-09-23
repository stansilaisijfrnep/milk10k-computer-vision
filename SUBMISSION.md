# Submission — Sessions 1–3 Homework + Course Project Milestone 1

**Name:** Stanislaus Lattorff
**Course:** Computer Vision and Speech Recognition · EADA Business School
**Repository:** <https://github.com/stansilaisijfrnep/milk10k-computer-vision>
**Dataset:** MILK10k (10,480 images / 5,240 lesions) — not committed, see [`data/README.md`](data/README.md)

> Start here: [`README.md`](README.md) — what the project is, how to run it, and why the repo
> is laid out the way it is.

---

## Part A — the homework notebook

| Deliverable | File |
|---|---|
| **Notebook** (runs top to bottom, outputs saved) | [`notebooks/homework_part_a.ipynb`](notebooks/homework_part_a.ipynb) |
| **Exported PDF** | [`reports/homework_part_a.pdf`](reports/homework_part_a.pdf) |
| Notebook source (one file per exercise) | [`notebooks/parts/`](notebooks/parts/) |
| Assembler / single-section runner | [`scripts/build_part_a.py`](scripts/build_part_a.py) · [`scripts/run_part.py`](scripts/run_part.py) |

### Exercise index

| Ex. | Title | Source |
|---|---|---|
| A1.1 | Cross-check the two label files | [`a1_1.py`](notebooks/parts/a1_1.py) |
| A1.2 | Ask the data five questions | [`a1_2.py`](notebooks/parts/a1_2.py) |
| A1.3 | Build a lesion-level table | [`a1_3.py`](notebooks/parts/a1_3.py) |
| A1.4 | Task framing (max 200 words) | [`a1_4.py`](notebooks/parts/a1_4.py) |
| A2.1 | Arrays, views and memory | [`a2_1.py`](notebooks/parts/a2_1.py) |
| A2.2 | JPEG compression and the file-size shortcut | [`a2_2.py`](notebooks/parts/a2_2.py) |
| A3.1 | Measure the leak | [`a3_1.py`](notebooks/parts/a3_1.py) |
| A3.2 | A reusable three-way split + property test | [`a3_2.py`](notebooks/parts/a3_2.py) |
| A3.3 | Why StratifiedGroupKFold? | [`a3_3.py`](notebooks/parts/a3_3.py) |
| A3.4 | Metrics and the cost of errors | [`a3_4.py`](notebooks/parts/a3_4.py) |
| A3.5 | Is this augmentation label-safe? | [`a3_5.py`](notebooks/parts/a3_5.py) |
| A3.6 | A lesion-level Dataset | [`a3_6.py`](notebooks/parts/a3_6.py) |

---

## Part B — Milestone 1: the data pipeline

### B0 · Repository structure and README
- [`README.md`](README.md) — task, dataset, setup, commands, folder tree with one line per
  entry, and the paragraph explaining **why** it is organised that way.
- [`requirements.txt`](requirements.txt)

### B1 · Full-dataset integrity check
- [`scripts/run_integrity_check.py`](scripts/run_integrity_check.py)
- [`src/milk10k/quality.py`](src/milk10k/quality.py) — `verify_all_images()`, `image_size_summary()`
- [`artifacts/tables/image_size_summary.csv`](artifacts/tables/image_size_summary.csv)
- **Result:** 10,480/10,480 files present and verified, **0 missing, 0 unreadable**, a single
  resolution (600×450). Missing files raise `FileNotFoundError` — they are never skipped.

### B2 · Label-centred EDA
- [`reports/figures/milestone1/b2_class_distribution.png`](reports/figures/milestone1/b2_class_distribution.png) — both schemes, log-scale 11-class panel
- [`reports/figures/milestone1/b2_label_mapping.png`](reports/figures/milestone1/b2_label_mapping.png) — the 11-class → `diagnosis_1` mapping
- [`reports/figures/milestone1/b2_class_gallery.png`](reports/figures/milestone1/b2_class_gallery.png) — the 3×4 gallery
- [`reports/findings_check.md`](reports/findings_check.md) — "earlier finding → still true on the full dataset? → consequence", 10 rows (8 verified, 1 nuanced, **1 refuted**)
- [`scripts/verify_findings.py`](scripts/verify_findings.py)

### B3 · Label strategy
- [`docs/label_strategy.md`](docs/label_strategy.md) — the decision, in full
- [`artifacts/label_map.json`](artifacts/label_map.json) — the mapping every later milestone loads
- [`src/milk10k/labels.py`](src/milk10k/labels.py)

### B4 · Data-quality report
- [`reports/data_quality_report.md`](reports/data_quality_report.md) · [`artifacts/tables/data_quality_report.csv`](artifacts/tables/data_quality_report.csv)
- [`scripts/run_quality_report.py`](scripts/run_quality_report.py)
- **Result:** 12/12 label-consistency checks pass; a written handling decision for each of the
  6 columns with missing values; shortcut cross-tabs; the excluded-columns list.

### B5 · Splits
- [`scripts/make_splits.py`](scripts/make_splits.py) · [`src/milk10k/splits.py`](src/milk10k/splits.py)
- [`artifacts/splits/train.csv`](artifacts/splits/train.csv) · [`val.csv`](artifacts/splits/val.csv) · [`test.csv`](artifacts/splits/test.csv) · [`split_manifest.json`](artifacts/splits/split_manifest.json)
- **Seed 42**, created **2026-09-22**. 3,667 / 786 / 787 lesions (70.0 / 15.0 / 15.0).
  Zero lesion overlap; every lesion keeps exactly 2 images; max class deviation 0.14 pp.

### B6 · Preprocessing decisions
- [`docs/preprocessing_decisions.md`](docs/preprocessing_decisions.md) — resolution and view-combination, with evidence
- [`artifacts/norm_stats.json`](artifacts/norm_stats.json) — **train-only** channel statistics

### B7 · Augmentation pipeline
- [`src/milk10k/transforms.py`](src/milk10k/transforms.py) — `train_transform` / `eval_transform` in an importable module
- [`reports/figures/milestone1/b7_augmentation_examples.png`](reports/figures/milestone1/b7_augmentation_examples.png) — 1 original + 7 augmented, for 3 classes including a rare one
- [`tests/test_transforms.py`](tests/test_transforms.py) — proves `eval_transform` is deterministic

### B8 · Dataset / DataLoader
- [`src/milk10k/datasets.py`](src/milk10k/datasets.py) — `MilkImageDataset`, `LesionDataset`, `build_dataloaders()`, `aggregate_predictions()`
- [`src/milk10k/imbalance.py`](src/milk10k/imbalance.py) — class weights **and** `WeightedRandomSampler`
- [`artifacts/class_weights.json`](artifacts/class_weights.json) — computed on **train only**
- [`scripts/run_pipeline.py`](scripts/run_pipeline.py) — the printed sanity checks
- [`reports/figures/milestone1/b8_transformed_batch.png`](reports/figures/milestone1/b8_transformed_batch.png) — 16 images after the transforms, labels as titles

### B9 · Milestone 1 report
- [`reports/milestone1_report.pdf`](reports/milestone1_report.pdf) — the printed version, **2 pages**
- [`reports/milestone1_report.md`](reports/milestone1_report.md) — the source (renders on GitHub)
- [`scripts/build_report_pdf.py`](scripts/build_report_pdf.py) — renders the PDF and fails if it exceeds 2 pages

---

## Session 2 homework — Extended EDA & data pipeline

| Deliverable | File |
|---|---|
| **Report (PDF, 4 pages)** | [`reports/Session2_MILK10k_EDA_Pipeline_Stanislaus_Lattorff.pdf`](reports/Session2_MILK10k_EDA_Pipeline_Stanislaus_Lattorff.pdf) |
| Report source (renders on GitHub) | [`reports/session2_report.md`](reports/session2_report.md) |
| Code — Parts 1–2 (metadata association, colour analysis) | [`src/milk10k/eda2.py`](src/milk10k/eda2.py) |
| Code — Parts 3–4 (`process_image`, `process_batch`, `BatchLoader`) | [`src/milk10k/pipeline.py`](src/milk10k/pipeline.py) |
| Code — Part 5 (grids, class balance, batch summary) | [`src/milk10k/viz.py`](src/milk10k/viz.py) |
| Runner / PDF builder | [`scripts/run_session2.py`](scripts/run_session2.py) · [`scripts/build_session2_pdf.py`](scripts/build_session2_pdf.py) |
| Figures · tables · numbers | `reports/figures/fig16`–`fig25` · `reports/tables/session2_*.csv` · [`reports/session2_summary.md`](reports/session2_summary.md) |

---

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

| File | Guards |
|---|---|
| [`tests/test_splits.py`](tests/test_splits.py) | splits are disjoint, correctly sized, reproducible (A3.2) |
| [`tests/test_transforms.py`](tests/test_transforms.py) | `eval_transform` determinism, train randomness, normalisation (B7) |
| [`tests/test_datasets.py`](tests/test_datasets.py) | batch shapes, ids, class counts, `aggregate_predictions` (A3.6) |
| [`tests/test_integration.py`](tests/test_integration.py) | end-to-end: tables → splits → weights → loaders |
| [`tests/test_session2.py`](tests/test_session2.py) | Session 2: preprocessing record, bad-file skipping, loader coverage, V / η² formulas |

---

## Notes for the oral check

- **No absolute paths.** Everything resolves through `src/milk10k/config.py`; the image folder
  can be moved with the `MILK10K_IMAGES_DIR` environment variable alone.
- **No copy-pasted logic.** The notebook and the pipeline scripts import the *same* functions
  from `src/milk10k/`. `scripts/` holds entry points only.
- **The raw images are not committed** (`data/` is in `.gitignore`). The split CSVs and the JSON
  artifacts are committed, because they are what makes a later result reproducible.
- **Reproducibility.** `python scripts/make_splits.py` with seed 42 regenerates the committed
  splits exactly.
