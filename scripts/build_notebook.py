"""Generate notebooks/01_eda_milk10k.ipynb from a single source of truth.

Keeping the notebook generated (rather than hand-edited) means the narrative and
the code stay in sync, and the notebook can always be rebuilt and re-executed.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_eda_milk10k.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip()}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n"),
    }


CELLS = [
    md("""
# MILK10k — Exploratory Data Analysis

**Computer Vision and Speech Recognition · Session 1 homework**

This notebook is the first milestone of the MILK10k course project. Before training
anything, we look at what is actually in the dataset: how many images and lesions
there are, how the labels are distributed, who the patients are, what the images
look like, and which of these facts will break a naive pipeline.

**The clinical task.** MILK10k contains photographs of skin lesions that a
dermatologist judged suspicious enough to look at closely. Each lesion was imaged
twice — once through a **dermatoscope** (polarised light, high magnification, sub-surface
structures visible) and once as a **clinical close-up** (what the naked eye sees).
Most were then biopsied and read by a pathologist, which is where the labels come from.

We want to predict, from the images, whether a lesion is **Benign**, **Malignant** or
**Indeterminate** (`diagnosis_1`), with an optional finer 11-class diagnosis as a
stretch goal. The point is triage: skin cancer is highly curable when caught early and
often lethal when missed, so a useful model is one that **rarely misses a malignant
lesion**, even at the cost of some false alarms. A full write-up is in
[`docs/clinical_task.md`](../docs/clinical_task.md).
"""),
    md("""
## 0. Setup

All reusable code lives in `src/milk10k/` so later sessions can import it rather than
copy-pasting cells. The notebook only orchestrates and narrates.
"""),
    code("""
import sys
from pathlib import Path

# Make the project's `src/` importable from the notebook.
PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from milk10k import config, data, plots

config.apply_style()
config.ensure_dirs()

print("pandas", pd.__version__, "| numpy", np.__version__)
print("Dataset directory:", config.DATA_DIR)
print("Dataset present:", config.METADATA_CSV.exists())
"""),
    md("""
> If `Dataset present` is `False`, run `bash scripts/download_data.sh` from the project
> root first. The data (~345 MB, CC-BY-NC) is deliberately not committed to git.
"""),
    md("""
## 1. Loading the four tables

MILK10k ships **two different row units**, and confusing them is the single easiest way
to ruin this project:

| File | Row unit | Rows |
|---|---|---|
| `metadata.csv` | one **image** | 10,480 |
| `supplements/training_gt.csv` | one **lesion** | 5,240 |
| `supplements/training_input.csv` | one **image** | 10,480 |
| `supplements/training_supp.csv` | one **image** | 10,480 |

That 2:1 ratio is not an accident — it is the two-images-per-lesion structure.
"""),
    code("""
meta = data.load_metadata()          # one row per image
gt = data.load_training_gt()         # one row per lesion, 11-class one-hot
inp = data.load_training_input()     # one row per image, MONET concepts + skin tone
supp = data.load_training_supp()     # one row per image, full-text diagnosis

for name, frame in [("metadata", meta), ("training_gt", gt),
                    ("training_input", inp), ("training_supp", supp)]:
    print(f"{name:16s} {frame.shape[0]:>6,} rows x {frame.shape[1]:>2} cols")

print(f"\\nunique images : {meta['isic_id'].nunique():,}")
print(f"unique lesions: {meta['lesion_id'].nunique():,}")
print(f"images/lesion : {len(meta) / meta['lesion_id'].nunique():.0f}")
"""),
    code("""
meta.head(3)
"""),
    md("""
### What the columns mean

`metadata.csv` carries a four-level diagnosis hierarchy (`diagnosis_1` … `diagnosis_4`),
going from the coarse triage label down to a specific subtype.
"""),
    code("""
meta[["isic_id", "lesion_id", "image_type",
      "diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_4"]].head(5)
"""),
    md("""
## 2. Building the two analysis views

`build_image_level()` joins the 11-class lesion label and the per-image MONET scores onto
the 10,480-row image grain. `build_lesion_level()` then collapses to the 5,240 lesions,
which is the grain any honest train/test split has to work on.
"""),
    code("""
df = data.build_image_level(meta, gt, inp)      # 10,480 rows — one per image
lesion = data.build_lesion_level(df)            # 5,240 rows — one per lesion

print("image level :", df.shape)
print("lesion level:", lesion.shape)
df[["isic_id", "lesion_id", "image_type", "diagnosis_1",
    "label_11", "skin_tone_class"]].head()
"""),
    md("""
## 3. Do not trust the structure — check it

Everything downstream rests on a few structural claims. Verify them from the data
rather than believing the slides.
"""),
    code("""
checks = data.check_lesion_consistency(df)
checks
"""),
    code("""
# Is the 11-class ground truth really single-label?
labels = data.gt_to_label(gt)
print(labels["n_positive"].value_counts().to_string())
print(f"\\nLesions without exactly one positive class: {(labels['n_positive'] != 1).sum()}")
"""),
    md("""
Every lesion has exactly two images, one of each modality; `isic_id` is unique; the
demographic and label columns are constant within a lesion; and the 11-class target is
genuinely single-label. **The structure holds.**
"""),
    md("""
## 4. Class distribution — the headline problem

This is the plot to draw before anything else.
"""),
    code("""
primary = lesion["diagnosis_1"].value_counts()
primary_pct = (primary / primary.sum() * 100).round(1)
pd.DataFrame({"lesions": primary, "share_%": primary_pct})
"""),
    code("""
plots.plot_primary_class_distribution(lesion, save_as="fig01_class_distribution_primary.png")
plt.show()
"""),
    md("""
**This is the opposite of what people expect.** Most skin-lesion datasets are dominated by
benign nevi. MILK10k is dominated by *malignant* lesions, because it samples lesions a
dermatologist already found suspicious enough to biopsy — a **selection bias baked into the
data by design**.

The practical consequence: a model that outputs "Malignant" unconditionally scores
~69% accuracy while being clinically worthless. Accuracy is the wrong headline metric here.
"""),
    code("""
majority = primary.max() / primary.sum()
print(f"Always-predict-'{primary.idxmax()}' baseline accuracy: {majority:.1%}")
print(f"...its recall on Benign lesions:                    0.0%")
print(f"\\nImbalance ratio (largest:smallest) = {primary.max() / primary.min():.1f} : 1")
"""),
    code("""
eleven = lesion["label_11"].value_counts()
pd.DataFrame({
    "diagnosis": [config.DIAGNOSIS_CODES[c] for c in eleven.index],
    "lesions": eleven.values,
    "share_%": (eleven.values / eleven.sum() * 100).round(2),
}, index=eleven.index)
"""),
    code("""
plots.plot_11_class_distribution(lesion, save_as="fig02_class_distribution_11.png")
plt.show()
"""),
    md("""
The 11-class target is far harsher: **BCC has 2,522 lesions, MAL_OTH has 9** — a 280:1
imbalance. Four classes have fewer than 55 lesions, which after a test split leaves barely
a handful of examples. Any 11-class attempt will need class weighting, oversampling, or
merging the rare tail.
"""),
    md("""
### Are the two label schemes consistent?

The 3-class and 11-class targets should be nested. Checking rather than assuming turns up
a genuine subtlety.
"""),
    code("""
hierarchy = data.check_label_hierarchy(lesion)
hierarchy
"""),
    code("""
plots.plot_diagnosis_hierarchy(lesion, save_as="fig03_diagnosis_hierarchy.png")
plt.show()

print("Codes mapping to more than one diagnosis_1 value:", data.ambiguous_codes(lesion))
"""),
    md("""
Ten of the eleven codes map cleanly onto a single `diagnosis_1` value. **AKIEC does not** —
it splits into 123 *Indeterminate* and 180 *Malignant* lesions. That is clinically sensible
(actinic keratosis and intraepithelial carcinoma sit on a continuum rather than either side
of a line), but it means you **cannot** derive `diagnosis_1` from the 11-class code with a
lookup table, and the *Indeterminate* class is essentially "AKIEC we could not commit on".
"""),
    md("""
## 5. Missing values

Missingness here is structural, not random.
"""),
    code("""
missing = df.isna().mean().sort_values(ascending=False)
missing[missing > 0].to_frame("missing_share").style.format("{:.1%}")
"""),
    code("""
plots.plot_missing_values(df, save_as="fig04_missing_values.png")
plt.show()
"""),
    md("""
- `melanocytic` is missing 77% of the time — but it is never `False`, only `True` or blank.
  It is an *annotation*, not a boolean feature; treating blank as "not melanocytic" would be wrong.
- `anatom_site_general` is missing for 37% of images, so body site can only be used with an
  explicit "unknown" category.
- `diagnosis_4` is missing 86% of the time simply because most diagnoses do not go four
  levels deep.
"""),
    md("""
## 6. Who is in this dataset?

Demographics matter because a model inherits whatever population it was trained on.
"""),
    code("""
lesion[["age_approx"]].describe().T
"""),
    code("""
plots.plot_demographics(lesion, save_as="fig05_demographics.png")
plt.show()
"""),
    code("""
pd.crosstab(lesion["sex"], lesion["diagnosis_1"], normalize="index").style.format("{:.1%}")
"""),
    md("""
Median age is 65 and 60% of lesions come from men — an older, male-skewed population,
consistent with a specialist skin-cancer clinic rather than the general public.
Malignancy rate is visibly higher in men.
"""),
    code("""
plots.plot_anatomic_site(lesion, save_as="fig06_anatomic_site.png")
plt.show()
"""),
    md("""
Head/neck is both the most common site and one of the most malignant — sun-exposed skin.
Note the large `(missing)` bar: site is unavailable for a substantial share of lesions.
"""),
    md("""
## 7. Skin tone — the fairness slice

This is the check that dermatology AI has historically failed, and it is the reason
MILK10k ships a skin tone class at all.
"""),
    code("""
tone = lesion["skin_tone_class"].value_counts().sort_index()
pd.DataFrame({
    "skin_tone": [config.SKIN_TONE_LABELS.get(int(i), i) for i in tone.index],
    "lesions": tone.values,
    "share_%": (tone.values / tone.sum() * 100).round(1),
}).set_index("skin_tone")
"""),
    code("""
plots.plot_skin_tone(lesion, save_as="fig07_skin_tone.png")
plt.show()
"""),
    md("""
**61% of lesions are a single skin tone class (III)**, and the lightest and darkest tones
together are under 10%. Two consequences:

1. Aggregate metrics will be dominated by type III/IV skin. Per-group recall must be
   reported separately, or failures on under-represented tones stay invisible.
2. The apparent trend of higher malignancy in darker classes is **confounded**, not causal —
   it reflects who gets referred and which body sites get imaged, not skin tone itself.
   With only 105 type-I lesions, the leftmost bars are noisy.
"""),
    md("""
## 8. The images themselves

Metadata is only half the story. Open the actual JPEGs and measure them.
"""),
    code("""
# Reading all 10,480 files is slow; 400 is plenty for a first estimate.
stats = data.sample_image_stats(df, n=400, seed=0)
stats[["width", "height", "megapixels", "file_size_kb", "mean_intensity"]].describe().T.round(1)
"""),
    code("""
stats.groupby("image_type").agg(
    n=("isic_id", "count"),
    mean_width=("width", "mean"),
    mean_height=("height", "mean"),
    mean_megapixels=("megapixels", "mean"),
    mean_file_kb=("file_size_kb", "mean"),
).round(1)
"""),
    code("""
print("Distinct (width, height) pairs in the sample:", stats.groupby(["width", "height"]).ngroups)
print("Width range :", stats["width"].min(), "-", stats["width"].max())
print("Height range:", stats["height"].min(), "-", stats["height"].max())
"""),
    code("""
plots.plot_image_properties(stats, save_as="fig08_image_properties.png")
plt.show()
"""),
    md("""
Images are **not** a uniform size, so a resize / crop step is mandatory before batching.
"""),
    code("""
plots.plot_color_statistics(stats, save_as="fig09_color_statistics.png")
plt.show()
"""),
    md("""
Dermoscopic and clinical images have measurably different colour and contrast distributions.
They are two modalities, not two samples of one. Mixing them blindly in a single batch means
the model spends capacity learning "which camera took this" before it learns "is this cancer".
"""),
    md("""
### Looking at one image as numbers

The bridge from "a picture" to "a matrix a model can use".
"""),
    code("""
row = df.iloc[0]
img = data.load_image(row["isic_id"])
arr = np.array(img)

print("isic_id   :", row["isic_id"])
print("diagnosis :", row["diagnosis_1"], "/", row["diagnosis_2"])
print("image_type:", row["image_type"])
print("shape     :", arr.shape, "(height, width, channels)")
print("dtype     :", arr.dtype)
print("min / max :", arr.min(), "/", arr.max())

fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.imshow(img)
ax.set_title(f"{row['isic_id']} — {row['diagnosis_1']}", fontsize=10)
ax.axis("off"); ax.grid(False)
plt.show()
"""),
    md("""
### One example per diagnosis class
"""),
    code("""
plots.plot_class_gallery(df, save_as="fig10_class_gallery.png")
plt.show()
"""),
    md("""
Note how little separates some of these visually. BCC (the majority class) can be a
barely-pigmented pink patch, while a benign nevus is often the *more* dramatic-looking
lesion. This is why the task is hard and why colour/size heuristics fail.
"""),
    md("""
### Two images, one lesion

The most consequential structural fact in the dataset.
"""),
    code("""
plots.plot_lesion_pairs(df, n_lesions=3, save_as="fig11_lesion_pairs.png")
plt.show()
"""),
    md("""
Same lesion, same label, two very different images. If a random **row** split puts the
dermoscopic image in train and the clinical image in test, the model has effectively seen the
test case. Scores go up; real performance does not. **Split on `lesion_id`.**
"""),
    code("""
plots.show_samples(df, n=6, diagnosis="Malignant", save_as="fig12_samples_malignant.png")
plt.show()
plots.show_samples(df, n=6, diagnosis="Benign", save_as="fig13_samples_benign.png")
plt.show()
"""),
    md("""
## 9. MONET concept scores

`training_input.csv` ships seven per-image scores from MONET, an image–text foundation
model. Each score says how strongly a human-readable visual concept ("hair", "ulceration",
"pigmented") is present. They are useful both as tabular features and as an interpretability
handle.
"""),
    code("""
monet_cols = [c for c in df.columns if c.startswith("MONET_")]
df.groupby("image_type")[monet_cols].mean().T.round(3)
"""),
    code("""
plots.plot_monet_by_image_type(df, save_as="fig14_monet_by_image_type.png")
plt.show()
"""),
    md("""
`vasculature_vessels` averages 0.22 on dermoscopic images against 0.02 on clinical ones —
exactly what a dermatoscope is *for*: it makes sub-surface vessels visible. The
`gel_water_drop` concept behaves the same way, because immersion fluid is part of the
dermoscopic procedure. These scores are modality-dependent and must not be pooled naively.
"""),
    code("""
plots.plot_monet_by_class(df, save_as="fig15_monet_by_class.png")
plt.show()
"""),
    md("""
Read row by row: VASC scores highest on `vasculature_vessels`, INF on `erythema`,
NV on `pigmented`. The concepts behave the way a dermatologist would expect, which is a
good sign that they carry real signal rather than noise.
"""),
    md("""
## 10. What the EDA tells us to do next

| # | Finding | Consequence for the pipeline |
|---|---|---|
| 1 | Every lesion has exactly 2 images | Split on `lesion_id` with a grouped split. A random row split leaks every lesion. |
| 2 | 69.4% of lesions are Malignant | Never report bare accuracy. Use balanced accuracy, macro-F1, per-class recall, and malignant sensitivity. |
| 3 | 11-class imbalance is 280:1 | Class weights / oversampling, or merge the rare tail. Report per-class recall. |
| 4 | Two imaging modalities with different pixel statistics | Treat `image_type` explicitly — stratify, condition, or use two heads. |
| 5 | AKIEC straddles Indeterminate and Malignant | The 3-class target is not derivable from the 11-class code. Pick one target and be explicit. |
| 6 | Site missing 37%, melanocytic missing 77% | Metadata features need an explicit "unknown" category, not imputation-by-mode. |
| 7 | 61% of lesions are one skin tone class | Report per-skin-tone metrics. Aggregate numbers will hide group failures. |
| 8 | Images vary in size | Resize/crop is mandatory; decide and document the target resolution. |
| 9 | Dataset is biopsy-referred, not population-representative | A model here estimates "malignant *given a specialist was already worried*" — state this in any claim. |

**Next session:** image representation and normalisation, then the first baseline —
with a grouped split and malignant recall as the metric that counts.
"""),
]

# Stable cell ids keep nbformat happy and make diffs readable.
for i, cell in enumerate(CELLS):
    cell["id"] = f"cell-{i:02d}"

notebook = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1))
print(f"Wrote {OUT} ({len(CELLS)} cells)")
