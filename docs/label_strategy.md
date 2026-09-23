# Label strategy (Milestone 1, B3)

The machine-readable version of everything on this page is
[`artifacts/label_map.json`](../artifacts/label_map.json), written by
`milk10k.labels.write_label_map()`. Every later milestone loads that file rather
than re-deciding.

---

## 1. Two label schemes, and why they are not interchangeable

MILK10k ships the label twice:

| Scheme | Column | Grain | Classes |
|---|---|---|---|
| **Primary** | `diagnosis_1` (metadata.csv) | per image, constant within a lesion | Benign / Indeterminate / Malignant |
| **Stretch** | 11-class one-hot (training_gt.csv) | per lesion | AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC |

The obvious assumption is that the 11-class code determines `diagnosis_1`, so you
could predict 11 classes and look up the coarse label. **That assumption is false**,
and checking it is exercise A1.1(c):

| 11-class code | Benign | Indeterminate | Malignant |
|---|---:|---:|---:|
| **AKIEC** | 0 | **123** | **180** |
| BCC | 0 | 0 | 2,522 |
| MEL | 0 | 0 | 450 |
| SCCKA | 0 | 0 | 473 |
| MAL_OTH | 0 | 0 | 9 |
| NV | 746 | 0 | 0 |
| BKL | 544 | 0 | 0 |
| DF | 52 | 0 | 0 |
| INF | 50 | 0 | 0 |
| VASC | 47 | 0 | 0 |
| BEN_OTH | 44 | 0 | 0 |

Ten of the eleven codes map cleanly. **AKIEC does not** — and it is not a labelling
bug. AKIEC bundles actinic keratosis with squamous cell carcinoma *in situ*: a
biological continuum where the pathologist's call of "pre-malignant" versus
"malignant *in situ*" is a genuine judgement, not a lookup. Every one of the
dataset's 123 Indeterminate lesions is an AKIEC.

**Consequence:** `diagnosis_1` cannot be derived from an 11-class prediction for
303 lesions (5.8% of the dataset), and those 303 are exactly the diagnostically
hardest ones. The two schemes are trained and evaluated as separate targets. The
project never converts one into the other.

---

## 2. Primary target — keep all three classes

**Decision: keep Benign / Indeterminate / Malignant as three classes.**

Counts per lesion: Malignant 3,634 (69.4%), Benign 1,483 (28.3%),
Indeterminate 123 (2.3%).

The alternatives and why they lose:

- **Merge Indeterminate into Benign.** This is the dangerous one. Indeterminate
  means the pathologist could not rule malignancy out. Folding it into Benign
  trains the model to produce a *reassuring* output on precisely the lesions
  where nobody was reassured. The label would no longer mean what its name says.
- **Merge Indeterminate into Malignant.** Defensible clinically — under a triage
  policy both get excised — but it destroys the ability to measure the model on
  the ambiguous cases separately, which is the thing a clinician would most want
  to know before trusting it.
- **Drop the 123 lesions.** Silently narrows the task to the easy part of the
  problem and inflates every metric. If the model never sees an AKIEC-as-
  Indeterminate, it will still be handed one in the clinic.

Keeping three classes costs us a 2.3% class. That cost is paid with class
weights and a `WeightedRandomSampler` (both computed on train only), not with
deletion. The honest caveat travels with the result: with 88 Indeterminate
lesions in train, 18 in val and 17 in test, per-class recall on Indeterminate
has a confidence interval wide enough to drive a truck through, and it is
reported as indicative rather than as a measurement.

A **binary Malignant-vs-rest** view is kept as a documented secondary framing for
the triage metric (sensitivity at a fixed specificity), because that is the shape
the clinical decision actually takes. It is a reporting view, not a retraining.

---

## 3. Stretch target — keep all eleven classes

**Decision: keep all 11 codes. Do not merge the rare tail.**

The five classes under ~55 lesions: MAL_OTH 9, BEN_OTH 44, VASC 47, INF 50, DF 52.
Imbalance ratio BCC:MAL_OTH = 280:1 on the full set (~252:1 on train).

The tempting move is to merge them into a single "other" bucket. **That is the
wrong merge**, because it is not a clinically coherent group: it would put
MAL_OTH (malignant) in the same bucket as VASC, DF, INF and BEN_OTH (all benign).
A model that predicts "other" would be telling a clinician nothing, and the
confusion matrix would lose the only thing that makes it readable — whether the
mistake crossed the benign/malignant line. A merge that mixes malignant and
benign into one label is a merge that hides the error that matters.

So the tail is kept and handled with:

1. **Class weights** in the loss (`class_weights.json`, computed on train only).
   Both `balanced` and `inverse_sqrt` weights are stored. `balanced` gives
   MAL_OTH roughly 50x BCC's weight, which is unstable at 6 training examples;
   `inverse_sqrt` is the softer default and the trade-off is recorded rather than
   hidden.
2. **A `WeightedRandomSampler`** so rare classes appear in most batches.
3. **Per-class recall** always reported — never a single macro number alone.

The limit, stated plainly: **MAL_OTH has 9 lesions in the entire dataset.** After
the committed 70/15/15 lesion-level split (seed 42) that is 7 / 1 / 1. No amount of weighting
fixes this. Exercise A3.2 measures it directly — over 10 seeds, MAL_OTH is
frequently absent from val or test entirely. Its per-class metrics are reported
with an explicit "n<5, indicative only" marker, and it is excluded from the
headline macro-F1 average, with that exclusion stated.

---

## 4. Columns that are never model inputs

Recorded in `config.LEAKY_COLUMNS` and re-derived with evidence in the data-quality
report (B4):

| Column | Why it is excluded |
|---|---|
| `diagnosis_1/2/3/4` | They *are* the label, at different resolutions. |
| `diagnosis_confirm_type` | The loudest leak in the dataset: histopathology rows are 72% Malignant, clinical-assessment rows 87% Benign. It records *that a biopsy happened*, which is downstream of the suspicion that produced the label. Unavailable at prediction time. |
| `diagnosis_full` | Free-text diagnosis. |
| `invasion_thickness_interval` | Breslow thickness — a histopathology measurement that only exists once the lesion is already known to be melanoma. |
| `melanocytic` | Derived from the diagnosis; 77% missing, and the missingness itself is informative of the class. |
| `concomitant_biopsy` | Records a clinical action taken because of suspicion. |
| `lesion_id` | The grouping key. As a feature it is a direct index into the answer. |

`image_manipulation` and `image_type` are *not* excluded, but are audited as
possible shortcuts in B4 — `image_type` turns out to be perfectly flat across
`diagnosis_1` (28.3 / 2.3 / 69.4 for both modalities), which is the useful
negative result that makes both views usable.
