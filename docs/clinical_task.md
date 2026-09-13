# The clinical task

_What problem is this project actually solving, and what would "good" mean?_

## 1. The medical problem

Skin cancer is the most commonly diagnosed cancer worldwide, and its outcomes depend
almost entirely on **when** it is found. Melanoma caught while still confined to the
epidermis is cured by a simple excision; the same tumour found after it has spread has a
dramatically worse prognosis. Basal cell and squamous cell carcinomas are rarely fatal but
become disfiguring and expensive to treat when they are allowed to grow, particularly on
the face.

The bottleneck is not treatment — it is **triage**. Far more lesions look worrying than
turn out to be cancer, and the specialists who can tell the difference are scarce. A
dermatologist examining a suspicious lesion has to decide: reassure the patient, monitor,
or cut it out and send it to pathology. That decision is made mostly by eye, often with a
**dermatoscope** — a handheld device that presses a lens against the skin under polarised
light, revealing pigment networks, vessels and structures invisible to the naked eye.

## 2. What MILK10k contains

MILK10k is built from exactly that clinical moment. It holds **10,480 images of 5,240
lesions**, and every lesion was photographed **twice**:

| Modality | What it is | Why it matters |
|---|---|---|
| **Clinical close-up** | An ordinary photograph under visible light | Shows the lesion in context: size, borders, surrounding skin |
| **Dermoscopic** | Taken through a dermatoscope, polarised, magnified | Shows sub-surface structure — pigment networks, vessels, crusting |

The labels are not opinions. **95.7% of lesions were confirmed by histopathology** — a
pathologist looked at the excised tissue under a microscope. The remaining 4.3% rest on a
single clinician's assessment. That makes this ground truth unusually trustworthy for a
dermatology dataset, and it also explains the dataset's most surprising property, below.

## 3. The prediction task

**Primary target — `diagnosis_1`, three classes:**

| Class | Lesions | Share | Meaning |
|---|---|---|---|
| Malignant | 3,634 | 69.4% | Cancer — needs excision |
| Benign | 1,483 | 28.3% | Harmless — reassure or monitor |
| Indeterminate | 123 | 2.3% | The pathology itself was not conclusive |

**Stretch target — 11 diagnosis codes** (`training_gt.csv`): BCC, NV, BKL, SCCKA, MEL,
AKIEC, DF, INF, VASC, BEN_OTH, MAL_OTH. One label per lesion, verified.

Input is one or both images of a lesion. It is a **classification** task — one label per
lesion, no bounding box, no segmentation mask.

## 4. The thing everyone gets wrong about this dataset

**MILK10k is 69% malignant.** That is the reverse of the general population, where the
overwhelming majority of pigmented lesions are harmless moles.

This is not a bug in the data — it is **referral bias**, and it is baked in by design. These
are lesions that already survived two filters: a patient thought it was worth showing
someone, and a dermatologist thought it was worth biopsying. Nobody biopsies an obvious
freckle, so obvious freckles are not in the dataset.

Two consequences follow, and both matter more than any modelling choice:

1. **A model trained here does not estimate "is this skin cancer?"** It estimates
   *"given that a dermatologist was already worried enough to biopsy this, is it cancer?"*
   Deployed as a phone app for the general public, its positive predictive value would
   collapse, because the population it would see is nothing like its training distribution.
2. **Accuracy is a meaningless headline number.** Predicting "Malignant" for every lesion
   scores 69.4% accuracy and would send every patient to surgery. Any reported accuracy
   below ~70% is worse than a constant; any above it needs the per-class breakdown to mean
   anything at all.

## 5. What "good" means here

The costs of the two errors are wildly asymmetric:

- **False negative** (a cancer called benign): the patient goes home, the tumour grows,
  and the diagnosis arrives at a later stage. In melanoma this can be fatal.
- **False positive** (a benign lesion called malignant): an unnecessary excision — a scar,
  a cost, some anxiety, a follow-up appointment.

A missed melanoma is not comparable to an extra biopsy. So the metrics that count are:

| Metric | Why |
|---|---|
| **Recall (sensitivity) on Malignant** | The number that maps to patient harm. Report it first. |
| **Balanced accuracy / macro-F1** | Treats the small Benign and Indeterminate classes as equally important. |
| **Per-class recall** | The only way to see whether the rare classes are being predicted at all. |
| **Per-skin-tone recall** | 61% of lesions are one skin tone class. Aggregate metrics hide group failures. |
| **ROC-AUC / PR-AUC** | Threshold-free view — and lets us pick an operating point that favours sensitivity. |

Plain accuracy is reported, if at all, only next to the 69.4% majority baseline.

## 6. Honest deployment framing

The realistic role for a model like this is **decision support, not diagnosis**: flagging
lesions for a specialist's attention, prioritising a queue, or offering a second opinion
that a clinician can overrule. Three limits should travel with any result produced in this
project:

- **Population**: biopsy-referred lesions from a specialist setting, median patient age 65,
  60% male. Not the general public.
- **Skin tone**: 61% of lesions are skin tone class III, and the lightest and darkest tones
  are each under 8%. Performance on under-represented tones is genuinely unknown and must
  be reported separately rather than assumed.
- **Modality**: results depend on having a dermatoscope. A clinical-photo-only model is a
  different, harder problem and should be evaluated as such.

## 7. Sources

- MILK10k dataset: <https://api.isic-archive.com/doi/milk10k/> (CC-BY-NC)
- Dataset description: <https://doi.org/10.1016/j.jid.2025.06.1594>
- ISIC Archive: <https://www.isic-archive.com/>
