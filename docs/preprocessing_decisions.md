# Preprocessing decisions (Milestone 1, B6)

Three decisions have to be made before a single image reaches a network: how big
the input is, how a lesion's two photographs become training and evaluation
examples, and what the pixels are normalised with. This page decides each one,
with the evidence it was decided on.

The machine-readable versions are `config.IMAGE_SIZE`, `config.IMAGENET_MEAN` /
`config.IMAGENET_STD` and [`artifacts/norm_stats.json`](../artifacts/norm_stats.json);
the code is `milk10k.transforms`, `milk10k.preprocessing` and `milk10k.datasets`.
Nothing on this page is a preference — each decision names the number it rests on
and the experiment that would overturn it.

---

## 1. Input resolution — **224 x 224**

### The evidence

From [`artifacts/tables/image_size_summary.csv`](../artifacts/tables/image_size_summary.csv),
produced by opening **all 10,480 files**, not a sample:

| metric | value |
|---|---|
| n_images / n_ok / n_missing / n_unreadable | 10,480 / 10,480 / 0 / 0 |
| width min / median / max | 600 / 600 / 600 |
| height min / median / max | 450 / 450 / 450 |
| n_distinct_resolutions | **1** (600x450, 100% of images) |
| share with both sides >= 224px | **100%** |
| share with both sides >= 256px | **100%** |

The dataset is dimensionally uniform. This also **refutes** the Session 1 claim
that the images vary in resolution — that came from a 400-image sample (see
[`reports/findings_check.md`](../reports/findings_check.md), row 7).

Two things follow immediately. First, neither 224 nor 256 upsamples anything:
every image is larger than both on both sides, so the data does not pick the
resolution for us and there is no special case to handle. Second, the resize has
no branches — no padding path, no "if the image is small" path, no per-image
aspect logic. Every image takes the same code path, which is one less thing that
can silently differ between train and test.

### The decision and why

**224 x 224.** Three reasons, in order of weight:

1. **It matches the pretrained weights.** Milestone 2 fine-tunes ImageNet
   backbones, whose native training resolution is 224. Feeding a different size
   works mechanically but moves the input off the distribution the filters and
   the positional embeddings were learned on, so the first epochs are spent
   re-adapting rather than transferring.
2. **It is cheaper.** 224² = 50,176 pixels vs 256² = 65,536, i.e. **1.31x less
   work per image** for a convolutional backbone. For a ViT-B/16 the token count
   goes 196 -> 256, so the self-attention term alone grows **1.71x**. With
   7,334 training images and a laptop-class budget, that is the difference
   between an overnight sweep and a single run.
3. **It keeps the whole lesion in frame.** The images are 4:3 (600x450) and the
   pipelines end at a square, so something has to give. `transforms.eval_transform`
   uses `Resize((224, 224))` — a squash — rather than a short-side resize plus
   centre crop, because a centre crop of a 4:3 frame throws away a quarter of the
   width, and the lesion *border* (the A and B of ABCD: asymmetry, border
   irregularity) is diagnostic. Mild anisotropy is the smaller error: the frame
   is scaled 2.68x horizontally and 2.01x vertically, and it is scaled the same
   way in every image, so the network never sees a train/test mismatch.
   `train_transform`'s `RandomResizedCrop` uses an aspect band centred on the
   native 4:3 for exactly that reason.

### The cost, stated honestly

600x450 -> 224x224 keeps **18.6% of the pixels** and discards the rest (a 5.38x
reduction). What gets discarded is not neutral: the fine dermoscopic structures a
dermatologist actually grades — pigment network, dots and globules, vessel
patterns — live in exactly that high-frequency detail. A bilinear downscale
blurs them; at 2.68x a fine network can be smoothed into flat pigment.

So 224 is a **budget decision, not a scientific one**, and it is not closed.
Milestone 2 carries a named experiment: fine-tune the same backbone at 384 and
compare malignant sensitivity and macro-F1 on the same lesion-level test split.
Because every image is 600x450, 384 is still a pure downscale and the comparison
needs no change to the data pipeline — only `config.IMAGE_SIZE`.

---

## 2. Combining the two images per lesion — **(a) for training, (c) for evaluation**

Every lesion has exactly one dermoscopic and one clinical close-up image; this
was re-verified on all 5,240 lesions (`reports/findings_check.md`, row 1). Three
options exist:

| Option | What it means | Verdict |
|---|---|---|
| **(a)** | Treat the two images as independent samples, each carrying the lesion's label | **Training** |
| **(b)** | Use one modality only (dermoscopic *or* clinical) | Rejected |
| **(c)** | Keep both views per lesion and produce one prediction per lesion | **Evaluation** |

### Why (a) for training

It doubles the effective training set — 3,667 training lesions become 7,334
training images — and it is a *free, real* augmentation rather than a synthetic
one: the two views are genuinely different photographs of the same skin, taken
with different instruments at different magnifications. No rotation or colour
jitter buys that kind of variation.

**What makes (a) safe is the split.** Two images of the same lesion are near
duplicates; if one landed in train and the other in val, the score would be
memorisation. The split is therefore grouped on `lesion_id`
(`splits.split_lesions`), and the check is not assumed but run: lesion overlap
between train/val, train/test and val/test is **0 / 0 / 0**, and every lesion
keeps both of its images inside its own split. Option (a) is only defensible on
top of that guarantee.

### Why (c) for evaluation

The clinical unit is the lesion. A dermatologist does not ask "what is this
photograph?", they ask "does this lesion need excising?" — so the reported number
must be one prediction per lesion.

This is not a fudge, because the two are different questions asked at different
times. Training optimises a per-image loss because that is where the gradient
signal and the data volume are; evaluation reports a per-lesion metric because
that is the decision being supported. Nothing about the model changes between
them — only how its outputs are counted.

### The leakage risk of evaluating per image, explicitly

Reporting per-image metrics on val or test is wrong in two distinct ways:

1. **It double-counts every lesion.** 1,572 val images are 786 lesions; 1,574
   test images are 787. The effective sample size is *half* the denominator, so
   every confidence interval computed on n=1,574 is roughly √2 too narrow and
   every significance claim is overstated. The errors are not independent — the
   two rows of a lesion are the same skin, the same patient, the same
   photographer.
2. **It awards half credit for self-contradiction.** A model that calls the
   dermoscopic view malignant and the clinical view benign scores 50% on that
   lesion. Clinically that outcome does not exist: the lesion is either excised
   or it is not. Per-image scoring rewards a model for being confidently wrong
   half the time in a way that never shows up as a real decision.

### What the code does

- `datasets.MilkImageDataset` + `datasets.build_dataloaders` — image-level, one
  row per photograph, used for training (and for the forward pass at evaluation).
- `datasets.aggregate_predictions` — averages a lesion's per-image probability
  vectors into one row per lesion. Run on the val split it turns 1,572 image rows
  into 786 lesion rows, each with `n_views = 2`. Averaging probabilities rather
  than voting on hard labels keeps the model's confidence: a confident malignant
  call is not cancelled by a hesitant benign one.
- `datasets.LesionDataset` — one item per lesion with both views as named
  tensors, for a Milestone 2 two-branch model that consumes the pair directly.

### Why not (b)

- **Dermoscopic only** halves the data to 5,240 images and throws away the
  modality a general-practice deployment would actually have. Most first-contact
  photographs are clinical close-ups taken without a dermatoscope; a model that
  only works on dermoscopy is a model for the specialist who needs it least.
- **Clinical only** throws away the higher-information view — dermoscopy exists
  precisely because it resolves structures the naked eye cannot.

There is also no shortcut argument for dropping a modality: `image_type` is
perfectly flat against the target (28.3 / 2.3 / 69.4% Benign / Indeterminate /
Malignant for *both* modalities, max gap 0.00 pp), so neither view is
contaminated. The modalities do differ in pixels — mean blue 0.537 dermoscopic vs
0.415 clinical over all 10,480 images — which is why they are treated as two
views of one lesion and never as two independent samples of a population.

---

## 3. Normalisation — **ImageNet mean/std, with the train-only statistics on file**

**Decision: normalise with `config.IMAGENET_MEAN = (0.485, 0.456, 0.406)` and
`config.IMAGENET_STD = (0.229, 0.224, 0.225)`.**

The reason is the same as for 224: Milestone 2 starts from pretrained weights,
and those filters expect inputs in the distribution they were trained on.
Changing the normalisation is the same class of change as changing the input
size — it should be an experiment with a number attached, not a default.

### The alternative is computed, not hand-waved

`artifacts/norm_stats.json` holds the MILK10k statistics, computed on
**TRAIN images only** (n = 1,500 sampled from the 7,334 training images, seed 42,
at 224px):

| | R | G | B |
|---|---|---|---|
| MILK10k train mean | 0.6790 | 0.5290 | 0.4785 |
| ImageNet mean | 0.485 | 0.456 | 0.406 |
| MILK10k train std | 0.1304 | 0.1360 | 0.1558 |
| ImageNet std | 0.229 | 0.224 | 0.225 |

**Train-only, because anything else is preprocessing leakage.** Normalisation
statistics are fitted parameters, exactly like a `StandardScaler`. Computing them
over all 10,480 images lets the mean brightness of the val and test photographs
decide how the training images are scaled; the reported score then comes out
flattering and does not survive contact with genuinely unseen data. The same
fitted numbers are applied unchanged to val and test.

### Why this is worth an A/B test in Milestone 2

The two distributions are not close. MILK10k is markedly **redder and brighter**
— the red mean is +0.194 above ImageNet's and the red-minus-blue gap is 0.200 vs
ImageNet's 0.079, which is simply what a dataset of nothing but skin looks like.
It is also much **flatter**: the MILK10k standard deviations are 0.57 / 0.61 /
0.69 of ImageNet's, so after ImageNet normalisation the inputs occupy well under
unit variance instead of filling it.

That is a concrete, testable hypothesis — dataset statistics may condition
better — and the file is already written, so the experiment is a one-line change
in `transforms.train_transform(mean=..., std=...)` with everything else held
fixed. Until that number exists, compatibility with the pretrained weights wins.

---

## Summary

| Decision | Choice | Rests on | Reopened by |
|---|---|---|---|
| Input resolution | 224 x 224, squashed from 600x450 | all 10,480 images are 600x450; 100% >= 224 and >= 256; 224 matches the backbones and is 1.31x cheaper than 256 | Milestone 2 experiment at 384 |
| Two images per lesion | (a) independent samples for training, (c) aggregated per lesion for evaluation | 7,334 train images from 3,667 lesions; 0 lesion overlap between splits; 1,572 val images -> 786 lesion predictions | A two-branch model (`LesionDataset`) if the pair beats the average |
| Normalisation | ImageNet mean/std | pretrained-weight compatibility; train-only MILK10k stats stored for comparison | Milestone 2 A/B against `artifacts/norm_stats.json` |
