# %% [markdown]
# ## A3.6 A lesion-level `Dataset`
#
# - Training works on one *image* at a time, but the label, the split and the clinical decision are per *lesion*.
# - Two parts of `milk10k.datasets` connect the two, and I test both:
#   - `LesionDataset`: both photos of one lesion as one item
#   - `aggregate_predictions`: turns two per-image predictions into one per lesion
# - Everything is asserted, so the section fails loudly instead of printing a nice-looking wrong number.

# %%
import time
from collections import Counter

import torch
from torch.utils.data import DataLoader

from milk10k import datasets

transforms.seed_everything(config.SEED)

lesion_table = data.build_lesion_table()
LABEL_COL = config.PRIMARY_LABEL_COL

print(f"lesion table : {len(lesion_table):,} rows x {lesion_table.shape[1]} columns")
print(lesion_table.head(3).to_string(index=False))

# %% [markdown]
# ### (a) One batch, and three things that can go wrong silently
#
# - ids no longer matching the labels (you'd only notice in the confusion matrix)
# - a missing file being skipped instead of raising (the dataset quietly shrinks)
# - both views getting the same augmentation draw

# %%
BATCH = 8
batch_table = lesion_table.head(BATCH * 2).reset_index(drop=True)

# The full signature the brief specifies: (lesion_table, img_dir, label_col, transform, views).
eval_ds = datasets.LesionDataset(
    batch_table, img_dir=config.IMG_DIR, label_col=LABEL_COL,
    transform=transforms.eval_transform(), views=("derm", "clinical"),
)
loader = DataLoader(eval_ds, batch_size=BATCH, shuffle=False)
batch = next(iter(loader))

print("batch keys:", list(batch))
for view in eval_ds.views:
    tensor = batch[view]
    print(f"  {view:<10} {tuple(tensor.shape)}  {tensor.dtype}")
print(f"  {'label':<10} {tuple(batch['label'].shape)}  {batch['label'].dtype}")
print(f"  {'lesion_id':<10} list of {len(batch['lesion_id'])} str")
print("\nlabel tensor :", batch["label"])
print("class names  :", [eval_ds.inverse_label_map[i] for i in batch["label"].tolist()])
print("lesion ids   :", list(batch["lesion_id"]))

expected_shape = (BATCH, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
for view in eval_ds.views:
    assert tuple(batch[view].shape) == expected_shape, (
        f"{view}: got {tuple(batch[view].shape)}, expected {expected_shape}"
    )
    assert batch[view].dtype is torch.float32, f"{view} is not float32"

# The ids and the labels must come from the SAME rows of the table, in order --
# a shuffled or off-by-one id column is invisible until error analysis.
expected_ids = batch_table["lesion_id"].head(BATCH).tolist()
expected_labels = [eval_ds.label_map[v] for v in batch_table[LABEL_COL].head(BATCH)]
assert list(batch["lesion_id"]) == expected_ids, "lesion_ids do not match the table"
assert batch["label"].tolist() == expected_labels, "labels do not match the table"
print("\nshapes, ids and labels all match the lesion table.")

# %%
# A missing file must crash, not be skipped: a dataset that quietly loses rows
# changes the denominator of every metric without telling anyone.
BOGUS_ID = "ISIC_0000000"
assert not (config.IMG_DIR / f"{BOGUS_ID}.jpg").exists(), "pick an id that really is absent"

bogus_table = lesion_table.head(1).copy()
bogus_table["derm_id"] = BOGUS_ID
bogus_ds = datasets.LesionDataset(
    bogus_table, label_col=LABEL_COL, transform=transforms.eval_transform()
)

try:
    bogus_ds[0]
except FileNotFoundError as exc:
    message = str(exc)
    assert BOGUS_ID in message, "the error does not name the offending id"
    print("FileNotFoundError raised, and it names the id:")
    print(" ", message.split(". ")[0])
else:
    raise AssertionError("a missing image file did not raise FileNotFoundError")

# %%
# Is the transform drawn independently per view? Asking a real lesion is not a
# test -- its two tensors differ because they are two different photographs. So
# request the SAME photograph as both views: any difference that survives is
# then purely the randomness of the augmentation.
same_photo = lesion_table.head(1).copy()
same_photo["clinical_id"] = same_photo["derm_id"]

deterministic = datasets.LesionDataset(
    same_photo, label_col=LABEL_COL, transform=transforms.eval_transform()
)[0]
assert torch.equal(deterministic["derm"], deterministic["clinical"]), (
    "eval_transform is not deterministic"
)

transforms.seed_everything(config.SEED)
augmented = datasets.LesionDataset(
    same_photo, label_col=LABEL_COL, transform=transforms.train_transform()
)[0]
gap = (augmented["derm"] - augmented["clinical"]).abs().max().item()
assert not torch.equal(augmented["derm"], augmented["clinical"]), (
    "train_transform produced identical tensors for two independent draws"
)

real_lesion = datasets.LesionDataset(
    lesion_table.head(1), label_col=LABEL_COL, transform=transforms.train_transform()
)[0]
assert not torch.equal(real_lesion["derm"], real_lesion["clinical"])

print(f"same photograph, eval_transform  : tensors identical  ({config.IMAGE_SIZE}px, deterministic)")
print(f"same photograph, train_transform : tensors differ, max |diff| = {gap:.3f}")
print("real lesion,    train_transform : tensors differ (two photographs AND two draws)")

# %% [markdown]
# **What (a) shows**
#
# - Batch is `(8, 3, 224, 224)` per view, `float32`; labels are `int64`; 8 ids, all matching the lesion table.
# - A missing file raises `FileNotFoundError` with the id.
# - Same photo as both views → identical with `eval_transform`, different with `train_transform`, so the transform runs once per view.
# - Why I apply it independently: from what I understood, the dermoscopic and clinical images are two different photos (different instrument and zoom), so forcing the same rotation or crop on both makes no sense.

# %% [markdown]
# ### (b) One epoch over 800 lesions
#
# - After a full shuffled pass, the class counts handed out must equal the class counts of the table.
# - Otherwise rows were dropped, repeated or relabelled somewhere.

# %%
N_LESIONS = 800
subset = lesion_table.sample(n=N_LESIONS, random_state=config.SEED).reset_index(drop=True)

epoch_ds = datasets.LesionDataset(
    subset, label_col=LABEL_COL, transform=transforms.eval_transform()
)
n_workers = datasets.default_num_workers()
generator = torch.Generator()
generator.manual_seed(config.SEED)
epoch_loader = DataLoader(
    epoch_ds, batch_size=16, shuffle=True, num_workers=n_workers, generator=generator
)

seen_counts: Counter = Counter()
seen_ids: list[str] = []
started = time.perf_counter()
for epoch_batch in epoch_loader:
    for code in epoch_batch["label"].tolist():
        seen_counts[epoch_ds.inverse_label_map[code]] += 1
    seen_ids.extend(epoch_batch["lesion_id"])
elapsed = time.perf_counter() - started

n_images = len(seen_ids) * len(epoch_ds.views)
print(
    f"one epoch: {len(seen_ids):,} lesions = {n_images:,} image decodes in "
    f"{elapsed:.1f}s  ({n_images / elapsed:.0f} img/s, num_workers={n_workers})"
)

comparison = pd.DataFrame(
    {"table": subset[LABEL_COL].value_counts(), "loader": pd.Series(seen_counts)}
).fillna(0).astype(int).sort_index()
comparison.index.name = "class"
print()
print(comparison.to_string())

assert len(seen_ids) == N_LESIONS, f"saw {len(seen_ids)} lesions, expected {N_LESIONS}"
assert sorted(seen_ids) == sorted(subset["lesion_id"]), (
    "the epoch did not visit each lesion exactly once"
)
assert comparison["table"].equals(comparison["loader"]), (
    "class counts seen by the loader differ from the subset's own counts"
)
print("\nevery lesion seen exactly once; class counts match the subset exactly.")

# %% [markdown]
# **What (b) shows**
#
# - All 800 lesions appear exactly once and the class counts match the subset exactly. Shuffling only changes the order.
# - `default_num_workers()` is 0 on macOS on purpose: worker processes use `spawn`, which hangs inside Jupyter. So the timing above is what one epoch costs on this machine without extra workers.

# %% [markdown]
# ### (c) Two views in, one prediction out
#
# - `aggregate_predictions` averages the two probability vectors of a lesion.
# - I use synthetic probabilities where I computed the right answer by hand first.
# - Three cases: views agree / views disagree and the average flips the answer / only one view present.

# %%
CLASSES = list(config.PRIMARY_LABELS)  # index == the integer label, from labels.py

synthetic_images = pd.DataFrame(
    {
        "isic_id": ["IMG_A_d", "IMG_A_c", "IMG_B_d", "IMG_B_c", "IMG_C_d"],
        "lesion_id": ["LES_AGREE", "LES_AGREE", "LES_FLIP", "LES_FLIP", "LES_ONE_VIEW"],
        "view": ["derm", "clinical", "derm", "clinical", "derm"],
    }
)
#                              Benign  Indeterminate  Malignant
synthetic_probs = np.array(
    [
        [0.05, 0.05, 0.90],  # LES_AGREE    derm      -> Malignant
        [0.10, 0.10, 0.80],  # LES_AGREE    clinical  -> Malignant
        [0.60, 0.05, 0.35],  # LES_FLIP     derm      -> Benign, mildly wrong
        [0.02, 0.03, 0.95],  # LES_FLIP     clinical  -> Malignant, confidently right
        [0.05, 0.90, 0.05],  # LES_ONE_VIEW derm      -> Indeterminate, no second view
    ]
)

per_image_pred = [CLASSES[i] for i in synthetic_probs.argmax(axis=1)]
print("per-image argmax:", dict(zip(synthetic_images["isic_id"], per_image_pred)))

lesion_pred = datasets.aggregate_predictions(
    synthetic_probs, synthetic_images, class_names=CLASSES
)
print()
print(lesion_pred.to_string(index=False))

# %%
# Hand-computed expectations. These are the arithmetic means of the rows above;
# writing them out literally is the whole point -- an expectation derived with
# the same code being tested would pass no matter what that code did.
expected_probs = np.array(
    [
        [0.075, 0.075, 0.850],  # LES_AGREE     (0.05+0.10)/2, (0.05+0.10)/2, (0.90+0.80)/2
        [0.310, 0.040, 0.650],  # LES_FLIP      (0.60+0.02)/2, (0.05+0.03)/2, (0.35+0.95)/2
        [0.050, 0.900, 0.050],  # LES_ONE_VIEW  single view, passed through unchanged
    ]
)
expected_pred = ["Malignant", "Malignant", "Indeterminate"]

assert lesion_pred["lesion_id"].tolist() == ["LES_AGREE", "LES_FLIP", "LES_ONE_VIEW"]
assert lesion_pred["n_views"].tolist() == [2, 2, 1]
assert np.allclose(lesion_pred[CLASSES].to_numpy(), expected_probs), (
    "averaged probabilities differ from the hand-computed values"
)
assert lesion_pred["pred"].tolist() == expected_pred
assert lesion_pred["pred_index"].tolist() == [CLASSES.index(p) for p in expected_pred]

# The point of LES_FLIP: the dermoscopic view alone would have called it Benign,
# and a hard vote over the two views is a 1-1 tie. Averaging the probabilities
# resolves it towards the confident view, which is the behaviour we want.
assert per_image_pred[2] == "Benign" and per_image_pred[3] == "Malignant"
assert lesion_pred.loc[1, "pred"] == "Malignant"
# A single-view lesion is averaged over what exists, i.e. left untouched.
assert np.allclose(lesion_pred.loc[2, CLASSES].to_numpy(dtype=float), synthetic_probs[4])

print("all three synthetic cases match the hand-computed answer:")
print("  LES_AGREE    both views Malignant            -> Malignant   (n_views=2)")
print("  LES_FLIP     derm Benign, clinical Malignant -> Malignant   (n_views=2, argmax flipped)")
print("  LES_ONE_VIEW one view only                   -> Indeterminate (n_views=1, unchanged)")

# %% [markdown]
# **What (c) shows**
#
# - One row per lesion, with an `n_views` column, so a lesion with only one photo stays visible.
# - `LES_FLIP` is the interesting one: dermoscopic says Benign (0.60), clinical says Malignant (0.95).
#   - voting on labels → 1–1 tie
#   - averaging probabilities → 0.65 Malignant, following the confident view
# - That's why I think averaging probabilities is better than voting: the vote throws away the confidence that breaks the tie.

# %% [markdown]
# ### (d) The unit of evaluation
#
# **My answer**
#
# - I think the unit of evaluation has to be the lesion, because that's what the clinical decision is about, and the two photos are two views of the same lesion, not two separate cases.
# - Scoring per image counts every lesion twice, so the real sample size is half of what it looks like and the confidence intervals come out too narrow.
# - It also gives "half credit" when one view is right and the other wrong, which isn't a real clinical outcome, and it hides exactly the disagreements a doctor would want to see.

