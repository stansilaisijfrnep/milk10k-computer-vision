# %% [markdown]
# ## A2.2 JPEG compression and the file-size shortcut
#
# MILK10k ships as JPEG, so every pixel the model ever sees has already been
# through a lossy codec. This section does three things: measure what further
# compression costs (part a), reason about what an already-compressed dataset
# implies for the pipeline (part b), and test whether file size leaks the label
# (part c).

# %% [markdown]
# ### (a) Re-encoding one image at five quality levels
#
# One real dermoscopic melanoma, re-encoded in memory at quality 95, 75, 50, 25
# and 10. Melanoma is the right subject here: pigment network, dots and globules
# are exactly the high-frequency structure a quantised DCT throws away first.
# PSNR is implemented below rather than imported.

# %%
import io

from PIL import Image

from milk10k import config


def psnr(reference: np.ndarray, test: np.ndarray, max_value: float = 255.0) -> float:
    """Peak signal-to-noise ratio between two uint8 images, in decibels.

    PSNR = 10 * log10(MAX^2 / MSE),  MAX = 255 for uint8.

    The MSE is accumulated in float64: uint8 arithmetic would wrap around on the
    difference, and float32 loses precision once the squared errors are summed
    over 810,000 values.
    """
    if reference.shape != test.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {test.shape}")

    mse = np.mean((reference.astype(np.float64) - test.astype(np.float64)) ** 2)

    # MSE == 0 means the two images are bit-identical. The ratio is then
    # infinite, so return inf explicitly instead of letting numpy divide by zero
    # and emit a warning alongside the right answer.
    if mse == 0.0:
        return float("inf")

    return float(10.0 * np.log10(max_value ** 2 / mse))


# Deterministic subject: the first melanoma lesion by lesion_id, dermoscopic view.
lesion_table = data.build_lesion_table()
subject_id = (
    lesion_table.loc[lesion_table["dx"] == "MEL"]
    .sort_values("lesion_id")["derm_id"]
    .iloc[0]
)
subject_path = config.IMG_DIR / f"{subject_id}.jpg"

original = np.asarray(Image.open(subject_path).convert("RGB"))
assert original.dtype == np.uint8 and original.shape == (450, 600, 3)

original_bytes = subject_path.stat().st_size
raw_bytes = int(original.size)  # H * W * 3 uint8 samples, i.e. uncompressed

print(f"subject      : {subject_id}  (melanoma, dermoscopic)")
print(f"raw pixels   : {raw_bytes:,} bytes uncompressed")
print(f"on disk      : {original_bytes:,} bytes "
      f"({original_bytes / 1024:.1f} KB, {raw_bytes / original_bytes:.1f}x)")

# %%
QUALITIES = [95, 75, 50, 25, 10]

rows = []
decoded = {}  # keep the pixels so the crop figure below re-uses this encoding
for quality in QUALITIES:
    buffer = io.BytesIO()
    # Encode into memory, never to disk: writing files would pollute the repo
    # and the point is the byte count, which BytesIO reports exactly.
    Image.fromarray(original).save(buffer, format="JPEG", quality=quality)
    n_bytes = buffer.getbuffer().nbytes

    buffer.seek(0)
    decoded[quality] = np.asarray(Image.open(buffer).convert("RGB"))

    rows.append({
        "quality": quality,
        "bytes": n_bytes,
        "kb": n_bytes / 1024.0,
        "compression_ratio": raw_bytes / n_bytes,
        "psnr_db": psnr(original, decoded[quality]),
    })

compression = pd.DataFrame(rows)
print(compression.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

# %% [markdown]
# The original is **already a JPEG**, so every row above is a *re-encoding*.
# The q=95 PSNR is measured against an already-lossy reference, not against
# ground truth — the information the first encoder discarded is gone and no
# quality setting brings it back.
#
# That also explains the one result that looks wrong: **q=75 scores higher than
# q=95**. The next cell checks why.

# %%
# Why q=75 is near-lossless: read the quantisation table the shipped file
# carries and compare it with what Pillow writes at each quality setting.
shipped_qt = tuple(tuple(t) for t in Image.open(subject_path).quantization.values())

matches = []
for quality in QUALITIES:
    buffer = io.BytesIO()
    Image.fromarray(original).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    probe = Image.open(buffer)
    matches.append((quality, tuple(tuple(t) for t in probe.quantization.values()) == shipped_qt))

print("shipped quantisation table matches Pillow quality:",
      [q for q, same in matches if same] or "none of the five")
print("chroma subsampling (component, h, v, tbl):", Image.open(subject_path).layer)

# The table is a property of the ISIC export, not of this one image: check every
# file rather than assuming it. Only the JPEG header is parsed (~2 s for all).
probe_ids = data.load_metadata()["isic_id"]
n_same = 0
for i in probe_ids:
    with Image.open(config.IMG_DIR / f"{i}.jpg") as im:
        n_same += tuple(tuple(t) for t in im.quantization.values()) == shipped_qt
print(f"same quantisation table in {n_same:,} of {len(probe_ids):,} images")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

axes[0].plot(compression["quality"], compression["psnr_db"], marker="o", color=config.ORANGE)
axes[0].set_xlabel("JPEG quality")
axes[0].set_ylabel("PSNR vs original (dB)")
axes[0].set_title("Fidelity of the re-encoding")

axes[1].plot(compression["quality"], compression["kb"], marker="o", color=config.ORANGE)
axes[1].axhline(original_bytes / 1024.0, color=config.GREY, ls="--", lw=1)
axes[1].text(12, original_bytes / 1024.0 + 1.5, "original on disk",
             color=config.GREY, fontsize=9)
axes[1].set_xlabel("JPEG quality")
axes[1].set_ylabel("file size (KB)")
axes[1].set_title("Cost of the re-encoding")

fig.suptitle(f"{subject_id} re-encoded at five JPEG qualities", y=1.04)
plt.show()

# %%
# Zoom on a patch with real lesion structure, chosen by luminance variance so
# the choice is reproducible rather than eyeballed: flat skin has low variance,
# the pigmented border and the globules have high variance.
CROP = 100
STRIDE = 20

luma = original.mean(axis=2)
best_std, best_y, best_x = -1.0, 0, 0
for y in range(0, luma.shape[0] - CROP + 1, STRIDE):
    for x in range(0, luma.shape[1] - CROP + 1, STRIDE):
        patch_std = float(luma[y:y + CROP, x:x + CROP].std())
        if patch_std > best_std:
            best_std, best_y, best_x = patch_std, y, x

print(f"crop at (x={best_x}, y={best_y}), {CROP}x{CROP}, luminance std {best_std:.1f}")


def zoom(arr: np.ndarray) -> Image.Image:
    """Crop and upsample 4x with NEAREST so 8x8 JPEG blocks stay square-edged."""
    patch = arr[best_y:best_y + CROP, best_x:best_x + CROP]
    return Image.fromarray(patch).resize((CROP * 4, CROP * 4), Image.NEAREST)


fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))

axes[0].imshow(original)
axes[0].add_patch(plt.Rectangle((best_x, best_y), CROP, CROP,
                                fill=False, edgecolor=config.BENIGN_MALIGNANT_COLORS["Malignant"],
                                linewidth=2))
axes[0].set_title("full image + crop")

panels = [
    (original, "original (as shipped)"),
    (decoded[95], f"q=95 · {compression.loc[0, 'psnr_db']:.1f} dB"),
    (decoded[10], f"q=10 · {compression.loc[4, 'psnr_db']:.1f} dB"),
]
for ax, (arr, title) in zip(axes[1:], panels):
    ax.imshow(zoom(arr))
    ax.set_title(title)

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

fig.suptitle("Best vs worst re-encoding, 4x nearest-neighbour zoom", y=1.04)
plt.show()

# %% [markdown]
# **Answer (a).** Re-encoding is not free and it is not monotone in the quality
# slider. Quality 75 is *near-lossless* (PSNR ~71 dB) because the shipped files
# already carry the standard quality-75 quantisation table — verified above, and
# identical in all 10,480 files — so re-encoding at 75 re-quantises
# coefficients that already sit on that grid. Quality 95 scores **worse**
# (~54 dB) while producing a file ~1.7x larger than the original: a finer grid
# cannot recover information the first encoder already destroyed, and the
# decode -> DCT -> 4:2:0 chroma round trip adds error of its own. Below 50 the
# loss becomes real: q=10 drops to ~32 dB and the zoom shows flat 8x8 blocks
# where the pigment border was. The practical rule is that a higher quality
# setting on a re-encode buys bytes, not fidelity.

# %% [markdown]
# ### (b) What "already JPEG" implies for the model
#
# **The artefacts are part of the data.** Every image was quantised on an 8x8
# DCT grid with 4:2:0 chroma subsampling — the same table for all 10,480 files.
# Blocking edges and desaturated chroma transitions are therefore present in
# every training, validation and test image, and a convolutional network will
# happily learn filters that respond to them. They are not noise the model
# averages out; they are a consistent texture it can key on.
#
# **Never round-trip through JPEG again.** Any augmentation that writes a JPEG
# and reads it back applies a second lossy pass on top of the first — double
# compression, and it compounds across epochs if the cache is reused. Part (a)
# quantifies the cost: even the best-case re-encode loses measurable fidelity,
# and the q=95 row shows the loss can grow while the file does. The pipeline
# therefore decodes once and keeps arrays and tensors in memory from that point
# on; `preprocessing.preprocess_image` and the torchvision transforms in
# `transforms.py` both work on decoded pixels and never re-serialise.
#
# **Resizing resamples the artefacts, not the signal.** Downscaling happens
# after compression, so the 224x224 input is an average of already-quantised
# pixels. That cuts both ways: the interpolation partially smooths the 8x8
# blocking away, which is a mild argument for downscaling 600x450 -> 224 rather
# than cropping at native resolution and feeding the blocks in untouched. It is
# an argument about artefacts only — the reason `preprocess_image` resizes
# instead of centre-cropping is that the lesion border is diagnostic.
#
# **The risk this creates.** If compression strength correlated with class —
# different clinic, different camera, different export setting per diagnosis —
# the model could score well by reading the codec instead of the pathology.
# That is a testable claim, and part (c) tests it.

# %% [markdown]
# ### (c) Does file size leak the label?
#
# File size is a free proxy for how hard an image was to compress. If malignant
# lesions were systematically harder, a model could reach a respectable score
# without looking at the lesion at all. `os.stat` over all 10,480 files costs no
# decoding, so the test is cheap — `preprocessing.image_file_sizes()` does it in
# one directory scan.

# %%
from sklearn.metrics import roc_auc_score

meta = data.load_metadata()
sizes = preprocessing.image_file_sizes(meta)
sized = meta[["isic_id", "image_type", "diagnosis_1"]].merge(
    sizes, on="isic_id", validate="one_to_one"
)
assert len(sized) == 10_480 and sized["file_size_bytes"].notna().all()

# "Malignant vs everything else" — the clinically meaningful one-vs-rest split.
# Indeterminate (2.3% of lesions) sits with the negatives; the robustness check
# below repeats the AUC without it.
sized["is_malignant"] = (sized["diagnosis_1"] == "Malignant").astype(int)

by_modality = (
    sized.groupby("image_type", as_index=False)["file_size_kb"]
    .agg(n="size", median="median", mean="mean", std="std")
)
print(by_modality.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

# %%
MODALITIES = ["dermoscopic", "clinical: close-up"]

print("ROC-AUC of file size as a malignancy score")
for modality in MODALITIES:
    subset = sized.loc[sized["image_type"] == modality]
    auc = roc_auc_score(subset["is_malignant"], subset["file_size_kb"])
    print(f"  {modality:<20s} n={len(subset):,}  AUC = {auc:.3f}")

auc_all = roc_auc_score(sized["is_malignant"], sized["file_size_kb"])
print(f"  {'all images':<20s} n={len(sized):,} AUC = {auc_all:.3f}")

# How far from 0.5 is "far"? A 95% bootstrap interval, resampling LESIONS would be
# stricter, but within one modality each lesion has exactly one image, so resampling
# rows is resampling lesions here.
rng = np.random.default_rng(config.SEED)
print("\n95% bootstrap CI (1,000 resamples)")
for modality in MODALITIES:
    subset = sized.loc[sized["image_type"] == modality]
    y, x = subset["is_malignant"].to_numpy(), subset["file_size_kb"].to_numpy()
    boot = [roc_auc_score(y[idx], x[idx])
            for idx in (rng.integers(0, len(y), len(y)) for _ in range(1000))]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {modality:<20s} [{lo:.3f}, {hi:.3f}]")

# Robustness: drop Indeterminate so the comparison is strictly Benign vs Malignant.
binary = sized.loc[sized["diagnosis_1"].isin(["Benign", "Malignant"])]
print("\nsame, Benign vs Malignant only (Indeterminate dropped)")
for modality in MODALITIES:
    subset = binary.loc[binary["image_type"] == modality]
    print(f"  {modality:<20s} n={len(subset):,}  "
          f"AUC = {roc_auc_score(subset['is_malignant'], subset['file_size_kb']):.3f}")

# What file size DOES predict: which of the two cameras took the picture.
auc_modality = roc_auc_score(
    (sized["image_type"] == "clinical: close-up").astype(int), sized["file_size_kb"]
)
print(f"\nROC-AUC of file size for predicting image_type = {auc_modality:.3f}")

# Is modality itself informative about the label? Each lesion has one image of each
# type, so the diagnosis mix must be identical in both — printed, not assumed.
print((pd.crosstab(sized["image_type"], sized["diagnosis_1"], normalize="index") * 100)
      .round(1).to_string())

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: the question asked — does malignancy shift the distribution?
for ax, (title, group_col, groups) in zip(
    axes,
    [
        ("File size by diagnosis (Benign vs Malignant)", "diagnosis_1",
         ["Benign", "Malignant"]),
        ("File size by modality", "image_type", MODALITIES),
    ],
):
    data_groups = [
        sized.loc[sized[group_col] == g, "file_size_kb"].to_numpy() for g in groups
    ]
    bp = ax.boxplot(data_groups, tick_labels=groups, showfliers=False,
                    patch_artist=True, widths=0.5, medianprops={"color": "black"})
    colors = ([config.BENIGN_MALIGNANT_COLORS[g] for g in groups]
              if group_col == "diagnosis_1" else [config.ORANGE, config.GREY])
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.set_ylabel("file size (KB)")
    ax.set_title(title)

fig.suptitle("File size separates the cameras, not the diagnoses", y=1.02)
plt.show()

# %% [markdown]
# **Answer (c). There is no file-size shortcut for malignancy.** As a malignancy
# score, file size gives AUC **0.464** on dermoscopic images, **0.511** on
# clinical close-ups and **0.499** on all 10,480 images — chance is 0.500, and
# dropping the Indeterminate lesions barely moves it (0.481 and 0.505). The
# dermoscopic value is honestly *not* pure noise: its bootstrap interval printed
# above excludes 0.5, so smaller files are very slightly more often malignant.
# But read the other way round it is an AUC of about 0.54 — a signal so weak that
# no model could score anything useful from it — and the clinical value sits at
# chance. The boxplot agrees: the Benign and Malignant distributions sit on top
# of each other. That is what "no shortcut" means here: not zero, but negligible.
#
# This is a negative result and it is worth stating plainly, because it is what
# lets the project stop worrying about this particular leak. Had the AUC come
# back at, say, 0.70, the honest response would have been to treat compression
# as a confound — re-encode every image to one fixed quality before training,
# and re-check the headline metric against that baseline. It did not, so no such
# correction is needed. It also matches what part (a) found: every file carries
# the *same* quantisation table, so ISIC re-encoded the whole archive with one
# setting and there is no per-class compression difference left to exploit.
#
# **What file size does separate is modality.** Median 26.0 KB (KiB, 1 KB = 1,024
# bytes) dermoscopic against 38.8 KB clinical, and an AUC of **0.865** for predicting `image_type`.
# The reason is acquisition, not pathology: a clinical close-up is a hand-held
# photograph containing surrounding skin texture, hair, background and uneven
# lighting — all high-frequency content that survives the DCT and costs bits. A
# dermoscopic image is a contact-immersion field: smooth, evenly lit, filling the
# frame with one lesion, so far more of its energy sits in coefficients the
# quantiser zeroes out.
#
# **That modality signal does not become a label shortcut**, because `image_type`
# is perfectly balanced across `diagnosis_1` — both modalities are 28.3% Benign,
# 2.3% Indeterminate, 69.4% Malignant (crosstab printed above), since every
# lesion contributes exactly one of each. A feature that predicts modality therefore carries no information
# about the diagnosis. The conclusion for the pipeline is that compression is a
# modality marker to be aware of when mixing the two views in one model, not a
# leak to defend against.
