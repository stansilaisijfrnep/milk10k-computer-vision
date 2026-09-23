# %% [markdown]
# ## A3.5 Is this augmentation label-safe? A quantitative audit
#
# An augmentation is a claim: *this transformation never changes the diagnosis*. For a
# geometric operation that claim is easy to defend. For a colour operation it is not —
# pigmentation and erythema are themselves diagnostic criteria, so a large enough hue or
# brightness shift changes the lesion while leaving the label on disk untouched.
#
# This section turns that worry into a number. We measure how far Benign and Malignant
# lesions actually sit apart in mean hue and mean brightness, then measure how far each
# `ColorJitter` setting moves the *same* statistic on the *same* images. Anything that moves
# colour further than the classes are apart is a candidate for relabelling an image by
# accident.

# %%
from matplotlib.colors import rgb_to_hsv
from PIL import Image

# Colour statistics do not need full resolution; 128px keeps the whole audit to a few seconds.
WORK_SIZE = 128
CROP_FRAC = 0.50                                  # central half of each side = 25% of the frame
HUE_SETTINGS = [0.02, 0.05, 0.10, 0.50]           # ColorJitter hue -> hue * 360 degrees
BRIGHTNESS_SETTINGS = [0.10, 0.20, 0.30, 0.60]    # 0.20 was the original train setting; this audit cut it to 0.10
N_PER_GROUP = 100

# The audit is about the TRAINING pipeline, so it runs on the training split only: using
# val/test pixels to justify a training choice would be a small leak. Dermoscopic images
# only, because the proxy below assumes a centred, framed lesion — true of dermoscopy and
# not of the clinical close-ups.
train = splits.load_split("train")
pool = train[(train["image_type"] == "dermoscopic")
             & train["diagnosis_1"].isin(["Benign", "Malignant"])]

sample = pd.concat(
    [pool[pool["diagnosis_1"] == g].sample(n=N_PER_GROUP, random_state=config.SEED)
     for g in ("Benign", "Malignant")],
    ignore_index=True,
)

assert len(sample) == 2 * N_PER_GROUP, "sample is not 100 + 100"
assert (sample["image_type"] == "dermoscopic").all()
assert sample["lesion_id"].is_unique, "a lesion would be counted twice"

print(f"pool (train split, dermoscopic): {len(pool):,} images "
      f"-> {pool['diagnosis_1'].value_counts().to_dict()}")
print(f"audited: {len(sample)} images, {N_PER_GROUP} per group, seed={config.SEED}, "
      f"working size {WORK_SIZE}x{WORK_SIZE}")

# %% [markdown]
# ### The lesion-pixel proxy — and why it is only a proxy
#
# The brief asks for lesion pixels, but MILK10k ships no segmentation masks. Rather than
# invent one, we use a rule transparent enough to be argued with:
#
# 1. keep the **central 50% of each side** (25% of the frame) — in dermoscopy the operator
#    centres the lesion under the contact plate;
# 2. inside that crop, keep the pixels **darker than that crop's own median V** — pigment is
#    darker than the surrounding skin.
#
# This is a proxy, not a segmentation, and it fails in a predictable way: for amelanotic
# lesions — pink, pearly BCC being the commonest malignant class here — the lesion is *not*
# darker than perilesional skin, so the darker-half rule picks up some surrounding skin
# instead. The figure shows the proxy on one lesion from each class so the reader can judge
# that failure rather than take our word for it. Because the proxy is arguable, every
# statistic below is also computed on the **whole frame** as a control, and the two answers
# are reported side by side.

# %%
def lesion_proxy_mask(v: np.ndarray, crop_frac: float = CROP_FRAC) -> np.ndarray:
    """Boolean mask: darker-than-median pixels inside a centred crop of a V channel."""
    h, w = v.shape
    y0, y1 = int(h * (1 - crop_frac) / 2), int(h * (1 + crop_frac) / 2)
    x0, x1 = int(w * (1 - crop_frac) / 2), int(w * (1 + crop_frac) / 2)
    core = v[y0:y1, x0:x1]
    mask = np.zeros_like(v, dtype=bool)
    mask[y0:y1, x0:x1] = core < np.median(core)
    return mask


def circular_mean_deg(degrees) -> float:
    """Mean of angles: atan2(mean sin, mean cos), wrapped into [0, 360).

    Hue is an angle on a colour wheel, so the arithmetic mean is simply wrong: it averages a
    pinkish-red 355 deg and an orange-brown 5 deg to a cyan 180 deg, the opposite side of the
    wheel. Roughly a third of the lesion pixels here sit above 180 deg, so that is not a
    hypothetical failure mode on this dataset.
    """
    rad = np.deg2rad(np.asarray(degrees, dtype=float))
    return float(np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360.0)


def circular_sd_deg(degrees) -> float:
    """Circular standard deviation, so a hue GAP can be read against hue SPREAD."""
    rad = np.deg2rad(np.asarray(degrees, dtype=float))
    resultant = np.hypot(np.sin(rad).mean(), np.cos(rad).mean())
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(max(resultant, 1e-12)))))


def circular_diff_deg(a, b) -> float:
    """Signed a - b wrapped into (-180, 180]: the short way round the wheel."""
    return float((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def hue_and_value(rgb_u8: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Circular-mean hue (degrees) and mean V over the masked pixels of one image."""
    hsv = rgb_to_hsv(rgb_u8.astype(np.float32) / 255.0)
    return circular_mean_deg(hsv[..., 0][mask] * 360.0), float(hsv[..., 2][mask].mean())


def working_image(isic_id: str) -> tuple[np.ndarray, Image.Image]:
    """One image at WORK_SIZE, as uint8 array and as PIL (ColorJitter needs PIL)."""
    # preprocess_image is the project's own resize, so the audit sees exactly the geometry
    # the training pipeline sees rather than a second, slightly different downscale.
    arr = preprocessing.preprocess_image(isic_id, size=WORK_SIZE, to_float=False)
    return arr, Image.fromarray(arr)


# --- Show the proxy on one lesion from each class --------------------------
examples = [sample[sample["diagnosis_1"] == g].iloc[0] for g in ("Benign", "Malignant")]
fig, axes = plt.subplots(2, 3, figsize=(9.0, 6.2))

for row, ex in zip(axes, examples):
    arr, _ = working_image(ex["isic_id"])
    hsv = rgb_to_hsv(arr.astype(np.float32) / 255.0)
    mask = lesion_proxy_mask(hsv[..., 2])
    lo, hi = int(WORK_SIZE * (1 - CROP_FRAC) / 2), int(WORK_SIZE * (1 + CROP_FRAC) / 2)

    # Rejected pixels are washed towards white so the kept ones read at a glance.
    dimmed = np.where(mask[..., None], arr,
                      0.82 * 255 + 0.18 * arr.astype(np.float32)).astype(np.uint8)

    row[0].imshow(arr)
    row[0].add_patch(plt.Rectangle((lo, lo), hi - lo, hi - lo, fill=False,
                                   edgecolor=config.ORANGE, linewidth=2))
    row[0].set_title(f"{ex['diagnosis_1']} ({ex['dx']})\noriginal + central crop", fontsize=9)
    row[1].imshow(arr[lo:hi, lo:hi])
    row[1].set_title("central crop", fontsize=9)
    row[2].imshow(dimmed)
    row[2].set_title(f"proxy lesion pixels ({mask.mean():.1%} of frame)", fontsize=9)
    for ax in row:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

fig.suptitle("The lesion-pixel proxy: centred crop, then darker than that crop's median",
             fontsize=11)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Measuring the class gap and the jitter-induced shift
#
# Each image is measured once in its original state and once per jitter setting. Two details
# make the numbers comparable rather than merely adjacent:
#
# * the mask is computed **on the original image and then reused** for every jittered
#   version. Re-deriving it after a brightness change would silently select a different set
#   of pixels and mix a mask effect into the colour measurement;
# * each jitter is applied at its **maximum** shift, not at a random draw, by passing a
#   degenerate range to `ColorJitter` (`hue=(h, h)`, `brightness=(1-b, 1-b)`). For brightness
#   we take the darkening end, because the brightening end clips at 255 and would understate
#   the shift the transform can actually produce.

# %%
records = []
for isic_id, group in zip(sample["isic_id"], sample["diagnosis_1"]):
    arr, pil = working_image(isic_id)
    hsv = rgb_to_hsv(arr.astype(np.float32) / 255.0)
    mask = lesion_proxy_mask(hsv[..., 2])
    frame = np.ones_like(mask, dtype=bool)          # the control: the same stats, all pixels

    hue0, v0 = hue_and_value(arr, mask)
    hue0_f, v0_f = hue_and_value(arr, frame)
    rec = {"isic_id": isic_id, "group": group, "hue": hue0, "v": v0,
           "hue_frame": hue0_f, "v_frame": v0_f, "mask_frac": float(mask.mean())}

    for h in HUE_SETTINGS:
        jittered = np.asarray(transforms.colorjitter_probe(pil, hue=(h, h)))
        hue1, v1 = hue_and_value(jittered, mask)
        rec[f"hue={h:g}|dhue"] = abs(circular_diff_deg(hue1, hue0))
        rec[f"hue={h:g}|dv"] = abs(v1 - v0)

    for b in BRIGHTNESS_SETTINGS:
        jittered = np.asarray(transforms.colorjitter_probe(pil, brightness=(1 - b, 1 - b)))
        hue1, v1 = hue_and_value(jittered, mask)
        rec[f"brightness={b:g}|dhue"] = abs(circular_diff_deg(hue1, hue0))
        rec[f"brightness={b:g}|dv"] = abs(v1 - v0)

    records.append(rec)

per_image = pd.DataFrame(records)


def group_summary(hue_col: str, v_col: str) -> pd.DataFrame:
    """Per-class mean and spread of the two statistics, for one pixel definition."""
    return pd.DataFrame([
        {"group": g, "n": len(sub),
         "mean hue (deg)": circular_mean_deg(sub[hue_col]),
         "hue sd (deg)": circular_sd_deg(sub[hue_col]),
         "mean V": sub[v_col].mean(), "V sd": sub[v_col].std()}
        for g, sub in per_image.groupby("group")
    ])


print("proxy lesion pixels")
print(group_summary("hue", "v").round(3).to_string(index=False))
print("\nwhole frame (control)")
print(group_summary("hue_frame", "v_frame").round(3).to_string(index=False))

ben = per_image[per_image["group"] == "Benign"]
mal = per_image[per_image["group"] == "Malignant"]


def class_gap(hue_col: str, v_col: str) -> tuple[float, float]:
    """|Benign - Malignant| in circular-mean hue (deg) and in mean V."""
    return (abs(circular_diff_deg(circular_mean_deg(ben[hue_col]),
                                  circular_mean_deg(mal[hue_col]))),
            abs(float(ben[v_col].mean() - mal[v_col].mean())))


GAP_HUE, GAP_V = class_gap("hue", "v")
GAP_HUE_F, GAP_V_F = class_gap("hue_frame", "v_frame")
print(f"\nclass gap, proxy lesion pixels: hue {GAP_HUE:6.2f} deg   V {GAP_V:.4f}")
print(f"class gap, whole frame        : hue {GAP_HUE_F:6.2f} deg   V {GAP_V_F:.4f}")
print(f"mean proxy mask size          : {100 * per_image['mask_frac'].mean():.1f}% of the frame")

# %% [markdown]
# The two pixel definitions disagree sharply, which is the first real finding and the reason
# both are carried through the table below. Before reading it, one check that the instrument
# works: a `hue=h` jitter shifts every pixel by exactly `h * 360` degrees, and a
# `brightness=b` jitter at its darkening end multiplies V by `1-b`, so the measured shifts
# have to land on those numbers. Both are asserted rather than eyeballed.

# %%
rows = []
for h in HUE_SETTINGS:
    dh = per_image[f"hue={h:g}|dhue"].mean()
    rows.append({"transform": f"ColorJitter hue={h:g}", "nominal": f"{h * 360:.1f} deg",
                 "d hue (deg)": dh, "d V": per_image[f"hue={h:g}|dv"].mean(),
                 "x gap (proxy)": dh / GAP_HUE, "x gap (frame)": dh / GAP_HUE_F})
    # torchvision's 8-bit HSV round trip loses a little of the shift, but not much.
    assert abs(dh - h * 360) < 0.25 * h * 360, f"hue={h} moved {dh:.2f} deg, expected ~{h * 360:.1f}"

mean_v = per_image["v"].mean()
for b in BRIGHTNESS_SETTINGS:
    dv = per_image[f"brightness={b:g}|dv"].mean()
    rows.append({"transform": f"ColorJitter brightness={b:g}"
                              + ("  <- project setting" if b == 0.20 else ""),
                 "nominal": f"{b * mean_v:.3f} V",
                 "d hue (deg)": per_image[f"brightness={b:g}|dhue"].mean(), "d V": dv,
                 "x gap (proxy)": dv / GAP_V, "x gap (frame)": dv / GAP_V_F})
    assert abs(dv - b * mean_v) < 0.15 * b * mean_v, f"brightness={b} moved V by {dv:.3f}"

gaps = [
    {"transform": "class gap: proxy lesion pixels", "nominal": "measured",
     "d hue (deg)": GAP_HUE, "d V": GAP_V, "x gap (proxy)": 1.0, "x gap (frame)": np.nan},
    {"transform": "class gap: whole frame (control)", "nominal": "measured",
     "d hue (deg)": GAP_HUE_F, "d V": GAP_V_F, "x gap (proxy)": np.nan, "x gap (frame)": 1.0},
]
audit = pd.DataFrame([*gaps, *rows])

# The ratio columns already use the channel each knob actually moves, so the verdict is
# just how many of the two definitions the setting exceeds.
over = (audit["x gap (proxy)"] > 1).astype(int) + (audit["x gap (frame)"] > 1).astype(int)
audit["exceeds"] = np.where(audit["nominal"] == "measured", "-",
                            np.select([over == 2, over == 1], ["both", "one"], "neither"))

print(audit.round({"d hue (deg)": 2, "d V": 4, "x gap (proxy)": 2, "x gap (frame)": 2})
           .to_string(index=False, na_rep=""))

# %%
LEVELS = {2: ("exceeds both definitions", config.BENIGN_MALIGNANT_COLORS["Malignant"]),
          1: ("exceeds one", config.ORANGE),
          0: ("below both", "#BFBFBF")}

fig, (ax_h, ax_v) = plt.subplots(1, 2, figsize=(11, 4.4))

panels = [
    (ax_h, [f"{h:g}" for h in HUE_SETTINGS],
     [per_image[f"hue={h:g}|dhue"].mean() for h in HUE_SETTINGS],
     GAP_HUE, GAP_HUE_F, "ColorJitter hue", "hue shift (degrees, log scale)",
     "Hue shift vs the Benign/Malignant hue gap", "{:.1f}", "{:.2f}", True),
    (ax_v, [f"{b:g}" for b in BRIGHTNESS_SETTINGS],
     [per_image[f"brightness={b:g}|dv"].mean() for b in BRIGHTNESS_SETTINGS],
     GAP_V, GAP_V_F, "ColorJitter brightness", "brightness shift (V)",
     "Brightness shift vs the Benign/Malignant V gap", "{:.3f}", "{:.3f}", False),
]

for ax, names, values, gap_p, gap_f, xlabel, ylabel, title, fmt, gap_fmt, log in panels:
    levels = [int(x > gap_p) + int(x > gap_f) for x in values]
    ax.bar(names, values, color=[LEVELS[lv][1] for lv in levels])
    ax.axhline(gap_p, color=config.GREY, linestyle="--", linewidth=1.8,
               label=f"class gap, lesion proxy ({gap_fmt.format(gap_p)})")
    ax.axhline(gap_f, color=config.GREY, linestyle=":", linewidth=1.8,
               label=f"class gap, whole frame ({gap_fmt.format(gap_f)})")
    if log:
        ax.set_yscale("log")   # the bars span 6 to 178 degrees; linear would hide all but one
    for i, x in enumerate(values):
        ax.text(i, x * 1.08 if log else x + 0.01, fmt.format(x), ha="center", fontsize=9)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, (_, c) in sorted(LEVELS.items())]
fig.legend(handles, [LEVELS[k][0] for k in sorted(LEVELS)], loc="lower center",
           ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.06))
plt.tight_layout()
plt.show()

# %%
# What those degrees look like: the same lesion under each setting, so the table can be
# checked against an eye.
_, demo_pil = working_image(examples[1]["isic_id"])
strip = [("original", demo_pil)]
strip += [(f"hue={h:g} ({h * 360:.0f} deg)", transforms.colorjitter_probe(demo_pil, hue=(h, h)))
          for h in (0.02, 0.10, 0.50)]
strip += [(f"brightness={b:g}", transforms.colorjitter_probe(demo_pil, brightness=(1 - b, 1 - b)))
          for b in (0.20, 0.60)]

fig, axes = plt.subplots(1, len(strip), figsize=(2.05 * len(strip), 2.6))
for ax, (title, img) in zip(axes, strip):
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.suptitle(f"One {examples[1]['dx']} lesion at each jitter setting, maximum shift",
             fontsize=11)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Answer (a) — which parameters change colour more than the class gap?** Measured on 100
# Benign and 100 Malignant dermoscopic training images, over the proxy lesion pixels, the
# classes sit **0.54 degrees apart in circular-mean hue** and **0.142 apart in mean V**. On
# that yardstick **all four hue settings exceed the gap** — 0.02 shifts hue by 6.30 degrees
# (11.57x the gap), 0.05 by 16.05 (29.50x), 0.1 by 34.25 (62.95x), 0.5 by 178.06 (327.30x) —
# while for brightness
# **0.3 (0.177 V, 1.24x) and 0.6 (0.351 V, 2.47x) exceed it** and 0.1 (0.060 V, 0.42x) and
# 0.2 (0.118 V, 0.83x) stay under.
#
# That is not the expected result for hue, and the reason is worth stating plainly: the hue
# gap is essentially zero. Benign lesions average 359.9 degrees and Malignant 0.4 degrees —
# both classes are red — so "bigger than the hue gap" is a test that any non-zero hue jitter
# passes and that therefore certifies nothing. Mean lesion hue does not separate these
# classes at all. That negative result is still useful twice over: it says a model cannot be
# taking a shortcut on average lesion hue, and it warns that a near-zero denominator makes
# the ratio column meaningless on its own.
#
# **The proxy changes the answer, so both definitions are reported.** On the whole frame the
# gaps are 10.22 degrees and 0.0398 V — the hue gap is 19x larger and the V gap 3.6x smaller
# than on the proxy pixels. Under that definition the verdicts flip: hue=0.02 falls *below*
# the gap (0.62x) while brightness=0.1 and 0.2 rise *above* it (1.51x and 2.97x). **No tested
# setting is below the class gap under both definitions.** Without a real segmentation mask
# this audit can bound the risk but cannot settle it, and that limitation is a finding rather
# than a failure — it is exactly why the decision belongs to a clinician.
#
# **A sturdier yardstick, and the consequence for `train_transform` (hue=0.02,
# brightness=0.2).** A difference of means over 100 images is fragile; the within-class
# spread is not. Per-image hue varies with a circular sd of 28.9 degrees (Benign) and 19.3
# (Malignant), and per-image mean V with an sd of 0.158 and 0.119. Against that, hue=0.02's
# 6.30 degrees is between a fifth and a third of the natural within-class variation and is
# defensible — it is also the size of drift a different dermatoscope's white balance would
# produce — whereas hue=0.1
# (34.25 degrees) exceeds the spread of both classes and hue=0.5 (178.06 degrees) rotates a
# red lesion to cyan, as the strip above shows. So the measurements **support hue=0.02** and
# rule out the torchvision-tutorial default of 0.5 outright. They do **not** comfortably
# support brightness=0.2: its 0.1182 shift is 83% of the entire proxy class gap, 2.97x the
# whole-frame gap, and 0.75 to 0.99 of a within-class standard deviation (0.158 Benign, 0.119
# Malignant), which means a single augmented image can be shifted about as far as a typical
# Benign/Malignant brightness difference. **Tighten brightness
# from 0.2 to 0.1**; even 0.1 stays above the whole-frame gap, so the residual risk is a
# question for a clinician rather than for another parameter sweep. *This recommendation
# was acted on:* `transforms.JITTER_BRIGHTNESS` is now 0.1, so the pipeline follows its own
# audit. The 0.2 row above is kept because it is the setting the audit rejected.

# %% [markdown]
# **Answer (b).** The geometric augmentations are safe here — horizontal and vertical flips,
# rotation, and mild scale or crop — because a dermatoscope is applied at whatever angle is
# convenient and skin has no canonical "up", so every transformed image is one that could
# genuinely have been captured; that is a domain-specific ruling and not a universal one,
# since a vertical flip of a chest X-ray produces an anatomically impossible image. Anything
# touching colour is risky, and so is aggressive cropping: pigmentation, erythema and the
# blue-white veil are diagnostic criteria in their own right, a hue shift large enough to
# move a lesion across the class colour gap has changed the label while leaving the filename
# alone, and a hard crop can cut off the lesion border, which is half of the ABCD rule. The
# audit above is why the colour knobs sit where they do rather than at library defaults, and
# it says the brightness setting is the one running closest to its limit and should come
# down. Approval is not an engineering call: every augmentation asserts that a transformation
# never changes the diagnosis, which is a medical claim, so a dermatologist or
# dermatopathologist has to sign it off. The engineer's job is the one done here — quantify
# the shift in the same units as the real class difference, and be explicit about what the
# measurement cannot settle — so that the clinician rules on a number instead of a vibe, and
# so the choice is still auditable after everyone who made it has left the project.
