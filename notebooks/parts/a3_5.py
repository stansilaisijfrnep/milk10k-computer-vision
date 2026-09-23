# %% [markdown]
# ## A3.5 Is this augmentation label-safe?
#
# - Every augmentation claims "this never changes the diagnosis".
# - For flips and rotations that's easy to believe. For colour it isn't, because pigment and redness *are* part of the diagnosis.
# - So I measure (1) how far apart Benign and Malignant lesions really are in hue and brightness, and (2) how far each `ColorJitter` setting moves the same images.
# - If a jitter moves colour more than the class gap, it could effectively relabel an image.

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
# ### My "lesion pixels" are only a proxy
#
# - MILK10k has no segmentation masks, so I use a simple rule:
#   1. keep the **central 50% of each side**, since the lesion is usually centred in dermoscopy
#   2. inside that, keep pixels **darker than the crop's median V**, since pigment is darker than skin
# - Where it fails: pink lesions that aren't darker (e.g. many BCCs), where it picks up surrounding skin.
# - The figure shows it on one lesion per class so you can judge it.
# - Because it's debatable, I also compute everything on the **whole image** and show both.

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
# ### Measuring the class gap and the jitter shift
#
# - The mask is computed on the original and **reused** for every jittered version; otherwise a brightness change would select different pixels.
# - Each jitter is applied at its **maximum** (`hue=(h, h)`, `brightness=(1-b, 1-b)`). For brightness I take the darkening side, because brightening clips at 255 and would understate the shift.

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
# - The two pixel definitions disagree a lot, which is my first finding and why I keep both in the table.
# - Sanity check: hue `h` should shift by exactly `h * 360` degrees and brightness `b` should multiply V by `1-b`. Both are asserted.

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
# **My answer (a): which settings change colour more than the class gap?**
#
# On the lesion proxy (100 Benign + 100 Malignant dermoscopic training images), the gap is **0.54° in hue** and **0.142 in V**.
#
# - **Hue:** all four settings exceed it: 0.02 → 6.30° (11.57x), 0.05 → 16.05° (29.50x), 0.1 → 34.25° (62.95x), 0.5 → 178.06° (327.30x).
# - **Brightness:** 0.3 (0.177, 1.24x) and 0.6 (0.351, 2.47x) exceed it; 0.1 (0.060, 0.42x) and 0.2 (0.118, 0.83x) stay under.
#
# What I make of it:
#
# - The hue gap is basically zero (Benign 359.9°, Malignant 0.4°, both red), so "bigger than the hue gap" says nothing. Average lesion hue simply doesn't separate the classes, which also means a model can't cheat with it.
# - On the whole image the gaps are 10.22° and 0.0398 V, and the verdicts flip: hue 0.02 is *below* the gap (0.62x), brightness 0.1 and 0.2 are *above* (1.51x, 2.97x). **No setting is below the gap under both definitions.** Without real masks I can't fully settle it.
# - A steadier comparison is the spread *within* each class: hue sd 28.9° (Benign) / 19.3° (Malignant), V sd 0.158 / 0.119.
#   - **hue 0.02** (6.30°) is a fifth to a third of that, so I think it's fine; hue 0.1 exceeds it and hue 0.5 turns red lesions cyan.
#   - **brightness 0.2** shifts 0.1182, i.e. 83% of the proxy gap and 0.75–0.99 of a within-class sd, which is too much in my opinion.
# - **Decision:** keep hue 0.02, lower brightness from 0.2 to 0.1. I did that: `transforms.JITTER_BRIGHTNESS` is now 0.1. I kept the 0.2 row to show what was rejected.

# %% [markdown]
# **My answer (b)**
#
# - **Safe, I think:** flips, rotation and mild crops, because a dermatoscope can be held at any angle and skin has no "up" (unlike a chest X-ray, where a vertical flip would be anatomically impossible).
# - **Risky:** anything touching colour, since pigment, redness and the blue-white veil are diagnostic, so a big enough shift changes the lesion but not the label.
# - **Also risky:** aggressive crops, because they can cut off the lesion border, which is half of the ABCD rule.
# - **From my audit:** brightness was the setting closest to the limit, which is why I lowered it to 0.1.
# - **Who approves:** a dermatologist or dermatopathologist, because "this never changes the diagnosis" is a medical claim. My job is to measure the shift in the same units as the class difference, so they can decide based on numbers.

