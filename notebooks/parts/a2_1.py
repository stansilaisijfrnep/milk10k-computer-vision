# %% [markdown]
# ## A2.1 Arrays, views and memory
#
# - An image is just a NumPy array, and flips and rotations are just indexing.
# - Here I (a) redo Pillow's four transposes with slicing, (b) check which ones are views vs copies, (c) work out how much RAM the whole dataset would need.

# %% [markdown]
# ### (a) Four transforms in NumPy, checked against Pillow
#
# - Each one is an indexing expression, compared pixel by pixel with Pillow.
# - Two are easy to get wrong:
#   - Pillow's `ROTATE_90` goes **counter-clockwise**
#   - transpose of an `H x W x 3` image swaps only the first two axes; `arr.T` would also flip the channels

# %%
from PIL import Image

# One real image, chosen deterministically so the section reproduces exactly.
meta = data.load_metadata()
isic_id = (meta.loc[meta["image_type"] == data.DERMOSCOPIC_TYPE, "isic_id"]
           .sort_values().iloc[0])

# preprocessing.load_image fails loudly on a missing file and guarantees RGB.
img = preprocessing.load_image(isic_id)
# np.array (not np.asarray) copies, so the array is writeable — part (b) needs
# a writeable base to show aliasing on.
arr = np.array(img)

print(f"image      : {isic_id}")
print(f"array shape: {arr.shape}  dtype: {arr.dtype}")
assert arr.shape == (450, 600, 3) and arr.dtype == np.uint8

# Slicing expressions. A negative step reverses an axis; transpose(1, 0, 2)
# swaps rows and columns and leaves the channel axis alone.
variants = {
    "left-right flip": ("arr[:, ::-1]",           arr[:, ::-1],            Image.Transpose.FLIP_LEFT_RIGHT),
    "up-down flip":    ("arr[::-1]",              arr[::-1],               Image.Transpose.FLIP_TOP_BOTTOM),
    "rotate 90 CCW":   ("np.rot90(arr, k=1)",     np.rot90(arr, k=1),      Image.Transpose.ROTATE_90),
    "transpose":       ("arr.transpose(1, 0, 2)", arr.transpose(1, 0, 2),  Image.Transpose.TRANSPOSE),
}

for name, (expr, mine, pil_op) in variants.items():
    theirs = np.asarray(img.transpose(pil_op))
    assert np.array_equal(mine, theirs), f"{name} does not match Pillow"
    print(f"identical to Pillow: {name:<16} {expr:<24} {str(mine.shape):<14} OK")

# %%
# The rotation written as pure slicing: reverse the columns, then swap the row
# and column axes. Both orderings give the same counter-clockwise rotation.
assert np.array_equal(np.rot90(arr, k=1), arr[:, ::-1].transpose(1, 0, 2))
assert np.array_equal(np.rot90(arr, k=1), arr.transpose(1, 0, 2)[::-1])

# The two traps, stated as assertions rather than as prose.
# 1. A transpose is a mirror across the diagonal, NOT a rotation.
assert not np.array_equal(arr.transpose(1, 0, 2), np.rot90(arr, k=1))
# 2. arr.T reverses ALL axes, so the channels end up as the leading axis.
print(f"arr.T.shape                = {arr.T.shape}   <- channels first, not an image")
print(f"arr.transpose(1, 0, 2).shape = {arr.transpose(1, 0, 2).shape}   <- what PIL TRANSPOSE means")

# Clockwise would be np.rot90(arr, k=-1); PIL's ROTATE_90 is counter-clockwise,
# which is why k=+1 is the matching call.
assert not np.array_equal(np.rot90(arr, k=1), np.rot90(arr, k=-1))

# %%
fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
panels = [("original", "arr", arr)] + [(name, expr, v)
                                       for name, (expr, v, _) in variants.items()]

for ax, (title, expr, im) in zip(axes, panels):
    ax.imshow(im)
    ax.set_title(f"{title}\n{expr}\n{im.shape[0]}x{im.shape[1]}", fontsize=8)
    ax.axis("off")

fig.suptitle(f"A2.1a — NumPy transforms of {isic_id} (each verified identical to Pillow)",
             fontsize=11)
plt.tight_layout()
plt.show()

# %% [markdown]
# **My answer (a)**
#
# - All four match Pillow exactly (`np.array_equal` is `True`, asserts pass).
# - `ROTATE_90` = `np.rot90(arr, k=1)`; `k=-1` would be clockwise and different.
# - `TRANSPOSE` = `arr.transpose(1, 0, 2)`; `arr.T` gives `(3, 600, 450)`, which isn't an image any more.
# - Transpose is not a rotation, it mirrors over the diagonal.

# %% [markdown]
# ### (b) View or copy?
#
# - If indexing only changes the strides, NumPy gives back a new object on the same memory.
# - `np.shares_memory` tells me which case I'm in.

# %%
lr = arr[:, ::-1]
pil_roundtrip = np.asarray(img.transpose(Image.Transpose.FLIP_LEFT_RIGHT))

checks = {
    "arr[:, ::-1]":                     lr,
    "arr[::-1]":                        arr[::-1],
    "np.rot90(arr, k=1)":               np.rot90(arr, k=1),
    "arr.transpose(1, 0, 2)":           arr.transpose(1, 0, 2),
    "arr[:, ::-1].copy()":              lr.copy(),
    "np.ascontiguousarray(arr[:, ::-1])": np.ascontiguousarray(lr),
    "np.asarray(pil.transpose(...))":   pil_roundtrip,
}

rows = [
    {
        "expression": expr,
        "shape": str(a.shape),
        "shares_memory": np.shares_memory(arr, a),
        "view_or_copy": "view" if np.shares_memory(arr, a) else "copy",
        "C_contiguous": a.flags["C_CONTIGUOUS"],
    }
    for expr, a in checks.items()
]
print(pd.DataFrame(rows).to_string(index=False))

# %%
# Two lines: take a view, write into it. Nothing else touches `demo`.
demo = arr.copy()
before = demo.copy()          # kept only so the "before" panel is honest

MAGENTA = (255, 0, 255)       # a colour no skin photograph contains

ud_view = demo[::-1]              # 1. a view — no pixels copied
ud_view[40:80, :, :] = MAGENTA    # 2. the write lands in `demo` itself

# The stripe was written at rows 40:80 of the flipped view, so in the original
# it appears at the mirrored position, counted from the bottom.
h = demo.shape[0]
assert (demo[h - 80:h - 40] == MAGENTA).all()
assert not (before[h - 80:h - 40] == MAGENTA).all()
print(f"magenta stripe written at rows 40:80 of the view -> appears at rows "
      f"{h - 80}:{h - 40} of the original (which was never assigned to)")

fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
for ax, (title, im) in zip(axes, [("original, before", before),
                                  ("the flipped VIEW, after the write", ud_view),
                                  ("original, after — corrupted", demo)]):
    ax.imshow(im)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
fig.suptitle("A2.1b — writing into a view writes into the original", fontsize=11)
plt.tight_layout()
plt.show()

# %% [markdown]
# **My answer (b)**
#
# - All four operations from (a) return **views**: `np.shares_memory` is `True` for both flips, the rotation and the transpose.
# - A **copy** only happens when I force it: `.copy()`, `np.ascontiguousarray`, or going through Pillow.
# - My experiment: I wrote a magenta stripe into rows 40:80 of the flipped view, and it showed up in rows 370:410 of the original.
# - Why I think this matters: if a pipeline caches images and an augmentation writes into a view, it silently damages the cached original for every later epoch. So copy before writing in place.

# %% [markdown]
# ### (c) Memory budget for the whole dataset

# %%
n_images = len(meta)
assert n_images == 10480

# Row labels use the WxH convention images are quoted in; the tuples are the
# array's own HxWxC shape, taken from the image we actually decoded.
shapes = {
    "600x450x3 (original)": arr.shape,
    "224x224x3 (model input)": (config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
}
dtypes = ["uint8", "float32"]


def dataset_bytes(n: int, shape: tuple[int, ...], dtype: str) -> int:
    """Bytes to hold n decoded images of this shape and dtype, all at once."""
    return n * int(np.prod(shape)) * np.dtype(dtype).itemsize


budget = pd.DataFrame(
    [[dataset_bytes(n_images, shape, dt) for dt in dtypes] for shape in shapes.values()],
    index=list(shapes), columns=dtypes,
)

print(f"{n_images:,} images\n")
print("RAM if fully decoded and held in memory — GB (10^9 bytes)")
print((budget / 1e9).round(2).to_string(), "\n")
print("...the same thing in MB (10^6 bytes)")
print((budget / 1e6).round(0).to_string())

# %%
# What the dataset actually costs on disk, as JPEG — the comparison that makes
# lazy decoding attractive. Read from the directory listing, nothing is decoded.
disk_bytes = preprocessing.image_file_sizes(meta)["file_size_bytes"].sum()
uint8_full = dataset_bytes(n_images, arr.shape, "uint8")

print(f"JPEGs on disk           : {disk_bytes / 1e9:.2f} GB")
print(f"decoded uint8 @600x450  : {uint8_full / 1e9:.2f} GB "
      f"({uint8_full / disk_bytes:.0f}x larger than on disk)")

# A batch is the unit that actually has to be float32, and it is tiny.
batch_size = 32
batch_mb = dataset_bytes(batch_size, shapes["224x224x3 (model input)"], "float32") / 1e6
print(f"one float32 batch of {batch_size} @224: {batch_mb:.1f} MB")

# The machine this notebook ran on. os.sysconf works on macOS and Linux; on
# Windows it is missing, so fall back to a stated 16 GB assumption.
import os
try:
    ram_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    ram_src = "this machine"
except (AttributeError, ValueError, OSError):
    ram_bytes, ram_src = 16e9, "assumed 16 GB laptop"
print(f"\nphysical RAM ({ram_src}): {ram_bytes / 1e9:.1f} GB")
for shape_name in budget.index:
    for dt in dtypes:
        share = budget.loc[shape_name, dt] / ram_bytes
        print(f"  {shape_name:<24} {dt:<8} {100 * share:5.0f}% of RAM  "
              f"-> {'fits' if share < 0.5 else 'does NOT fit comfortably'}")

# %% [markdown]
# **My answer (c)**
#
# - Full dataset in RAM:
#   - 600x450: **8.49 GB** as uint8, **33.96 GB** as float32
#   - 224x224: **1.58 GB** as uint8, **6.31 GB** as float32
# - On the 51.5 GB machine I ran this on, everything except float32 at full size fits.
# - On a normal 16 GB laptop, I think only the 224 versions are realistic.
# - The dtype hurts most: float32 is 4x uint8.
# - **What I'd do instead:** lazy loading, i.e. a `Dataset` that reads one JPEG at a time, with DataLoader workers preparing the next batch. That's what my `MilkImageDataset` does, and the JPEGs are only 0.35 GB on disk.
# - Alternatives: pre-resize once to a 224 uint8 cache (fast, but fixes the resolution), or decode on the fly (flexible, costs CPU).
# - Either way, convert to float32 per batch (one batch of 32 at 224 is 19.3 MB), never for the whole dataset.

