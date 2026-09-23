# %% [markdown]
# ## A2.1 Arrays, views and memory
#
# An image is a NumPy array, and the cheap geometric augmentations are array
# indexing. This section does three things: reproduce Pillow's four transposes
# with nothing but slicing, find out which of those operations hand back a view
# into the same memory rather than a new array, and compute what it would cost
# in RAM to hold the whole dataset decoded.

# %% [markdown]
# ### (a) Four transforms in NumPy, checked against Pillow
#
# Each transform is written as an indexing expression and then compared
# element-by-element with the Pillow operation it is supposed to reproduce.
# Two of the four are easy to get wrong, so they are worth stating explicitly:
# Pillow's `ROTATE_90` turns **counter-clockwise**, and "transpose" of an
# `H x W x 3` array swaps the first two axes only — `arr.T` would also reverse
# the channel axis and give a `3 x W x H` array that is not an image any more.

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
# **Answer (a).** All four transforms reproduce Pillow exactly — `np.array_equal`
# is `True` for every pair, so the assertions pass. The two that need care:
# Pillow's `ROTATE_90` rotates counter-clockwise, which is `np.rot90(arr, k=1)`
# (`k=-1` is the clockwise one and gives a different array); and `TRANSPOSE`
# means swapping rows and columns only, `arr.transpose(1, 0, 2)`, because `arr.T`
# reverses all three axes and returns `(3, 600, 450)` — channels first, no longer
# an image. Transpose is also not a rotation: it mirrors across the main
# diagonal, which the assertion above confirms.

# %% [markdown]
# ### (b) View or copy?
#
# Indexing that only changes strides does not touch the pixels at all — NumPy
# returns a second object pointing at the same buffer. `np.shares_memory` says
# which is which.

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
# **Answer (b).** All four transforms from part (a) return **views**:
# `np.shares_memory(arr, ...)` is `True` for the two flips, the rotation and the
# transpose. None of them copies a pixel — they only rewrite the strides, which
# is why all four come back non-C-contiguous. A **copy** happens only when you
# force one: `.copy()`, `np.ascontiguousarray` on a non-contiguous view, or a
# round trip through Pillow. That is a cheap-augmentation win (a flip costs no
# memory and no pixel traffic) and a bug waiting to happen: the two-line
# experiment above writes a magenta stripe into rows 40:80 of the flipped view,
# and it appears at rows 370:410 of the original, which was never assigned to.
# In a preprocessing pipeline that caches decoded images, an in-place
# augmentation applied to a view of a cached array silently corrupts the cached
# original, so every later epoch trains on the damaged image and nothing raises.
# The fix is to copy before writing in place — or not to write in place at all.

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
# **Answer (c).** Decoded, this dataset is large: 8.49 GB as uint8 at the native
# 600x450 and 33.96 GB as float32, more than most laptops have. Resizing to the 224x224
# model input cuts that to 1.58 GB (uint8) and 6.31 GB (float32). Can it be
# pre-loaded? On the 51.5 GB machine this ran on, everything except float32 at
# full resolution technically fits (table above). On a typical 16 GB laptop only
# the 224 versions fit: full-resolution uint8 would take over half the RAM before
# the OS, Python and the model get any, and full-resolution float32 is twice the
# machine. Either way it is the dtype, not the pixel count, that does the most
# damage — float32 is a flat 4x on every cell. And even where it fits, a RAM
# cache written by one notebook is not something a DataLoader should rely on.
#
# What we do instead is lazy loading: a `Dataset` that decodes one JPEG per
# `__getitem__` and hands back a tensor, with DataLoader workers decoding the
# next batch while the GPU works on the current one. That is exactly what this
# project's `milk10k.datasets.MilkImageDataset` does, and it is affordable
# because the JPEGs on disk are only 0.35 GB in total — 24x smaller than the
# decoded array, and the operating system's page cache keeps the hot ones in
# memory for free. The honest alternatives: pre-resize once to 224 uint8 and
# memory-map that cache (fastest per epoch, costs 1.58 GB and an extra
# preprocessing pass, and it freezes the resize so resolution experiments need
# rebuilding), or decode on the fly (flexible, costs CPU per epoch). Either way
# the float32 conversion belongs at the end of the per-batch transform — one
# batch of 32 at 224 is 19.3 MB — never applied to the dataset as a whole.
