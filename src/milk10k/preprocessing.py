"""Turning MILK10k JPEGs into arrays, and checking that they are really there.

This is the layer that sits between the CSV tables and anything that takes
tensors. It does four jobs, and deliberately nothing else:

1. **Availability** — which ``isic_id`` actually has a file on disk
   (:func:`available_subset`, :func:`missing_images`).
2. **Loading** — one image, failing loudly when the file is absent
   (:func:`load_image`).
3. **Preprocessing** — one image or a batch, to a fixed-size array
   (:func:`preprocess_image`, :func:`preprocess_batch`).
4. **Normalisation statistics** — per-channel mean/std computed by streaming
   over the images, never by holding them all in memory
   (:func:`compute_channel_stats`).

Two rules run through the whole module. First, a *missing* file is an error,
not something to skip quietly: silently dropping images changes the class
balance behind your back. Second, the batch path calls the single-image path,
so training and inference cannot drift apart as the code evolves.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import config

# Resize filter. PIL's ``Image.resize`` applies a proper antialiasing
# convolution for BILINEAR when downscaling, so 600x450 -> 224x224 averages the
# pixels it throws away instead of point-sampling them (NEAREST would alias the
# fine dermoscopic texture, which is exactly the signal we care about).
# BILINEAR is also what torchvision's ``Resize`` uses by default, so the numpy
# path here and the tensor path in ``transforms.py`` agree.
RESAMPLE = Image.BILINEAR

# Suffixes we accept when a caller hands us a path instead of an ``isic_id``.
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# --- Where the files are ---------------------------------------------------
def _resolve_dir(img_dir: str | Path | None) -> Path:
    """Fall back to the configured image folder. No path is ever hard-coded.

    Guards against being handed an image *file* where a directory is expected —
    passing ``image_path(...)`` instead of the folder is an easy slip, and left
    unchecked it surfaces much later as a confusing doubled path
    (``.../images/ISIC_123.jpg/ISIC_123.jpg``) raised from inside a DataLoader
    worker. Failing here names the real mistake instead.
    """
    resolved = Path(config.IMG_DIR if img_dir is None else img_dir)
    if resolved.is_file():
        raise NotADirectoryError(
            f"img_dir must be the image FOLDER, but {resolved} is a file. "
            "Pass the directory (e.g. config.IMG_DIR), not a path to one image."
        )
    return resolved


def image_path(isic_id: str, img_dir: str | Path | None = None) -> Path:
    """Path a given ``isic_id`` is expected at. Says nothing about existence."""
    return _resolve_dir(img_dir) / f"{isic_id}.jpg"


def available_ids(img_dir: str | Path | None = None) -> set[str]:
    """The ids that have a JPEG, read from a single directory listing.

    One ``os.scandir`` beats 10,480 ``Path.exists()`` calls: each ``exists()``
    is its own system call, whereas the listing is one pass over the directory
    that the operating system already has cached. The result is a ``set`` so
    that membership testing is O(1).
    """
    directory = _resolve_dir(img_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {directory}")

    with os.scandir(directory) as entries:
        return {
            Path(e.name).stem
            for e in entries
            if e.is_file() and Path(e.name).suffix.lower() in _IMAGE_SUFFIXES
        }


def available_subset(
    df: pd.DataFrame, img_dir: str | Path | None = None
) -> pd.DataFrame:
    """Keep only the rows whose image file exists on disk.

    Used when working from a partial download. Filtering is a vectorised
    ``isin`` against the directory listing, so cost is one listing plus one
    pass over the column, not one file check per row.

    Note what this function is *for*: exploring an incomplete copy of the
    dataset. It must never be used to paper over missing files in a real
    training run — that is what :func:`missing_images` is for.
    """
    present = available_ids(img_dir)
    return df[df["isic_id"].isin(present)].copy()


def missing_images(
    df: pd.DataFrame, img_dir: str | Path | None = None
) -> list[str]:
    """The ids in ``df`` with no file on disk — the complement of the above.

    Returned as a plain list so it can be printed, counted or asserted empty in
    the data-quality report.
    """
    present = available_ids(img_dir)
    return df.loc[~df["isic_id"].isin(present), "isic_id"].tolist()


# --- Loading ---------------------------------------------------------------
def load_image(
    isic_id: str,
    img_dir: str | Path | None = None,
    allow_missing: bool = False,
) -> Image.Image | None:
    """Open one image as RGB, raising if the file is not there.

    Failing loudly is the point. A pipeline that skips unreadable files quietly
    trains on a different dataset than the one you documented, and the gap only
    shows up as an unexplained metric later. The error names both the id and the
    full path so the cause (wrong ``MILK10K_IMAGES_DIR``, incomplete download,
    typo in the id) is obvious from the message alone.

    Pass ``allow_missing=True`` — and only then — to get ``None`` back instead,
    for code that is explicitly auditing which files exist.

    ``.convert("RGB")`` guarantees three channels even if a source file is
    greyscale or carries an alpha channel, so downstream shapes are stable.
    """
    path = image_path(isic_id, img_dir)
    if not path.is_file():
        if allow_missing:
            return None
        raise FileNotFoundError(
            f"no image file for isic_id {isic_id!r}: expected {path}"
        )
    with Image.open(path) as im:
        return im.convert("RGB")


def _as_pil(img: Image.Image | str | Path, img_dir: str | Path | None) -> Image.Image:
    """Accept a PIL image, a file path or a bare ``isic_id`` and return a PIL image.

    Having one place that decides what a caller meant keeps the three public
    entry points from each inventing their own slightly different rule.
    """
    if isinstance(img, Image.Image):
        return img.convert("RGB")

    if isinstance(img, Path) or (
        isinstance(img, str) and Path(img).suffix.lower() in _IMAGE_SUFFIXES
    ):
        path = Path(img)
        if not path.is_file():
            raise FileNotFoundError(f"image file not found: {path}")
        with Image.open(path) as im:
            return im.convert("RGB")

    if isinstance(img, str):
        return load_image(img, img_dir=img_dir)

    raise TypeError(
        f"expected a PIL image, a path or an isic_id string, got {type(img).__name__}"
    )


# --- Preprocessing ---------------------------------------------------------
def preprocess_image(
    img: Image.Image | str | Path,
    size: int = config.IMAGE_SIZE,
    to_float: bool = True,
    grayscale: bool = False,
    img_dir: str | Path | None = None,
) -> np.ndarray:
    """Resize one image to ``size`` x ``size`` and return it as a numpy array.

    Accepts a PIL image, a path, or an ``isic_id``, so exploratory code and the
    dataloaders can share it.

    Every MILK10k file is 600x450, so this is a pure downscale to a square. The
    square distorts the aspect ratio by 4:3 rather than cropping, which is the
    safer trade here: a centre crop would cut the left and right edges off, and
    a lesion's border is diagnostic — an asymmetric or ragged edge is precisely
    what separates melanoma from a nevus. Mild horizontal squashing keeps the
    whole lesion in frame; the network sees it consistently in every image.

    ``to_float`` scales to [0, 1] float32, which is what torch expects before
    normalisation; leave it off to inspect the raw uint8 pixels. ``grayscale``
    returns HxW instead of HxWxC, for the colour-ablation experiment.

    Returns: (size, size, 3) — or (size, size) when ``grayscale`` — as float32
    in [0, 1] if ``to_float``, else uint8.
    """
    pil = _as_pil(img, img_dir)
    pil = pil.resize((size, size), RESAMPLE)
    if grayscale:
        # Convert after resizing: the filter then averages true colour pixels,
        # and "L" uses the luminance weights rather than a flat channel mean.
        pil = pil.convert("L")

    arr = np.asarray(pil, dtype=np.uint8)
    if to_float:
        # /255 puts the data in [0, 1]; float32 (not float64) because that is
        # what torch and every GPU kernel downstream actually use.
        return arr.astype(np.float32) / 255.0
    return arr


def preprocess_batch(
    isic_ids,
    size: int = config.IMAGE_SIZE,
    to_float: bool = True,
    grayscale: bool = False,
    allow_missing: bool = False,
    img_dir: str | Path | None = None,
) -> np.ndarray:
    """Preprocess many ids and stack them into one (N, H, W, C) array.

    It calls :func:`preprocess_image` per image on purpose: if the two ever
    diverged, a model would be trained on one preprocessing and evaluated on
    another, which is the kind of bug that produces a quietly wrong score
    instead of a crash.

    ``allow_missing=False`` (the default) means one absent file aborts the whole
    batch. With ``allow_missing=True`` the missing ids are dropped, and the
    returned array is shorter than ``isic_ids`` — so the caller must not assume
    the two line up positionally.

    Memory: N x size x size x 3 float32. At 224 that is ~600 KB per image, so a
    few thousand images is fine and the whole dataset is not — use
    :func:`compute_channel_stats` or a DataLoader for that.
    """
    ids = list(isic_ids)
    arrays = []

    for isic_id in ids:
        # Check the path rather than loading the image, so a skipped id costs a
        # stat call and a kept one is decoded exactly once.
        if allow_missing and not image_path(isic_id, img_dir).is_file():
            continue
        arrays.append(
            preprocess_image(
                isic_id, size=size, to_float=to_float,
                grayscale=grayscale, img_dir=img_dir,
            )
        )

    if not arrays:
        # An empty stack would raise something unhelpful from numpy, so return a
        # correctly-shaped empty array instead.
        channels = () if grayscale else (3,)
        dtype = np.float32 if to_float else np.uint8
        return np.empty((0, size, size, *channels), dtype=dtype)

    return np.stack(arrays, axis=0)


# --- Normalisation statistics ----------------------------------------------
def compute_channel_stats(
    isic_ids,
    size: int = config.IMAGE_SIZE,
    max_images: int | None = None,
    seed: int = config.SEED,
    img_dir: str | Path | None = None,
) -> dict:
    """Per-channel mean and std in [0, 1], computed one image at a time.

    **Pass the TRAINING ids only.** Normalisation statistics are learned
    parameters: computing them over the whole dataset lets the mean brightness
    of the validation and test images influence how the training images are
    scaled. That is preprocessing leakage — the reported score comes out
    flattering and does not survive contact with genuinely unseen data. The same
    numbers are then applied unchanged to val and test, exactly like a fitted
    scaler.

    The accumulation is streaming: we keep a running sum and sum-of-squares per
    channel (six float64 numbers) and never hold more than one image in memory,
    so this scales to the full training split. float64 for the accumulators
    because summing millions of float32 values loses precision.

    Std comes from ``E[x^2] - E[x]^2``. That shortcut can go very slightly
    negative through floating-point error when a channel is nearly constant, so
    it is clipped at zero before the square root.

    ``max_images`` subsamples (seeded, so it is reproducible) when a quick
    estimate is enough; the exact count used is returned.

    Returns ``{"mean": [r, g, b], "std": [r, g, b], "n_images": int,
    "image_size": int}``.
    """
    ids = list(isic_ids)
    if max_images is not None and len(ids) > max_images:
        rng = np.random.default_rng(seed)
        picked = rng.choice(len(ids), size=max_images, replace=False)
        # Sort the positions so we read the ids in their original order — this
        # keeps the result independent of how numpy happened to order the draw.
        ids = [ids[i] for i in sorted(picked)]

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sqsum = np.zeros(3, dtype=np.float64)
    n_pixels = 0

    for isic_id in ids:
        arr = preprocess_image(isic_id, size=size, to_float=True, img_dir=img_dir)
        flat = arr.reshape(-1, 3).astype(np.float64)
        channel_sum += flat.sum(axis=0)
        channel_sqsum += (flat ** 2).sum(axis=0)
        n_pixels += flat.shape[0]

    if n_pixels == 0:
        raise ValueError("compute_channel_stats got no images to read")

    mean = channel_sum / n_pixels
    variance = np.clip(channel_sqsum / n_pixels - mean ** 2, 0.0, None)
    std = np.sqrt(variance)

    return {
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "n_images": len(ids),
        "image_size": int(size),
    }


def write_norm_stats(stats: dict, path: Path = config.NORM_STATS_JSON) -> Path:
    """Save the statistics to JSON so every later run normalises identically.

    Recomputing them per run would make two runs subtly incomparable; a
    committed file makes the numbers auditable and quotable in the report.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2) + "\n")
    return path


def load_norm_stats(path: Path = config.NORM_STATS_JSON) -> dict:
    """Read back the saved statistics, with a clear error if nobody wrote them."""
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"normalisation statistics not found at {path}; "
            "run compute_channel_stats + write_norm_stats on the TRAIN split first"
        )
    return json.loads(Path(path).read_text())


# --- File size (no decoding) -----------------------------------------------
def image_file_sizes(
    df: pd.DataFrame, img_dir: str | Path | None = None
) -> pd.DataFrame:
    """File size of each image in bytes and KB, read from the directory metadata.

    Deliberately never decodes a JPEG: ``os.scandir`` gives the size from the
    filesystem, so the full 10,480 files take well under a second.

    Why it matters (homework A2.2c): JPEG size is a compression artefact, not
    biology. If malignant lesions systematically produce larger files — because
    they were photographed with a different device, or are more textured and so
    compress worse — then a model can reach a decent score from acquisition
    trivia rather than from the lesion. Measuring the size lets us test that
    shortcut instead of assuming it away.

    Ids with no file get NaN rather than being dropped, so the row count still
    matches the input and the gap is visible.
    """
    directory = _resolve_dir(img_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {directory}")

    with os.scandir(directory) as entries:
        sizes = {
            Path(e.name).stem: e.stat().st_size
            for e in entries
            if e.is_file() and Path(e.name).suffix.lower() in _IMAGE_SUFFIXES
        }

    out = pd.DataFrame({"isic_id": df["isic_id"].to_numpy()})
    # One vectorised lookup against the listing beats a stat call per row.
    out["file_size_bytes"] = out["isic_id"].map(sizes)
    out["file_size_kb"] = out["file_size_bytes"] / 1024.0
    return out
