"""Session 2 homework, Parts 3-4: a configurable image pipeline and a batch loader.

This module is the framework-free layer that every exploratory script, figure
and future experiment calls to turn MILK10k files into arrays. It deliberately
sits *on top of* :mod:`milk10k.preprocessing` instead of beside it:

* the resize is :func:`preprocessing.preprocess_image` itself (same BILINEAR
  filter, same 224x224 target), so an image processed here is pixel-identical to
  one processed by the training pipeline before normalisation;
* the "which files are really on disk" check is
  :func:`preprocessing.available_ids` (one directory listing, not one
  ``Path.exists()`` per row);
* the z-score statistics default to ``artifacts/norm_stats.json``, which was
  computed on the **training split only**.

What this layer adds is what the Session 2 brief asks for and the lower layer
does not do: a choice of colour space and normalisation, a record of what was
done to each image, a batch function that *reports* failures instead of either
crashing or silently dropping them, and a loader that yields
``(images, labels)`` batches on the fly.

The torch path for training (augmentation, weighted sampling, workers) remains
:func:`milk10k.datasets.build_dataloaders`. :class:`BatchLoader` is the plain
numpy loader for analysis, debugging and the visualiser in :mod:`milk10k.viz`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from . import config, preprocessing

COLOR_SPACES = ("rgb", "gray", "hsv", "lab")
NORMALIZATIONS = ("minmax", "zscore", "none")

# Luminosity weights (ITU-R BT.601) — the formula from the Session 2 slides.
# Humans are most sensitive to green and least to blue, so a flat (R+G+B)/3
# would make blue-heavy dermoscopic backgrounds look brighter than they are.
GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# Upper bound of each channel for the 8-bit encodings cv2 produces. Used by
# "minmax" so every colour space lands in [0, 1]. OpenCV stores 8-bit hue as
# 0-179 (degrees / 2) so it fits in a byte; everything else is 0-255.
_CHANNEL_MAX = {
    "rgb": (255.0, 255.0, 255.0),
    "gray": (255.0,),
    "hsv": (179.0, 255.0, 255.0),
    "lab": (255.0, 255.0, 255.0),
}


# --- Results ---------------------------------------------------------------
@dataclass(frozen=True)
class ProcessedImage:
    """One preprocessed image plus a record of exactly what was done to it.

    ``image`` is the array to use. Everything else answers "what am I looking
    at?" without re-reading the code: the final ``shape`` and ``dtype``, the
    actual ``value_range`` after normalisation, and ``steps`` — a human-readable
    audit trail such as ``["resize 600x450 -> 224x224", "colour: rgb",
    "minmax: x / 255", "layout: HWC -> CHW"]``.
    """

    image: np.ndarray
    shape: tuple[int, ...]
    dtype: str
    value_range: tuple[float, float]
    color_space: str
    normalization: str
    channels_first: bool
    original_size: tuple[int, int] | None  # (width, height) of the source, PIL order
    steps: tuple[str, ...]
    source: str | None = None


@dataclass(frozen=True)
class BatchResult:
    """Output of :func:`process_batch`.

    ``images`` is one stacked array of shape ``(N, ...)`` when every image came
    out the same shape (always the case with a fixed ``size``), otherwise a list.
    ``ids[i]`` names ``images[i]`` — after skips the two stay aligned, but they
    are no longer aligned with the *input* order, so always use ``ids``.
    ``skipped`` lists ``(identifier, reason)`` for every input that failed.
    """

    images: np.ndarray | list[np.ndarray]
    ids: list[str]
    skipped: list[tuple[str, str]]
    n_requested: int
    info: ProcessedImage | None = None  # the record of the first successful image

    @property
    def n_ok(self) -> int:
        return len(self.ids)


# --- Part 3, task 9: one image ---------------------------------------------
def default_zscore_stats(color_space: str = "rgb") -> tuple[list[float], list[float]]:
    """Mean/std for z-scoring, taken from the training split (``norm_stats.json``).

    For grayscale the RGB statistics are collapsed with the same luminosity
    weights used to make the grayscale image, so the two stay consistent.
    Using train-only statistics is the whole point: statistics computed over
    validation and test images would leak their brightness into training.
    """
    stats = preprocessing.load_norm_stats()
    mean, std = np.asarray(stats["mean"]), np.asarray(stats["std"])
    if color_space == "rgb":
        return mean.tolist(), std.tolist()
    if color_space == "gray":
        # The weighted sum of the channel stds is exact when the channels are
        # perfectly correlated and an upper bound otherwise; RGB channels of
        # skin images are highly correlated, so it is a close, safe estimate.
        return [float(GRAY_WEIGHTS @ mean)], [float(GRAY_WEIGHTS @ std)]
    raise ValueError(
        f"no stored z-score statistics for colour space {color_space!r}; "
        "pass mean= and std= explicitly (computed on training images only)"
    )


def _to_pil(source, img_dir) -> tuple[Image.Image, str | None]:
    """Accept a path, an isic_id, a PIL image or an array; return (PIL RGB, label)."""
    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.dtype != np.uint8:
            if arr.min() < 0 or arr.max() > 255:
                raise ValueError("float input must already be in [0, 1] or [0, 255]")
            arr = arr * 255.0 if arr.max() <= 1.0 else arr
            arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB"), None
    if isinstance(source, Image.Image):
        return source.convert("RGB"), None
    label = source.stem if isinstance(source, Path) else Path(str(source)).stem
    # preprocessing._as_pil already knows how to tell a path from an isic_id
    # and fails loudly on a missing file; reuse it rather than re-deciding.
    return preprocessing._as_pil(source, img_dir), label


def process_image(
    source: str | Path | np.ndarray | Image.Image,
    size: int | None = config.IMAGE_SIZE,
    color_space: str = "rgb",
    normalize: str = "minmax",
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    channels_first: bool = False,
    img_dir: str | Path | None = None,
) -> ProcessedImage:
    """Resize, convert colour space and normalise one image.

    ``source`` can be a file path, a bare ``isic_id``, a PIL image or an array
    (HWC or CHW, uint8 or float in [0, 1]).

    Steps, in this order (the order matters — see the notes):

    1. **Resize** to ``size`` x ``size`` with :func:`preprocessing.preprocess_image`
       (BILINEAR, antialiased). ``size=None`` keeps the original resolution.
       Every MILK10k file is 600x450, so 224x224 squashes the 4:3 aspect ratio
       rather than cropping it — cropping would cut off the lesion border, which
       is diagnostic. Resizing happens on 8-bit RGB, *before* any conversion,
       so interpolation averages real colours.
    2. **Colour space**: ``"rgb"`` (default — colour carries diagnostic signal),
       ``"gray"`` (luminosity 0.299R + 0.587G + 0.114B), ``"hsv"`` or ``"lab"``
       (via OpenCV). Grayscale returns a single channel.
    3. **Normalise**:

       * ``"minmax"`` (default) — divide by each channel's fixed maximum (255
         for RGB), giving [0, 1]. The bounds are the *encoding's* bounds, not the
         image's own min and max: per-image min-max would stretch every image to
         full contrast and erase exactly the brightness and contrast differences
         Part 2 of the homework measures. It needs no fitted statistics, so it
         cannot leak anything between splits.
       * ``"zscore"`` — ``(x/255 - mean) / std`` per channel, for models that
         expect zero-centred input. ``mean``/``std`` default to the
         training-split statistics (:func:`default_zscore_stats`); for a
         pretrained backbone pass ``config.IMAGENET_MEAN``/``IMAGENET_STD``.
       * ``"none"`` — float32 in the raw 0-255 range.

       Values are cast to float32 *before* dividing (dividing uint8 can
       silently produce integers in some libraries).
    4. **Layout**: HWC by default (matplotlib, numpy); ``channels_first=True``
       gives CHW (what torch models consume).

    Raises on unreadable input — use :func:`process_batch` for tolerant
    processing of many files.
    """
    if color_space not in COLOR_SPACES:
        raise ValueError(f"color_space must be one of {COLOR_SPACES}, got {color_space!r}")
    if normalize not in NORMALIZATIONS:
        raise ValueError(f"normalize must be one of {NORMALIZATIONS}, got {normalize!r}")
    if normalize == "zscore" and color_space not in ("rgb", "gray") and (mean is None or std is None):
        raise ValueError("z-scoring hsv/lab needs explicit mean= and std=")

    pil, label = _to_pil(source, img_dir)
    original_size = pil.size  # PIL reports (width, height)
    steps: list[str] = []

    # 1. Resize — reuse the project's single resize implementation.
    if size is not None:
        arr = preprocessing.preprocess_image(pil, size=size, to_float=False)
        steps.append(f"resize {original_size[0]}x{original_size[1]} -> {size}x{size} (bilinear)")
    else:
        arr = np.asarray(pil, dtype=np.uint8)
        steps.append(f"no resize (kept {original_size[0]}x{original_size[1]})")

    # 2. Colour space.
    if color_space == "rgb":
        out = arr.astype(np.float32)
    elif color_space == "gray":
        out = (arr.astype(np.float32) @ GRAY_WEIGHTS)[..., None]
    elif color_space == "hsv":
        out = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    else:
        out = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    steps.append(f"colour: {color_space}" + (" (0.299R+0.587G+0.114B)" if color_space == "gray" else ""))

    # 3. Normalise.
    if normalize == "minmax":
        out = out / np.asarray(_CHANNEL_MAX[color_space], dtype=np.float32)
        steps.append("minmax: x / channel max (fixed bounds) -> [0, 1]")
    elif normalize == "zscore":
        if mean is None or std is None:
            mean, std = default_zscore_stats(color_space)
            origin = "train-split stats"
        else:
            origin = "given stats"
        m = np.asarray(mean, dtype=np.float32)
        s = np.asarray(std, dtype=np.float32)
        if m.size != out.shape[-1] or s.size != out.shape[-1]:
            raise ValueError(
                f"mean/std have {m.size}/{s.size} values but the image has {out.shape[-1]} channel(s)"
            )
        if np.any(s <= 0):
            raise ValueError("std must be strictly positive")
        out = (out / 255.0 - m) / s
        steps.append(f"zscore: (x/255 - mean) / std using {origin} "
                     f"mean={np.round(m, 3).tolist()} std={np.round(s, 3).tolist()}")
    else:
        steps.append("no normalisation (float32, 0-255)")

    # 4. Layout. Grayscale keeps an explicit channel axis so shapes are uniform
    # across colour spaces: (H, W, 1) or (1, H, W).
    if channels_first:
        out = np.transpose(out, (2, 0, 1))
        steps.append("layout: HWC -> CHW")
    out = np.ascontiguousarray(out, dtype=np.float32)

    return ProcessedImage(
        image=out,
        shape=tuple(out.shape),
        dtype=str(out.dtype),
        value_range=(float(out.min()), float(out.max())),
        color_space=color_space,
        normalization=normalize,
        channels_first=channels_first,
        original_size=original_size,
        steps=tuple(steps),
        source=label,
    )


# --- Part 3, task 10: many images ------------------------------------------
def _identifiers(sources, id_column: str) -> list:
    if isinstance(sources, pd.DataFrame):
        if id_column not in sources.columns:
            raise KeyError(f"DataFrame has no {id_column!r} column")
        return sources[id_column].tolist()
    if isinstance(sources, (str, Path)):
        raise TypeError("pass a list of paths/ids, not a single one — use process_image for that")
    return list(sources)


def process_batch(
    sources: Sequence[str | Path] | pd.DataFrame,
    *,
    id_column: str = "isic_id",
    stack: bool = True,
    **process_kwargs,
) -> BatchResult:
    """Apply :func:`process_image` to many images without letting one bad file stop it.

    ``sources`` is a list of file paths or ``isic_id`` strings, or a metadata
    DataFrame (its ``id_column`` is used). ``process_kwargs`` are forwarded to
    :func:`process_image` unchanged, so a batch is guaranteed to be processed
    exactly like a single image.

    Every per-image failure — missing file, truncated or non-image JPEG, wrong
    array shape — is caught, and recorded in ``skipped`` as
    ``(identifier, "ExceptionType: message")``. The rest of the batch carries on.
    Configuration errors (an unknown ``color_space``, say) are *not* swallowed:
    they would fail for every image, so they are raised once, up front.

    With ``stack=True`` the results are stacked into one ``(N, ...)`` array when
    all shapes agree (they always do with a fixed ``size``); otherwise, or with
    ``stack=False``, a list is returned.
    """
    # Validate the configuration once so a typo is an error, not N skips.
    if process_kwargs.get("color_space", "rgb") not in COLOR_SPACES:
        raise ValueError(f"color_space must be one of {COLOR_SPACES}")
    if process_kwargs.get("normalize", "minmax") not in NORMALIZATIONS:
        raise ValueError(f"normalize must be one of {NORMALIZATIONS}")

    items = _identifiers(sources, id_column)
    images: list[np.ndarray] = []
    ids: list[str] = []
    skipped: list[tuple[str, str]] = []
    first: ProcessedImage | None = None

    for item in items:
        name = item.stem if isinstance(item, Path) else (
            Path(item).stem if isinstance(item, str) else f"<{type(item).__name__}>"
        )
        try:
            result = process_image(item, **process_kwargs)
        except Exception as exc:  # noqa: BLE001 — the brief: skip it and report why
            skipped.append((name, f"{type(exc).__name__}: {exc}"))
            continue
        images.append(result.image)
        ids.append(result.source or name)
        first = first or result

    if stack and images and len({im.shape for im in images}) == 1:
        batch: np.ndarray | list[np.ndarray] = np.stack(images, axis=0)
    elif stack and not images:
        batch = np.empty((0,), dtype=np.float32)
    else:
        batch = images

    return BatchResult(images=batch, ids=ids, skipped=skipped,
                       n_requested=len(items), info=first)


# --- Part 4: the loader ----------------------------------------------------
@dataclass(frozen=True)
class LoaderBatch:
    """One batch from :class:`BatchLoader`.

    ``images``      float32, ``(B, C, H, W)`` by default (``(B, H, W, C)`` with
                    ``channels_first=False``)
    ``labels``      int64 ``(B,)`` class indices — see ``BatchLoader.classes``
    ``label_names`` the same labels as strings, for plots and debugging
    ``ids``         the ``isic_id`` of each image, aligned with ``images``
    ``skipped``     ``(isic_id, reason)`` for rows in this chunk that failed

    ``B`` can be smaller than ``batch_size`` for the last batch, and whenever
    files fail to load.
    """

    images: np.ndarray
    labels: np.ndarray
    label_names: list[str]
    ids: list[str]
    skipped: list[tuple[str, str]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ids)


class BatchLoader:
    """Yield ``(processed images, labels)`` batches from the metadata table, on the fly.

    Construct with a metadata table (image-level, one row per ``isic_id``) and
    optionally an image directory. Iterate to get :class:`LoaderBatch` objects.
    Images are read and processed *per batch* with :func:`process_batch`, so
    memory use is one batch, never the dataset.

    Only rows whose JPEG actually exists are used: the directory is listed once
    at construction (:func:`preprocessing.available_ids`) and the table filtered
    with ``isin``. ``n_missing`` / ``missing_ids`` report what was dropped.

    Example::

        loader = BatchLoader(split_table, batch_size=32, shuffle=True)
        for batch in loader:          # len(loader) batches per epoch
            x, y = batch.images, batch.labels   # (B,3,224,224) float32, (B,) int64

    Labels are integer indices into ``classes`` (default
    ``config.PRIMARY_LABELS`` = Benign, Indeterminate, Malignant — the same
    encoding as ``artifacts/label_map.json``), because that is what a loss
    function consumes; ``label_names`` keeps the strings for display.

    With ``shuffle=True`` the order is re-drawn every epoch from
    ``seed + epoch``, so epochs differ but a run is exactly reproducible.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        img_dir: str | Path | None = None,
        *,
        batch_size: int = 32,
        label_col: str = config.PRIMARY_LABEL_COL,
        classes: Sequence[str] | None = None,
        shuffle: bool = False,
        seed: int = config.SEED,
        drop_last: bool = False,
        channels_first: bool = True,
        **process_kwargs,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        for col in ("isic_id", label_col):
            if col not in metadata.columns:
                raise KeyError(f"metadata has no {col!r} column")

        self.classes = list(classes) if classes is not None else list(config.PRIMARY_LABELS)
        self.class_to_index = {c: i for i, c in enumerate(self.classes)}
        unknown = set(metadata[label_col].dropna()) - set(self.classes)
        if unknown:
            raise ValueError(f"labels not in classes: {sorted(unknown)}")
        if metadata[label_col].isna().any():
            raise ValueError(f"{label_col!r} has missing labels")

        # Task 13: keep only rows whose file exists — one directory listing.
        self.img_dir = img_dir
        present = preprocessing.available_ids(img_dir)
        on_disk = metadata["isic_id"].isin(present)
        self.missing_ids = metadata.loc[~on_disk, "isic_id"].tolist()
        self.table = metadata.loc[on_disk].reset_index(drop=True)

        self.batch_size = batch_size
        self.label_col = label_col
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.process_kwargs = {"channels_first": channels_first, "img_dir": img_dir, **process_kwargs}
        self.epoch = 0
        self.skipped: list[tuple[str, str]] = []  # accumulated over the last full epoch

    @property
    def n_images(self) -> int:
        """Rows available on disk (after the availability filter)."""
        return len(self.table)

    @property
    def n_missing(self) -> int:
        """Metadata rows dropped because their file is not on disk."""
        return len(self.missing_ids)

    def __len__(self) -> int:
        """Number of batches per epoch."""
        full, rest = divmod(self.n_images, self.batch_size)
        return full if (self.drop_last or rest == 0) else full + 1

    def _order(self) -> np.ndarray:
        idx = np.arange(self.n_images)
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(idx)
        return idx

    def __iter__(self) -> Iterator[LoaderBatch]:
        order = self._order()
        epoch_skips: list[tuple[str, str]] = []
        for b in range(len(self)):
            rows = self.table.iloc[order[b * self.batch_size:(b + 1) * self.batch_size]]
            result = process_batch(rows["isic_id"].tolist(), **self.process_kwargs)
            epoch_skips.extend(result.skipped)
            name_of = dict(zip(rows["isic_id"], rows[self.label_col]))
            names = [name_of[i] for i in result.ids]
            images = result.images if isinstance(result.images, np.ndarray) and result.images.ndim > 1 \
                else np.empty((0,), dtype=np.float32)
            yield LoaderBatch(
                images=images,
                labels=np.array([self.class_to_index[n] for n in names], dtype=np.int64),
                label_names=names,
                ids=result.ids,
                skipped=result.skipped,
            )
        self.skipped = epoch_skips
        self.epoch += 1

    def describe(self) -> str:
        """One-paragraph summary: sizes, class counts, preprocessing settings."""
        counts = self.table[self.label_col].value_counts().reindex(self.classes, fill_value=0)
        settings = {k: v for k, v in self.process_kwargs.items() if k != "img_dir"}
        return (
            f"BatchLoader: {self.n_images:,} images on disk ({self.n_missing:,} metadata rows "
            f"without a file), batch_size={self.batch_size}, {len(self)} batches/epoch, "
            f"shuffle={self.shuffle}, classes={ {k: int(v) for k, v in counts.items()} }, preprocessing={settings}"
        )
