"""Session 2 homework, Part 5: visualisers for images, batches and class balance.

Everything here accepts the objects the pipeline produces — a list of arrays, a
:class:`~milk10k.pipeline.BatchResult`, or a
:class:`~milk10k.pipeline.LoaderBatch` straight out of
:class:`~milk10k.pipeline.BatchLoader` — so "what is my loader actually feeding
the model?" is one function call.

The load-bearing piece is :func:`to_displayable`. Loader output is float32,
channels-first and possibly z-scored (negative values); ``plt.imshow`` expects
HWC in [0, 1] or uint8. Handing it a z-scored CHW tensor gives either an error or
a plausible-looking but wrong image, which is exactly how a broken pipeline goes
unnoticed. Every grid in this module goes through that one conversion.

Figures follow the repo convention: return ``plt.Figure``; ``save_as`` writes to
``reports/figures/`` via :func:`milk10k.plots._save`.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, pipeline
from .plots import _save

_CLASS_COLORS = config.BENIGN_MALIGNANT_COLORS


def _class_color(name: str) -> str:
    return _CLASS_COLORS.get(name, config.ORANGE)


# --- The one conversion every display goes through -------------------------
def to_displayable(
    img: np.ndarray,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> np.ndarray:
    """Turn any pipeline array into something ``plt.imshow`` shows correctly.

    Accepts HWC or CHW, one or three channels, uint8 or float. If ``mean`` and
    ``std`` are given the z-scoring is undone first (``x * std + mean``), giving
    back the true [0, 1] image. Otherwise values already in [0, 1] (min-max) are
    left alone, and anything else is rescaled by its own min and max *for display
    only* — the fallback makes a z-scored image visible, but its contrast is then
    not comparable across images, so pass ``mean``/``std`` when you know them.

    Returns float in [0, 1], shape ``(H, W, 3)`` or ``(H, W)`` for one channel.
    """
    arr = np.asarray(img)
    was_uint8 = arr.dtype == np.uint8
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    arr = arr.astype(np.float32)

    if mean is not None and std is not None:
        arr = arr * np.asarray(std, dtype=np.float32) + np.asarray(mean, dtype=np.float32)
    else:
        looks_8bit = arr.min() >= 0 and 1.0 < arr.max() <= 255
        if was_uint8 or looks_8bit:
            arr = arr / 255.0
    # Anything still outside [0, 1] (e.g. z-scored without stats): stretch for display.
    if arr.min() < -1e-6 or arr.max() > 1.0 + 1e-6:
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo if hi > lo else 1.0)

    arr = np.clip(arr, 0.0, 1.0)
    return arr[..., 0] if arr.ndim == 3 and arr.shape[-1] == 1 else arr


def _unpack(images, labels):
    """Accept a batch object or (images, labels); return (list of arrays, labels)."""
    if isinstance(images, pipeline.LoaderBatch):
        return list(images.images), labels if labels is not None else images.label_names
    if isinstance(images, pipeline.BatchResult):
        return list(images.images), labels if labels is not None else images.ids
    return list(images), labels


# --- Task 15: image grid ---------------------------------------------------
def show_image_grid(
    images,
    labels: Sequence[str] | None = None,
    n: int | None = None,
    n_cols: int = 4,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    title: str | None = None,
    save_as: str | None = None,
) -> plt.Figure:
    """Grid of images with their labels as titles.

    ``images`` is a list/array of images, or a ``LoaderBatch`` (titles default to
    its ``label_names``) or ``BatchResult`` (titles default to its ids). Class
    names are coloured with the project's Benign/Indeterminate/Malignant palette.
    Pass the ``mean``/``std`` used for z-scoring so images are shown as they
    really are rather than contrast-stretched.
    """
    arrays, labels = _unpack(images, labels)
    if n is not None:
        arrays, labels = arrays[:n], (list(labels)[:n] if labels is not None else None)
    if not arrays:
        raise ValueError("no images to show")

    n_cols = min(n_cols, len(arrays))
    n_rows = int(np.ceil(len(arrays) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.5 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, (ax, arr) in enumerate(zip(axes.ravel(), arrays)):
        shown = to_displayable(arr, mean, std)
        ax.imshow(shown, cmap="gray" if shown.ndim == 2 else None, vmin=0, vmax=1)
        if labels is not None:
            name = str(list(labels)[i])
            ax.set_title(name, fontsize=9, color=_class_color(name), fontweight="bold")
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, save_as)


def show_batch(
    batch: pipeline.LoaderBatch,
    n: int = 8,
    n_cols: int = 4,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    save_as: str | None = None,
) -> plt.Figure:
    """Sanity-check a loader batch: the first ``n`` images titled ``label · isic_id``."""
    titles = [f"{name}\n{iid}" for name, iid in zip(batch.label_names, batch.ids)]
    shape = "x".join(map(str, batch.images.shape))
    fig = show_image_grid(
        list(batch.images), titles, n=n, n_cols=n_cols, mean=mean, std=std,
        title=f"One batch from BatchLoader — images {shape} {batch.images.dtype}, "
              f"labels {batch.labels[:n].tolist()}",
    )
    for ax, name in zip(fig.axes, batch.label_names[:n]):
        ax.title.set_color(_class_color(name))
    return _save(fig, save_as)


# --- Task 16: class balance ------------------------------------------------
def plot_class_balance(
    labels: Sequence[str] | pd.Series | pd.DataFrame,
    classes: Sequence[str] | None = None,
    label_col: str = config.PRIMARY_LABEL_COL,
    group_col: str | None = None,
    title: str = "Class balance",
    save_as: str | None = None,
) -> plt.Figure:
    """Bar chart of how many items fall in each class, with counts and shares.

    ``labels`` is a sequence of class names or a DataFrame (``label_col`` is
    counted). With a DataFrame and ``group_col`` (e.g. ``"lesion_id"``), a second
    panel counts unique groups per class — images and lesions tell different
    stories in MILK10k, where every lesion has two images. The imbalance ratio
    (largest / smallest class) is printed on the chart.
    """
    classes = list(classes) if classes is not None else list(config.PRIMARY_LABELS)
    panels: list[tuple[str, pd.Series]] = []
    if isinstance(labels, pd.DataFrame):
        panels.append(("images", labels[label_col].value_counts()))
        if group_col is not None:
            panels.append((f"{group_col.replace('_id', '')}s",
                           labels.groupby(label_col)[group_col].nunique()))
    else:
        panels.append(("items", pd.Series(list(labels)).value_counts()))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.6), squeeze=False)
    for ax, (unit, counts) in zip(axes[0], panels):
        counts = counts.reindex(classes, fill_value=0)
        total = int(counts.sum())
        bars = ax.bar(classes, counts.values, color=[_class_color(c) for c in classes])
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:,}\n({v / total:.1%})",
                    ha="center", va="bottom", fontsize=9)
        nonzero = counts[counts > 0]
        ratio = nonzero.max() / nonzero.min() if len(nonzero) else float("nan")
        ax.set_title(f"{unit}: n = {total:,} · imbalance {ratio:.0f}:1", fontsize=10)
        ax.set_ylabel(f"number of {unit}")
        ax.set_ylim(0, counts.max() * 1.25)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, save_as)


# --- Task 17: pipeline debugging -------------------------------------------
def plot_batch_summary(
    batch: pipeline.LoaderBatch | np.ndarray,
    bins: int = 80,
    save_as: str | None = None,
) -> plt.Figure:
    """Check that normalisation did what you think: value histogram per channel.

    Left: pixel-value distribution of the whole batch, one line per channel,
    with the overall min/max/mean/std written on the plot — a min-max batch must
    sit inside [0, 1]; a z-scored batch should be centred near 0 with std near 1.
    Right: per-image mean vs std, which exposes a single broken image (all black,
    all white, wrongly scaled) that the pooled histogram would hide.
    """
    x = batch.images if isinstance(batch, pipeline.LoaderBatch) else np.asarray(batch)
    if x.ndim != 4:
        raise ValueError(f"expected a (B, C, H, W) or (B, H, W, C) batch, got shape {x.shape}")
    chw = x.shape[1] in (1, 3) and x.shape[-1] not in (1, 3)
    channels = x.shape[1] if chw else x.shape[-1]
    per_channel = [(x[:, c] if chw else x[..., c]).ravel() for c in range(channels)]
    names = ["R", "G", "B"] if channels == 3 else ["gray"]
    colors = ["#C1272D", "#2E8B84", "#1a5fb4"] if channels == 3 else [config.GREY]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})
    lo, hi = float(x.min()), float(x.max())
    edges = np.linspace(lo, hi, bins + 1)
    for vals, name, col in zip(per_channel, names, colors):
        levels, counts = np.unique(vals, return_counts=True)
        if len(levels) <= 512:
            # Pixels come from 8-bit files, so each channel holds at most 256
            # distinct values. Fixed-width bins would alias against those levels
            # (a comb pattern); plotting each level's share is exact instead.
            share = counts / counts.sum() / np.median(np.diff(levels)) if len(levels) > 1 else counts
            ax1.plot(levels, share, color=col, lw=1.4, label=name)
        else:
            h, _ = np.histogram(vals, bins=edges, density=True)
            ax1.plot((edges[:-1] + edges[1:]) / 2, h, color=col, lw=1.6, label=name)
    ax1.axvline(0, color="#999", lw=0.8, ls="--")
    ax1.set_xlabel("pixel value after preprocessing")
    ax1.set_ylabel("density")
    ax1.legend(loc="upper right")
    ax1.set_title("Pixel values in this batch", fontsize=10)
    ax1.text(0.02, 0.97,
             f"shape {tuple(x.shape)}  {x.dtype}\nmin {lo:.3f}  max {hi:.3f}\n"
             f"mean {x.mean():.3f}  std {x.std():.3f}",
             transform=ax1.transAxes, va="top", fontsize=8.5, family="monospace",
             bbox=dict(boxstyle="round", fc="white", ec="#ddd"))

    axes_mean = tuple(range(1, 4))
    ax2.scatter(x.mean(axis=axes_mean), x.std(axis=axes_mean), s=18, color=config.ORANGE)
    ax2.set_xlabel("per-image mean")
    ax2.set_ylabel("per-image std")
    ax2.set_title("One point per image", fontsize=10)
    fig.tight_layout()
    return _save(fig, save_as)


def plot_raw_vs_processed(
    isic_ids: Sequence[str],
    configs: Sequence[dict] | None = None,
    labels: Sequence[str] | None = None,
    save_as: str | None = None,
) -> plt.Figure:
    """Before/after grid: each row is one image, raw first, then each processing config.

    ``labels`` (e.g. the class of each image) are added to the raw panel's title.

    Every processed panel is titled with the result's shape and value range, so
    the figure shows *what the array is*, not just what it looks like. Z-scored
    panels are shown by undoing the z-score with the same statistics.
    """
    configs = list(configs) if configs is not None else [
        {"normalize": "minmax"},
        {"color_space": "gray", "normalize": "minmax"},
        {"normalize": "zscore"},
    ]
    n_cols = len(configs) + 1
    fig, axes = plt.subplots(len(isic_ids), n_cols, figsize=(2.55 * n_cols, 2.35 * len(isic_ids)),
                             squeeze=False)
    for r, iid in enumerate(isic_ids):
        raw = pipeline.process_image(iid, size=None, normalize="none")
        axes[r, 0].imshow(raw.image.astype(np.uint8))
        tag = f"{labels[r]} · " if labels is not None else ""
        axes[r, 0].set_title(f"{tag}{iid}\nraw {raw.original_size[0]}x{raw.original_size[1]} uint8", fontsize=8,
                             color=_class_color(labels[r]) if labels is not None else None)
        for c, cfg in enumerate(configs, start=1):
            res = pipeline.process_image(iid, **cfg)
            mean = std = None
            if res.normalization == "zscore":
                mean, std = cfg.get("mean"), cfg.get("std")
                if mean is None:
                    mean, std = pipeline.default_zscore_stats(res.color_space)
            shown = to_displayable(res.image, mean, std)
            axes[r, c].imshow(shown, cmap="gray" if shown.ndim == 2 else None, vmin=0, vmax=1)
            shape = "x".join(map(str, res.shape))
            axes[r, c].set_title(
                f"{res.color_space} · {res.normalization}\n{shape} [{res.value_range[0]:.2f}, "
                f"{res.value_range[1]:.2f}]" + ("\n(displayed un-z-scored)" if mean is not None else ""),
                fontsize=8)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    return _save(fig, save_as)
