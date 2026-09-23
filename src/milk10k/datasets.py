"""Dataset and DataLoader layer for the MILK10k project.

The dataset has two grains and this module has two Dataset classes, on purpose:

* :class:`MilkImageDataset` serves ONE IMAGE per item. This is what the
  Milestone 1 (B8) training loop actually consumes — a CNN backbone looks at
  one photograph at a time, and with 10,480 photographs we want every one of
  them to be a training example.
* :class:`LesionDataset` serves ONE LESION per item, i.e. both photographs of
  the same lesion together (homework A3.6). A lesion is the unit a clinician
  diagnoses and the unit the ground truth is recorded on, so it is the grain a
  two-view model would be trained and reported on.

The bridge between the two is :func:`aggregate_predictions`: train per image,
predict per image, then average a lesion's two views back into one prediction
per lesion. This is only sound because every split in this project is made on
``lesion_id`` (see ``splits.py``), so a lesion's two images always land in the
same split — otherwise averaging at test time would be reading the training
set through the back door.

Both classes hand back the id next to the tensor and the label. That is not
decoration: without the id you cannot find out *which* images a model got
wrong, and the Milestone 2 error analysis depends on being able to.

Nothing here decodes, resizes, normalises, weights or splits anything by hand.
Those jobs live in ``preprocessing.py``, ``transforms.py``, ``imbalance.py``
and ``splits.py``; this module only wires them together.
"""

from __future__ import annotations

import os
import platform
import random
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from . import config, labels, preprocessing

# --- Constants -------------------------------------------------------------
# Which column of the lesion table holds the id for each photographic view.
# ``data.build_lesion_table`` produces exactly these two columns.
VIEW_COLUMNS: dict[str, str] = {"derm": "derm_id", "clinical": "clinical_id"}

# The two label schemes and the column each one lives in, so a caller only has
# to say "primary" or "stretch" instead of remembering column names. Re-exported
# from labels.py rather than redefined: labels.py owns everything about the
# target, and a second copy here would be free to drift.
SCHEME_LABEL_COLUMNS: dict[str, str] = labels.SCHEME_LABEL_COLUMNS


# --- Small shared helpers --------------------------------------------------
def _as_table(table: pd.DataFrame | str | Path, what: str) -> pd.DataFrame:
    """Accept either a DataFrame or a path to a CSV and return a DataFrame.

    Taking both keeps notebooks convenient (pass the frame you already have)
    and scripts honest (pass the committed split CSV), without every caller
    writing the same ``pd.read_csv`` line.
    """
    if isinstance(table, (str, Path)):
        path = Path(table)
        if not path.exists():
            raise FileNotFoundError(
                f"{what}: no CSV at {path}. Run `splits.write_splits(...)` first "
                "to generate the committed splits."
            )
        return pd.read_csv(path)

    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"{what}: expected a DataFrame or a path, got {type(table)!r}.")

    # reset_index returns a new frame, so the caller's object is never mutated
    # and positional indexing in __getitem__ lines up with row order.
    return table.reset_index(drop=True)


def _require_columns(table: pd.DataFrame, needed: Sequence[str], what: str) -> None:
    """Fail immediately, with names, if the table is missing a required column.

    Catching this in ``__init__`` rather than in ``__getitem__`` means the error
    appears before a training run starts, not three minutes into epoch one.
    """
    missing = [c for c in needed if c not in table.columns]
    if missing:
        raise KeyError(
            f"{what}: missing column(s) {missing}. Available columns: "
            f"{list(table.columns)}"
        )


def _check_scheme_matches_column(scheme: str, label_col: str, what: str) -> None:
    """Refuse a scheme and a label column that describe different targets.

    ``MilkImageDataset(table, label_col="dx")`` with the default
    ``scheme="primary"`` is an easy mistake to make and a miserable one to
    debug: the 11 codes are then looked up in a 3-class map and the failure
    surfaces as "value not in the label map" with no hint as to why. Naming the
    contradiction here turns it into a one-line fix.

    A column that belongs to no scheme is left alone, because a caller passing
    their own ``label_map`` for a bespoke column is doing something legitimate.
    """
    expected = labels.SCHEME_LABEL_COLUMNS.get(scheme)
    known_columns = set(labels.SCHEME_LABEL_COLUMNS.values())
    if label_col in known_columns and label_col != expected:
        raise ValueError(
            f"{what}: scheme={scheme!r} expects label_col={expected!r}, but "
            f"label_col={label_col!r} belongs to the "
            f"{labels.scheme_for_column(label_col)!r} scheme. Pass both "
            "consistently."
        )


def _resolve_label_map(
    scheme: str, label_map: Mapping[str, int] | None = None
) -> dict[str, int]:
    """Get the label-name -> integer mapping, from ``labels.py`` unless overridden.

    ``labels.py`` owns the mapping so that the training code, the saved
    ``label_map.json`` and the confusion-matrix axes can never drift apart. The
    ``label_map`` argument exists only so a caller can pass an explicit mapping
    (tests, or a model trained on a subset of the classes).
    """
    if label_map is not None:
        return {str(k): int(v) for k, v in label_map.items()}
    return labels.label_map(scheme)


def _encode_labels(values: pd.Series, mapping: Mapping[str, int]) -> np.ndarray:
    """Turn a column of class names into the int64 targets a loss function wants.

    An already-numeric column is passed through (a caller may have encoded it
    earlier); everything else goes through ``labels.encode``, which is the one
    place in the project that maps a class name to an integer and raises on
    anything it does not recognise. A silently dropped class is a bug you only
    notice in the confusion matrix weeks later.
    """
    if pd.api.types.is_numeric_dtype(values):
        codes = values.to_numpy(dtype=np.int64)
        out_of_range = codes[(codes < 0) | (codes >= len(mapping))]
        if out_of_range.size:
            raise ValueError(
                f"label column holds integer codes outside 0..{len(mapping) - 1}: "
                f"{sorted(set(out_of_range.tolist()))}"
            )
        return codes

    return labels.encode(values, mapping=dict(mapping)).to_numpy(dtype=np.int64)


def _open_rgb(img_dir: Path, isic_id: str, context: str) -> Image.Image:
    """Open one JPEG as RGB, failing loudly and informatively if it is not there.

    The actual read is ``preprocessing.load_image``, the project's single image
    loader — three Dataset classes each opening their own ``Image.open`` is how
    a colour-mode or resampling change ends up applied in two places out of
    three. All this wrapper adds is the row's context (which dataset, which
    lesion, which view) in front of the error, because the message is what turns
    a missing file into a row you can find in the split table.
    """
    try:
        return preprocessing.load_image(isic_id, img_dir=img_dir)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{context}: {exc}. Check MILK10K_IMAGES_DIR, or pass "
            "allow_missing=True to skip rows whose file is absent."
        ) from exc


def _default_transform(image_size: int):
    """Fall back to the deterministic eval transform when none is given.

    A Dataset with no transform would return PIL images, which cannot be
    stacked into a batch, so "no transform" has to mean something sensible.
    The *eval* transform is the safe default: resize + normalise, no randomness.
    """
    from . import transforms

    return transforms.eval_transform(image_size=image_size)


# --- Image-level dataset (Milestone 1, B8) ---------------------------------
class MilkImageDataset(Dataset):
    """One IMAGE per item: ``(image_tensor, int_label, isic_id)``.

    This is the workhorse of the training pipeline. It is deliberately thin: it
    holds a split table, turns a row index into a file path, opens the JPEG and
    hands it to the transform. All the decisions — which rows, which
    augmentations, which integer for which class — were made elsewhere and are
    passed in, so the same class serves train, val and test without branching
    on the split name.

    Parameters
    ----------
    table
        A split DataFrame (or the path to a split CSV) with at least
        ``isic_id`` and the label column.
    img_dir
        Where the JPEGs live. Defaults to ``config.IMG_DIR``, which honours the
        ``MILK10K_IMAGES_DIR`` environment variable.
    label_col, scheme
        The column holding the class names, and which label scheme its values
        belong to (``"primary"`` = 3 classes, ``"stretch"`` = 11 codes).
    transform
        Callable applied to the PIL image. ``None`` means the deterministic
        eval transform.
    allow_missing
        ``False`` (default) means a missing file raises ``FileNotFoundError``
        naming the id and path: a silently shrinking dataset is far more
        dangerous than a crash. ``True`` drops rows without a file up front and
        records how many in ``n_dropped``.
    """

    def __init__(
        self,
        table: pd.DataFrame | str | Path,
        img_dir: str | Path | None = None,
        label_col: str = config.PRIMARY_LABEL_COL,
        scheme: str = "primary",
        transform: Any | None = None,
        allow_missing: bool = False,
        *,
        label_map: Mapping[str, int] | None = None,
        image_size: int = config.IMAGE_SIZE,
    ) -> None:
        self.img_dir = Path(img_dir) if img_dir is not None else config.IMG_DIR
        self.label_col = label_col
        self.scheme = scheme
        self.allow_missing = allow_missing

        if label_map is None:
            _check_scheme_matches_column(scheme, label_col, "MilkImageDataset")

        table = _as_table(table, "MilkImageDataset table")
        _require_columns(table, ["isic_id", label_col], "MilkImageDataset table")

        self.label_map = _resolve_label_map(scheme, label_map)
        self.inverse_label_map = {i: name for name, i in self.label_map.items()}

        if allow_missing:
            # preprocessing.available_subset already knows how to check a whole
            # frame against the image folder; no reason to stat files here too.
            kept = preprocessing.available_subset(table, img_dir=self.img_dir)
            self.n_dropped = len(table) - len(kept)
            table = kept.reset_index(drop=True)
        else:
            self.n_dropped = 0

        self.table = table
        self.isic_ids = table["isic_id"].astype(str).to_numpy()
        self.targets = _encode_labels(table[label_col], self.label_map)
        self.transform = (
            transform if transform is not None else _default_transform(image_size)
        )

    def __len__(self) -> int:
        return len(self.isic_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        """Return one transformed image, its integer label, and its id."""
        isic_id = str(self.isic_ids[index])
        image = _open_rgb(self.img_dir, isic_id, "MilkImageDataset")
        return self.transform(image), int(self.targets[index]), isic_id

    def class_counts(self) -> pd.Series:
        """Count items per class name — the one-line check that a split is sane."""
        names = pd.Series(self.targets).map(self.inverse_label_map)
        return names.value_counts().sort_index()

    def __repr__(self) -> str:
        return (
            f"MilkImageDataset(n={len(self)}, scheme={self.scheme!r}, "
            f"label_col={self.label_col!r}, img_dir={self.img_dir})"
        )


# --- Lesion-level dataset (homework A3.6) ----------------------------------
class LesionDataset(Dataset):
    """One LESION per item: both views of the same lesion, plus label and id.

    Each item is a dict, e.g.
    ``{"derm": tensor, "clinical": tensor, "label": 2, "lesion_id": "IL_1234"}``.
    A dict rather than a tuple because the views are named things, and a model
    that takes two inputs should read ``batch["derm"]`` instead of remembering
    that position 0 was the dermoscopic one.

    The transform is applied INDEPENDENTLY to each view, and that is the whole
    point: the dermoscopic and the clinical image are two *different
    photographs*, taken with different instruments at different magnifications,
    not two crops of one photo. Forcing them to share a random flip or crop
    would invent a geometric correspondence between them that does not exist,
    and would halve the augmentation diversity the two-view branch sees.

    Parameters
    ----------
    lesion_table
        One row per lesion (or a path to such a CSV), with ``lesion_id``, the
        label column, and the id column of every requested view — the output of
        ``data.build_lesion_table``.
    img_dir
        Folder holding the JPEGs.
    label_col
        Column with the class names (``diagnosis_1`` or ``dx``).
    transform
        Callable applied to each view separately. ``None`` means the eval
        transform.
    views
        Any subset of ``("derm", "clinical")``. Passing a single view turns
        this into a per-lesion single-image dataset, which is a useful ablation.
    """

    def __init__(
        self,
        lesion_table: pd.DataFrame | str | Path,
        img_dir: str | Path | None = None,
        label_col: str = config.PRIMARY_LABEL_COL,
        transform: Any | None = None,
        views: Iterable[str] = ("derm", "clinical"),
        *,
        scheme: str = "primary",
        label_map: Mapping[str, int] | None = None,
        image_size: int = config.IMAGE_SIZE,
        view_columns: Mapping[str, str] | None = None,
    ) -> None:
        self.img_dir = Path(img_dir) if img_dir is not None else config.IMG_DIR
        self.label_col = label_col
        self.scheme = scheme
        self.view_columns = dict(VIEW_COLUMNS if view_columns is None else view_columns)

        if label_map is None:
            _check_scheme_matches_column(scheme, label_col, "LesionDataset")

        self.views = tuple(views)
        if not self.views:
            raise ValueError("`views` must name at least one view; got an empty sequence.")
        unknown = [v for v in self.views if v not in self.view_columns]
        if unknown:
            raise ValueError(
                f"unknown view(s) {unknown}; valid views are "
                f"{sorted(self.view_columns)}"
            )

        table = _as_table(lesion_table, "LesionDataset lesion_table")
        id_cols = [self.view_columns[v] for v in self.views]
        _require_columns(
            table, ["lesion_id", label_col, *id_cols], "LesionDataset lesion_table"
        )

        # A lesion with no id for a requested view cannot produce an item, and
        # discovering that halfway through an epoch is worse than refusing now.
        for view, col in zip(self.views, id_cols):
            n_null = int(table[col].isna().sum())
            if n_null:
                raise ValueError(
                    f"LesionDataset: {n_null} lesion(s) have no '{view}' image id in "
                    f"column '{col}'. Drop them, or build the dataset with "
                    f"views={tuple(v for v in self.views if v != view)}."
                )

        self.label_map = _resolve_label_map(scheme, label_map)
        self.inverse_label_map = {i: name for name, i in self.label_map.items()}

        self.table = table
        self.lesion_ids = table["lesion_id"].astype(str).to_numpy()
        self.targets = _encode_labels(table[label_col], self.label_map)
        # Pull the id columns out once as plain arrays; per-item .iloc lookups
        # on a DataFrame are slow and would show up in the epoch timing.
        self.view_ids = {
            view: table[self.view_columns[view]].astype(str).to_numpy()
            for view in self.views
        }
        self.transform = (
            transform if transform is not None else _default_transform(image_size)
        )

    def __len__(self) -> int:
        return len(self.lesion_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one lesion: a tensor per requested view, its label and its id."""
        lesion_id = str(self.lesion_ids[index])
        item: dict[str, Any] = {}

        for view in self.views:
            isic_id = str(self.view_ids[view][index])
            image = _open_rgb(
                self.img_dir,
                isic_id,
                f"LesionDataset lesion '{lesion_id}' view '{view}'",
            )
            # A separate call per view = a separate draw of the random
            # augmentation parameters. See the class docstring for why.
            item[view] = self.transform(image)

        item["label"] = int(self.targets[index])
        item["lesion_id"] = lesion_id
        return item

    def __repr__(self) -> str:
        return (
            f"LesionDataset(n={len(self)}, views={self.views}, "
            f"label_col={self.label_col!r}, img_dir={self.img_dir})"
        )


# --- DataLoaders -----------------------------------------------------------
def default_num_workers() -> int:
    """Pick a safe number of worker processes for this machine.

    macOS and Windows start worker processes with ``spawn``, which re-imports
    the entry module in every worker. Inside a Jupyter kernel — how this project
    is actually run — that either hangs or dies, and the failure looks like a
    mysterious stall rather than an error. So those platforms get 0 workers
    (decode in the main process), which is fast enough here: the images are
    600x450 JPEGs downscaled to 224. Elsewhere, a few workers help, but never
    all the cores, because the main process still has to train.
    """
    if platform.system() in {"Darwin", "Windows"}:
        return 0
    cpus = os.cpu_count() or 1
    return max(0, min(4, cpus - 1))


def _seed_worker(worker_id: int) -> None:
    """Re-seed ``numpy`` and ``random`` inside each DataLoader worker process.

    PyTorch gives every worker its own torch seed derived from the loader's
    generator, but ``numpy`` and ``random`` are not seeded for us — without
    this, workers would produce the same augmentation stream every epoch (or
    differ between runs), and the whole run would stop being reproducible.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _resolve_split_tables(
    splits: Mapping[str, pd.DataFrame | str | Path] | None,
) -> dict[str, pd.DataFrame]:
    """Load the three split tables, defaulting to the committed split CSVs."""
    if splits is None:
        splits = {
            "train": config.TRAIN_SPLIT_CSV,
            "val": config.VAL_SPLIT_CSV,
            "test": config.TEST_SPLIT_CSV,
        }
    return {
        name: _as_table(table, f"{name} split") for name, table in splits.items()
    }


def build_dataloaders(
    batch_size: int = 32,
    num_workers: int | None = None,
    image_size: int = config.IMAGE_SIZE,
    scheme: str = "primary",
    use_sampler: bool = False,
    seed: int = config.SEED,
    *,
    shuffle: bool | None = None,
    label_col: str | None = None,
    img_dir: str | Path | None = None,
    splits: Mapping[str, pd.DataFrame | str | Path] | None = None,
    allow_missing: bool = False,
    pin_memory: bool = False,
    drop_last: bool = False,
    train_transform: Any | None = None,
    eval_transform: Any | None = None,
    label_map: Mapping[str, int] | None = None,
) -> dict[str, DataLoader]:
    """Build the ``{"train", "val", "test"}`` DataLoaders the training loop uses.

    The asymmetry between the splits is the substance of this function:
    train shuffles and augments, val and test never shuffle and always use the
    deterministic eval transform. Augmenting the validation set would make the
    reported score depend on a coin flip; shuffling it would only scramble the
    order in which the same number comes out.

    Reproducibility is set up in three places, because one is not enough:
    ``transforms.seed_everything`` seeds the process, a ``torch.Generator``
    seeds the shuffling and the sampler, and ``worker_init_fn`` seeds each
    worker process.

    Parameters
    ----------
    use_sampler
        Draw training batches with the class-balanced
        ``WeightedRandomSampler`` from ``imbalance.py`` instead of a uniform
        shuffle. A sampler already defines the order of an epoch, so PyTorch
        forbids combining it with ``shuffle=True``; leaving ``shuffle=None``
        (the default) resolves this automatically.
    splits
        Optional ``{"train": ..., "val": ..., "test": ...}`` of DataFrames or
        CSV paths. Defaults to the committed split CSVs from ``config``.
    train_transform, eval_transform
        Override the augmentation policy from ``transforms.py`` — useful for an
        ablation, and for testing this function without touching global state.
    """
    if scheme not in SCHEME_LABEL_COLUMNS:
        raise ValueError(
            f"unknown scheme {scheme!r}; expected one of {sorted(SCHEME_LABEL_COLUMNS)}"
        )
    if use_sampler and shuffle is True:
        raise ValueError(
            "use_sampler=True and shuffle=True cannot be combined: a "
            "WeightedRandomSampler already fixes the order of the epoch, and "
            "PyTorch rejects a DataLoader given both. Pass use_sampler=True on "
            "its own (shuffle then defaults to False), or shuffle=True without "
            "the sampler."
        )

    if label_col is None:
        label_col = SCHEME_LABEL_COLUMNS[scheme]
    if num_workers is None:
        num_workers = default_num_workers()
    train_shuffle = (not use_sampler) if shuffle is None else bool(shuffle)

    from . import transforms as milk_transforms

    # Seed the process first, so that building the transforms and the sampler
    # below already happens from a known state.
    milk_transforms.seed_everything(seed)
    if train_transform is None:
        train_transform = milk_transforms.train_transform(image_size=image_size)
    if eval_transform is None:
        eval_transform = milk_transforms.eval_transform(image_size=image_size)

    generator = torch.Generator()
    generator.manual_seed(seed)

    tables = _resolve_split_tables(splits)
    datasets = {
        name: MilkImageDataset(
            table,
            img_dir=img_dir,
            label_col=label_col,
            scheme=scheme,
            transform=train_transform if name == "train" else eval_transform,
            allow_missing=allow_missing,
            label_map=label_map,
            image_size=image_size,
        )
        for name, table in tables.items()
    }

    sampler = None
    if use_sampler:
        from . import imbalance

        sampler = imbalance.make_weighted_sampler(
            datasets["train"].table, label_col, scheme, generator=generator
        )

    loaders: dict[str, DataLoader] = {}
    for name, dataset in datasets.items():
        is_train = name == "train"
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train_shuffle if (is_train and sampler is None) else False,
            sampler=sampler if is_train else None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            # Dropping a short last batch is only ever acceptable in training;
            # on val/test it would quietly throw away evaluation examples.
            drop_last=drop_last if is_train else False,
            generator=generator if is_train else None,
            worker_init_fn=_seed_worker if num_workers > 0 else None,
        )

    return loaders


# --- Image predictions -> lesion predictions (homework A3.6c) --------------
def _class_columns(
    image_probs: pd.DataFrame | np.ndarray, class_names: Sequence[str] | None
) -> list[str]:
    """Work out sensible names for the probability columns.

    Explicit names win; a DataFrame's own columns come next; otherwise the
    number of columns identifies the scheme, because 3 and 11 are the only two
    label schemes in this project.
    """
    if class_names is not None:
        return [str(c) for c in class_names]
    if isinstance(image_probs, pd.DataFrame):
        return [str(c) for c in image_probs.columns]

    n_classes = np.asarray(image_probs).shape[1]
    if n_classes == len(config.PRIMARY_LABELS):
        return list(config.PRIMARY_LABELS)
    if n_classes == len(config.STRETCH_LABELS):
        return list(config.STRETCH_LABELS)
    return [f"class_{i}" for i in range(n_classes)]


def aggregate_predictions(
    image_probs: np.ndarray | pd.DataFrame,
    image_table: pd.DataFrame,
    class_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Average a lesion's per-image probabilities into ONE prediction per lesion.

    The model predicts per image, but the clinical question is asked per lesion,
    so the two views have to be combined. Averaging the probability vectors
    (rather than voting on the two hard labels) keeps the model's confidence:
    a confident malignant call on the dermoscopic view is not cancelled by a
    hesitant benign call on the clinical one.

    A lesion with only one available view is averaged over what exists, which
    for a single row means its probabilities are passed through unchanged. The
    ``n_views`` column records how many images each row was built from, so a
    half-observed lesion can be spotted in the results instead of silently
    looking like a normal one.

    Parameters
    ----------
    image_probs
        ``(N, C)`` array or DataFrame of class probabilities, row-aligned to
        ``image_table``.
    image_table
        The ``(N,)`` rows those probabilities belong to; needs ``lesion_id``
        (and usually ``isic_id``).

    Returns
    -------
    DataFrame
        One row per lesion: ``lesion_id``, ``n_views``, one averaged
        probability column per class, ``pred_index`` and ``pred``.
    """
    _require_columns(image_table, ["lesion_id"], "aggregate_predictions image_table")

    probs = np.asarray(
        image_probs.to_numpy() if isinstance(image_probs, pd.DataFrame) else image_probs,
        dtype=float,
    )
    if probs.ndim != 2:
        raise ValueError(f"image_probs must be 2-D (N, C); got shape {probs.shape}.")
    if len(probs) != len(image_table):
        raise ValueError(
            f"image_probs has {len(probs)} rows but image_table has "
            f"{len(image_table)}: the two must be row-aligned."
        )

    columns = _class_columns(image_probs, class_names)
    if len(columns) != probs.shape[1]:
        raise ValueError(
            f"got {len(columns)} class names for {probs.shape[1]} probability columns."
        )

    frame = pd.DataFrame(probs, columns=columns)
    frame.insert(0, "lesion_id", image_table["lesion_id"].to_numpy())

    # sort=False keeps lesions in the order they first appear in the image
    # table, which makes the output easy to line up with the split table.
    grouped = frame.groupby("lesion_id", sort=False)
    averaged = grouped[columns].mean()
    n_views = grouped.size().rename("n_views")

    winner = np.asarray(averaged.to_numpy()).argmax(axis=1)
    out = averaged.reset_index()
    out.insert(1, "n_views", n_views.to_numpy())
    out["pred_index"] = winner
    out["pred"] = np.asarray(columns, dtype=object)[winner]
    return out


# --- Sanity checks (Milestone 1, B8) ---------------------------------------
def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Get (images, labels) out of a batch from either Dataset class."""
    if isinstance(batch, dict):
        # LesionDataset: inspect the first view present.
        view = next(k for k in batch if isinstance(batch[k], torch.Tensor) and batch[k].ndim == 4)
        return batch[view], batch["label"]
    images, labels = batch[0], batch[1]
    return images, labels


def sanity_check(
    loaders: Mapping[str, DataLoader],
    n_batches: int = 20,
    split: str = "train",
    time_epoch: bool = True,
) -> dict[str, Any]:
    """Print and return the B8 checks: shapes, dtype, value range, balance, speed.

    These are the four questions worth asking before spending an hour on a
    training run. Is the batch the shape the model expects? Are the values
    normalised (after ImageNet normalisation the range is roughly -2.1..2.6,
    *not* 0..1 — seeing 0..1 here means the normalisation silently did not
    happen)? Do the labels still look like the class distribution we expect?
    And how long does one epoch of pure data loading cost, which is the floor
    under every training epoch.
    """
    loader = loaders[split]
    dataset = loader.dataset
    inverse = getattr(dataset, "inverse_label_map", {})

    first: dict[str, Any] = {}
    label_counts: dict[int, int] = {}
    seen_batches = 0

    for batch in loader:
        images, labels = _unpack_batch(batch)
        if not first:
            first = {
                "batch_shape": tuple(images.shape),
                "dtype": str(images.dtype),
                "min": float(images.min()),
                "max": float(images.max()),
                "mean": float(images.mean()),
                "label_dtype": str(labels.dtype),
            }
        values, counts = torch.unique(labels, return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            label_counts[value] = label_counts.get(value, 0) + count

        seen_batches += 1
        if seen_batches >= n_batches:
            break

    if not first:
        raise ValueError(f"the '{split}' loader produced no batches at all.")

    epoch_seconds = None
    n_epoch_batches = None
    n_epoch_images = None
    if time_epoch:
        start = time.perf_counter()
        n_epoch_batches = 0
        n_epoch_images = 0
        for batch in loaders["train"]:
            images, _ = _unpack_batch(batch)
            n_epoch_images += int(images.shape[0])
            n_epoch_batches += 1
        epoch_seconds = time.perf_counter() - start

    named_counts = {
        str(inverse.get(value, value)): count
        for value, count in sorted(label_counts.items())
    }
    result = {
        "split": split,
        "n_items": len(dataset),
        "n_batches_inspected": seen_batches,
        **first,
        "label_counts": named_counts,
        "epoch_seconds": epoch_seconds,
        "n_epoch_batches": n_epoch_batches,
        "n_epoch_images": n_epoch_images,
        "images_per_second": (
            None if not epoch_seconds else n_epoch_images / epoch_seconds
        ),
    }

    print(f"--- sanity_check on '{split}' ({len(dataset):,} items) ---")
    print(f"batch shape      : {result['batch_shape']}  dtype {result['dtype']}")
    print(
        f"value range      : min {result['min']:+.3f}  max {result['max']:+.3f}  "
        f"mean {result['mean']:+.3f}"
    )
    print(f"labels           : dtype {result['label_dtype']}")
    total = sum(named_counts.values()) or 1
    for name, count in named_counts.items():
        print(f"  {name:<15} {count:6,}  ({100 * count / total:5.1f}%)")
    print(f"  over {seen_batches} batch(es) of '{split}'")
    if epoch_seconds is not None:
        print(
            f"one train epoch  : {epoch_seconds:.2f} s for {n_epoch_images:,} images "
            f"in {n_epoch_batches:,} batches "
            f"({result['images_per_second']:.1f} img/s, "
            f"num_workers={loaders['train'].num_workers})"
        )
    return result
