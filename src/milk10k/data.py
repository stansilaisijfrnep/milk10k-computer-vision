"""Loading and joining the MILK10k tables.

The dataset ships two different row units, and keeping them straight is the
single most important thing in this project:

* ``metadata.csv`` / ``training_input.csv`` / ``training_supp.csv``
  -> one row per **image** (10,480 rows).
* ``training_gt.csv``
  -> one row per **lesion** (5,240 rows), 11-class one-hot.

Every lesion contributes exactly two images (one dermoscopic, one clinical
close-up), so any train/test split has to be done on ``lesion_id``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import config, preprocessing


# --- Raw loaders -----------------------------------------------------------
def load_metadata() -> pd.DataFrame:
    """One row per image: demographics, diagnosis hierarchy, image type."""
    return pd.read_csv(config.METADATA_CSV, low_memory=False)


def load_training_gt() -> pd.DataFrame:
    """One row per lesion: 11-class one-hot ground truth."""
    return pd.read_csv(config.TRAINING_GT_CSV)


def load_training_input() -> pd.DataFrame:
    """One row per image: MONET concept scores, skin tone class, body site."""
    return pd.read_csv(config.TRAINING_INPUT_CSV, low_memory=False)


def load_training_supp() -> pd.DataFrame:
    """One row per image: full-text diagnosis and confirmation type."""
    return pd.read_csv(config.TRAINING_SUPP_CSV)


# --- Derived views ---------------------------------------------------------
def gt_to_label(gt: pd.DataFrame) -> pd.DataFrame:
    """Collapse the 11-class one-hot block into a single categorical column.

    Returns a frame with ``lesion_id``, ``label_11`` (the code) and
    ``n_positive`` so the single-label assumption can be verified rather than
    assumed.
    """
    # labels.py owns the one-hot -> code step for the whole project; this
    # function only adds the exploratory columns around it. The import is local
    # because labels.py imports this module, and a top-level import would make
    # the two files circular.
    from . import labels as labels_module

    onehot = gt[config.STRETCH_LABELS]

    out = pd.DataFrame(
        {
            "lesion_id": gt["lesion_id"],
            "label_11": labels_module.codes_from_onehot(gt),
            # Measured, not assumed: the EDA's job is to *report* whether the
            # one-hot really is single-label, so this counts rather than asserts.
            "n_positive": onehot.sum(axis=1),
        }
    )
    out["label_11_name"] = out["label_11"].map(config.DIAGNOSIS_CODES)
    out["code_group"] = out["label_11"].map(config.code_group)
    out["is_malignant_11"] = out["label_11"].isin(config.MALIGNANT_CODES)
    return out


def check_label_hierarchy(lesion: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulate the 11-class code against the 3-class target.

    The two label schemes are nested, but not perfectly: this is how you find
    out which codes straddle a boundary instead of assuming a clean mapping.
    """
    ct = pd.crosstab(lesion["label_11"], lesion["diagnosis_1"])
    cols = [c for c in config.PRIMARY_CLASSES if c in ct.columns]
    return ct[cols].reindex(lesion["label_11"].value_counts().index)


def ambiguous_codes(lesion: pd.DataFrame) -> list[str]:
    """11-class codes that map to more than one value of ``diagnosis_1``."""
    ct = check_label_hierarchy(lesion)
    return ct.index[(ct > 0).sum(axis=1) > 1].tolist()


def build_image_level(
    meta: pd.DataFrame | None = None,
    gt: pd.DataFrame | None = None,
    inp: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join everything onto the per-image grain (10,480 rows).

    The 11-class lesion label is broadcast to both images of a lesion, and the
    per-image MONET/skin-tone columns are merged on ``isic_id``.
    """
    meta = load_metadata() if meta is None else meta
    gt = load_training_gt() if gt is None else gt
    inp = load_training_input() if inp is None else inp

    labels = gt_to_label(gt)

    monet_cols = [c for c in inp.columns if c.startswith("MONET_")]
    extra = inp[["isic_id", "skin_tone_class", "site", *monet_cols]]

    df = meta.merge(labels, on="lesion_id", how="left", validate="many_to_one")
    df = df.merge(extra, on="isic_id", how="left", validate="one_to_one")

    df["skin_tone_label"] = df["skin_tone_class"].map(config.SKIN_TONE_LABELS)
    return df


def build_lesion_level(image_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Collapse to one row per lesion (5,240 rows) — the real modelling unit.

    Lesion-level attributes (diagnosis, age, sex, site, skin tone) are constant
    across a lesion's two images, so taking the first value is safe; this is
    asserted in :func:`check_lesion_consistency`.
    """
    df = build_image_level() if image_df is None else image_df

    agg = {
        "diagnosis_1": "first",
        "diagnosis_2": "first",
        "diagnosis_3": "first",
        "label_11": "first",
        "label_11_name": "first",
        "is_malignant_11": "first",
        "age_approx": "first",
        "sex": "first",
        "anatom_site_general": "first",
        "skin_tone_class": "first",
        "skin_tone_label": "first",
        "diagnosis_confirm_type": "first",
        "melanocytic": "first",
        "isic_id": "count",
    }
    lesion = df.groupby("lesion_id", as_index=False).agg(agg)
    return lesion.rename(columns={"isic_id": "n_images"})


# --- Integrity checks ------------------------------------------------------
def check_lesion_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Verify the structural claims the whole project depends on.

    Returns a tidy pass/fail table instead of printing, so the result can be
    saved into the report.
    """
    per_lesion = df.groupby("lesion_id").size()
    types_per_lesion = df.groupby("lesion_id")["image_type"].nunique()

    varying = []
    for col in ["diagnosis_1", "label_11", "age_approx", "sex"]:
        n_varying = (df.groupby("lesion_id")[col].nunique(dropna=False) > 1).sum()
        varying.append((col, n_varying))

    checks = [
        ("Every lesion has exactly 2 images", bool((per_lesion == 2).all()),
         f"{(per_lesion == 2).sum():,} / {len(per_lesion):,} lesions"),
        ("Each lesion has 2 distinct image types", bool((types_per_lesion == 2).all()),
         f"{(types_per_lesion == 2).sum():,} / {len(types_per_lesion):,} lesions"),
        ("isic_id is unique", bool(df["isic_id"].is_unique),
         f"{df['isic_id'].nunique():,} unique / {len(df):,} rows"),
        ("No duplicated image rows", bool(not df.duplicated("isic_id").any()),
         f"{int(df.duplicated('isic_id').sum())} duplicates"),
    ]
    checks += [
        (f"'{col}' is constant within a lesion", n == 0, f"{n} lesions vary")
        for col, n in varying
    ]

    return pd.DataFrame(checks, columns=["check", "passed", "detail"])


# --- Image helpers ---------------------------------------------------------
# Both helpers below are thin aliases. ``preprocessing`` owns reading images
# from disk for the whole project — it knows about MILK10K_IMAGES_DIR overrides
# and it raises an error that names the offending id. These names stay because
# the Session 1 notebook and ``plots.py`` call them, but they must not become a
# second implementation.
def image_path(isic_id: str) -> Path:
    """Absolute path to the JPEG for one image id (see ``preprocessing``)."""
    return preprocessing.image_path(isic_id)


def load_image(isic_id: str) -> Image.Image:
    """Open one image as RGB (guarantees 3 channels regardless of source file)."""
    return preprocessing.load_image(isic_id)


def sample_image_stats(
    df: pd.DataFrame, n: int = 400, seed: int = 0
) -> pd.DataFrame:
    """Measure real pixel properties on a random sample of images.

    Reading every one of the 10,480 JPEGs is slow and unnecessary for a first
    look, so we sample. Returns width, height, megapixels, aspect ratio, file
    size and per-channel mean intensity.
    """
    sample = df.sample(n=min(n, len(df)), random_state=seed)
    rows = []

    for _, row in sample.iterrows():
        path = image_path(row["isic_id"])
        with Image.open(path) as im:
            width, height = im.size
            arr = np.asarray(im.convert("RGB"))

        rows.append(
            {
                "isic_id": row["isic_id"],
                "image_type": row["image_type"],
                "diagnosis_1": row["diagnosis_1"],
                "label_11": row.get("label_11"),
                "width": width,
                "height": height,
                "megapixels": width * height / 1e6,
                "aspect_ratio": width / height,
                "file_size_kb": path.stat().st_size / 1024,
                "mean_r": float(arr[:, :, 0].mean()),
                "mean_g": float(arr[:, :, 1].mean()),
                "mean_b": float(arr[:, :, 2].mean()),
                "mean_intensity": float(arr.mean()),
                "std_intensity": float(arr.std()),
            }
        )

    return pd.DataFrame(rows)


# ===========================================================================
# Milestone 1 / homework A1.3 — the lesion table
# ===========================================================================
# Everything below is appended for Milestone 1. It does not change any of the
# functions above; it adds the one table the rest of the pipeline is built on.

# The two literal values of ``image_type``. They are spelled out once here so
# that a typo shows up as an immediate KeyError instead of a silently empty
# column.
DERMOSCOPIC_TYPE = "dermoscopic"
CLINICAL_TYPE = "clinical: close-up"

# Lesion-level attributes and the column names we want them under. These are
# constant across a lesion's two images (asserted in check_lesion_consistency),
# which is what makes a groupby-first legitimate here.
_LESION_ATTRS = {
    "diagnosis_1": "diagnosis_1",
    "age_approx": "age",
    "sex": "sex",
    "anatom_site_general": "site",
}

LESION_TABLE_COLUMNS = [
    "lesion_id", "derm_id", "clinical_id", "diagnosis_1", "dx",
    "age", "sex", "site",
]


def build_lesion_table(
    meta: pd.DataFrame | None = None,
    gt: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per lesion, with both of its image ids side by side.

    This is the table the whole pipeline splits, weights and loads from. The
    modelling unit is the *lesion*, not the image: a lesion contributes one
    dermoscopic and one clinical close-up photo of the same piece of skin, so
    putting one of them in train and the other in test would leak the answer.
    Having ``derm_id`` and ``clinical_id`` in the same row makes that mistake
    structurally impossible — you can only ever move a whole lesion.

    The reshape is a ``pivot``: ``image_type`` is the thing that distinguishes
    the two rows of a lesion, so it becomes the column axis and ``isic_id``
    becomes the value. No Python loop over rows is involved, which matters
    because a row loop over 10,480 rows is both slow and easy to get subtly
    wrong. ``pivot`` also fails loudly if a lesion ever had two dermoscopic
    images, because the (lesion_id, image_type) pair would no longer be unique.

    Columns: lesion_id, derm_id, clinical_id, diagnosis_1, dx, age, sex, site.
    ``dx`` is the 11-class code from ``training_gt.csv``; ``diagnosis_1`` is the
    coarse 3-class target from ``metadata.csv``. On the shipped dataset the
    result has 5,240 rows.
    """
    # Remember whether we loaded the shipped files ourselves: only then do we
    # know the exact row count the result must have.
    using_full_dataset = meta is None

    meta = load_metadata() if meta is None else meta
    gt = load_training_gt() if gt is None else gt

    # --- The two image ids, one per column ---------------------------------
    images = meta.pivot(index="lesion_id", columns="image_type", values="isic_id")
    missing_types = [t for t in (DERMOSCOPIC_TYPE, CLINICAL_TYPE)
                     if t not in images.columns]
    if missing_types:
        raise KeyError(f"image_type values not found in metadata: {missing_types}")

    images = images[[DERMOSCOPIC_TYPE, CLINICAL_TYPE]]
    images.columns = ["derm_id", "clinical_id"]
    images = images.rename_axis(columns=None).reset_index()

    # --- Lesion-level attributes -------------------------------------------
    # "first" is safe because these columns do not vary within a lesion; it also
    # skips NaN, so if only one of the two image rows carries an age we keep it.
    attrs = (
        meta.groupby("lesion_id", as_index=False)[list(_LESION_ATTRS)]
        .first()
        .rename(columns=_LESION_ATTRS)
    )

    # --- The 11-class code --------------------------------------------------
    # Taken from labels.py rather than derived here: that module is the single
    # owner of the target, and it also checks the single-positive assumption.
    from . import labels as labels_module

    labels = labels_module.lesion_labels(gt)

    table = (
        images.merge(attrs, on="lesion_id", how="left", validate="one_to_one")
        .merge(labels, on="lesion_id", how="left", validate="one_to_one")
    )
    table = table[LESION_TABLE_COLUMNS]

    # --- Assertions the rest of the project relies on ----------------------
    assert table["lesion_id"].is_unique, "duplicate lesion_id in the lesion table"
    assert len(table) == meta["lesion_id"].nunique(), (
        "the pivot lost or invented lesions"
    )
    assert table["derm_id"].notna().all(), (
        f"{int(table['derm_id'].isna().sum())} lesions have no dermoscopic image"
    )
    assert table["clinical_id"].notna().all(), (
        f"{int(table['clinical_id'].isna().sum())} lesions have no clinical image"
    )
    if using_full_dataset:
        assert len(table) == 5_240, f"expected 5,240 lesions, got {len(table):,}"

    return table


def save_lesion_table(
    table: pd.DataFrame | None = None,
    path: Path = config.LESION_TABLE_CSV,
) -> Path:
    """Write the lesion table to CSV so later milestones load it, not rebuild it.

    Rebuilding is cheap, but committing the file means the splits, the class
    weights and the dataloaders all provably refer to the same 5,240 lesions.
    """
    table = build_lesion_table() if table is None else table
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path
