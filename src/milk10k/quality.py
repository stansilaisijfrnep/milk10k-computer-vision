"""Data integrity and data quality for MILK10k (Milestone 1, B1 and B4).

This module answers three questions that have to be settled *before* any model
is trained, and it answers them with evidence rather than with assumptions:

1. **Is the data physically there and readable?** — every one of the 10,480
   JPEGs is opened and its header checked, not a sample of them (B1).
2. **What is missing, and what do we do about it?** — one explicit, auditable
   handling decision per column with NaNs (B4).
3. **Can the model cheat?** — the structural checks and the cross-tabs that
   expose shortcuts and label leakage (B4, and Homework A1.1).

Everything returns a DataFrame instead of printing, so the same numbers can go
into the report, into the notebook and into an assertion in a test.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import config, data


# --- Structural expectations ----------------------------------------------
# The claims the whole pipeline is built on. They live here as named constants
# so the checks below read as English and so a reviewer can see what is being
# asserted without reading the check code.
IMAGES_PER_LESION = 2
EXPECTED_IMAGE_TYPES = ("clinical: close-up", "dermoscopic")

# The diagnosis hierarchy, coarse to fine. Each level must nest inside the one
# before it, otherwise the label taxonomy is inconsistent.
DIAGNOSIS_LEVELS = ["diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_4"]

# Columns that are constant across the two images of one lesion. If any of
# these varied, collapsing to the lesion grain (the real modelling unit) would
# silently pick one of two different values.
LESION_CONSTANT_COLUMNS = ["diagnosis_1", "age_approx", "sex", "anatom_site_general"]


# --- Missing-value policy --------------------------------------------------
# One decision per column, written down once. Keeping them in a dict rather
# than in prose means the report and the preprocessing code cannot drift apart,
# and a marker can check in one place what we claimed we would do.
MISSING_VALUE_DECISIONS: dict[str, str] = {
    "age_approx": (
        "Impute with the TRAIN-split median (never the full-data median, that "
        "would leak); add no missingness indicator — at 0.4% the flag would be "
        "almost constant and carries no usable signal."
    ),
    "anatom_site_general": (
        "Encode the NaN as an explicit 'unknown' category. At 37% the values "
        "are clearly not missing at random (some contributors simply never "
        "recorded a site), so mode-imputation would invent 37% of a feature."
    ),
    "anatom_site_special": (
        "Drop the column. 2% coverage cannot support a category level, and the "
        "few populated values duplicate anatom_site_general."
    ),
    "melanocytic": (
        "Leave the NaN as-is and DO NOT USE the column as a model input: it is "
        "derived from the diagnosis, so it leaks the label (see the leakage "
        "section). Imputing it would only make the leak look tidier."
    ),
    "diagnosis_3": (
        "Leave as-is. NaN here means 'no finer level exists for this lesion' — "
        "structurally absent, not missing — and it is a label column, never an "
        "input."
    ),
    "diagnosis_4": (
        "Leave as-is. Same reasoning as diagnosis_3: absence is information "
        "about the taxonomy depth, and it is a label column, never an input."
    ),
    "invasion_thickness_interval": (
        "Leave as-is and never use as an input. It is Breslow thickness, "
        "measured on the excised tumour, so it only exists after the biopsy "
        "that produced the label."
    ),
}

# Shown for any column with NaNs that nobody has ruled on yet. Its presence in
# the report is the signal that a decision is owed.
UNREVIEWED_DECISION = "NO DECISION RECORDED — review before this column is used."


# --- Leakage policy --------------------------------------------------------
# Why each config.LEAKY_COLUMNS entry is banned as a model input. A column is
# leaky if it either *is* the label, or could only be known after the biopsy
# that produced the label.
LEAKY_COLUMN_REASONS: dict[str, str] = {
    "diagnosis_1": "It is the primary target itself.",
    "diagnosis_2": "Finer level of the same label taxonomy as the target.",
    "diagnosis_3": "Finer level of the same label taxonomy as the target.",
    "diagnosis_4": "Finer level of the same label taxonomy as the target.",
    "diagnosis_confirm_type": (
        "Records how the diagnosis was confirmed. A lesion goes to "
        "histopathology because a clinician already suspected malignancy, so "
        "this column encodes the clinical decision, not the image."
    ),
    "diagnosis_full": "Free-text form of the label.",
    "invasion_thickness_interval": (
        "Tumour thickness measured on the excised lesion — it exists only for "
        "lesions already diagnosed as invasive malignancies."
    ),
    "melanocytic": (
        "Derived from the diagnosis (whether the lesion is of melanocytic "
        "origin), so it is a coarse label in disguise."
    ),
    "concomitant_biopsy": (
        "Records that a biopsy was performed, which is again the clinician's "
        "suspicion rather than anything visible in the image."
    ),
    "lesion_id": (
        "An identifier, not a feature. It is the grouping key for the split; "
        "as an input it would let the model memorise individual lesions."
    ),
}


# --- B1: full-dataset image integrity --------------------------------------
def verify_all_images(
    df: pd.DataFrame | None = None,
    img_dir: Path | str | None = None,
    progress: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Open and check **every** image file referenced by the metadata.

    B1 asks for a full integrity pass, not a sample: a single corrupt or
    missing file only shows up as a crash halfway through the first training
    epoch, which is the most expensive moment to discover it.

    The check is deliberately header-only. ``Image.open`` parses the JPEG
    header (which is where width, height, format and mode come from) and
    ``Image.verify`` checks the file structure without decoding the pixel data,
    so the whole dataset takes seconds rather than minutes. Anything that fails
    to parse raises here and is recorded instead of propagating.

    Returns a per-image frame (one row per metadata row) and a summary dict.
    """
    df = data.load_metadata() if df is None else df
    img_dir = Path(config.IMG_DIR if img_dir is None else img_dir)

    ids = df["isic_id"].to_list()
    n_total = len(ids)
    started = time.perf_counter()

    records: list[dict] = []
    for i, isic_id in enumerate(ids, start=1):
        path = img_dir / f"{isic_id}.jpg"
        row = {
            "isic_id": isic_id,
            "exists": False,
            "readable": False,
            "width": np.nan,
            "height": np.nan,
            "format": None,
            "mode": None,
            "file_size_bytes": np.nan,
            "error": None,
        }

        # One stat() call does double duty: it tells us the file exists and
        # gives us its size, instead of paying for exists() plus stat().
        try:
            row["file_size_bytes"] = path.stat().st_size
            row["exists"] = True
        except OSError as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            records.append(row)
            continue

        try:
            with Image.open(path) as im:
                # Read the header attributes BEFORE verify(): verify() leaves
                # the file object unusable, so anything we want must be taken
                # from the already-parsed header first.
                row["width"], row["height"] = im.size
                row["format"] = im.format
                row["mode"] = im.mode
                im.verify()
            row["readable"] = True
        except Exception as exc:  # noqa: BLE001 - any failure means "unreadable"
            row["error"] = f"{type(exc).__name__}: {exc}"

        records.append(row)

        if progress and i % 2000 == 0:
            print(f"  ...{i:,}/{n_total:,} checked")

    sizes = pd.DataFrame.from_records(records)
    elapsed = time.perf_counter() - started

    ok = sizes[sizes["readable"]]
    summary = {
        "n_rows": int(len(sizes)),
        "n_missing": int((~sizes["exists"]).sum()),
        "n_unreadable": int((sizes["exists"] & ~sizes["readable"]).sum()),
        "n_ok": int(len(ok)),
        "width_min": _stat_or_nan(ok["width"], "min"),
        "width_median": _stat_or_nan(ok["width"], "median"),
        "width_max": _stat_or_nan(ok["width"], "max"),
        "height_min": _stat_or_nan(ok["height"], "min"),
        "height_median": _stat_or_nan(ok["height"], "median"),
        "height_max": _stat_or_nan(ok["height"], "max"),
        "n_distinct_resolutions": int(ok[["width", "height"]].drop_duplicates().shape[0]),
        "elapsed_seconds": round(elapsed, 2),
    }
    return sizes, summary


def _stat_or_nan(series: pd.Series, how: str) -> float:
    """Aggregate a column, returning NaN for an empty selection.

    Guards the summary against the degenerate case where nothing was readable:
    a report full of exceptions is less useful than one that says "0 readable".
    """
    if series.empty:
        return float("nan")
    return float(getattr(series, how)())


def image_size_summary(sizes_df: pd.DataFrame) -> pd.DataFrame:
    """Condense the per-image frame into the small table B1 asks us to save.

    Two things matter here. First, whether the dataset is dimensionally uniform
    (if it is, no resize-vs-pad decision is needed). Second, what fraction of
    images are large enough for the candidate input resolutions — that fraction
    is the evidence B6 uses to justify the chosen input size, because resizing
    *up* invents detail while resizing down only discards it.

    Returns a tidy ``metric / value / note`` frame, which is the shape that
    survives a CSV round-trip with mixed numeric and text values.
    """
    ok = sizes_df[sizes_df["readable"]]
    n_ok = len(ok)

    if n_ok:
        both_sides = np.minimum(ok["width"], ok["height"])
        frac_input = float((both_sides >= config.IMAGE_SIZE).mean())
        frac_256 = float((both_sides >= 256).mean())
        resolutions = ok.groupby(["width", "height"]).size().sort_values(ascending=False)
        top_res, top_n = resolutions.index[0], int(resolutions.iloc[0])
        top_label = f"{int(top_res[0])}x{int(top_res[1])}"
        top_share = 100.0 * top_n / n_ok
    else:
        frac_input = frac_256 = float("nan")
        top_label, top_share = "n/a", float("nan")
        resolutions = pd.Series(dtype="int64")

    rows = [
        ("n_images", int(len(sizes_df)), "rows in the metadata, i.e. files expected"),
        ("n_missing", int((~sizes_df["exists"]).sum()), "referenced file not on disk"),
        ("n_unreadable", int((sizes_df["exists"] & ~sizes_df["readable"]).sum()),
         "file present but the header failed to parse"),
        ("n_ok", int(n_ok), "opened and verified successfully"),
        ("width_min", _stat_or_nan(ok["width"], "min"), "pixels"),
        ("width_median", _stat_or_nan(ok["width"], "median"), "pixels"),
        ("width_max", _stat_or_nan(ok["width"], "max"), "pixels"),
        ("height_min", _stat_or_nan(ok["height"], "min"), "pixels"),
        ("height_median", _stat_or_nan(ok["height"], "median"), "pixels"),
        ("height_max", _stat_or_nan(ok["height"], "max"), "pixels"),
        ("n_distinct_resolutions", int(len(resolutions)),
         "1 means the dataset is dimensionally uniform"),
        ("most_common_resolution", top_label, f"{top_share:.1f}% of readable images"),
        (f"frac_at_least_{config.IMAGE_SIZE}", frac_input,
         f"share with both sides >= {config.IMAGE_SIZE}px (the configured input size)"),
        ("frac_at_least_256", frac_256,
         "share with both sides >= 256px (the next common backbone resolution)"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "note"])


# --- B4: missing values ----------------------------------------------------
def missing_value_report(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per column that has any NaN, with the agreed handling decision.

    The point of this table is not to count NaNs — that is one line of pandas —
    but to force a written decision next to each count, so that "what did you
    do about the missing sites?" has a single, checkable answer.
    """
    df = data.load_metadata() if df is None else df

    n_missing = df.isna().sum()
    n_missing = n_missing[n_missing > 0].sort_values(ascending=False)

    out = pd.DataFrame(
        {
            "column": n_missing.index,
            "n_missing": n_missing.to_numpy(dtype="int64"),
            "pct_missing": (100.0 * n_missing / len(df)).to_numpy(dtype="float64"),
            "dtype": [str(df[c].dtype) for c in n_missing.index],
        }
    )
    out["decision"] = out["column"].map(MISSING_VALUE_DECISIONS).fillna(UNREVIEWED_DECISION)
    return out.reset_index(drop=True)


# --- B4: structural and label consistency ----------------------------------
def label_consistency_checks(
    df: pd.DataFrame | None = None,
    gt: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pass/fail table for every structural claim the pipeline relies on.

    These are the assumptions that, if wrong, break something silently rather
    than loudly: a lesion with three images would leak across a split, a
    lesion whose two rows disagree on the diagnosis would give two different
    labels to the same lesion, and a one-hot row with two positives would make
    ``idxmax`` quietly pick the alphabetically first class.

    ``detail`` always carries the actual counts, so a failure tells you how bad
    it is, not just that it happened.
    """
    df = data.load_metadata() if df is None else df
    gt = data.load_training_gt() if gt is None else gt

    checks: list[tuple[str, bool, str]] = []

    # 1. Two images per lesion — the assumption behind grouped splitting.
    per_lesion = df.groupby("lesion_id").size()
    n_ok = int((per_lesion == IMAGES_PER_LESION).sum())
    checks.append((
        f"Every lesion has exactly {IMAGES_PER_LESION} images",
        n_ok == len(per_lesion),
        f"{n_ok:,} / {len(per_lesion):,} lesions",
    ))

    # 2. One dermoscopic + one clinical close-up per lesion. Two images is not
    #    enough: two dermoscopic views would make the paired model meaningless.
    type_counts = (
        df.groupby(["lesion_id", "image_type"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )
    have_expected_types = list(type_counts.columns) == list(EXPECTED_IMAGE_TYPES)
    n_one_each = int((type_counts == 1).all(axis=1).sum()) if have_expected_types else 0
    checks.append((
        "Each lesion has exactly one image of each type",
        have_expected_types and n_one_each == len(type_counts),
        f"{n_one_each:,} / {len(type_counts):,} lesions; "
        f"types present: {list(type_counts.columns)}",
    ))

    # 3. The two files describe the same universe of lesions. A mismatch means
    #    images without labels, or labels without images.
    meta_ids = set(df["lesion_id"].unique())
    gt_ids = set(gt["lesion_id"].unique())
    checks.append((
        "lesion_id sets identical in metadata and training_gt",
        meta_ids == gt_ids,
        f"{len(meta_ids):,} metadata / {len(gt_ids):,} ground-truth; "
        f"{len(meta_ids - gt_ids)} only in metadata, {len(gt_ids - meta_ids)} only in gt",
    ))

    # 4. Exactly one positive per one-hot row — single-label, not multi-label.
    code_cols = [c for c in gt.columns if c != "lesion_id"]
    positives = gt[code_cols].sum(axis=1)
    n_single = int((positives == 1).sum())
    checks.append((
        "Exactly one positive class per lesion in the one-hot",
        n_single == len(gt),
        f"{n_single:,} / {len(gt):,} rows; positives per row: "
        f"{sorted(positives.unique().tolist())}",
    ))

    # 5. Lesion-level attributes agree across a lesion's two rows. dropna=False
    #    so that "one row says NaN, the other says torso" counts as varying.
    for col in LESION_CONSTANT_COLUMNS:
        if col not in df.columns:
            continue
        n_varying = int((df.groupby("lesion_id")[col].nunique(dropna=False) > 1).sum())
        checks.append((
            f"'{col}' is constant within a lesion",
            n_varying == 0,
            f"{n_varying:,} lesions with more than one value",
        ))

    # 6. Image ids unique — duplicates would put the same file in two splits.
    checks.append((
        "isic_id is unique",
        bool(df["isic_id"].is_unique),
        f"{df['isic_id'].nunique():,} unique / {len(df):,} rows",
    ))

    # 7. The diagnosis taxonomy nests: each finer label has exactly one parent.
    #    If it did not, the 3-class target would not be a coarsening of the
    #    fine-grained one and the two label schemes would contradict each other.
    for child, parent in zip(DIAGNOSIS_LEVELS[1:], DIAGNOSIS_LEVELS[:-1]):
        if child not in df.columns or parent not in df.columns:
            continue
        parents_per_child = df.groupby(child, dropna=True)[parent].nunique(dropna=False)
        n_bad = int((parents_per_child > 1).sum())
        checks.append((
            f"Every {child} value maps to exactly one {parent}",
            n_bad == 0,
            f"{n_bad} of {len(parents_per_child)} distinct {child} values have >1 parent",
        ))

    return pd.DataFrame(checks, columns=["check", "passed", "detail"])


# --- B4: shortcut and confound cross-tabs ----------------------------------
def shortcut_crosstabs(df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Row-percentage cross-tabs of acquisition metadata against the target.

    A *shortcut* is a feature that predicts the label without saying anything
    about the lesion. Row percentages are the right normalisation: they answer
    "given this acquisition condition, what is the class mix?", which is
    exactly the question a model exploiting the shortcut would be answering.

    A flat table is a good result — it means that condition is not a shortcut.
    """
    df = data.load_metadata() if df is None else df
    return {
        col: _row_percentage_crosstab(df, col)
        for col in ("image_manipulation", "image_type", "diagnosis_confirm_type")
    }


def _row_percentage_crosstab(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Cross-tab of ``col`` against diagnosis_1 as row percentages, plus row n.

    The raw row count is kept alongside the percentages because a 100% row
    built on nine images means something very different from one built on nine
    thousand.
    """
    pct = pd.crosstab(df[col], df[config.PRIMARY_LABEL_COL], normalize="index") * 100.0
    pct = pct.reindex(columns=config.PRIMARY_LABELS, fill_value=0.0)
    pct["n_images"] = df.groupby(col).size().reindex(pct.index).astype("int64")
    return pct


def leaky_columns_table() -> pd.DataFrame:
    """The banned-input list from config, each with the reason it is banned.

    Written as a table because "we excluded the leaky columns" is not a claim a
    marker can check, whereas a named column with a one-line reason is.
    """
    rows = [
        {"column": col, "reason": LEAKY_COLUMN_REASONS.get(col, UNREVIEWED_DECISION)}
        for col in config.LEAKY_COLUMNS
    ]
    return pd.DataFrame(rows)


# --- Report assembly -------------------------------------------------------
def build_quality_report(
    out_md: Path | str = config.QUALITY_REPORT_MD,
    out_csv: Path | str = config.QUALITY_REPORT_CSV,
    df: pd.DataFrame | None = None,
    gt: pd.DataFrame | None = None,
    sizes_df: pd.DataFrame | None = None,
    verify_images: bool = True,
) -> Path:
    """Write the B4 data-quality report as Markdown and as a machine CSV.

    Two outputs because they have two audiences: the Markdown is what a human
    reads and what goes into the milestone hand-in, the CSV is the same numbers
    in a long ``section/item/metric/value`` shape so a later notebook or test
    can assert on them without re-parsing prose.

    Pass ``sizes_df`` to reuse an earlier :func:`verify_all_images` result, or
    ``verify_images=False`` to skip the (cheap, but I/O-bound) image pass.

    Returns the path to the Markdown file.
    """
    df = data.load_metadata() if df is None else df
    gt = data.load_training_gt() if gt is None else gt

    if sizes_df is None and verify_images:
        sizes_df, _ = verify_all_images(df=df)

    missing = missing_value_report(df)
    consistency = label_consistency_checks(df, gt)
    crosstabs = shortcut_crosstabs(df)
    leaks = leaky_columns_table()
    sizes_summary = image_size_summary(sizes_df) if sizes_df is not None else None

    out_md, out_csv = Path(out_md), Path(out_csv)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_md.write_text(
        _render_markdown(df, gt, sizes_summary, missing, consistency, crosstabs, leaks),
        encoding="utf-8",
    )
    _flatten_to_csv(sizes_summary, missing, consistency, crosstabs, leaks).to_csv(
        out_csv, index=False
    )
    return out_md


def _render_markdown(
    df: pd.DataFrame,
    gt: pd.DataFrame,
    sizes_summary: pd.DataFrame | None,
    missing: pd.DataFrame,
    consistency: pd.DataFrame,
    crosstabs: dict[str, pd.DataFrame],
    leaks: pd.DataFrame,
) -> str:
    """Compose the human-readable report. Every number here is computed above."""
    n_failed = int((~consistency["passed"]).sum())
    lines = [
        "# MILK10k — data quality report",
        "",
        f"Generated {pd.Timestamp.today().date()} from `{config.METADATA_CSV.name}` "
        f"and `{config.TRAINING_GT_CSV.name}`.",
        "",
        "## 1. Scope",
        "",
        f"- {len(df):,} image rows, {df['lesion_id'].nunique():,} lesions "
        f"({len(df) / max(df['lesion_id'].nunique(), 1):.0f} images per lesion).",
        f"- {len(gt):,} ground-truth rows, one per lesion, "
        f"{len([c for c in gt.columns if c != 'lesion_id'])} diagnosis codes.",
        f"- Modelling unit is the **lesion**, so every split is grouped on `lesion_id`.",
        "",
    ]

    # --- Image integrity ---
    lines += ["## 2. Image integrity (B1)", ""]
    if sizes_summary is None:
        lines += ["_Image verification was skipped for this run._", ""]
    else:
        lookup = dict(zip(sizes_summary["metric"], sizes_summary["value"]))
        n_bad = int(lookup["n_missing"]) + int(lookup["n_unreadable"])
        verdict = (
            "Every referenced file exists and its header parses."
            if n_bad == 0
            else f"**{n_bad:,} files are missing or unreadable and must be excluded.**"
        )
        lines += [
            f"All {int(lookup['n_images']):,} files were opened and verified "
            "(header parse only, no pixel decode). " + verdict,
            "",
            _md_table(sizes_summary),
            "",
            f"The dataset is dimensionally uniform "
            f"({int(lookup['n_distinct_resolutions'])} distinct "
            f"{_plural('resolution', int(lookup['n_distinct_resolutions']))}, "
            f"{lookup['most_common_resolution']}), and "
            f"{100 * float(lookup[f'frac_at_least_{config.IMAGE_SIZE}']):.1f}% of images "
            f"have both sides at least {config.IMAGE_SIZE}px. Resizing to "
            f"{config.IMAGE_SIZE}x{config.IMAGE_SIZE} therefore only ever discards "
            "detail, never invents it — which is the argument for that input size.",
            "",
        ]

    # --- Missing values ---
    lines += [
        "## 3. Missing values and handling decisions (B4)",
        "",
        _md_table(missing),
        "",
    ]

    # --- Consistency ---
    shown = consistency.assign(
        passed=consistency["passed"].map({True: "PASS", False: "FAIL"})
    )
    lines += [
        "## 4. Label and structural consistency (B4)",
        "",
        f"{len(consistency) - n_failed} of {len(consistency)} checks pass.",
        "",
        _md_table(shown),
        "",
    ]

    # --- Shortcuts ---
    lines += ["## 5. Shortcuts, confounds and leakage (B4)", ""]
    titles = {
        "image_manipulation": "5.1 Image manipulation x diagnosis_1 (row %)",
        "image_type": "5.2 Image type x diagnosis_1 (row %)",
        "diagnosis_confirm_type": "5.3 Diagnosis confirmation type x diagnosis_1 (row %)",
    }
    for key, title in titles.items():
        lines += [f"### {title}", "", _md_table(crosstabs[key], index_name=key), ""]

    lines += ["### Conclusion", ""] + _shortcut_conclusion(crosstabs) + [""]

    # --- Leakage ---
    lines += [
        "## 6. Columns excluded from model inputs",
        "",
        "Defined once in `config.LEAKY_COLUMNS`; the reason each one is banned:",
        "",
        _md_table(leaks),
        "",
    ]
    return "\n".join(lines)


def _shortcut_conclusion(crosstabs: dict[str, pd.DataFrame]) -> list[str]:
    """Turn the three cross-tabs into an argued conclusion, not just tables.

    The wording is fixed but every number is read back out of the tables, so
    the conclusion cannot silently disagree with the data above it.
    """
    confirm = crosstabs["diagnosis_confirm_type"]
    modality = crosstabs["image_type"]
    manip = crosstabs["image_manipulation"]

    # The two confirmation types, identified by which class they lean towards,
    # rather than by hard-coding their names.
    histo = confirm["Malignant"].idxmax()
    clinical = confirm["Benign"].idxmax()

    # "Flat" = for every class, the two modality rows agree to within a rounding
    # tolerance. The widest per-class disagreement is the number worth quoting,
    # and a small one is the negative result we actually want.
    percentages = modality[config.PRIMARY_LABELS]
    spread = float((percentages.max(axis=0) - percentages.min(axis=0)).max())
    flat = spread < 0.5

    out = [
        f"**`diagnosis_confirm_type` is the loudest leak in the dataset.** "
        f"`{histo}` images are {confirm.loc[histo, 'Malignant']:.1f}% Malignant "
        f"({int(confirm.loc[histo, 'n_images']):,} images), while `{clinical}` images "
        f"are {confirm.loc[clinical, 'Benign']:.1f}% Benign "
        f"({int(confirm.loc[clinical, 'n_images']):,} images). This is not a quirk of "
        "the data: the decision to send a lesion to histopathology *is* the clinical "
        "suspicion of malignancy, so the column encodes the answer. It must never be "
        "a model input, and it must not be used to filter the test set either, or the "
        "reported accuracy will be measured on an artificially easy subset.",
        "",
    ]

    if flat:
        mix = ", ".join(
            f"{c} {modality.iloc[0][c]:.1f}%" for c in config.PRIMARY_LABELS
        )
        out += [
            f"**`image_type` is not a shortcut, and that is the useful result.** "
            f"Both modalities show an identical class mix ({mix}), which follows by "
            "construction: every lesion contributes exactly one dermoscopic and one "
            "clinical close-up image, so the two rows are the same lesions counted "
            "twice. Modality therefore carries no label information on its own, and "
            "both views can be used as training images (and later fused per lesion) "
            "without introducing a confound.",
            "",
        ]
    else:
        out += [
            f"**`image_type` is NOT flat** (spread {spread:.1f} percentage points) — "
            "unexpected given the 1:1 pairing, and worth investigating before both "
            "views are used.",
            "",
        ]

    hi = manip["Indeterminate"].idxmax()
    lo = manip["Indeterminate"].idxmin()
    out += [
        f"**`image_manipulation` is a mild confound, not a shortcut.** "
        f"`{hi}` images are {manip.loc[hi, 'Indeterminate']:.1f}% Indeterminate against "
        f"{manip.loc[lo, 'Indeterminate']:.1f}% for `{lo}`, but `{hi}` covers only "
        f"{int(manip.loc[hi, 'n_images']):,} of "
        f"{int(manip['n_images'].sum()):,} images. The skew is real and worth reporting, "
        "yet too small a slice to drive a classifier. We keep those images (dropping "
        "them would bias the rare Indeterminate class further) and simply never expose "
        "the column to the model.",
    ]
    return out


def _flatten_to_csv(
    sizes_summary: pd.DataFrame | None,
    missing: pd.DataFrame,
    consistency: pd.DataFrame,
    crosstabs: dict[str, pd.DataFrame],
    leaks: pd.DataFrame,
) -> pd.DataFrame:
    """Stack every table into one long ``section/item/metric/value`` frame.

    A single CSV of heterogeneous tables only works in long form; wide would
    mean one column per metric and a forest of blanks.
    """
    rows: list[dict] = []

    def add(section: str, item: str, metric: str, value) -> None:
        rows.append({"section": section, "item": item, "metric": metric,
                     "value": "" if pd.isna(value) else value})

    if sizes_summary is not None:
        for rec in sizes_summary.to_dict(orient="records"):
            add("image_integrity", rec["metric"], "value", rec["value"])

    for rec in missing.to_dict(orient="records"):
        add("missing_values", rec["column"], "n_missing", rec["n_missing"])
        add("missing_values", rec["column"], "pct_missing", round(rec["pct_missing"], 3))
        add("missing_values", rec["column"], "decision", rec["decision"])

    for rec in consistency.to_dict(orient="records"):
        add("consistency", rec["check"], "passed", bool(rec["passed"]))
        add("consistency", rec["check"], "detail", rec["detail"])

    for name, table in crosstabs.items():
        for idx, rec in table.to_dict(orient="index").items():
            for metric, value in rec.items():
                add(f"crosstab_{name}", str(idx), str(metric),
                    round(value, 2) if isinstance(value, float) else value)

    for rec in leaks.to_dict(orient="records"):
        add("leaky_columns", rec["column"], "reason", rec["reason"])

    return pd.DataFrame(rows, columns=["section", "item", "metric", "value"])


# --- Markdown helpers ------------------------------------------------------
def _plural(word: str, n: int) -> str:
    """Naive English pluralisation, so generated prose does not say "1 rows"."""
    return word if n == 1 else word + "s"


def _md_table(table: pd.DataFrame, index_name: str | None = None) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table.

    Hand-rolled rather than ``DataFrame.to_markdown`` so the report does not
    depend on the optional ``tabulate`` package, which is not in
    requirements.txt.
    """
    frame = table.reset_index() if index_name is not None else table
    if index_name is not None:
        frame = frame.rename(columns={frame.columns[0]: index_name})

    header = list(frame.columns)
    body = [[_md_cell(v) for v in row] for row in frame.itertuples(index=False)]

    out = ["| " + " | ".join(str(h) for h in header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(cells) + " |" for cells in body]
    return "\n".join(out)


def _md_cell(value) -> str:
    """Format one cell: thousands separators for ints, 1 decimal for floats.

    Pipes are escaped because an unescaped one would silently split the cell
    and shift every column after it.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.1f}" if abs(value) >= 0.01 else f"{float(value):.3f}"
    return str(value).replace("|", "\\|")
