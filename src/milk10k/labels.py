"""The single source of truth for "what is the label" in this project.

Two targets are defined, and every later milestone reads them from here rather
than re-deriving them:

* **primary** — ``diagnosis_1`` from ``metadata.csv``, three classes
  (Benign / Indeterminate / Malignant). This is the coarse triage decision.
* **stretch** — the 11 diagnosis codes one-hot encoded in ``training_gt.csv``,
  collapsed into a single ``dx`` column.

Both are defined on the **lesion**, not on the image: a lesion contributes two
photographs (dermoscopic + clinical close-up) that share one biopsy result, so
the label lives at lesion grain and is broadcast to images downstream.

The integer encodings are written once to ``artifacts/label_map.json``
(:func:`write_label_map`) together with the counts they were derived from and a
written rationale, so that training, evaluation and the report can never drift
apart on what "class 2" means.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config, data


# --- The documented decision (Milestone 1, B3) -----------------------------
# These strings are written verbatim into label_map.json. They live in code so
# that the artefact and the report cannot disagree about why the target looks
# the way it does.
PRIMARY_STRATEGY = (
    "All three diagnosis_1 classes are kept. 'Indeterminate' (123 lesions, "
    "2.3%) is a real clinical state - the pathologist could not commit - and "
    "those are precisely the lesions that warrant excision. Merging it into "
    "Benign would train the model to reassure on the ambiguous cases, and "
    "dropping it would silently narrow the task; its small size is handled "
    "with class weights and a balanced sampler, not by deletion."
)

STRETCH_STRATEGY = (
    "All 11 codes are kept. The five rare classes (MAL_OTH 9, BEN_OTH 44, "
    "VASC 47, INF 50, DF 52 lesions) are not merged into an 'other' bucket, "
    "because a bucket mixing malignant MAL_OTH with benign VASC/DF destroys "
    "the clinical meaning of the confusion matrix; they are handled with "
    "class weights and a WeightedRandomSampler instead. Honest caveat: "
    "MAL_OTH (9 lesions) cannot be evaluated reliably at any split size, so "
    "its per-class metrics must be reported as indicative only."
)

# A class below this many lesions cannot fill a 15% test split with enough
# examples for a stable per-class score, so it is flagged as rare in the
# artefact rather than quietly treated like the head classes.
RARE_CLASS_THRESHOLD = 100

# Structural facts this module refuses to proceed without. They are asserted,
# not trusted: a silently mis-shaped ground-truth file would poison every
# downstream milestone.
N_LESIONS = 5_240

# Which column each labelling scheme lives in. Defined here, in the module that
# owns the labels, and imported by ``imbalance`` and ``datasets`` instead of
# being retyped there: two modules with their own copy of this mapping is
# exactly how "scheme='primary', label_col='dx'" ends up silently encoding the
# wrong target.
SCHEME_LABEL_COLUMNS: dict[str, str] = {
    "primary": config.PRIMARY_LABEL_COL,   # "diagnosis_1" -> 3 classes
    "stretch": config.STRETCH_LABEL_COL,   # "dx"          -> 11 codes
}


# --- Lesion-level labels ---------------------------------------------------
def codes_from_onehot(gt: pd.DataFrame) -> pd.Series:
    """Turn the 11-column one-hot block into one diagnosis code per row.

    This is the single implementation of that step in the project. ``data.py``
    needs it for the exploratory view and this module needs it for the target,
    and two ``idxmax`` calls written independently is precisely how two parts of
    a pipeline end up disagreeing about what a lesion is.

    The columns are taken in ``config.STRETCH_LABELS`` order rather than in
    whatever order the CSV happens to use, so the code ``idxmax`` returns for a
    tie — or for a column that was added to the file later — cannot depend on
    the file's layout. Returns a Series aligned to ``gt``'s index; it makes no
    claim about how many positives a row has, which is the caller's business.
    """
    missing = [c for c in config.STRETCH_LABELS if c not in gt.columns]
    assert not missing, f"training_gt.csv is missing one-hot columns: {missing}"

    return gt[config.STRETCH_LABELS].idxmax(axis=1)


def lesion_labels(gt: pd.DataFrame | None = None) -> pd.DataFrame:
    """Collapse the 11-column one-hot ground truth into one ``dx`` code per lesion.

    ``training_gt.csv`` stores the stretch target as 11 indicator columns. A
    model needs a single categorical column, and the whole project assumes that
    every lesion carries exactly one diagnosis. That assumption is checked here
    instead of downstream: a row with zero positives would become a silent
    wrong label via ``idxmax``, and a row with two positives would mean the
    lesion is genuinely multi-label and the task definition is wrong.

    Returns a frame with ``lesion_id`` and ``dx``.
    """
    # The 5,240-row check only makes sense for the shipped file. A caller who
    # hands us a subset (a split, or a five-row fixture) is doing something
    # legitimate, so remember which case we are in rather than rejecting both.
    using_full_dataset = gt is None
    gt = data.load_training_gt() if gt is None else gt

    onehot = gt[config.STRETCH_LABELS]
    n_positive = onehot.sum(axis=1)
    bad = gt.loc[n_positive != 1, "lesion_id"]
    assert bad.empty, (
        f"{len(bad)} lesions do not have exactly one positive label, "
        f"e.g. {bad.head(5).tolist()}"
    )

    out = pd.DataFrame(
        {
            "lesion_id": gt["lesion_id"].to_numpy(),
            "dx": codes_from_onehot(gt).to_numpy(),
        }
    )

    if using_full_dataset:
        assert len(out) == N_LESIONS, (
            f"expected {N_LESIONS:,} lesions in training_gt.csv, got {len(out):,}"
        )
    assert out["lesion_id"].is_unique, "lesion_id is duplicated in training_gt.csv"
    return out


def _lesion_primary(meta: pd.DataFrame | None = None) -> pd.DataFrame:
    """Lift the per-image ``diagnosis_1`` column up to one row per lesion.

    ``metadata.csv`` repeats the diagnosis on both images of a lesion. Taking
    the first value is only safe if the column really is constant within a
    lesion, so that is verified before de-duplicating.
    """
    meta = data.load_metadata() if meta is None else meta

    varying = meta.groupby("lesion_id")[config.PRIMARY_LABEL_COL].nunique(dropna=False)
    assert (varying == 1).all(), (
        f"{int((varying > 1).sum())} lesions disagree with themselves on "
        f"{config.PRIMARY_LABEL_COL}"
    )

    out = meta[["lesion_id", config.PRIMARY_LABEL_COL]].drop_duplicates("lesion_id")
    return out.reset_index(drop=True)


# --- Integer encodings -----------------------------------------------------
def label_map(scheme: str = "primary") -> dict[str, int]:
    """Class name -> integer code for one labelling scheme.

    The integer is the position in the configured class list, which is why that
    list is ordered and fixed in ``config``: torch's loss functions index into
    the model's output columns, so the meaning of column *k* must be stable
    across training runs, checkpoints and the report.
    """
    names = _scheme_classes(scheme)
    return {name: i for i, name in enumerate(names)}


def inverse_label_map(scheme: str = "primary") -> dict[int, str]:
    """Integer code -> class name, for turning predictions back into words."""
    return {i: name for name, i in label_map(scheme).items()}


def _scheme_classes(scheme: str) -> list[str]:
    """Resolve a scheme name to its ordered class list, or fail loudly."""
    if scheme == "primary":
        return list(config.PRIMARY_LABELS)
    if scheme == "stretch":
        return list(config.STRETCH_LABELS)
    raise ValueError(
        f"unknown label scheme {scheme!r}; expected 'primary' or 'stretch'"
    )


def scheme_for_column(label_col: str) -> str:
    """Which scheme a label column belongs to, or a clear error if it is unknown.

    Guessing here would be worse than failing: inferring "primary" for an
    unrecognised column would hand the caller a 3-element weight vector for an
    11-class target, and the shapes happen to be compatible often enough that
    the mistake survives until the confusion matrix looks strange.
    """
    for scheme, column in SCHEME_LABEL_COLUMNS.items():
        if column == label_col:
            return scheme
    raise ValueError(
        f"cannot infer the label scheme from column {label_col!r}; "
        f"known label columns are {sorted(SCHEME_LABEL_COLUMNS.values())}. "
        "Pass scheme='primary' or scheme='stretch' explicitly."
    )


def encode(
    labels: pd.Series,
    scheme: str = "primary",
    mapping: dict[str, int] | None = None,
) -> pd.Series:
    """Encode a column of class names as integer codes.

    Deliberately strict: ``Series.map`` turns anything outside the map into
    NaN, and a NaN label that survives into a DataLoader is how a model ends up
    training on garbage without anyone noticing. Unknown values (including
    missing ones) therefore raise, naming the offenders.

    ``mapping`` overrides the scheme's own map. It exists so that the Dataset
    classes in ``datasets.py``, which allow a caller to pass a custom label map
    (for a model trained on a subset of the classes), can still route their
    encoding through this one function instead of reimplementing the strictness.
    """
    if not isinstance(labels, pd.Series):
        labels = pd.Series(list(labels))  # tolerate lists/arrays from callers

    explicit_map = mapping is not None
    mapping = label_map(scheme) if mapping is None else dict(mapping)
    codes = labels.map(mapping)

    unmapped = codes.isna()
    if unmapped.any():
        offenders = sorted({repr(v) for v in labels[unmapped]})
        where = "the label map given" if explicit_map else f"the {scheme!r} scheme"
        raise ValueError(
            f"{int(unmapped.sum())} label(s) are not part of {where}: "
            f"{', '.join(offenders)}. Valid classes: {list(mapping)}"
        )

    return codes.astype("int64")


# --- The 11-class / 3-class relationship -----------------------------------
def code_to_primary_table(
    gt: pd.DataFrame | None = None, meta: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Show how each 11-class code sits inside the 3-class primary target.

    The two schemes are *nearly* nested, and the exception matters: AKIEC
    straddles Indeterminate and Malignant, so no lookup table can convert a
    stretch prediction into a primary one. This table is what makes that
    visible in the report (A1.1c, B2) instead of being papered over by a
    hand-written mapping.

    One row per code with the lesion count, the per-``diagnosis_1`` breakdown,
    the primary class(es) it maps to, and an ``ambiguous`` flag.
    """
    labels = lesion_labels(gt)
    primary = _lesion_primary(meta)

    lesions = labels.merge(primary, on="lesion_id", how="left", validate="one_to_one")
    assert lesions[config.PRIMARY_LABEL_COL].notna().all(), (
        "some lesions in training_gt.csv have no diagnosis_1 in metadata.csv"
    )

    counts = pd.crosstab(lesions["dx"], lesions[config.PRIMARY_LABEL_COL])
    counts = counts.reindex(
        index=config.STRETCH_LABELS, columns=config.PRIMARY_LABELS, fill_value=0
    )

    positive = counts.gt(0)
    table = pd.DataFrame(
        {
            "dx": counts.index,
            "name": [config.DIAGNOSIS_CODES[c] for c in counts.index],
            "n_lesions": counts.sum(axis=1).to_numpy(),
            **{cls: counts[cls].to_numpy() for cls in config.PRIMARY_LABELS},
            # Only 11 rows, so a row-wise join of the positive column names is
            # clearer here than any vectorised trick.
            "maps_to": positive.apply(
                lambda row: " + ".join(positive.columns[row]), axis=1
            ).to_numpy(),
            "ambiguous": positive.sum(axis=1).gt(1).to_numpy(),
        }
    )

    return table.sort_values("n_lesions", ascending=False).reset_index(drop=True)


def ambiguous_stretch_codes(table: pd.DataFrame | None = None) -> list[str]:
    """The 11-class codes that map to more than one ``diagnosis_1`` value."""
    table = code_to_primary_table() if table is None else table
    return table.loc[table["ambiguous"], "dx"].tolist()


# --- The artefact ----------------------------------------------------------
def write_label_map(
    path: Path | str = config.LABEL_MAP_JSON,
    gt: pd.DataFrame | None = None,
    meta: pd.DataFrame | None = None,
) -> dict:
    """Write ``label_map.json``: both encodings, the real counts, and the why.

    This is the file every later milestone loads instead of hard-coding class
    orders. Counts are recomputed from the CSVs on every write rather than
    typed in, so the artefact can never describe a dataset that is not the one
    on disk. Returns the dictionary it wrote.
    """
    labels = lesion_labels(gt)
    primary = _lesion_primary(meta)
    table = code_to_primary_table(gt=gt, meta=meta)

    primary_counts = _ordered_counts(
        primary[config.PRIMARY_LABEL_COL], config.PRIMARY_LABELS
    )
    stretch_counts = _ordered_counts(labels["dx"], config.STRETCH_LABELS)

    # Rare = too few lesions to survive a 15% test split intact. Listed
    # smallest-first because that is the order in which they stop being
    # measurable.
    rare = [c for c, n in sorted(stretch_counts.items(), key=lambda kv: kv[1])
            if n < RARE_CLASS_THRESHOLD]

    payload = {
        "created": config.SPLIT_DATE,
        "seed": config.SEED,
        "n_lesions": int(len(labels)),
        "primary": {
            "column": config.PRIMARY_LABEL_COL,
            "classes": label_map("primary"),
            "n_lesions": primary_counts,
            "strategy": PRIMARY_STRATEGY,
        },
        "stretch": {
            "column": config.STRETCH_LABEL_COL,
            "classes": label_map("stretch"),
            "n_lesions": stretch_counts,
            "rare_classes": rare,
            "rare_threshold_lesions": RARE_CLASS_THRESHOLD,
            "strategy": STRETCH_STRATEGY,
        },
        "code_to_diagnosis_1": _code_to_diagnosis_1(table),
        "ambiguous_codes": ambiguous_stretch_codes(table),
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_label_map(path: Path | str = config.LABEL_MAP_JSON) -> dict:
    """Read back the label map written by :func:`write_label_map`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist yet - run labels.write_label_map() first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


# --- Small helpers ---------------------------------------------------------
def _ordered_counts(values: pd.Series, classes: list[str]) -> dict[str, int]:
    """Count class occurrences in the configured class order, as plain ints.

    The order matches the integer encoding so the JSON reads in the same order
    as the model's output columns, and ``int`` conversion keeps numpy types out
    of the JSON encoder.
    """
    counts = values.value_counts()
    return {cls: int(counts.get(cls, 0)) for cls in classes}


def _code_to_diagnosis_1(table: pd.DataFrame) -> dict[str, dict]:
    """Serialise the code -> diagnosis_1 relationship, ambiguity included.

    Kept as a nested object rather than a flat ``code -> class`` string map on
    purpose: a flat map would have to invent a single answer for AKIEC, which
    is exactly the fiction this project refuses to tell.
    """
    ordered = table.sort_values("dx")
    return {
        row.dx: {
            "diagnosis_1": row.maps_to,
            "ambiguous": bool(row.ambiguous),
            "n_lesions": int(row.n_lesions),
            "n_by_diagnosis_1": {
                cls: int(getattr(row, cls)) for cls in config.PRIMARY_LABELS
            },
        }
        for row in ordered.itertuples(index=False)
    }
