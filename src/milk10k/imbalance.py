"""Class-imbalance handling for the MILK10k classifier (Milestone 1, B8).

MILK10k is severely unbalanced at the lesion level: BCC has 2,522 lesions and
MAL_OTH has 9. A model trained on the raw distribution can score ~48% accuracy
by predicting BCC for everything, which is worthless for a triage task where
the rare classes are the ones that matter.

Two standard remedies are implemented here, and the course asks for BOTH so
they can be compared:

1. **Class-weighted loss** — every class keeps its natural frequency in the
   data, but a mistake on a rare class costs the loss function more.
2. **Weighted sampling** — the data loader draws rare lesions more often, so
   each epoch sees a roughly balanced stream of examples.

Everything in this module is computed on the **train split only**. Weights
derived from val/test counts would leak information about the evaluation sets
into training, which is exactly the kind of quiet mistake that invalidates a
result.

The canonical class order comes from ``config.PRIMARY_LABELS`` and
``config.STRETCH_LABELS``; the integer label of a class is its position in
that list. ``labels.py`` builds its label map from the same two lists, so the
tensor produced here lines up with the integer targets produced there.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

from . import config, labels


# --- Scheme helpers --------------------------------------------------------
def class_order(scheme: str) -> list[str]:
    """Return the canonical class names of a scheme, in integer-label order.

    Every count, weight vector and tensor in this module is built against this
    list, so a class that is missing from some split still occupies its own
    slot instead of shifting every later class by one.

    The order is read straight out of ``labels.label_map`` rather than from
    ``config`` directly, because that map is what the Dataset classes encode
    with and what ``label_map.json`` records. Position *k* in the weight tensor
    has to mean the same class as column *k* of the model's output, and the only
    way to guarantee that is to have one function decide it.
    """
    return list(labels.label_map(scheme))


def resolve_scheme(label_col: str, scheme: str | None = None) -> str:
    """Work out which label scheme a column belongs to.

    An explicit ``scheme`` always wins; otherwise the column name decides. We
    refuse to guess for an unfamiliar column, because guessing wrong would
    silently produce weights for the wrong number of classes.
    """
    if scheme is not None:
        class_order(scheme)  # validates, raises for a typo like "stretchy"
        return scheme
    return labels.scheme_for_column(label_col)


# --- Counting --------------------------------------------------------------
def class_counts(
    table: pd.DataFrame, label_col: str, scheme: str | None = None
) -> pd.Series:
    """Count rows per class, in canonical order, keeping empty classes as 0.

    ``value_counts`` alone would drop a class that has no rows at all, and the
    weight vector would then be one element too short — a bug that only shows
    up as a shape error much later, or worse, as a silent off-by-one in the
    class indices. A class with zero training examples is something we want to
    *see*, so it stays in the index with a count of 0.

    Raises if the column contains a value outside the scheme (including NaN),
    because an unrecognised label must never be quietly ignored.
    """
    scheme = resolve_scheme(label_col, scheme)
    order = class_order(scheme)

    raw = table[label_col].value_counts(dropna=False)

    unknown = [str(v) for v in raw.index if v not in order]
    if unknown:
        raise ValueError(
            f"column {label_col!r} contains labels outside the {scheme} scheme: "
            f"{sorted(unknown)}"
        )

    counts = raw.reindex(order, fill_value=0).astype(int)
    counts.index.name = label_col
    counts.name = "n"
    return counts


def imbalance_ratio(
    table: pd.DataFrame, label_col: str, scheme: str | None = None
) -> float:
    """Size of the largest class divided by the size of the smallest one.

    This is the single number that says how skewed a split is: 1.0 is perfectly
    balanced, and MILK10k's 11-class target sits near 280.

    Only classes with at least one member are considered. A class with zero
    members would make the ratio infinite, i.e. undefined — the ratio silently
    ignores such a class, so always read it next to :func:`class_counts`, which
    shows the zero.
    """
    counts = class_counts(table, label_col, scheme)
    present = counts[counts > 0]
    if present.empty:
        raise ValueError(f"column {label_col!r} has no labelled rows to compare")
    return float(present.max() / present.min())


# --- Loss weights ----------------------------------------------------------
def compute_class_weights(
    table: pd.DataFrame,
    label_col: str,
    scheme: str,
    method: str = "balanced",
) -> dict[str, float]:
    """Per-class loss weights, computed from one split (in practice: train).

    Two methods, and the choice between them is a real modelling decision:

    ``"balanced"``
        ``n_samples / (n_classes * count)``, the formula sklearn's
        ``compute_class_weight(class_weight="balanced")`` uses. It makes every
        class contribute the same total mass to the loss, so the ratio between
        two weights is exactly the inverse ratio of their counts. On our train
        split that is brutal: MAL_OTH has ~5 lesions against BCC's ~1,800, so
        MAL_OTH's weight comes out ~360x BCC's. Five lesions then steer the
        gradient as hard as eighteen hundred, and one unlucky outlier among
        them can dominate training and destabilise it.

    ``"inverse_sqrt"``
        ``1 / sqrt(count)``, rescaled so the mean weight over non-empty classes
        is 1. Taking the square root compresses the same ordering into a much
        gentler range: the 360x above becomes ~19x. It is the usual fallback
        when balanced weights get extreme — it still pushes the model off the
        majority class, but it does not bet the whole run on five examples.

        The honest trade-off: inverse_sqrt does *not* equalise the classes, so
        the model keeps a bias towards the frequent ones. Which is better is an
        empirical question, decided on validation macro-F1, not by assertion.

    Classes absent from ``table`` get weight 0.0: there is nothing to learn
    from them, 0 keeps the loss finite, and the zero is visible in the stored
    JSON rather than hidden as an ``inf``.

    ``n_classes`` is the number of canonical classes, present or not, so the
    weights of the remaining classes do not quietly rescale when a rare class
    is missing from a fold.
    """
    order = class_order(scheme)
    counts = class_counts(table, label_col, scheme)
    present = counts > 0

    if method == "balanced":
        n_samples = int(counts.sum())
        n_classes = len(order)
        raw = np.where(present, n_samples / (n_classes * counts.replace(0, 1)), 0.0)
    elif method == "inverse_sqrt":
        raw = np.where(present, 1.0 / np.sqrt(counts.replace(0, 1)), 0.0)
        # Normalise to mean 1 over the classes that exist, so that switching
        # method changes the RELATIVE weighting without also changing the
        # overall scale of the loss (and with it the effective learning rate).
        raw = raw / raw[present.to_numpy()].mean()
    else:
        raise ValueError(
            f"unknown method {method!r}; expected 'balanced' or 'inverse_sqrt'"
        )

    return {cls: float(w) for cls, w in zip(order, raw)}


def weights_tensor(weights: dict[str, float], scheme: str) -> torch.Tensor:
    """Turn a {class: weight} dict into a tensor in integer-label order.

    ``nn.CrossEntropyLoss(weight=...)`` indexes this tensor by the integer
    target, so the order has to be exactly the canonical one — a dict has no
    order you can rely on, which is why this conversion is explicit and not
    left to ``list(weights.values())``.
    """
    order = class_order(scheme)
    missing = [cls for cls in order if cls not in weights]
    if missing:
        raise ValueError(f"no weight given for classes {missing}")
    return torch.tensor([weights[cls] for cls in order], dtype=torch.float32)


# --- Persistence -----------------------------------------------------------
def write_class_weights(
    train_table: pd.DataFrame,
    path: Path = config.CLASS_WEIGHTS_JSON,
    primary_col: str = config.PRIMARY_LABEL_COL,
    stretch_col: str = config.STRETCH_LABEL_COL,
    split_name: str = "train",
    row_unit: str = "lesion",
) -> dict:
    """Compute and store the weights for both schemes and both methods.

    Milestone 1 asks for ``class_weights.json`` as a deliverable, and later
    milestones load it instead of recomputing: the training run must use the
    exact numbers that were reported, not numbers that drift when the split
    code changes.

    Both methods are stored even though training uses one of them, so the
    comparison in the report is backed by the file rather than by memory. The
    counts travel with the weights so anyone can re-derive them by hand.
    """
    payload: dict = {
        "generated_on": date.today().isoformat(),
        "split_date": config.SPLIT_DATE,
        "seed": config.SEED,
        "computed_on_split": split_name,
        "row_unit": row_unit,
        "n_rows": int(len(train_table)),
        "note": (
            f"Weights are computed on the {split_name.upper()} split ONLY "
            f"({row_unit}-level rows). Using val/test counts would leak the "
            "evaluation distribution into training."
        ),
        "schemes": {},
    }

    for scheme, col in (("primary", primary_col), ("stretch", stretch_col)):
        counts = class_counts(train_table, col, scheme)
        order = class_order(scheme)
        payload["schemes"][scheme] = {
            "label_col": col,
            "classes": order,
            # Straight from labels.py, so class_weights.json and label_map.json
            # cannot disagree about which integer means which class.
            "label_map": labels.label_map(scheme),
            "counts": {cls: int(n) for cls, n in counts.items()},
            "imbalance_ratio": round(imbalance_ratio(train_table, col, scheme), 3),
            "weights": {
                "balanced": compute_class_weights(
                    train_table, col, scheme, method="balanced"
                ),
                "inverse_sqrt": compute_class_weights(
                    train_table, col, scheme, method="inverse_sqrt"
                ),
            },
        }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_class_weights(path: Path = config.CLASS_WEIGHTS_JSON) -> dict:
    """Read back the JSON written by :func:`write_class_weights`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --- Weighted sampling -----------------------------------------------------
def make_weighted_sampler(
    table: pd.DataFrame,
    label_col: str,
    scheme: str,
    generator: torch.Generator | None = None,
    replacement: bool = True,
) -> WeightedRandomSampler:
    """Build a sampler that draws a roughly class-balanced stream of rows.

    Each row gets weight ``1 / count(its class)``, so the total weight of every
    class is 1 and each class is equally likely to be drawn. ``num_samples`` is
    ``len(table)``, which keeps an "epoch" the same length as before — what
    changes is the composition of that epoch, not its cost.

    The consequence the course slides flag, and it is not a footnote:
    oversampling does not create new information. With ~6 MAL_OTH lesions the
    sampler will show the *same six* lesions hundreds of times per epoch, so the
    network can simply memorise them — training accuracy on the rare classes
    looks excellent while validation accuracy does not move. That is why the
    sampler is paired with augmentation (``transforms.train_transform``), which
    at least makes each repeat a different view, and why the rare-class result
    is always read off the validation split. Oversampling is a trade of variance
    for bias, not a free win.

    ``replacement=False`` is allowed but turns the sampler into a weighted
    shuffle of the existing rows: with no repeats there is no oversampling, so
    keep the default unless you know why you want that.
    """
    counts = class_counts(table, label_col, scheme)
    per_row_counts = table[label_col].map(counts)

    # 1/count: rare class -> large per-row weight. Counts are guaranteed > 0
    # here because every count came from the rows we are weighting. copy=True
    # because a pandas 3 view is read-only, and torch warns when it wraps one.
    weights = (1.0 / per_row_counts).to_numpy(dtype=float, copy=True)

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(table),
        replacement=replacement,
        generator=generator,
    )


def sampler_report(
    table: pd.DataFrame,
    label_col: str,
    scheme: str,
    n_batches: int = 20,
    batch_size: int = 32,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Draw real batches through the sampler and compare the two histograms.

    Claiming the sampler balances the classes is cheap; this actually pulls
    ``n_batches * batch_size`` indices out of it and counts what came back. The
    ``lift`` column is the empirical share divided by the natural share: a
    sampler that works pushes every class towards an equal share, so ``lift``
    goes far above 1 for the rare classes and below 1 for BCC.

    Sampled counts are drawn with replacement, so ``sampled_n`` for a rare
    class counts repeats of a handful of lesions — read it as "how often the
    model sees this class", not "how many distinct lesions it sees".
    """
    generator = torch.Generator().manual_seed(seed)
    sampler = make_weighted_sampler(table, label_col, scheme, generator=generator)

    wanted = n_batches * batch_size
    drawn: list[int] = []
    while len(drawn) < wanted:
        # One pass over the sampler yields len(table) indices; for a large
        # request we simply start another pass, exactly as a DataLoader would
        # across epochs.
        drawn.extend(int(i) for i in sampler)
    drawn = drawn[:wanted]

    # The sampler yields POSITIONS, not DataFrame index labels, so go through
    # the positional numpy view rather than .loc.
    labels = table[label_col].to_numpy()[drawn]
    sampled = pd.Series(labels).value_counts()

    order = class_order(scheme)
    raw = class_counts(table, label_col, scheme)
    sampled = sampled.reindex(order, fill_value=0).astype(int)

    report = pd.DataFrame(
        {
            "raw_n": raw.to_numpy(),
            "raw_pct": 100 * raw.to_numpy() / raw.sum(),
            "sampled_n": sampled.to_numpy(),
            "sampled_pct": 100 * sampled.to_numpy() / wanted,
        },
        index=pd.Index(order, name=label_col),
    )
    report["lift"] = report["sampled_pct"] / report["raw_pct"].replace(0, np.nan)
    return report.round(3)
