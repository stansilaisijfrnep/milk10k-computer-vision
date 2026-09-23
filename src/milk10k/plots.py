"""Every figure in the MILK10k EDA report.

Each function takes the already-loaded dataframes, draws one figure and
returns the matplotlib ``Figure`` so it renders inline in a notebook. Passing
``save_as`` also writes it to ``reports/figures/``.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

from . import config, data


def _save(fig: plt.Figure, save_as: str | None) -> plt.Figure:
    if save_as:
        config.ensure_dirs()
        fig.savefig(config.FIGURES_DIR / save_as)
    return fig


def _bar_labels(ax: plt.Axes, values, total: int, fmt: str = "{:,} ({:.1%})") -> None:
    """Annotate bars with count and share of total."""
    for patch, value in zip(ax.patches, values):
        ax.annotate(
            fmt.format(int(value), value / total),
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 2),
            textcoords="offset points",
        )


# --- 1. Class distribution -------------------------------------------------
def plot_primary_class_distribution(
    lesion: pd.DataFrame, save_as: str | None = None
) -> plt.Figure:
    """The required 3-class target, counted at the lesion level."""
    counts = lesion["diagnosis_1"].value_counts()
    colors = [config.BENIGN_MALIGNANT_COLORS.get(c, config.ORANGE) for c in counts.index]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(counts.index, counts.values, color=colors, width=0.6)
    _bar_labels(ax, counts.values, counts.sum())
    ax.set_ylabel("Number of lesions")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.set_title("Primary target — diagnosis_1 (3 classes, lesion level)")
    ax.text(
        0.99,
        0.95,
        f"Imbalance ratio {counts.max() / counts.min():.1f} : 1",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        style="italic",
    )
    fig.tight_layout()
    return _save(fig, save_as)


def plot_11_class_distribution(
    lesion: pd.DataFrame, save_as: str | None = None
) -> plt.Figure:
    """The stretch target — 11 diagnosis codes, coloured by malignancy."""
    counts = lesion["label_11"].value_counts()
    colors = [
        config.BENIGN_MALIGNANT_COLORS[config.code_group(code)] for code in counts.index
    ]

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.bar(counts.index, counts.values, color=colors, width=0.65)
    for patch, value in zip(ax.patches, counts.values):
        ax.annotate(
            f"{value:,}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=8.5,
            xytext=(0, 2),
            textcoords="offset points",
        )
    ax.set_yscale("log")
    ax.set_ylabel("Number of lesions (log scale)")
    ax.set_title("Stretch target — 11-class scheme (training_gt.csv, lesion level)")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=config.BENIGN_MALIGNANT_COLORS[g])
        for g in config.PRIMARY_CLASSES
    ]
    ax.legend(handles, [f"{g} codes" for g in config.PRIMARY_CLASSES], loc="upper right")
    ax.text(
        0.5,
        -0.22,
        f"Most common class is {counts.idxmax()} with {counts.max():,} lesions; "
        f"rarest is {counts.idxmin()} with {counts.min():,} "
        f"— an imbalance of {counts.max() / counts.min():.0f} : 1.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        style="italic",
    )
    fig.tight_layout()
    return _save(fig, save_as)


# --- 2. Missing values -----------------------------------------------------
def plot_missing_values(df: pd.DataFrame, save_as: str | None = None) -> plt.Figure:
    """Share of missing values per column, image level."""
    missing = df.isna().mean().sort_values(ascending=True)
    missing = missing[missing > 0]

    fig, ax = plt.subplots(figsize=(8, max(3.2, 0.32 * len(missing) + 1)))
    ax.barh(missing.index, missing.values * 100, color=config.ORANGE, height=0.65)
    for y, value in enumerate(missing.values):
        ax.annotate(
            f"{value:.1%}",
            (value * 100, y),
            va="center",
            fontsize=9,
            xytext=(4, 0),
            textcoords="offset points",
        )
    ax.set_xlabel("Missing (%)")
    ax.set_xlim(0, 105)
    ax.set_title("Missing values by column (image level)")
    fig.tight_layout()
    return _save(fig, save_as)


# --- 3. Demographics -------------------------------------------------------
def plot_demographics(lesion: pd.DataFrame, save_as: str | None = None) -> plt.Figure:
    """Age distribution, sex balance and age split by diagnosis."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    ages = lesion["age_approx"].dropna()
    ax.hist(ages, bins=16, color=config.ORANGE, edgecolor="white")
    ax.axvline(ages.median(), color=config.GREY, linestyle="--", linewidth=1.4)
    ax.annotate(
        f"median {ages.median():.0f}",
        (ages.median(), ax.get_ylim()[1] * 0.92),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=9,
    )
    ax.set_xlabel("Approximate age (years)")
    ax.set_ylabel("Number of lesions")
    ax.set_title("Age distribution")

    ax = axes[1]
    sex = lesion["sex"].value_counts()
    ax.bar(sex.index, sex.values, color=[config.ORANGE, "#9C6B2F"][: len(sex)], width=0.5)
    _bar_labels(ax, sex.values, sex.sum())
    ax.set_ylim(0, sex.max() * 1.18)
    ax.set_ylabel("Number of lesions")
    ax.set_title("Sex")

    ax = axes[2]
    order = [c for c in config.PRIMARY_CLASSES if c in lesion["diagnosis_1"].unique()]
    grouped = [lesion.loc[lesion["diagnosis_1"] == c, "age_approx"].dropna() for c in order]
    bp = ax.boxplot(grouped, patch_artist=True, widths=0.55, tick_labels=order)
    for patch, name in zip(bp["boxes"], order):
        patch.set_facecolor(config.BENIGN_MALIGNANT_COLORS[name])
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("white")
        median.set_linewidth(1.8)
    ax.set_ylabel("Approximate age (years)")
    ax.set_title("Age by diagnosis_1")

    fig.suptitle("MILK10k demographics (lesion level)", fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, save_as)


# --- 4. Anatomic site ------------------------------------------------------
def plot_anatomic_site(lesion: pd.DataFrame, save_as: str | None = None) -> plt.Figure:
    """Where lesions come from, and how malignancy rate varies by site."""
    site = lesion["anatom_site_general"].fillna("(missing)")
    counts = site.value_counts()
    rate = (
        lesion.assign(site=site)
        .groupby("site")["diagnosis_1"]
        .apply(lambda s: (s == "Malignant").mean())
        .reindex(counts.index)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))

    ax = axes[0]
    ax.barh(counts.index[::-1], counts.values[::-1], color=config.ORANGE, height=0.6)
    for y, value in enumerate(counts.values[::-1]):
        ax.annotate(
            f"{value:,}",
            (value, y),
            va="center",
            fontsize=9,
            xytext=(4, 0),
            textcoords="offset points",
        )
    ax.set_xlabel("Number of lesions")
    ax.set_xlim(0, counts.max() * 1.15)
    ax.set_title("Anatomic site")

    ax = axes[1]
    ax.barh(
        rate.index[::-1],
        rate.values[::-1] * 100,
        color=config.BENIGN_MALIGNANT_COLORS["Malignant"],
        height=0.6,
    )
    for y, value in enumerate(rate.values[::-1]):
        ax.annotate(
            f"{value:.0%}",
            (value * 100, y),
            va="center",
            fontsize=9,
            xytext=(4, 0),
            textcoords="offset points",
        )
    ax.set_xlabel("Malignant share (%)")
    ax.set_xlim(0, 105)
    ax.set_title("Malignancy rate by site")

    fig.suptitle("Body site — sampling and risk are not uniform", fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, save_as)


# --- 5. Skin tone / fairness ----------------------------------------------
def plot_skin_tone(lesion: pd.DataFrame, save_as: str | None = None) -> plt.Figure:
    """Skin tone sampling and malignancy rate — the fairness slice."""
    tone = lesion["skin_tone_class"].dropna().astype(int)
    counts = tone.value_counts().sort_index()
    labels = [config.SKIN_TONE_LABELS.get(i, str(i)) for i in counts.index]

    sub = lesion.dropna(subset=["skin_tone_class"]).copy()
    sub["skin_tone_class"] = sub["skin_tone_class"].astype(int)
    rate = (
        sub.groupby("skin_tone_class")["diagnosis_1"]
        .apply(lambda s: (s == "Malignant").mean())
        .reindex(counts.index)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))

    ax = axes[0]
    ax.bar(labels, counts.values, color=config.ORANGE, width=0.6)
    _bar_labels(ax, counts.values, counts.sum())
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_ylabel("Number of lesions")
    ax.set_xlabel("Skin tone class")
    ax.set_title("Skin tone sampling")

    ax = axes[1]
    ax.bar(
        labels,
        rate.values * 100,
        color=config.BENIGN_MALIGNANT_COLORS["Malignant"],
        width=0.6,
    )
    for patch, value in zip(ax.patches, rate.values):
        ax.annotate(
            f"{value:.0%}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 2),
            textcoords="offset points",
        )
    ax.set_ylim(0, 105)
    ax.set_ylabel("Malignant share (%)")
    ax.set_xlabel("Skin tone class")
    ax.set_title("Malignancy rate by skin tone")

    fig.suptitle(
        "Skin tone — a model trained here sees almost no type I or V skin",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return _save(fig, save_as)


# --- 6. Image properties ---------------------------------------------------
def plot_image_properties(
    stats: pd.DataFrame, save_as: str | None = None
) -> plt.Figure:
    """Real pixel geometry measured from a sample of JPEGs."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    types = list(stats["image_type"].unique())
    palette = {types[0]: config.ORANGE, types[-1]: "#2E6F95"}

    ax = axes[0]
    for t in types:
        s = stats[stats["image_type"] == t]
        ax.scatter(s["width"], s["height"], s=18, alpha=0.55, label=t, color=palette[t])
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Image dimensions")
    ax.legend(fontsize=8.5)

    ax = axes[1]
    ax.hist(
        [stats.loc[stats["image_type"] == t, "megapixels"] for t in types],
        bins=20,
        color=[palette[t] for t in types],
        label=types,
        stacked=True,
    )
    ax.set_xlabel("Megapixels")
    ax.set_ylabel("Images")
    ax.set_title("Resolution")
    ax.legend(fontsize=8.5)

    ax = axes[2]
    grouped = [stats.loc[stats["image_type"] == t, "file_size_kb"] for t in types]
    bp = ax.boxplot(grouped, patch_artist=True, widths=0.5, tick_labels=types)
    for patch, t in zip(bp["boxes"], types):
        patch.set_facecolor(palette[t])
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("white")
        median.set_linewidth(1.8)
    ax.set_ylabel("File size (KB)")
    ax.set_title("File size")
    ax.tick_params(axis="x", labelsize=8.5)

    fig.suptitle(
        f"Image properties — measured on a {len(stats)}-image random sample",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return _save(fig, save_as)


def plot_color_statistics(
    stats: pd.DataFrame, save_as: str | None = None
) -> plt.Figure:
    """Dermoscopic and clinical images have genuinely different pixel statistics."""
    types = list(stats["image_type"].unique())
    channels = ["mean_r", "mean_g", "mean_b"]
    channel_names = ["Red", "Green", "Blue"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))

    ax = axes[0]
    x = np.arange(len(channels))
    width = 0.35
    for i, t in enumerate(types):
        s = stats[stats["image_type"] == t]
        means = [s[c].mean() for c in channels]
        errs = [s[c].std() for c in channels]
        ax.bar(
            x + (i - 0.5) * width,
            means,
            width,
            yerr=errs,
            capsize=3,
            label=t,
            color=[config.ORANGE, "#2E6F95"][i],
        )
    ax.set_xticks(x, channel_names)
    ax.set_ylabel("Mean channel intensity (0-255)")
    ax.set_title("Colour balance by image type")
    ax.legend(fontsize=8.5)

    ax = axes[1]
    for i, t in enumerate(types):
        s = stats[stats["image_type"] == t]
        ax.scatter(
            s["mean_intensity"],
            s["std_intensity"],
            s=18,
            alpha=0.55,
            label=t,
            color=[config.ORANGE, "#2E6F95"][i],
        )
    ax.set_xlabel("Mean intensity")
    ax.set_ylabel("Intensity std. dev. (contrast)")
    ax.set_title("Brightness vs contrast")
    ax.legend(fontsize=8.5)

    fig.suptitle(
        "Two imaging modalities, two pixel distributions", fontweight="bold", y=1.02
    )
    fig.tight_layout()
    return _save(fig, save_as)


# --- 7. Actual images ------------------------------------------------------
def show_samples(
    df: pd.DataFrame,
    n: int = 6,
    diagnosis: str | None = None,
    image_type: str | None = None,
    seed: int = 0,
    save_as: str | None = None,
) -> plt.Figure:
    """The reusable image-grid viewer used throughout the course."""
    subset = df
    if diagnosis is not None:
        subset = subset[subset["diagnosis_1"] == diagnosis]
    if image_type is not None:
        subset = subset[subset["image_type"] == image_type]

    sample = subset.sample(n=min(n, len(subset)), random_state=seed)
    fig, axes = plt.subplots(1, len(sample), figsize=(2.4 * len(sample), 3))
    axes = np.atleast_1d(axes)

    for ax, (_, row) in zip(axes, sample.iterrows()):
        ax.imshow(data.load_image(row["isic_id"]))
        ax.set_title(f"{row['diagnosis_1']}\n{row.get('label_11', '')}", fontsize=8)
        ax.axis("off")
        ax.grid(False)

    title = diagnosis or "all classes"
    if image_type:
        title += f" · {image_type}"
    fig.suptitle(f"MILK10k samples — {title}", fontweight="bold")
    fig.tight_layout()
    return _save(fig, save_as)


def plot_class_gallery(
    df: pd.DataFrame, seed: int = 1, save_as: str | None = None
) -> plt.Figure:
    """One dermoscopic example per 11-class code, ordered by frequency."""
    derm = df[df["image_type"] == "dermoscopic"]
    codes = derm["label_11"].value_counts().index.tolist()

    n_cols = 6
    n_rows = int(np.ceil(len(codes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.3 * n_cols, 2.75 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, code in zip(axes, codes):
        row = derm[derm["label_11"] == code].sample(1, random_state=seed).iloc[0]
        ax.imshow(data.load_image(row["isic_id"]))
        color = config.BENIGN_MALIGNANT_COLORS[config.code_group(code)]
        n = (derm["label_11"] == code).sum()
        ax.set_title(f"{code}  (n={n:,})", fontsize=9, color=color)
        ax.axis("off")
        ax.grid(False)

    for ax in axes[len(codes):]:
        ax.axis("off")
        ax.grid(False)

    fig.suptitle(
        "One dermoscopic example per diagnosis class\n"
        "(red = malignant, teal = benign, amber = indeterminate/mixed)",
        fontweight="bold",
        y=1.0,
    )
    fig.tight_layout()
    return _save(fig, save_as)


def plot_lesion_pairs(
    df: pd.DataFrame, n_lesions: int = 3, seed: int = 3, save_as: str | None = None
) -> plt.Figure:
    """The same lesion photographed twice — why splits must group by lesion_id."""
    lesions = (
        df["lesion_id"].drop_duplicates().sample(n=n_lesions, random_state=seed).tolist()
    )

    fig, axes = plt.subplots(n_lesions, 2, figsize=(5.6, 2.9 * n_lesions))
    axes = np.atleast_2d(axes)

    for r, lesion_id in enumerate(lesions):
        pair = df[df["lesion_id"] == lesion_id].sort_values("image_type")
        for ax, (_, row) in zip(axes[r], pair.iterrows()):
            ax.imshow(data.load_image(row["isic_id"]))
            ax.set_title(f"{row['image_type']}", fontsize=9)
            ax.axis("off")
            ax.grid(False)
        axes[r][0].set_ylabel(lesion_id)
        axes[r][0].text(
            -0.06,
            0.5,
            f"{lesion_id}\n{pair.iloc[0]['diagnosis_1']} · {pair.iloc[0]['label_11']}",
            transform=axes[r][0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=8,
        )

    fig.suptitle(
        "Two images, one lesion — group by lesion_id when splitting",
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, save_as)


# --- 8. MONET concept scores ----------------------------------------------
def plot_monet_by_image_type(
    df: pd.DataFrame, save_as: str | None = None
) -> plt.Figure:
    """MONET semantic-concept scores split by imaging modality."""
    monet_cols = [c for c in df.columns if c.startswith("MONET_")]
    types = list(df["image_type"].unique())

    fig, axes = plt.subplots(
        2, int(np.ceil(len(monet_cols) / 2)), figsize=(13, 6.4), sharex=False
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, monet_cols):
        for i, t in enumerate(types):
            values = df.loc[df["image_type"] == t, col].dropna()
            ax.hist(
                values,
                bins=30,
                alpha=0.6,
                label=t,
                color=[config.ORANGE, "#2E6F95"][i],
            )
        ax.set_title(col.replace("MONET_", "").replace("_", " "), fontsize=9.5)
        ax.tick_params(labelsize=8)

    for ax in axes[len(monet_cols):]:
        ax.axis("off")
        ax.grid(False)

    axes[0].legend(fontsize=8)
    fig.suptitle(
        "MONET concept scores by image type", fontweight="bold", y=1.01
    )
    fig.tight_layout()
    return _save(fig, save_as)


def plot_monet_by_class(df: pd.DataFrame, save_as: str | None = None) -> plt.Figure:
    """Mean MONET score per diagnosis class — a heatmap of visual concepts."""
    monet_cols = [c for c in df.columns if c.startswith("MONET_")]
    derm = df[df["image_type"] == "dermoscopic"]

    order = derm["label_11"].value_counts().index.tolist()
    matrix = derm.groupby("label_11")[monet_cols].mean().reindex(order)

    # z-score each concept so colours compare classes, not concept scales
    z = (matrix - matrix.mean()) / matrix.std()

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    im = ax.imshow(z.values, cmap="RdBu_r", aspect="auto", vmin=-2.2, vmax=2.2)
    ax.set_xticks(
        range(len(monet_cols)),
        [c.replace("MONET_", "").replace("_", " ") for c in monet_cols],
        rotation=35,
        ha="right",
        fontsize=8.5,
    )
    ax.set_yticks(range(len(order)), order, fontsize=9)
    ax.grid(False)

    for i in range(len(order)):
        for j in range(len(monet_cols)):
            ax.text(
                j,
                i,
                f"{matrix.values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black" if abs(z.values[i, j]) < 1.3 else "white",
            )

    fig.colorbar(im, ax=ax, label="z-score within concept", shrink=0.8)
    ax.set_title(
        "Mean MONET concept score per class (dermoscopic images)\n"
        "cell text = raw mean, colour = z-score across classes",
        fontsize=11,
    )
    fig.tight_layout()
    return _save(fig, save_as)


def plot_diagnosis_hierarchy(
    lesion: pd.DataFrame, save_as: str | None = None
) -> plt.Figure:
    """How the 3-class target maps onto the 11-class codes."""
    ct = pd.crosstab(lesion["label_11"], lesion["diagnosis_1"])
    ct = ct.reindex(lesion["label_11"].value_counts().index)
    cols = [c for c in config.PRIMARY_CLASSES if c in ct.columns]
    ct = ct[cols]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bottom = np.zeros(len(ct))
    for col in cols:
        ax.bar(
            ct.index,
            ct[col].values,
            bottom=bottom,
            label=col,
            color=config.BENIGN_MALIGNANT_COLORS[col],
            width=0.65,
        )
        bottom += ct[col].values

    ax.set_ylabel("Number of lesions")
    ax.set_yscale("symlog")
    ax.legend(title="diagnosis_1")
    ax.set_title("11-class code vs. 3-class target — the labels are nested, not independent")
    fig.tight_layout()
    return _save(fig, save_as)


# ===========================================================================
# Milestone 1 figures (B2, B7, B8)
# ===========================================================================
# Everything above is the Session 1 EDA. The figures below are the ones the
# Milestone 1 brief names, and they live in this module so there is still one
# place that knows the house style. They write into a sub-folder of
# reports/figures/ so the two sets can never overwrite each other, and they read
# the COMMITTED artifacts (the split CSVs, the label map) rather than rebuilding
# them — a figure that re-derives the split is a figure that can disagree with
# the pipeline it is supposed to document.

SPLIT_ORDER = ("train", "val", "test")


def _milestone1(save_as: str | None) -> str | None:
    """Route a Milestone 1 filename into ``config.MILESTONE1_FIGURES_DIR``.

    ``_save`` writes relative to ``config.FIGURES_DIR``, so a bare filename
    would land next to the Session 1 figures. Prefixing the sub-folder here,
    rather than changing ``_save``, keeps a single writer for every figure in
    the project. ``relative_to`` derives the prefix from config instead of
    spelling it out, so moving the folder cannot silently split the output in
    two — it raises.
    """
    if save_as is None:
        return None
    config.MILESTONE1_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return str(config.MILESTONE1_FIGURES_DIR.relative_to(config.FIGURES_DIR) / save_as)


def _label_color(name: str) -> str:
    """Colour for a class name in either labelling scheme.

    ``diagnosis_1`` values already *are* the three groups; an 11-class code is
    routed through ``config.code_group``. Both schemes therefore land on the
    same benign / indeterminate / malignant colour axis, so a reader only has
    to learn one legend for the whole report.
    """
    if name in config.BENIGN_MALIGNANT_COLORS:
        return config.BENIGN_MALIGNANT_COLORS[name]
    return config.BENIGN_MALIGNANT_COLORS[config.code_group(name)]


def _group_handles() -> tuple[list, list[str]]:
    """Legend handles and labels for the three diagnosis_1 groups."""
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=config.BENIGN_MALIGNANT_COLORS[g])
        for g in config.PRIMARY_LABELS
    ]
    return handles, list(config.PRIMARY_LABELS)


def _blank(ax: plt.Axes) -> plt.Axes:
    """Turn an axes into a plain drawing surface for one image tile.

    The anchor is the non-obvious part. ``imshow`` keeps the pixel aspect
    ratio, so an axes holding a 4:3 photograph shrinks vertically inside the
    cell the grid gave it and, by default, centres what is left. Its title then
    floats down with it, and in a row that mixes 4:3 originals with 224x224
    crops the column headers end up on two different heights. Anchoring north
    pins every tile to the top of its cell, which lines the headers up and puts
    each title directly above the image it names.
    """
    ax.axis("off")
    ax.grid(False)
    ax.set_anchor("N")
    return ax


def _split_lesions(
    split_tables: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """One row per lesion across the three committed splits.

    The split CSVs are image level — two rows per lesion — so counting classes
    straight out of them would double every number and describe a dataset that
    does not exist. The modelling unit is the lesion, so de-duplicate on
    ``lesion_id`` first. Columns: lesion_id, diagnosis_1, dx, split.
    """
    from . import splits as splits_module

    if split_tables is None:
        split_tables = {name: splits_module.load_split(name) for name in SPLIT_ORDER}

    frames = []
    for name, table in split_tables.items():
        lesions = table.drop_duplicates(subset="lesion_id").copy()
        lesions["split"] = name
        frames.append(lesions[["lesion_id", "diagnosis_1", "dx", "split"]])

    out = pd.concat(frames, ignore_index=True)
    assert out["lesion_id"].is_unique, (
        "a lesion_id appears in more than one split — the splits leak"
    )
    return out


def _one_image_per_class(
    lesions: pd.DataFrame,
    view: str = "derm_id",
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Pick one lesion per 11-class code, most common class first.

    Fixed seed, so the gallery in the report is the same gallery the marker
    reproduces. ``view`` selects which of the lesion's two photographs is used;
    dermoscopic is the default because it is the modality the diagnostic
    criteria (pigment network, vessel pattern) are actually defined on.
    """
    counts = lesions["dx"].value_counts()
    rows = [
        lesions[lesions["dx"] == code].sample(1, random_state=seed).iloc[0]
        for code in counts.index
    ]
    picked = pd.DataFrame(rows).reset_index(drop=True)
    picked["n_lesions"] = picked["dx"].map(counts)
    picked["isic_id"] = picked[view]
    return picked


# --- B2. Class distribution in both schemes --------------------------------
def plot_split_class_distribution(
    split_tables: dict[str, pd.DataFrame] | None = None,
    save_as: str | None = None,
) -> plt.Figure:
    """Both label schemes, counted on lesions, with the split breakdown (B2).

    Four panels because four separate questions have to be answered before the
    modelling choices in B8 make sense:

    * top-left — how imbalanced is the 3-class target we are actually graded on;
    * top-right — did stratifying the split preserve those proportions, or did
      the rare classes drift into one fold;
    * bottom-left — the 11-class target on a linear axis, which is the honest
      picture of why a linear axis is useless here: the rarest code is a single
      pixel next to BCC;
    * bottom-right — the same counts on a log axis, which is the readable one.
      The brief asks for the log version explicitly, and this is why.

    Counts are printed on every bar, because a log axis lets a reader see the
    ordering but not the magnitude.
    """
    lesions = _split_lesions(split_tables)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # --- Top-left: the primary target ---------------------------------------
    ax = axes[0, 0]
    primary = lesions["diagnosis_1"].value_counts().reindex(config.PRIMARY_LABELS)
    ax.bar(
        primary.index,
        primary.to_numpy(),
        color=[_label_color(c) for c in primary.index],
        width=0.6,
    )
    _bar_labels(ax, primary.to_numpy(), int(primary.sum()))
    ax.set_ylim(0, primary.max() * 1.18)
    ax.set_ylabel("Number of lesions")
    ax.set_title(
        f"Primary target — diagnosis_1  ({primary.sum():,} lesions, "
        f"{primary.max() / primary.min():.1f} : 1)"
    )

    # --- Top-right: did the stratification hold? ----------------------------
    # Shares rather than counts: train is ~4.7x the size of val, so counts would
    # only show that train is bigger, which nobody needs a figure for.
    ax = axes[0, 1]
    share = pd.crosstab(
        lesions["split"], lesions["diagnosis_1"], normalize="index"
    ).reindex(index=list(SPLIT_ORDER), columns=config.PRIMARY_LABELS) * 100
    overall = primary / primary.sum() * 100

    x = np.arange(len(SPLIT_ORDER))
    width = 0.26
    for i, cls in enumerate(config.PRIMARY_LABELS):
        offset = (i - (len(config.PRIMARY_LABELS) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            share[cls].to_numpy(),
            width,
            color=_label_color(cls),
            label=cls,
        )
        for patch, value in zip(bars, share[cls].to_numpy()):
            ax.annotate(
                f"{value:.1f}",
                (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )
    # One dashed line per class at the full-dataset share: any bar that does not
    # touch its line is a fold that drifted.
    for cls in config.PRIMARY_LABELS:
        ax.axhline(overall[cls], color=config.GREY, linestyle=":", linewidth=0.9)

    max_dev = float((share - overall).abs().to_numpy().max())
    ax.set_xticks(x, [f"{n}\n(n={int((lesions['split'] == n).sum()):,})"
                      for n in SPLIT_ORDER])
    ax.set_ylabel("Share of the split's lesions (%)")
    # Headroom for the legend and the note, so neither lands on a bar.
    ax.set_ylim(0, share.to_numpy().max() * 1.55)
    ax.legend(fontsize=8.5, ncol=3, loc="upper center")
    ax.set_title("Stratification check — class share per split")
    ax.text(
        0.5,
        0.84,
        f"dotted = full-dataset share  ·  largest deviation {max_dev:.2f} pp",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        style="italic",
    )

    # --- Bottom row: the 11-class stretch target ----------------------------
    stretch = lesions["dx"].value_counts()
    colors = [_label_color(code) for code in stretch.index]
    handles, group_labels = _group_handles()

    for ax, log in ((axes[1, 0], False), (axes[1, 1], True)):
        bars = ax.bar(stretch.index, stretch.to_numpy(), color=colors, width=0.65)
        for patch, value in zip(bars, stretch.to_numpy()):
            ax.annotate(
                f"{value:,}",
                (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2.5),
                textcoords="offset points",
            )
        ax.tick_params(axis="x", labelsize=8.5, rotation=45)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        if log:
            ax.set_yscale("log")
            # Headroom for the count labels: on a log axis a linear 1.18x margin
            # is invisible, so extend the limit multiplicatively instead.
            ax.set_ylim(1, stretch.max() * 4)
            ax.set_ylabel("Number of lesions (log scale)")
            ax.legend(handles, group_labels, fontsize=8.5, loc="upper right")
            ax.set_title(
                f"Stretch target — 11 codes, log scale  "
                f"({stretch.max() / stretch.min():.0f} : 1)"
            )
        else:
            ax.set_ylim(0, stretch.max() * 1.15)
            ax.set_ylabel("Number of lesions")
            ax.set_title("Stretch target — 11 codes, linear scale")
            rarest, n_rarest = stretch.index[-1], int(stretch.iloc[-1])
            ax.annotate(
                f"{rarest} = {n_rarest} lesions\n(one pixel tall next to BCC)",
                xy=(len(stretch) - 1, n_rarest),
                xytext=(len(stretch) - 3.6, stretch.max() * 0.42),
                arrowprops=dict(arrowstyle="->", color=config.GREY, linewidth=1.1),
                fontsize=8.5,
                ha="center",
            )

    fig.suptitle(
        "B2 — class distribution in both label schemes (lesion level, committed splits)",
        fontweight="bold",
        y=1.0,
    )
    fig.tight_layout()
    return _save(fig, _milestone1(save_as))


# --- B2 / B3. How the two schemes relate -----------------------------------
def plot_label_mapping(
    table: pd.DataFrame | None = None, save_as: str | None = None
) -> plt.Figure:
    """The 11-class codes crossed with diagnosis_1 — where the schemes disagree.

    The two targets are *nearly* nested, and the exception is the whole point:
    if every code mapped to exactly one ``diagnosis_1`` value, an 11-class model
    would give the 3-class answer for free by lookup. One code does not, so it
    does not, and the figure has to make that single row impossible to miss
    rather than leave it in a caption.

    Cell shading is logarithmic. The counts run from 2,522 down to 9, so a
    linear ramp would render nine of the eleven rows as the same empty white.
    """
    from . import labels as labels_module

    table = labels_module.code_to_primary_table() if table is None else table
    order = table["dx"].tolist()                      # already sorted by size
    counts = table.set_index("dx")[list(config.PRIMARY_LABELS)].to_numpy(dtype=float)
    totals = table["n_lesions"].to_numpy()

    # --- Colour: hue says which primary class, alpha says how many ----------
    scale = np.log10(counts + 1.0) / np.log10(counts.max() + 1.0)
    rgba = np.empty(counts.shape + (4,))
    for j, cls in enumerate(config.PRIMARY_LABELS):
        base = to_rgba(config.BENIGN_MALIGNANT_COLORS[cls])[:3]
        for i in range(counts.shape[0]):
            rgba[i, j] = (
                to_rgba("#F4F4F4")
                if counts[i, j] == 0
                else (*base, 0.16 + 0.84 * scale[i, j])
            )

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.imshow(rgba, aspect="auto")
    ax.grid(False)

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            value = int(counts[i, j])
            ax.text(
                j,
                i,
                f"{value:,}" if value else "—",
                ha="center",
                va="center",
                fontsize=10 if value else 9,
                fontweight="bold" if value else "normal",
                # White on a saturated cell, grey on a pale one: the alpha above
                # is exactly the number that decides which is legible.
                color="white" if (value and scale[i, j] > 0.70) else config.GREY,
            )

    ax.set_xticks(range(len(config.PRIMARY_LABELS)), config.PRIMARY_LABELS, fontsize=10)
    for tick, cls in zip(ax.get_xticklabels(), config.PRIMARY_LABELS):
        tick.set_color(_label_color(cls))
        tick.set_fontweight("bold")

    ax.set_yticks(
        range(len(order)),
        [
            f"{code} · {config.DIAGNOSIS_CODES[code]}  (n={total:,})"
            for code, total in zip(order, totals)
        ],
        fontsize=8.5,
    )
    for tick, code in zip(ax.get_yticklabels(), order):
        tick.set_color(_label_color(code))

    ax.set_xlabel("diagnosis_1 — the primary 3-class target")
    ax.set_title("Stretch code (11) x primary target (3) — lesion counts")

    # --- The one row that splits -------------------------------------------
    ambiguous = table.loc[table["ambiguous"], "dx"].tolist()
    for code in ambiguous:
        row = order.index(code)
        ax.add_patch(
            plt.Rectangle(
                (-0.5, row - 0.5),
                len(config.PRIMARY_LABELS),
                1,
                fill=False,
                edgecolor=config.BENIGN_MALIGNANT_COLORS["Indeterminate"],
                linewidth=2.6,
            )
        )
        ax.get_yticklabels()[row].set_fontweight("bold")
        ax.annotate(
            f"{code} splits",
            xy=(len(config.PRIMARY_LABELS) - 0.4, row),
            xytext=(len(config.PRIMARY_LABELS) + 0.15, row),
            va="center",
            fontsize=9,
            fontweight="bold",
            color=config.BENIGN_MALIGNANT_COLORS["Indeterminate"],
            annotation_clip=False,
        )

    if len(ambiguous) == 1:
        code = ambiguous[0]
        row = table.set_index("dx").loc[code]
        parts = ", ".join(
            f"{int(row[cls]):,} {cls}" for cls in config.PRIMARY_LABELS if row[cls]
        )
        caption = (
            f"{code} is the only code that straddles the primary target "
            f"({parts}). Every other code maps 1:1, so an 11-class prediction "
            "cannot be collapsed to diagnosis_1 by table lookup — the two "
            "targets are trained and reported separately."
        )
    else:
        caption = (
            f"{len(ambiguous)} codes map to more than one diagnosis_1 value: "
            f"{', '.join(ambiguous) or 'none'}."
        )
    ax.text(
        0.5,
        -0.16,
        textwrap.fill(caption, 118),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        style="italic",
    )

    fig.tight_layout()
    return _save(fig, _milestone1(save_as))


# --- B2. The 3x4 class gallery ---------------------------------------------
def plot_class_gallery_grid(
    lesions: pd.DataFrame | None = None,
    seed: int = config.SEED,
    save_as: str | None = None,
) -> plt.Figure:
    """One dermoscopic example per class in a 3x4 grid, with names and counts.

    Eleven classes in twelve cells: the spare one carries the legend rather
    than sitting blank, so the figure explains its own colour coding without a
    caption the reader has to look for somewhere else.

    Each title states the code, the full diagnosis name and the lesion count,
    because "BCC" is not a defence — knowing that it is basal cell carcinoma
    and that it is 48% of the dataset is.
    """
    lesions = data.build_lesion_table() if lesions is None else lesions
    picked = _one_image_per_class(lesions, view="derm_id", seed=seed)

    n_cols, n_rows = 4, 3
    # Cell height is set just above the 4:3 photograph plus its 2-3 line title;
    # a taller cell would open a band of white between every row.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.9 * n_cols, 2.9 * n_rows))
    flat = axes.ravel()

    for ax, (_, row) in zip(flat, picked.iterrows()):
        _blank(ax)
        ax.imshow(data.load_image(row["isic_id"]))
        code = row["dx"]
        ax.set_title(
            f"{code}  (n={int(row['n_lesions']):,})\n"
            + textwrap.fill(config.DIAGNOSIS_CODES[code], 28),
            fontsize=9,
            color=_label_color(code),
        )

    # --- The spare cell: legend + provenance --------------------------------
    legend_ax = _blank(flat[len(picked)])
    legend_ax.text(
        0.5, 0.92, "Title colour = diagnosis_1 group",
        ha="center", va="top", fontsize=9.5, fontweight="bold",
    )
    for i, group in enumerate(config.PRIMARY_LABELS):
        legend_ax.text(
            0.5,
            0.76 - 0.10 * i,
            group,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=config.BENIGN_MALIGNANT_COLORS[group],
        )
    legend_ax.text(
        0.5,
        0.40,
        textwrap.fill(
            f"One dermoscopic view per class, drawn with seed {seed}. "
            f"n counts LESIONS, not images — every lesion also has a clinical "
            f"close-up, so the dataset holds exactly 2n images.",
            32,
        ),
        ha="center",
        va="top",
        fontsize=8.5,
        style="italic",
    )

    for ax in flat[len(picked) + 1:]:
        _blank(ax)

    fig.suptitle(
        "B2 — one example per diagnosis class (dermoscopic view)",
        fontweight="bold",
        y=1.0,
    )
    fig.tight_layout()
    return _save(fig, _milestone1(save_as))


# --- B7. What the augmentations actually do --------------------------------
def plot_augmentation_examples(
    lesions: pd.DataFrame | None = None,
    codes: tuple[str, ...] = ("BCC", "MEL", "VASC"),
    n_augmented: int = 7,
    seed: int = config.SEED,
    save_as: str | None = None,
) -> plt.Figure:
    """One original plus ``n_augmented`` augmented views, for three classes (B7).

    The classes are chosen rather than sampled: the commonest code, the one
    that matters most clinically, and a rare one. The rare row is the one worth
    staring at — a class with 47 lesions is the class most dependent on
    augmentation, and also the class an over-aggressive crop would destroy
    fastest.

    Only the random half of the pipeline is applied, via
    ``transforms.augmentation_grid``, because after ``Normalize`` the pixels are
    signed floats and no longer displayable. That is the same object the
    training pipeline composes, so this figure cannot drift from what the
    network is fed.
    """
    from . import transforms as milk_transforms

    lesions = data.build_lesion_table() if lesions is None else lesions
    counts = lesions["dx"].value_counts()

    missing = [c for c in codes if c not in set(lesions["dx"])]
    if missing:
        raise ValueError(f"no lesions with dx in {missing}; available: {list(counts.index)}")

    n_cols = n_augmented + 1
    # Row height a little over the tile width: the tiles are square, so a
    # taller row would only add a band of white under every one of them.
    fig, axes = plt.subplots(
        len(codes), n_cols, figsize=(1.75 * n_cols, 2.05 * len(codes))
    )
    axes = np.atleast_2d(axes)

    for r, code in enumerate(codes):
        row = lesions[lesions["dx"] == code].sample(1, random_state=seed).iloc[0]
        original = data.load_image(row["derm_id"])
        views = milk_transforms.augmentation_grid(original, n=n_augmented, seed=seed)

        for c, (ax, (_, image)) in enumerate(zip(axes[r], views)):
            _blank(ax)
            ax.imshow(image)
            if r == 0:
                ax.set_title("original" if c == 0 else f"aug {c}", fontsize=9)

        axes[r][0].text(
            -0.12,
            0.5,
            f"{code}\nn={int(counts[code]):,}",
            transform=axes[r][0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
            color=_label_color(code),
        )

    fig.suptitle(
        "B7 — the training augmentations, applied to three classes",
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        textwrap.fill(
            "Column 1 is the native 600x450 frame; columns 2-8 are the 224x224 "
            "crops the network actually sees, drawn from train_transform with "
            f"seed {seed}. Rotation, crop and both flips are label-preserving "
            "in dermoscopy — skin has no canonical up — while the colour jitter "
            "is deliberately weak, because pigmentation is itself the diagnosis.",
            128,
        ),
        ha="center",
        va="top",
        fontsize=8.5,
        style="italic",
    )
    fig.tight_layout()
    return _save(fig, _milestone1(save_as))


# --- B8. A real batch, un-normalised ---------------------------------------
def plot_transformed_batch(
    n_images: int = 16,
    scheme: str = "primary",
    mean: tuple[float, float, float] = config.IMAGENET_MEAN,
    std: tuple[float, float, float] = config.IMAGENET_STD,
    seed: int = config.SEED,
    save_as: str | None = None,
) -> plt.Figure:
    """One real batch straight out of the train DataLoader, made viewable (B8).

    This is the end-to-end check: CSV row -> file path -> JPEG -> augmentation
    -> normalised tensor -> batch, with the integer label the loss function will
    receive printed next to the class name it stands for. If the wiring were
    wrong — a mismatched label map, a shuffled id column — it shows up here as
    an obviously mislabelled lesion, and nowhere else until the confusion matrix.

    The batch is pulled through ``datasets.build_dataloaders`` rather than
    rebuilt, so what is drawn is the object the training loop iterates.
    """
    from . import datasets, labels as labels_module
    import torch

    loaders = datasets.build_dataloaders(
        batch_size=n_images, scheme=scheme, seed=seed
    )
    images, targets, isic_ids = next(iter(loaders["train"]))
    inverse = labels_module.inverse_label_map(scheme)

    # --- Un-normalise, or every tile is noise -------------------------------
    # The loader returns (pixel - mean) / std, i.e. values roughly in
    # [-2.1, 2.7]. imshow clips float input to [0, 1], so drawing the tensor as
    # it comes out would flatten most of the range to pure black or white and
    # the figure would prove nothing. Inverting the normalisation exactly —
    # multiply by std, add the mean — returns the original [0, 1] pixels.
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)
    restored = images * std_t + mean_t

    # Bilinear resampling and the colour jitter can push a pixel a hair outside
    # [0, 1]; clamp those rather than let imshow's own clipping do it silently,
    # and report how many so the claim "this is the real image back" is checked
    # rather than asserted.
    n_outside = int(((restored < 0.0) | (restored > 1.0)).sum())
    viewable = restored.clamp(0.0, 1.0).permute(0, 2, 3, 1).numpy()  # NCHW -> NHWC

    n_cols = int(np.ceil(np.sqrt(len(viewable))))
    n_rows = int(np.ceil(len(viewable) / n_cols))
    # Cells are taller than wide on purpose. The tiles are square, so the extra
    # height becomes a gutter under each one (they are anchored north), which is
    # what keeps a two-line title attached to the image below it rather than
    # touching the image above.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.7 * n_cols, 3.3 * n_rows))
    flat = np.atleast_1d(axes).ravel()

    for ax, image, target, isic_id in zip(flat, viewable, targets, isic_ids):
        _blank(ax)
        ax.imshow(image)
        name = inverse[int(target)]
        ax.set_title(
            f"{int(target)} · {name}\n{isic_id}",
            fontsize=8.5,
            color=_label_color(name),
        )

    for ax in flat[len(viewable):]:
        _blank(ax)

    counts = pd.Series([inverse[int(t)] for t in targets]).value_counts()
    fig.suptitle(
        f"B8 — one training batch after the transforms "
        f"({len(viewable)} images, {tuple(images.shape[1:])} each)",
        fontweight="bold",
        y=1.0,
    )
    fig.text(
        0.5,
        -0.01,
        textwrap.fill(
            "Titles are the integer label the loss function receives, then the "
            f"class it decodes to. This batch: "
            + ", ".join(f"{n}x {c}" for c, n in counts.items())
            + ". Tensors were un-normalised for display (x * std + mean, then "
            "clipped to [0, 1]); "
            + (
                "not one of the "
                f"{restored.numel():,} pixel values needed clipping"
                if n_outside == 0
                else f"{n_outside:,} of {restored.numel():,} pixel values "
                "needed clipping"
            )
            + ", so what is shown is the input the network sees.",
            126,
        ),
        ha="center",
        va="top",
        fontsize=8.5,
        style="italic",
    )
    fig.tight_layout()
    # After tight_layout, because it sets the spacing itself: a sliver of
    # horizontal gutter so neighbouring lesions do not read as one photograph.
    fig.subplots_adjust(wspace=0.05)
    return _save(fig, _milestone1(save_as))
