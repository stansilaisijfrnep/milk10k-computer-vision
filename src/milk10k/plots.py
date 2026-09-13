"""Every figure in the MILK10k EDA report.

Each function takes the already-loaded dataframes, draws one figure and
returns the matplotlib ``Figure`` so it renders inline in a notebook. Passing
``save_as`` also writes it to ``reports/figures/``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
