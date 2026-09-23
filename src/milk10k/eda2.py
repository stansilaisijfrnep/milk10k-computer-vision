"""Session 2 homework, Parts 1-2: metadata-target association and dataset colour analysis.

Two rules shape every function here.

**The unit of analysis is the lesion.** ``metadata.csv`` has 10,480 rows but only
5,240 lesions: every lesion is photographed twice (dermoscopic + clinical
close-up) and every metadata field except ``image_type`` and
``image_manipulation`` is identical on both rows. Testing on image rows counts
each lesion twice, which doubles every chi-square statistic and makes weak
associations look twice as convincing. So lesion-constant fields are tested on
the 5,240 lesions; the two fields that genuinely vary per image are tested on
images, and the table says which unit each row used.

**Effect size ranks, the p-value does not.** With n = 5,240 almost any difference
is "significant". Cramer's V (categorical) and the correlation ratio eta
(numeric) are on the same 0-1 scale, so fields can be compared directly.

Part 2 applies the same thinking to pixels: per-class colour statistics are
computed *per imaging modality*. Dermoscopic and clinical images differ strongly
in the blue channel, and the classes separate far better in clinical photos than
in dermoscopy, so pooling the two modalities mixes two different pictures and
understates the class differences in both.

Computation functions return DataFrames; plotting functions return
``plt.Figure`` and take ``save_as`` (written to ``reports/figures/``).
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import config, data, preprocessing
from .plots import _save

TARGET = config.PRIMARY_LABEL_COL
CLASSES = list(config.PRIMARY_LABELS)  # Benign, Indeterminate, Malignant
MISSING = "(missing)"
MODALITIES = ["dermoscopic", "clinical: close-up"]

# --- Part 1, task 4: what each column is, and whether a model may use it ---
# Written as data, not prose, so the report table and the ranking plot use the
# same classification. Every claim here is checked in run_session2.py.
FIELD_ROLES: dict[str, tuple[str, str]] = {
    "age_approx": ("usable", "patient age at capture; known before any biopsy"),
    "sex": ("usable", "patient sex; known before any biopsy"),
    "anatom_site_general": ("usable", "body site; 37% missing, so the gap itself needs care"),
    "anatom_site_special": ("usable", "98% missing; the few values are acral / oral-genital"),
    "diagnosis_2": ("leakage", "finer level of the diagnosis hierarchy — contains the label"),
    "diagnosis_3": ("leakage", "finer level of the diagnosis hierarchy — contains the label"),
    "diagnosis_4": ("leakage", "finer level of the diagnosis hierarchy — contains the label"),
    "diagnosis_confirm_type": ("leakage", "biopsied or not — decided because of clinical suspicion"),
    "concomitant_biopsy": ("leakage", "identical to diagnosis_confirm_type (perfect 1:1 cross-tab)"),
    "melanocytic": ("leakage", "only recorded for melanocytic lesions — its presence names the family"),
    "image_manipulation": ("acquisition", "how the photo was taken/edited — a shortcut, not biology"),
    "image_type": ("acquisition", "imaging modality; every lesion has one of each"),
    "attribution": ("constant", "one value for all rows"),
    "copyright_license": ("constant", "one value for all rows"),
    "lesion_id": ("grouping", "identifier of the lesion — used to split, never as an input"),
}
ROLE_COLORS = {
    "usable": "#2E8B84",
    "leakage": "#C1272D",
    "acquisition": config.ORANGE,
    "constant": "#BFBFBF",
    "grouping": "#BFBFBF",
}


# ===========================================================================
# Part 1 — metadata vs target
# ===========================================================================
def lesion_table(meta: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per lesion with every lesion-constant metadata column.

    ``groupby().first()`` skips NaN, which is only safe because each of these
    columns is constant within a lesion — :func:`lesion_constancy` checks that.
    """
    meta = data.load_metadata() if meta is None else meta
    varying = set(lesion_constancy(meta).query("n_lesions_varying > 0")["column"])
    keep = [c for c in meta.columns if c not in varying | {"isic_id", "lesion_id"}]
    return meta.groupby("lesion_id", as_index=False)[keep].first()


def lesion_constancy(meta: pd.DataFrame) -> pd.DataFrame:
    """For each column: in how many lesions do the two images disagree?"""
    rows = []
    for col in meta.columns:
        if col in ("isic_id", "lesion_id"):
            continue
        n = int((meta.groupby("lesion_id")[col].nunique(dropna=False) > 1).sum())
        rows.append({"column": col, "n_lesions_varying": n})
    return pd.DataFrame(rows)


def _field_type(s: pd.Series) -> str:
    values = s.dropna()
    if values.nunique() <= 1:
        return "constant" if s.isna().sum() == 0 else "boolean (present/missing)"
    if pd.api.types.is_bool_dtype(values) or set(values.unique()) <= {True, False}:
        return "boolean"
    if pd.api.types.is_numeric_dtype(values):
        return "numeric"
    if values.nunique() > 50:
        return "identifier" if values.is_unique or values.nunique() > 1000 else "free text"
    return "categorical"


def metadata_inventory(meta: pd.DataFrame | None = None, target: str = TARGET) -> pd.DataFrame:
    """Task 1: every column other than the id and the target — type, missingness, role.

    ``unit`` says where the column lives: ``lesion`` (identical on both images of
    a lesion) or ``image`` (can differ between the two).
    """
    meta = data.load_metadata() if meta is None else meta
    varying = dict(lesion_constancy(meta)[["column", "n_lesions_varying"]].values)
    rows = []
    for col in meta.columns:
        if col in ("isic_id", target):
            continue
        s = meta[col]
        role, reason = FIELD_ROLES.get(col, ("unreviewed", ""))
        rows.append({
            "column": col,
            "type": "identifier" if col == "lesion_id" else _field_type(s),
            "n_unique": int(s.nunique()),
            "n_missing": int(s.isna().sum()),
            "pct_missing": round(100 * s.isna().mean(), 2),
            "unit": "image" if varying.get(col, 0) > 0 else "lesion",
            "role": role,
            "why": reason,
        })
    return pd.DataFrame(rows)


def class_distribution_table(
    df: pd.DataFrame, column: str, target: str = TARGET, missing_as_level: bool = True
) -> pd.DataFrame:
    """Task 2: class mix (row %) within each category of ``column``, plus the row count.

    Missing values become their own ``(missing)`` level by default, because in
    this dataset *whether* a field was recorded is itself informative.
    """
    col = df[column].astype("object")
    if missing_as_level:
        col = col.where(col.notna(), MISSING)
    pct = pd.crosstab(col, df[target], normalize="index").reindex(columns=CLASSES, fill_value=0) * 100
    pct["n"] = col.value_counts().reindex(pct.index).astype(int)
    overall = df[target].value_counts(normalize=True).reindex(CLASSES, fill_value=0) * 100
    pct.loc["(all)"] = [*overall.values, len(df)]
    return pct.round(1).astype({"n": int})


def categorical_association(df: pd.DataFrame, column: str, target: str = TARGET) -> dict:
    """Chi-square test of independence + Cramer's V for a categorical field.

    Why these: chi-square asks whether the class mix depends on the category;
    Cramer's V = sqrt(chi2 / (n (k - 1))) rescales it to 0-1 regardless of n and
    of how many categories the field has, so a 2-level and a 19-level field can
    be ranked on the same axis. Conventional reading: 0.1 small, 0.3 medium,
    0.5 large.

    Reliability: chi-square's p-value assumes expected counts of at least 5 in
    (nearly) every cell. ``pct_cells_expected_lt5`` and ``chi2_reliable``
    (Cochran's rule: <= 20% of cells below 5 and none below 1) flag where that
    fails — here, typically because Indeterminate has only 123 lesions. V is
    still a fair description of the observed table; the p-value is not.
    """
    col = df[column].astype("object")
    col = col.where(col.notna(), MISSING)
    table = pd.crosstab(col, df[target]).reindex(columns=CLASSES, fill_value=0)
    table = table.loc[:, table.sum() > 0]
    n = int(table.values.sum())
    if table.shape[0] < 2:
        return {"field": column, "test": "none (constant)", "n": n, "effect": np.nan,
                "effect_measure": "-", "p_value": np.nan, "levels": int(table.shape[0])}
    chi2, p, dof, expected = stats.chi2_contingency(table.values, correction=False)
    k = min(table.shape) - 1
    return {
        "field": column,
        "test": "chi-square",
        "n": n,
        "levels": int(table.shape[0]),
        "statistic": float(chi2),
        "dof": int(dof),
        "p_value": float(p),
        "effect": float(np.sqrt(chi2 / (n * k))),
        "effect_measure": "Cramer's V",
        "min_expected": float(expected.min()),
        "pct_cells_expected_lt5": round(100 * float((expected < 5).mean()), 1),
        "chi2_reliable": bool((expected < 5).mean() <= 0.2 and expected.min() >= 1),
    }


def numeric_association(df: pd.DataFrame, column: str, target: str = TARGET) -> dict:
    """One-way ANOVA + eta for a numeric field against the 3-class target.

    Why these: the question is whether the field's mean differs between classes,
    which is exactly one-way ANOVA. eta^2 = SS_between / SS_total is the share of
    the field's variance explained by class; its square root, eta (the
    correlation ratio), is on the same 0-1 scale as Cramer's V, so numeric and
    categorical fields sit on one ranking. Age is recorded in 5-year bins and is
    skewed, so the rank-based Kruskal-Wallis test is reported as a
    distribution-free check on the ANOVA conclusion.
    """
    sub = df[[column, target]].dropna()
    groups = [g.to_numpy(float) for _, g in sub.groupby(target)[column]]
    f, p = stats.f_oneway(*groups)
    h, p_kw = stats.kruskal(*groups)
    grand = sub[column].mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = float(((sub[column] - grand) ** 2).sum())
    eta2 = ss_between / ss_total
    return {
        "field": column,
        "test": "one-way ANOVA (+ Kruskal-Wallis)",
        "n": int(len(sub)),
        "levels": len(groups),
        "statistic": float(f),
        "dof": len(groups) - 1,
        "p_value": float(p),
        "effect": float(np.sqrt(eta2)),
        "effect_measure": "eta (sqrt of eta^2)",
        "eta_squared": float(eta2),
        "kruskal_H": float(h),
        "kruskal_p": float(p_kw),
    }


def association_summary(meta: pd.DataFrame | None = None, target: str = TARGET) -> pd.DataFrame:
    """Tasks 2-4: every metadata field tested against the target, ranked by effect size.

    Lesion-constant fields are tested on the lesion table; ``image_type`` and
    ``image_manipulation`` on image rows (``unit`` column). Constant columns and
    the grouping key are listed with no test so nothing silently disappears.
    """
    meta = data.load_metadata() if meta is None else meta
    lesions = lesion_table(meta)
    inv = metadata_inventory(meta, target).set_index("column")
    rows = []
    for col, info in inv.iterrows():
        if info["type"] in ("identifier", "constant"):
            rows.append({"field": col, "test": f"none ({info['type']})", "unit": info["unit"],
                         "effect": np.nan, "role": info["role"]})
            continue
        frame = lesions if info["unit"] == "lesion" else meta
        fn = numeric_association if info["type"] == "numeric" else categorical_association
        res = fn(frame, col, target)
        res.update(unit=info["unit"], role=info["role"])
        rows.append(res)
    out = pd.DataFrame(rows)
    out = out.sort_values("effect", ascending=False, na_position="last").reset_index(drop=True)
    cols = ["field", "role", "unit", "n", "test", "effect_measure", "effect", "statistic", "dof",
            "p_value", "levels", "min_expected", "pct_cells_expected_lt5", "chi2_reliable",
            "eta_squared", "kruskal_H", "kruskal_p"]
    return out[[c for c in cols if c in out.columns]]


def effect_label(v: float) -> str:
    """Conventional wording for V / eta: negligible, small, medium, large."""
    if pd.isna(v):
        return "n/a"
    return "large" if v >= 0.5 else "medium" if v >= 0.3 else "small" if v >= 0.1 else "negligible"


# --- Part 1 figures --------------------------------------------------------
def plot_association_ranking(summary: pd.DataFrame, save_as: str | None = None) -> plt.Figure:
    """Horizontal bars of Cramer's V / eta per field, coloured by role, reference lines at 0.1/0.3/0.5."""
    s = summary.dropna(subset=["effect"]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 0.36 * len(s) + 1.2))
    bars = ax.barh(s["field"], s["effect"], color=[ROLE_COLORS.get(r, config.GREY) for r in s["role"]])
    for bar, (_, r) in zip(bars, s.iterrows()):
        unit = "img" if r["unit"] == "image" else "les"
        measure = "η" if r["effect_measure"].startswith("eta") else "V"
        reliable = r.get("chi2_reliable")
        flag = " †" if isinstance(reliable, (bool, np.bool_)) and not reliable else ""
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{measure}={r['effect']:.2f} · n={int(r['n']):,} {unit}{flag}", va="center", fontsize=8)
    for x, lab in [(0.1, "small"), (0.3, "medium"), (0.5, "large")]:
        ax.axvline(x, color="#999", lw=0.8, ls=":")
        ax.text(x, len(s) - 0.35, lab, fontsize=7.5, color="#777", ha="center")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for r, c in ROLE_COLORS.items() if r in set(s["role"])]
    ax.legend(handles, [r for r in ROLE_COLORS if r in set(s["role"])], loc="lower right", fontsize=8)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("association with diagnosis_1 (Cramér's V, or η for age) — 0 none, 1 perfect")
    ax.set_title("Which metadata fields track the diagnosis?", fontsize=11)
    ax.grid(axis="y", visible=False)
    fig.text(0.01, -0.02, "† chi-square p-value unreliable (sparse cells); V still describes the table. "
             "les = tested on 5,240 lesions, img = on 10,480 images.", fontsize=7.5, color="#666")
    fig.tight_layout()
    return _save(fig, save_as)


def plot_class_mix(
    meta: pd.DataFrame, fields: Sequence[str], target: str = TARGET, save_as: str | None = None
) -> plt.Figure:
    """Task 2 chart: 100% stacked bars of the class mix within each category, one panel per field.

    A dashed line marks the overall class mix: a field is uninformative when
    every bar looks like the ``(all)`` bar.
    """
    lesions = lesion_table(meta)
    inv = metadata_inventory(meta, target).set_index("column")
    tables = {}
    for field in fields:
        frame = lesions if inv.loc[field, "unit"] == "lesion" else meta
        tables[field] = class_distribution_table(frame, field, target)
    short = {"single contributor clinical assessment": "clinical\nassessment", "histopathology": "histo-\npathology",
             "lower extremity": "lower\nlimb", "upper extremity": "upper\nlimb", "oral/genital": "oral/\ngenital",
             "head/neck": "head/\nneck", "instrument only": "instrument\nonly", MISSING: "(miss-\ning)"}
    # Two rows, split so each row holds about half the bars; panel width is
    # proportional to the number of categories so every bar is the same width.
    widths = {f: len(t) + 0.6 for f, t in tables.items()}
    order, half, acc, rows_ = list(fields), sum(widths.values()) / 2, 0.0, [[], []]
    for f in order:
        rows_[0 if acc + widths[f] / 2 <= half else 1].append(f)
        acc += widths[f]
    row_w = max(sum(widths[f] for f in r) for r in rows_)
    fig = plt.figure(figsize=(0.7 * row_w + 0.8, 6.6))
    subfigs = fig.subfigures(2, 1, hspace=0.04)
    axes_by_field = {}
    for sf, r in zip(subfigs, rows_):
        ratios = [widths[f] for f in r] + ([row_w - sum(widths[f] for f in r)] if sum(widths[f] for f in r) < row_w - 0.5 else [])
        axs = sf.subplots(1, len(ratios), gridspec_kw={"width_ratios": ratios}, squeeze=False)[0]
        for extra in axs[len(r):]:
            extra.axis("off")
        axes_by_field.update(zip(r, axs))
    for field in fields:
        ax = axes_by_field[field]
        t = tables[field]
        labels = [short.get(str(i), str(i)) for i in t.index]
        bottom = np.zeros(len(t))
        for c in CLASSES:
            ax.bar(labels, t[c], bottom=bottom, color=config.BENIGN_MALIGNANT_COLORS[c], label=c, width=0.75)
            bottom += t[c].to_numpy()
        for i, n in enumerate(t["n"]):
            ax.text(i, 101.5, f"{n:,}", ha="center", fontsize=6.8, color="#555")
        # Overall class boundaries: a category is informative when its bar
        # breaks away from these two lines.
        for y in (t.loc["(all)", "Benign"], 100 - t.loc["(all)", "Malignant"]):
            ax.axhline(y, color="#222", lw=0.8, ls="--")
        unit = "lesions" if inv.loc[field, "unit"] == "lesion" else "images"
        ax.set_title(f"{field}\n({unit}; n above bars)", fontsize=9)
        ax.set_ylim(0, 108)
        ax.tick_params(axis="x", labelsize=7.2, rotation=0)
        ax.set_ylabel("% of category" if field in (rows_[0][0], rows_[1][0]) else "")
        ax.grid(axis="x", visible=False)
    handles, names = axes_by_field[fields[0]].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.075))
    return _save(fig, save_as)


def plot_numeric_by_class(
    meta: pd.DataFrame, column: str = "age_approx", target: str = TARGET, save_as: str | None = None
) -> plt.Figure:
    """Task 3 chart: per-class boxplots and overlaid density histograms, with ANOVA/eta annotated."""
    lesions = lesion_table(meta)
    res = numeric_association(lesions, column, target)
    sub = lesions[[column, target]].dropna()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1.4]})
    data_ = [sub.loc[sub[target] == c, column] for c in CLASSES]
    bp = ax1.boxplot(data_, tick_labels=CLASSES, patch_artist=True, widths=0.55, showmeans=True,
                     meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="#333", markersize=5))
    for patch, c in zip(bp["boxes"], CLASSES):
        patch.set_facecolor(config.BENIGN_MALIGNANT_COLORS[c])
        patch.set_alpha(0.8)
    for i, d in enumerate(data_, start=1):
        ax1.text(i, d.min() - 6, f"n={len(d):,}\nmean {d.mean():.1f}", ha="center", fontsize=7.5, va="top")
    ax1.set_ylim(sub[column].min() - 20, sub[column].max() + 5)
    ax1.set_ylabel(column)
    ax1.set_title("Per class (◇ = mean)", fontsize=10)
    bins = np.arange(0, 95, 5)
    for c, d in zip(CLASSES, data_):
        ax2.hist(d, bins=bins, density=True, histtype="step", lw=2, color=config.BENIGN_MALIGNANT_COLORS[c], label=c)
    ax2.set_xlabel(f"{column} (5-year bins)")
    ax2.set_ylabel("density (each class sums to 1)")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.set_title(f"ANOVA F={res['statistic']:.0f}, η²={res['eta_squared']:.3f} (η={res['effect']:.2f}); "
                  f"Kruskal-Wallis H={res['kruskal_H']:.0f}", fontsize=9)
    fig.tight_layout()
    return _save(fig, save_as)


# ===========================================================================
# Part 2 — colour and histograms across classes
# ===========================================================================
def sample_balanced_lesions(
    meta: pd.DataFrame | None = None, n_per_class: int = 120, seed: int = config.SEED
) -> pd.DataFrame:
    """Draw ``n_per_class`` lesions per class and return BOTH images of each.

    Sampling lesions rather than images makes the dermoscopic and clinical sets
    the *same* lesions, so a difference between modalities cannot be a
    difference in which lesions were drawn. Only lesions whose two files exist
    are eligible. Capped at the smallest class (Indeterminate: 123 lesions).
    """
    meta = data.load_metadata() if meta is None else meta
    meta = preprocessing.available_subset(meta)
    complete = meta.groupby("lesion_id")["isic_id"].transform("size") == 2
    meta = meta[complete]
    lesions = meta.drop_duplicates("lesion_id")[["lesion_id", TARGET]]
    smallest = lesions[TARGET].value_counts().min()
    if n_per_class > smallest:
        raise ValueError(f"n_per_class={n_per_class} exceeds the smallest class ({smallest} lesions)")
    chosen = lesions.groupby(TARGET).sample(n=n_per_class, random_state=seed)
    return meta[meta["lesion_id"].isin(chosen["lesion_id"])].reset_index(drop=True)


def _gray(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def image_color_features(
    sample: pd.DataFrame, bins: int = 256, progress_every: int = 100
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Read every sampled image at full resolution and measure it.

    Returns ``(stats, gray_hist, rgb_hist)``:

    * ``stats`` — one row per image: mean and std of R, G, B and of luminosity
      grayscale (brightness and contrast), and R/G, a simple redness (erythema)
      proxy; plus ``isic_id``, ``lesion_id``, class and modality.
    * ``gray_hist`` — ``(N, bins)``, ``rgb_hist`` — ``(N, 3, bins)``: each image's
      histogram **normalised to a density before averaging**, so every image
      carries equal weight whatever its pixel count (all MILK10k images are
      600x450, so this equals summing counts today, but stays correct if a
      different-sized image is ever added).
    """
    rows, gh, ch = [], [], []
    edges = np.linspace(0, 256, bins + 1)
    for i, r in enumerate(sample.itertuples(index=False), start=1):
        arr = np.asarray(preprocessing.load_image(r.isic_id), dtype=np.uint8)
        g = _gray(arr)
        px = arr.reshape(-1, 3).astype(np.float32)
        mean, std = px.mean(axis=0), px.std(axis=0)
        rows.append({
            "isic_id": r.isic_id, "lesion_id": r.lesion_id, TARGET: getattr(r, TARGET),
            "image_type": r.image_type,
            "mean_r": mean[0], "mean_g": mean[1], "mean_b": mean[2],
            "std_r": std[0], "std_g": std[1], "std_b": std[2],
            "mean_gray": float(g.mean()), "std_gray": float(g.std()),
            "r_over_g": float(mean[0] / max(mean[1], 1e-6)),
        })
        gh.append(np.histogram(g, bins=edges, density=True)[0])
        ch.append([np.histogram(arr[..., c], bins=edges, density=True)[0] for c in range(3)])
        if progress_every and i % progress_every == 0:
            print(f"    colour features: {i}/{len(sample)} images")
    return pd.DataFrame(rows), np.asarray(gh), np.asarray(ch)


COLOR_FEATURES = ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
                  "mean_gray", "std_gray", "r_over_g"]


def color_stats_by_class(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Task 6 table: mean ± std of each per-image colour statistic, per modality and class."""
    g = stats_df.groupby(["image_type", TARGET])[COLOR_FEATURES]
    out = g.mean().round(1).astype(str) + " ± " + g.std().round(1).astype(str)
    out["n_images"] = g.size()
    return out.reindex(pd.MultiIndex.from_product([MODALITIES, CLASSES]))


def color_separability(stats_df: pd.DataFrame, features: Sequence[str] = COLOR_FEATURES) -> pd.DataFrame:
    """Task 8, backed by a number: eta^2 per colour feature, within each modality and pooled.

    eta^2 is the share of a feature's variance explained by the class — the same
    statistic as Part 1's age analysis. The ``pooled`` column shows what happens
    if the modality is ignored.
    """
    rows = []
    for feat in features:
        row = {"feature": feat}
        for label, frame in [*((m, stats_df[stats_df.image_type == m]) for m in MODALITIES),
                             ("pooled", stats_df)]:
            row[label] = numeric_association(frame, feat)["eta_squared"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("feature").round(4)


def color_only_classifier(stats_df: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    """How far do colour statistics alone get? Cross-validated balanced accuracy per modality.

    A standardised multinomial logistic regression on the nine colour features,
    5-fold stratified CV. Within one modality every lesion appears once, so the
    folds cannot leak a lesion. The class sample is balanced, so chance is 1/3.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = []
    for m in MODALITIES:
        frame = stats_df[stats_df.image_type == m]
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        scores = cross_val_score(model, frame[COLOR_FEATURES], frame[TARGET],
                                 cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                                 scoring="balanced_accuracy")
        rows.append({"modality": m, "n_images": len(frame), "balanced_accuracy": scores.mean(),
                     "fold_std": scores.std(), "chance": 1 / frame[TARGET].nunique()})
    return pd.DataFrame(rows)


def sample_confounders(sample: pd.DataFrame) -> pd.DataFrame:
    """Task 7: do the classes differ in things that change colour but are not the diagnosis?

    Per class, in the colour sample: share of each Fitzpatrick skin-tone class,
    share of head/neck lesions (sun-exposed, redder skin), and share of
    ``altered`` images. Uses the supplement join in :func:`data.build_image_level`.
    """
    full = data.build_image_level()[["isic_id", "skin_tone_class"]]
    s = sample.merge(full, on="isic_id", how="left").drop_duplicates("lesion_id")
    out = pd.crosstab(s[TARGET], s["skin_tone_class"].map(config.SKIN_TONE_LABELS),
                      normalize="index").mul(100).round(1)
    out.columns = [f"skin tone {c} %" for c in out.columns]
    out["head/neck %"] = s.groupby(TARGET)["anatom_site_general"].apply(
        lambda x: 100 * (x == "head/neck").mean()).round(1)
    out["site missing %"] = s.groupby(TARGET)["anatom_site_general"].apply(
        lambda x: 100 * x.isna().mean()).round(1)
    alt = sample.groupby(TARGET)["image_manipulation"].apply(lambda x: 100 * (x == "altered").mean())
    out["altered images %"] = alt.round(1)
    return out.reindex(CLASSES)


def average_histograms(stats_df: pd.DataFrame, hist: np.ndarray) -> dict[tuple[str, str], np.ndarray]:
    """Mean histogram per (modality, class) — works for gray (N, bins) and RGB (N, 3, bins)."""
    out = {}
    for m in MODALITIES:
        for c in CLASSES:
            mask = ((stats_df.image_type == m) & (stats_df[TARGET] == c)).to_numpy()
            out[(m, c)] = hist[mask].mean(axis=0)
    return out


# --- Part 2 figures --------------------------------------------------------
def plot_gray_histograms(stats_df: pd.DataFrame, gray_hist: np.ndarray, save_as: str | None = None) -> plt.Figure:
    """Task 5: average grayscale histogram per class, one line per class, one panel per modality."""
    avg = average_histograms(stats_df, gray_hist)
    x = np.arange(gray_hist.shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3), sharey=True)
    for ax, m in zip(axes, MODALITIES):
        for c in CLASSES:
            n = int(((stats_df.image_type == m) & (stats_df[TARGET] == c)).sum())
            ax.plot(x, avg[(m, c)], color=config.BENIGN_MALIGNANT_COLORS[c], lw=1.8, label=f"{c} (n={n})")
        ax.set_title(f"{m} images", fontsize=10)
        ax.set_xlabel("grayscale intensity (0 = black, 255 = white)")
        ax.set_xlim(0, 255)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("share of pixels (density)")
    fig.suptitle("Average grayscale histogram per class", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, save_as)


def plot_rgb_histograms(stats_df: pd.DataFrame, rgb_hist: np.ndarray, save_as: str | None = None) -> plt.Figure:
    """Task 5: average R, G, B histograms per class — rows = modality, columns = channel."""
    avg = average_histograms(stats_df, rgb_hist)
    x = np.arange(rgb_hist.shape[-1])
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 4.9), sharex=True, sharey="row")
    for r, m in enumerate(MODALITIES):
        for ch, name in enumerate(["Red", "Green", "Blue"]):
            ax = axes[r, ch]
            for c in CLASSES:
                ax.plot(x, avg[(m, c)][ch], color=config.BENIGN_MALIGNANT_COLORS[c], lw=1.5, label=c)
            ax.set_title(f"{name} · {m}", fontsize=9)
            ax.set_xlim(0, 255)
            if ch == 0:
                ax.set_ylabel("density")
    for ax in axes[1]:
        ax.set_xlabel("channel value")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Average per-channel histogram per class", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, save_as)


def plot_color_boxplots(stats_df: pd.DataFrame, save_as: str | None = None) -> plt.Figure:
    """Task 6: per-image colour statistics by class, dermoscopic and clinical side by side."""
    feats = [("mean_r", "mean R"), ("mean_g", "mean G"), ("mean_b", "mean B"),
             ("std_gray", "contrast (std of gray)"), ("r_over_g", "R / G (redness)")]
    fig, axes = plt.subplots(1, len(feats), figsize=(12, 3.4))
    for ax, (feat, title) in zip(axes, feats):
        pos, vals, cols = [], [], []
        for mi, m in enumerate(MODALITIES):
            for ci, c in enumerate(CLASSES):
                pos.append(mi * 4 + ci)
                vals.append(stats_df.loc[(stats_df.image_type == m) & (stats_df[TARGET] == c), feat])
                cols.append(config.BENIGN_MALIGNANT_COLORS[c])
        bp = ax.boxplot(vals, positions=pos, widths=0.8, patch_artist=True, showfliers=False)
        for patch, col in zip(bp["boxes"], cols):
            patch.set_facecolor(col)
            patch.set_alpha(0.85)
        ax.set_xticks([1, 5], ["dermo-\nscopic", "clinical"], fontsize=8)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=config.BENIGN_MALIGNANT_COLORS[c]) for c in CLASSES]
    fig.legend(handles, CLASSES, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, save_as)
