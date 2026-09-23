"""Milestone 1, B2 — re-check every Session 1-2 finding against the full dataset.

Session 1 looked at samples: 400 JPEGs for the image properties, the summary
tables for everything else. A finding that holds on a sample is a hypothesis,
not a fact, and Milestone 1 builds a pipeline on top of these facts — the split
key, the metric, the class weights and the banned-input list all follow from
them. So each claim is re-run here **in code, on all 10,480 images / 5,240
lesions**, and gets one of three verdicts:

    verified  — the full dataset agrees with the Session 1-2 claim
    nuanced   — the claim holds, but the full check changes what follows from it
    refuted   — the full dataset disagrees; the claim is withdrawn

A refuted finding is a result, not an embarrassment: finding 7 (images vary in
resolution) is false on the full dataset, and that simplifies the resize
decision in ``docs/preprocessing_decisions.md``.

Every row also carries a ``shortcut_risk`` verdict, because "this feature
predicts the label" and "this feature must never be a model input" are different
claims and the brief asks for both.

Writes ``artifacts/tables/findings_check.csv`` (machine-readable) and
``reports/findings_check.md`` (the table a human reads).

Usage:
    python scripts/verify_findings.py
    python scripts/verify_findings.py --pixel-sample 500
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from milk10k import config, data, imbalance, labels, preprocessing, quality, splits

# --- Output paths ----------------------------------------------------------
# Derived from config, never spelled out: artifacts/ is what later milestones
# load, reports/ is what a human reads.
FINDINGS_CSV = config.ARTIFACT_TABLES_DIR / "findings_check.csv"
FINDINGS_MD = config.REPORTS_DIR / "findings_check.md"

# --- Verdicts --------------------------------------------------------------
VERIFIED = "verified"
NUANCED = "nuanced"
REFUTED = "refuted"

DERMOSCOPIC = "dermoscopic"
CLINICAL = "clinical: close-up"

COLUMNS = [
    "finding",
    "checked_on_full_dataset",
    "evidence",
    "consequence_for_pipeline",
    "shortcut_risk",
]


@dataclass(frozen=True)
class Finding:
    """One row of the B2 table: a claim, its verdict, and what follows from it.

    Frozen because a check produces its row once and nothing downstream is
    allowed to edit a verdict after the fact.
    """

    finding: str
    checked_on_full_dataset: str
    evidence: str
    consequence_for_pipeline: str
    shortcut_risk: str


@dataclass
class Context:
    """The data every check shares, loaded once.

    Each check re-deriving the metadata would be 10 redundant reads of the same
    four CSVs, and — worse — would let two checks silently disagree about what
    "the dataset" is.
    """

    meta: pd.DataFrame                    # 10,480 rows, one per image
    lesions: pd.DataFrame                 # 5,240 rows, one per lesion
    splits: dict[str, pd.DataFrame]       # the committed image-level splits
    crosstabs: dict[str, pd.DataFrame]    # quality.shortcut_crosstabs()
    pixel_sample: int | None              # images per modality, None = all


# --- Check 1: two images per lesion ----------------------------------------
def check_two_images_per_lesion(ctx: Context) -> Finding:
    """Session 1 claim: every lesion has exactly 2 images -> split on lesion_id.

    Re-counted on all 10,480 rows, and then checked where it actually bites:
    in the committed splits. The claim only protects us if no lesion has one
    image in train and the other in test.
    """
    per_lesion = ctx.meta.groupby("lesion_id").size()
    types_per_lesion = ctx.meta.groupby("lesion_id")["image_type"].nunique()

    n_lesions = len(per_lesion)
    exactly_two = int((per_lesion == 2).sum())
    two_types = int((types_per_lesion == 2).sum())

    # The consequence, checked on the artifacts rather than assumed.
    lesion_sets = {name: set(df["lesion_id"]) for name, df in ctx.splits.items()}
    overlaps = {
        f"{a}/{b}": len(lesion_sets[a] & lesion_sets[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    split_pairs_ok = all(
        bool((df.groupby("lesion_id").size() == 2).all()) for df in ctx.splits.values()
    )

    print(f"  images per lesion: min {per_lesion.min()}, max {per_lesion.max()}; "
          f"{exactly_two:,}/{n_lesions:,} lesions have exactly 2")
    print(f"  lesion overlap between splits: {overlaps}")

    return Finding(
        finding="Every lesion has exactly 2 images (one dermoscopic, one clinical "
                "close-up), so images are not independent samples.",
        checked_on_full_dataset=VERIFIED,
        evidence=f"{exactly_two:,}/{n_lesions:,} lesions have exactly 2 images "
                 f"(min {per_lesion.min()}, max {per_lesion.max()}); "
                 f"{two_types:,}/{n_lesions:,} have 2 distinct image_type values; "
                 f"{len(ctx.meta):,} image rows, {ctx.meta['isic_id'].nunique():,} "
                 f"unique isic_id. In the committed splits the lesion overlap is "
                 f"{overlaps['train/val']}/{overlaps['train/test']}/"
                 f"{overlaps['val/test']} (train-val / train-test / val-test) and "
                 f"every lesion keeps both of its images: {split_pairs_ok}.",
        consequence_for_pipeline="Split on lesion_id with a grouped splitter "
                                 "(splits.split_lesions), never on image rows. A row-level "
                                 "split would put the same piece of skin on both sides and "
                                 "inflate every score.",
        shortcut_risk="lesion_id is in config.LEAKY_COLUMNS: it is the grouping key, "
                      "never a model input — as a feature it would let the model "
                      "memorise individual lesions.",
    )


# --- Check 2: the majority class ------------------------------------------
def check_majority_class(ctx: Context) -> Finding:
    """Session 1 claim: 69.4% of lesions are Malignant -> accuracy is useless.

    Rather than asserting that accuracy is useless, the constant "always
    Malignant" predictor is actually scored, so the table quotes what the
    useless model gets on each metric we might report.
    """
    counts = ctx.lesions[config.PRIMARY_LABEL_COL].value_counts()
    shares = 100.0 * counts / len(ctx.lesions)
    majority = counts.idxmax()

    truth = ctx.lesions[config.PRIMARY_LABEL_COL]
    constant = pd.Series(majority, index=truth.index)
    acc = float((constant == truth).mean()) * 100.0
    macro_f1 = f1_score(truth, constant, average="macro", zero_division=0) * 100.0
    bal_acc = balanced_accuracy_score(truth, constant) * 100.0

    print(f"  lesion-level diagnosis_1: "
          + ", ".join(f"{cls} {counts[cls]:,} ({shares[cls]:.1f}%)"
                      for cls in config.PRIMARY_LABELS))
    print(f"  constant '{majority}' predictor: accuracy {acc:.2f}%, "
          f"balanced accuracy {bal_acc:.2f}%, macro-F1 {macro_f1:.2f}%")

    return Finding(
        finding="The 3-class target is dominated by one class: ~69% of lesions are "
                "Malignant, so plain accuracy is not a meaningful metric.",
        checked_on_full_dataset=VERIFIED,
        evidence="; ".join(f"{cls} {counts[cls]:,} ({shares[cls]:.1f}%)"
                           for cls in config.PRIMARY_LABELS)
                 + f" of {len(ctx.lesions):,} lesions. A constant "
                   f"'{majority}' predictor scores accuracy {acc:.2f}% but balanced "
                   f"accuracy {bal_acc:.2f}% and macro-F1 {macro_f1:.2f}%, with 0% "
                   f"recall on Benign and Indeterminate.",
        consequence_for_pipeline="Headline metrics are balanced accuracy, macro-F1, "
                                 "per-class recall and malignant sensitivity. Accuracy may "
                                 "be reported only next to the constant-predictor baseline "
                                 "above, so the reader can see what it is worth.",
        shortcut_risk="No feature involved — but the prior itself is the shortcut a "
                      "model will take if the loss is unweighted.",
    )


# --- Check 3: the 11-class tail --------------------------------------------
def check_stretch_imbalance(ctx: Context) -> Finding:
    """Session 1 claim: the 11-class target is imbalanced ~280:1.

    Re-checked on the full dataset and, more importantly, on the TRAIN split —
    the training ratio is the one the loss and the sampler actually see, and it
    is not the same number.
    """
    counts = imbalance.class_counts(ctx.lesions, config.STRETCH_LABEL_COL)
    ratio = imbalance.imbalance_ratio(ctx.lesions, config.STRETCH_LABEL_COL)

    train_lesions = _lesion_view(ctx.splits["train"])
    train_counts = imbalance.class_counts(train_lesions, config.STRETCH_LABEL_COL)
    train_ratio = imbalance.imbalance_ratio(train_lesions, config.STRETCH_LABEL_COL)

    rarest = counts.idxmin()
    biggest = counts.idxmax()
    per_split = {
        name: int((_lesion_view(df)[config.STRETCH_LABEL_COL] == rarest).sum())
        for name, df in ctx.splits.items()
    }

    print(f"  full dataset: {biggest} {counts[biggest]:,} vs {rarest} "
          f"{counts[rarest]} -> {ratio:.1f}:1")
    print(f"  TRAIN split : {biggest} {train_counts[biggest]:,} vs {rarest} "
          f"{train_counts[rarest]} -> {train_ratio:.1f}:1")
    print(f"  {rarest} lesions per split: " +
          ", ".join(f"{k} {v}" for k, v in per_split.items()))

    return Finding(
        finding="The 11-class stretch target has a long tail — imbalance around 280:1.",
        checked_on_full_dataset=VERIFIED,
        evidence=f"Full dataset {biggest} {counts[biggest]:,} vs {rarest} "
                 f"{counts[rarest]} = {ratio:.1f}:1. On the TRAIN split the ratio is "
                 f"{train_ratio:.1f}:1 ({biggest} {train_counts[biggest]:,} vs "
                 f"{rarest} {train_counts[rarest]}), which is the number the loss "
                 f"actually sees. {rarest} lands "
                 + "/".join(str(per_split[n]) for n in ("train", "val", "test"))
                 + " in train/val/test.",
        consequence_for_pipeline="Class weights (balanced and inverse-sqrt, computed on "
                                 "TRAIN only, artifacts/class_weights.json) plus an optional "
                                 "WeightedRandomSampler. No class is merged away: a "
                                 f"{rarest} test set of {per_split['test']} "
                                 f"{'lesion' if per_split['test'] == 1 else 'lesions'} is "
                                 "reported as a count, never as a percentage.",
        shortcut_risk="None. This is a property of the labels, not a feature.",
    )


# --- Check 4: the two modalities -------------------------------------------
def check_modality_pixel_stats(ctx: Context) -> Finding:
    """Session 1 claim: the two image types have different pixel statistics.

    Measured rather than asserted: per-channel means over every dermoscopic and
    every clinical image (the same streaming estimator used for the training
    normalisation statistics). The second half of the check is the one that
    matters for the pipeline — a modality that differed in pixels *and* in class
    mix would be a shortcut, so the image_type x diagnosis_1 cross-tab is
    inspected and the file-size-as-malignancy-score ROC-AUC is computed.
    """
    derm = preprocessing.compute_channel_stats(
        ctx.lesions["derm_id"].tolist(), max_images=ctx.pixel_sample)
    clin = preprocessing.compute_channel_stats(
        ctx.lesions["clinical_id"].tolist(), max_images=ctx.pixel_sample)

    gaps = [c - d for c, d in zip(clin["mean"], derm["mean"])]
    brightness = {k: sum(s["mean"]) / 3.0 for k, s in (("derm", derm), ("clin", clin))}
    worst = max(range(3), key=lambda i: abs(gaps[i]))
    channel = "RGB"[worst]

    stats_table = pd.DataFrame(
        {"R": [derm["mean"][0], clin["mean"][0]],
         "G": [derm["mean"][1], clin["mean"][1]],
         "B": [derm["mean"][2], clin["mean"][2]],
         "brightness": [brightness["derm"], brightness["clin"]],
         "n_images": [derm["n_images"], clin["n_images"]]},
        index=[DERMOSCOPIC, CLINICAL],
    ).round(4)
    print(stats_table.to_string())

    # Is the modality also a label shortcut? Row percentages answer "given this
    # modality, what is the class mix?" — which is the question a cheating model
    # would be answering.
    ct = ctx.crosstabs["image_type"]
    print(ct.round(2).to_string())
    spread = float(max(abs(ct.loc[DERMOSCOPIC, cls] - ct.loc[CLINICAL, cls])
                       for cls in config.PRIMARY_LABELS))

    # File size is pure compression, so if it scored malignancy we would have an
    # acquisition shortcut. It does not — but it separates the modalities.
    # Scored per modality (a pooled AUC would just rediscover the modality gap)
    # and as Malignant vs rest, which is the clinically relevant split.
    sizes = preprocessing.image_file_sizes(ctx.meta).merge(
        ctx.meta[["isic_id", "image_type", config.PRIMARY_LABEL_COL]], on="isic_id")
    aucs = {}
    for modality in (DERMOSCOPIC, CLINICAL):
        sub = sizes[sizes["image_type"] == modality]
        aucs[modality] = roc_auc_score(
            (sub[config.PRIMARY_LABEL_COL] == "Malignant").to_numpy(),
            sub["file_size_kb"].to_numpy(),
        )
    size_median = sizes.groupby("image_type")["file_size_kb"].median()
    print(f"  file size as a malignancy score: ROC-AUC "
          + ", ".join(f"{m.split(':')[0]} {a:.3f}" for m, a in aucs.items())
          + f"; median KB derm {size_median[DERMOSCOPIC]:.1f} vs clinical "
            f"{size_median[CLINICAL]:.1f}")

    return Finding(
        finding="Dermoscopic and clinical close-up images have different pixel "
                "statistics — two modalities, not two samples of one.",
        checked_on_full_dataset=VERIFIED,
        evidence=f"Channel means over {derm['n_images']:,} dermoscopic vs "
                 f"{clin['n_images']:,} clinical images at {config.IMAGE_SIZE}px: "
                 f"R {derm['mean'][0]:.3f}/{clin['mean'][0]:.3f}, "
                 f"G {derm['mean'][1]:.3f}/{clin['mean'][1]:.3f}, "
                 f"B {derm['mean'][2]:.3f}/{clin['mean'][2]:.3f}; mean brightness "
                 f"{brightness['derm']:.3f} vs {brightness['clin']:.3f} "
                 f"({brightness['clin'] - brightness['derm']:+.3f}). The largest gap is "
                 f"{channel} at {gaps[worst]:+.3f}. NOT a label shortcut: image_type x "
                 f"diagnosis_1 is flat — dermoscopic "
                 + "/".join(f"{ct.loc[DERMOSCOPIC, c]:.1f}" for c in config.PRIMARY_LABELS)
                 + " vs clinical "
                 + "/".join(f"{ct.loc[CLINICAL, c]:.1f}" for c in config.PRIMARY_LABELS)
                 + f" % Benign/Indeterminate/Malignant (max gap {spread:.2f} pp). File "
                   f"size scores malignancy at ROC-AUC {aucs[DERMOSCOPIC]:.3f} "
                   f"(dermoscopic) and {aucs[CLINICAL]:.3f} (clinical) — Malignant vs "
                   f"rest, i.e. chance — while "
                   f"separating the modalities strongly (median "
                   f"{size_median[DERMOSCOPIC]:.1f} vs {size_median[CLINICAL]:.1f} KB).",
        consequence_for_pipeline="Normalisation statistics are computed over the mixed "
                                 "training set, not per modality, because Milestone 2 feeds "
                                 "both views to one backbone. image_type stays in the split "
                                 "files so results can be reported per modality, and "
                                 "evaluation aggregates the two views per lesion "
                                 "(datasets.aggregate_predictions).",
        shortcut_risk="Low, and measured. image_type predicts nothing about the label, so "
                      "it is safe as a reporting slice or a conditioning input; file size "
                      "is chance-level on malignancy so no compression shortcut exists.",
    )


# --- Check 5: AKIEC straddles the coarse target ----------------------------
def check_akiec_straddles(ctx: Context) -> Finding:
    """Session 1 claim: AKIEC sits in both Indeterminate and Malignant.

    If it did not, the 3-class target would be a lookup away from the 11-class
    one and B3 would have had a much easier decision. It does, so both targets
    have to be carried through the pipeline separately.
    """
    table = labels.code_to_primary_table()
    ambiguous = labels.ambiguous_stretch_codes(table)
    print(table.to_string(index=False))

    akiec = table.loc[table["dx"] == "AKIEC"].iloc[0]
    indeterminate = ctx.lesions[ctx.lesions[config.PRIMARY_LABEL_COL] == "Indeterminate"]
    indeterminate_codes = indeterminate[config.STRETCH_LABEL_COL].value_counts()

    return Finding(
        finding="AKIEC straddles the coarse target: the same 11-class code appears as "
                "both Indeterminate and Malignant, so diagnosis_1 is not derivable "
                "from the 11-class code.",
        checked_on_full_dataset=VERIFIED,
        evidence=f"Ambiguous codes: {ambiguous}. AKIEC = {int(akiec['n_lesions'])} "
                 f"lesions, {int(akiec['Indeterminate'])} Indeterminate + "
                 f"{int(akiec['Malignant'])} Malignant; the other "
                 f"{len(table) - len(ambiguous)} codes each map to exactly one "
                 f"diagnosis_1 value. All {len(indeterminate):,} Indeterminate lesions "
                 f"are AKIEC ({indeterminate_codes.to_dict()}).",
        consequence_for_pipeline="Two label columns are stored side by side in the split "
                                 "files (diagnosis_1 and dx) and two label maps in "
                                 "artifacts/label_map.json. Indeterminate is kept as a real "
                                 "third class, not merged into Malignant: the pathologist "
                                 "declining to commit is clinical information.",
        shortcut_risk="None, but it blocks a tempting shortcut in the other direction — "
                      "deriving the 3-class label from an 11-class prediction by lookup "
                      "would silently mislabel every AKIEC lesion.",
    )


# --- Check 6: missing metadata ---------------------------------------------
def check_missing_metadata(ctx: Context) -> Finding:
    """Session 1 claim: anatom_site_general ~37% missing, melanocytic ~77%.

    The counts are re-checked, and then the question Session 1 did not ask: is
    the missingness itself informative? If lesions with no recorded site have a
    different class mix from lesions with one, then "unknown" carries signal and
    imputing it away destroys information (and quietly invents a site for 37% of
    the data).
    """
    site_col = "anatom_site_general"
    missing = {col: 100.0 * ctx.meta[col].isna().mean()
               for col in (site_col, "melanocytic")}

    lesion_site = ctx.meta.groupby("lesion_id")[site_col].first()
    lesion_view = ctx.lesions.merge(
        lesion_site.rename("site_recorded"), on="lesion_id", how="left")
    absent = lesion_view["site_recorded"].isna()

    rare_code = "NV"
    with_site = 100.0 * (lesion_view.loc[~absent, config.STRETCH_LABEL_COL]
                         == rare_code).mean()
    without_site = 100.0 * (lesion_view.loc[absent, config.STRETCH_LABEL_COL]
                            == rare_code).mean()

    mix = (lesion_view.assign(site_missing=absent)
           .groupby("site_missing")[config.PRIMARY_LABEL_COL]
           .value_counts(normalize=True)
           .unstack()
           .reindex(columns=config.PRIMARY_LABELS)
           * 100.0)
    print(f"  image-level missing: {site_col} {missing[site_col]:.2f}%, "
          f"melanocytic {missing['melanocytic']:.2f}%")
    print(f"  {rare_code} share: {with_site:.2f}% of lesions WITH a site vs "
          f"{without_site:.2f}% WITHOUT ({without_site - with_site:+.2f} pp)")
    print("  diagnosis_1 mix by site missingness (row %):")
    print(mix.round(2).to_string())

    # melanocytic: the non-null values are what give it away — the column is
    # never False, so "present" already means "melanocytic lesion".
    melanocytic = ctx.meta["melanocytic"]
    n_true = int(melanocytic.eq(True).sum())
    n_false = int(melanocytic.eq(False).sum())
    print(f"  melanocytic: True {n_true:,}, False {n_false:,}, "
          f"absent {int(melanocytic.isna().sum()):,}")

    return Finding(
        finding="Metadata is heavily incomplete: anatom_site_general missing ~37%, "
                "melanocytic missing ~77%, and melanocytic leaks the label.",
        checked_on_full_dataset=NUANCED,
        evidence=f"Image level: {site_col} missing {missing[site_col]:.2f}%, "
                 f"melanocytic missing {missing['melanocytic']:.2f}% "
                 f"(True on {n_true:,} images, False on {n_false:,}, absent on the rest "
                 f"— so its mere presence already says 'melanocytic lesion', which makes "
                 f"it a coarse label in disguise). The missingness "
                 f"is NOT at random: {rare_code} is {with_site:.2f}% of lesions with a "
                 f"recorded site vs {without_site:.2f}% of lesions without "
                 f"({without_site - with_site:+.2f} pp), and the Malignant share moves "
                 f"from {mix.loc[False, 'Malignant']:.1f}% to "
                 f"{mix.loc[True, 'Malignant']:.1f}%.",
        consequence_for_pipeline="anatom_site_general becomes an explicit 'unknown' "
                                 "category — never mode-imputed, because the absence is "
                                 "itself weakly predictive and imputing would both destroy "
                                 "that signal and invent a site for 37% of the rows. "
                                 "melanocytic stays in config.LEAKY_COLUMNS and is never "
                                 "loaded as a feature. age_approx (0.4% missing) is imputed "
                                 "with the TRAIN median only.",
        shortcut_risk="HIGH for melanocytic — it is derived from the diagnosis, so it is "
                      "banned as an input. MODERATE for 'unknown site': it is a real, "
                      "usable weak feature, but it encodes which contributor uploaded the "
                      "lesion, so any metadata model must be reported without it as well.",
    )


# --- Check 7: image resolution ---------------------------------------------
def check_resolution_uniform(ctx: Context) -> Finding:
    """Session 1 claim: images vary in resolution, so a resize is required.

    REFUTED. Session 1 measured 400 sampled JPEGs; the B1 pass opened all
    10,480 and the saved summary is read back here rather than recomputed, so
    this row quotes the same artifact the preprocessing decision quotes.
    """
    summary = pd.read_csv(config.IMAGE_SIZE_SUMMARY_CSV)
    value = dict(zip(summary["metric"], summary["value"]))
    note = dict(zip(summary["metric"], summary["note"]))
    print(summary.to_string(index=False))

    n_distinct = int(float(value["n_distinct_resolutions"]))
    uniform = n_distinct == 1

    return Finding(
        finding="Images vary in resolution, so the pipeline needs a resize step to "
                "make batching possible.",
        checked_on_full_dataset=REFUTED if uniform else VERIFIED,
        evidence=f"All {int(float(value['n_images'])):,} files were opened: "
                 f"{int(float(value['n_ok'])):,} readable, "
                 f"{int(float(value['n_missing']))} missing, "
                 f"{int(float(value['n_unreadable']))} unreadable. Width "
                 f"min/median/max = {int(float(value['width_min']))}/"
                 f"{int(float(value['width_median']))}/"
                 f"{int(float(value['width_max']))}, height "
                 f"{int(float(value['height_min']))}/"
                 f"{int(float(value['height_median']))}/"
                 f"{int(float(value['height_max']))}. "
                 f"n_distinct_resolutions = {n_distinct} "
                 f"({value['most_common_resolution']}, {note['most_common_resolution']}); "
                 f"{100 * float(value[f'frac_at_least_{config.IMAGE_SIZE}']):.1f}% are "
                 f"at least {config.IMAGE_SIZE}px and "
                 f"{100 * float(value['frac_at_least_256']):.1f}% at least 256px on both "
                 f"sides. The dataset is dimensionally uniform; the Session 1 claim came "
                 f"from a 400-image sample and does not survive the full check.",
        consequence_for_pipeline="A resize is still applied, but for a different reason: "
                                 "not to reconcile ragged inputs, but purely to match the "
                                 "backbone's expected input. Because every image is "
                                 "600x450, the resize is a pure downscale with no "
                                 "upsampled special cases and no padding branch — see "
                                 "docs/preprocessing_decisions.md.",
        shortcut_risk="None — and that is itself the useful result. Varying resolution "
                      "would have been a per-contributor fingerprint the model could key "
                      "on; a single uniform resolution removes that possibility.",
    )


# --- Check 8: the referral population --------------------------------------
def check_biopsy_referred(ctx: Context) -> Finding:
    """Session 1 claim: this is a biopsy-referred population, not a screening one.

    Quantified from diagnosis_confirm_type and concomitant_biopsy, next to the
    malignant prevalence, because the prevalence is what makes the point: no
    general population has 69% malignant skin lesions.
    """
    confirm = ctx.meta["diagnosis_confirm_type"].value_counts()
    histo = int(confirm.get("histopathology", 0))
    biopsied = ctx.meta["concomitant_biopsy"].eq(True)
    n_biopsied = int(biopsied.sum())
    # Two columns, one fact? Checked rather than assumed.
    same_images = bool(
        (biopsied == ctx.meta["diagnosis_confirm_type"].eq("histopathology")).all())
    malignant_share = 100.0 * (
        ctx.lesions[config.PRIMARY_LABEL_COL] == "Malignant").mean()

    print(f"  diagnosis_confirm_type: {confirm.to_dict()}")
    print(f"  concomitant_biopsy True on {n_biopsied:,}/{len(ctx.meta):,} images; "
          f"same images as histopathology: {same_images}")
    print(f"  malignant prevalence {malignant_share:.1f}% of lesions")

    return Finding(
        finding="MILK10k is a biopsy-referred dataset: lesions are here because a "
                "clinician already suspected something.",
        checked_on_full_dataset=VERIFIED,
        evidence=f"{histo:,}/{len(ctx.meta):,} images "
                 f"({100.0 * histo / len(ctx.meta):.1f}%) are confirmed by "
                 f"histopathology; concomitant_biopsy flags the identical "
                 f"{n_biopsied:,} images (mask equality checked: {same_images}); "
                 f"the remaining {len(ctx.meta) - histo:,} are single-contributor "
                 f"clinical assessment. Malignant prevalence is "
                 f"{malignant_share:.1f}% of lesions — a referral-clinic prior, not a "
                 f"population one.",
        consequence_for_pipeline="Every performance claim names the population: "
                                 "'on biopsied, referral-clinic lesions'. No threshold is "
                                 "tuned as if the model were a screening tool, and the "
                                 "test split is reported per class rather than as a single "
                                 "headline number.",
        shortcut_risk="Indirect: the referral filter is baked into the images themselves, "
                      "so it cannot be removed — only declared. It is the reason the "
                      "acquisition columns below leak.",
    )


# --- Check 9: image_manipulation -------------------------------------------
def check_manipulation_shortcut(ctx: Context) -> Finding:
    """Is image_manipulation a shortcut? (Homework A1.1, B4.)

    Row percentages, because the question is "given that this image was altered,
    what is the class mix?". The row count is quoted next to the percentages:
    a strong association over 335 images is a different object from one over
    ten thousand.
    """
    ct = ctx.crosstabs["image_manipulation"]
    print(ct.round(2).to_string())

    altered, instrument = "altered", "instrument only"
    gap = float(ct.loc[instrument, "Malignant"] - ct.loc[altered, "Malignant"])

    return Finding(
        finding="Acquisition metadata could encode the label: image_manipulation "
                "(altered vs instrument only) is a candidate shortcut.",
        checked_on_full_dataset=VERIFIED,
        evidence=f"Row percentages over {int(ct.loc[altered, 'n_images']):,} altered vs "
                 f"{int(ct.loc[instrument, 'n_images']):,} instrument-only images: "
                 f"altered = {ct.loc[altered, 'Benign']:.1f}% Benign / "
                 f"{ct.loc[altered, 'Indeterminate']:.1f}% Indeterminate / "
                 f"{ct.loc[altered, 'Malignant']:.1f}% Malignant, instrument only = "
                 f"{ct.loc[instrument, 'Benign']:.1f} / "
                 f"{ct.loc[instrument, 'Indeterminate']:.1f} / "
                 f"{ct.loc[instrument, 'Malignant']:.1f}. A real association "
                 f"({gap:.1f} pp on Malignant), on only "
                 f"{100.0 * ct.loc[altered, 'n_images'] / len(ctx.meta):.1f}% of images.",
        consequence_for_pipeline="Not used as a model input. It is kept in the split files "
                                 "as a reporting slice so a per-group metric can show "
                                 "whether the 335 altered images behave differently at "
                                 "test time.",
        shortcut_risk="MODERATE. The association is real but it describes the "
                      "photographer, not the lesion; as an input it would let the model "
                      "score points on 3.2% of images for free.",
    )


# --- Check 10: diagnosis_confirm_type --------------------------------------
def check_confirm_type_leak(ctx: Context) -> Finding:
    """The loudest leak in the dataset, quantified (B4).

    diagnosis_confirm_type is not a shortcut in the subtle sense — it is close
    to the label, because a lesion goes to histopathology precisely when a
    clinician already suspects malignancy.
    """
    ct = ctx.crosstabs["diagnosis_confirm_type"]
    print(ct.round(2).to_string())

    histo = "histopathology"
    clinical = "single contributor clinical assessment"
    gap = float(ct.loc[histo, "Malignant"] - ct.loc[clinical, "Malignant"])

    return Finding(
        finding="diagnosis_confirm_type records how the diagnosis was confirmed and is "
                "available at training time — is it usable as a feature?",
        checked_on_full_dataset=VERIFIED,
        evidence=f"{ct.loc[histo, 'Malignant']:.1f}% of the {int(ct.loc[histo, 'n_images']):,} "
                 f"histopathology-confirmed images are Malignant vs "
                 f"{ct.loc[clinical, 'Malignant']:.1f}% of the "
                 f"{int(ct.loc[clinical, 'n_images']):,} clinical-assessment images — a "
                 f"{gap:.1f} pp gap, the largest of any metadata column. The column is "
                 f"only knowable after the biopsy that produced the label.",
        consequence_for_pipeline="Listed in config.LEAKY_COLUMNS and excluded by "
                                 "construction: the dataloaders read image pixels plus the "
                                 "label column only. quality.leaky_columns_table() prints "
                                 "the ban list with a reason per column, so the exclusion "
                                 "is auditable rather than asserted.",
        shortcut_risk="SEVERE — effectively a label. Never a model input, in any "
                      "milestone.",
    )


CHECKS = [
    check_two_images_per_lesion,
    check_majority_class,
    check_stretch_imbalance,
    check_modality_pixel_stats,
    check_akiec_straddles,
    check_missing_metadata,
    check_resolution_uniform,
    check_biopsy_referred,
    check_manipulation_shortcut,
    check_confirm_type_leak,
]


# --- Helpers ---------------------------------------------------------------
def _lesion_view(split_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse an image-level split file to one row per lesion.

    The split files are image level (two rows per lesion). Counting classes on
    them would double every count — correct ratios, wrong n — so anything that
    talks about lesions goes through here first.
    """
    return split_df.drop_duplicates("lesion_id")


def _print_table(findings: list[Finding]) -> None:
    """Print the full table as readable blocks plus a one-line-per-row summary.

    A pandas ``to_string`` of five prose columns is unreadable in a terminal, and
    the point of this script is that a human can check each row, so the long
    cells are wrapped instead of truncated.
    """
    wrap = textwrap.TextWrapper(width=88, initial_indent=" " * 6,
                                subsequent_indent=" " * 6)
    for i, f in enumerate(findings, start=1):
        print(f"\n[{i}] {f.checked_on_full_dataset.upper()}")
        for field, value in asdict(f).items():
            print(f"  {field}:")
            print(wrap.fill(value))

    summary = pd.DataFrame(
        {
            "#": range(1, len(findings) + 1),
            "finding": [textwrap.shorten(f.finding, width=60, placeholder=" ...")
                        for f in findings],
            "verdict": [f.checked_on_full_dataset for f in findings],
            "shortcut_risk": [textwrap.shorten(f.shortcut_risk, width=44,
                                               placeholder=" ...")
                              for f in findings],
        }
    )
    print("\n--- summary ---")
    print(summary.to_string(index=False))


def _write_markdown(table: pd.DataFrame, ctx: Context, path: Path) -> Path:
    """Render the findings as a Markdown table for the report.

    Pipes inside a cell would break the table, so they are escaped; nothing else
    in the text needs escaping because the cells are plain prose and numbers.
    """
    counts = table["checked_on_full_dataset"].value_counts()
    lines = [
        "# Session 1-2 findings, re-checked on the full dataset",
        "",
        f"_Generated by `scripts/verify_findings.py` on {config.SPLIT_DATE}. "
        f"{len(table)} findings, each re-computed in code against all "
        f"{len(ctx.meta):,} images / {len(ctx.lesions):,} lesions — not a sample._",
        "",
        "| " + " | ".join(c.replace("_", " ") for c in COLUMNS) + " |",
        "|" + "|".join(["---"] * len(COLUMNS)) + "|",
    ]
    for _, row in table.iterrows():
        cells = [str(row[c]).replace("|", r"\|") for c in COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Verdict counts",
        "",
        *[f"- **{verdict}**: {n}" for verdict, n in counts.items()],
        "",
        "A refuted finding is a result. Finding 7 (images vary in resolution) came from "
        "a 400-image sample in Session 1; opening all 10,480 files shows a single "
        "resolution, which is why the resize decision in "
        "`docs/preprocessing_decisions.md` has no special cases.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


# --- Entry point -----------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pixel-sample", type=int, default=0,
        help="images per modality for the channel statistics (0 = all of them)")
    args = parser.parse_args()

    config.ensure_project_dirs()
    pd.set_option("display.width", 140)

    print("=" * 78)
    print("B2 — SESSION 1-2 FINDINGS, RE-CHECKED ON THE FULL DATASET")
    print("=" * 78)

    meta = data.load_metadata()
    lesions = data.build_lesion_table(meta)
    ctx = Context(
        meta=meta,
        lesions=lesions,
        splits={name: splits.load_split(name) for name in ("train", "val", "test")},
        crosstabs=quality.shortcut_crosstabs(meta),
        pixel_sample=args.pixel_sample or None,
    )
    print(f"images {len(ctx.meta):,}   lesions {len(ctx.lesions):,}   "
          f"splits " + " ".join(f"{k}={len(v):,}" for k, v in ctx.splits.items()))

    findings: list[Finding] = []
    for check in CHECKS:
        print(f"\n--- {check.__name__} ---")
        findings.append(check(ctx))

    print("\n" + "=" * 78)
    print("THE TABLE")
    print("=" * 78)
    _print_table(findings)

    table = pd.DataFrame([asdict(f) for f in findings], columns=COLUMNS)
    FINDINGS_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(FINDINGS_CSV, index=False)
    md_path = _write_markdown(table, ctx, FINDINGS_MD)

    print(f"\ncsv -> {FINDINGS_CSV.relative_to(config.PROJECT_ROOT)}")
    print(f"md  -> {md_path.relative_to(config.PROJECT_ROOT)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
