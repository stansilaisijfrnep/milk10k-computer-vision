"""Session 2 homework — run every analysis and write every figure and table.

    python scripts/run_session2.py [--lesions-per-class 120] [--batch-size 16]

Parts 1-2 (metadata association, colour analysis) and the Part 3-5 demos
(preprocessing before/after, a real loader batch, batch summary, class balance)
all come from functions in ``src/milk10k/`` — this script only calls them and
saves the results:

* figures  -> reports/figures/fig16_* ... fig25_*
* tables   -> reports/tables/session2_*.csv
* numbers  -> reports/tables/session2_key_numbers.json  (quoted in the report)
* digest   -> reports/session2_summary.md

Every sentence written to the digest is derived from a number computed in the
same run; nothing states a conclusion the data has not been checked against.
Runtime is about a minute (it reads ~720 full-resolution JPEGs).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from milk10k import config, data, eda2, pipeline, preprocessing, viz  # noqa: E402

TABLES = config.TABLES_DIR
CLASS_MIX_FIELDS = ["sex", "diagnosis_confirm_type", "melanocytic", "anatom_site_general",
                    "image_manipulation"]


def main(lesions_per_class: int = 120, batch_size: int = 16, seed: int = config.SEED) -> None:
    if not config.METADATA_CSV.exists():
        raise SystemExit(f"Dataset not found at {config.DATA_DIR}. Run `bash scripts/download_data.sh` first.")
    config.apply_style()
    config.ensure_dirs()
    started = time.time()
    lines: list[str] = ["# Session 2 — generated summary", "",
                        "_Written by `scripts/run_session2.py`; every number below is computed in that run._", ""]
    log = lines.append
    key: dict = {}

    meta = data.load_metadata()
    lesions = eda2.lesion_table(meta)

    # ------------------------------------------------------------------ Part 1
    print("Part 1 — metadata vs diagnosis_1")
    constancy = eda2.lesion_constancy(meta)
    inv = eda2.metadata_inventory(meta)
    inv.to_csv(TABLES / "session2_metadata_inventory.csv", index=False)
    summary = eda2.association_summary(meta)
    summary.to_csv(TABLES / "session2_association.csv", index=False)
    for field in inv["column"]:
        if inv.set_index("column").loc[field, "type"] in ("identifier", "constant"):
            continue
        unit = inv.set_index("column").loc[field, "unit"]
        if inv.set_index("column").loc[field, "type"] == "numeric":
            continue
        eda2.class_distribution_table(lesions if unit == "lesion" else meta, field).to_csv(
            TABLES / f"session2_class_mix_{field}.csv")
    age = lesions.groupby(eda2.TARGET)["age_approx"].describe().reindex(eda2.CLASSES).round(1)
    age.to_csv(TABLES / "session2_age_by_class.csv")

    # Claims in eda2.FIELD_ROLES, checked rather than asserted:
    ct = pd.crosstab(meta["concomitant_biopsy"], meta["diagnosis_confirm_type"])
    same_variable = int((ct.values > 0).sum()) == 2 and ct.shape == (2, 2)
    mel_present = lesions["melanocytic"].notna()
    mel_classes = sorted(meta.loc[meta["melanocytic"].notna()].merge(
        data.gt_to_label(data.load_training_gt()), on="lesion_id")["label_11"].unique())
    varying = constancy.query("n_lesions_varying > 0").set_index("column")["n_lesions_varying"].to_dict()
    assert same_variable, "concomitant_biopsy is no longer a copy of diagnosis_confirm_type"
    assert set(varying) == {"image_manipulation", "image_type"}, varying
    assert set(inv["role"]) <= set(eda2.ROLE_COLORS), "a column has no reviewed role"

    ranked = summary.dropna(subset=["effect"])
    top_usable = ranked[ranked.role == "usable"].head(3)
    confirm = eda2.class_distribution_table(lesions, "diagnosis_confirm_type")
    altered = eda2.class_distribution_table(meta, "image_manipulation")
    key["part1"] = {
        "n_images": len(meta), "n_lesions": len(lesions),
        "ranking": ranked[["field", "role", "unit", "effect", "effect_measure"]].round(3).to_dict("records"),
        "age": eda2.numeric_association(lesions, "age_approx"),
        "melanocytic_present_codes": mel_classes,
        "clinical_assessment_malignant_pct": float(confirm.loc["single contributor clinical assessment", "Malignant"]),
        "clinical_assessment_n": int(confirm.loc["single contributor clinical assessment", "n"]),
        "altered_indeterminate_pct": float(altered.loc["altered", "Indeterminate"]),
        "overall_indeterminate_pct_images": float(altered.loc["(all)", "Indeterminate"]),
        "images_varying_within_lesion": varying,
    }

    log("## Part 1 — metadata vs diagnosis")
    log(f"- Unit of analysis: {len(lesions):,} lesions ({len(meta):,} images). Only "
        f"{', '.join(f'`{k}` ({v:,} lesions)' for k, v in varying.items())} differ between a lesion's two "
        "images; those two are tested on images, everything else on lesions.")
    log(f"- `concomitant_biopsy` vs `diagnosis_confirm_type`: {'identical (1:1)' if same_variable else 'NOT identical'}.")
    log(f"- `melanocytic` is recorded only for lesions with 11-class codes {mel_classes}.")
    log("- Ranking by effect size (Cramér's V, or η for age):")
    for _, r in ranked.iterrows():
        log(f"  - `{r.field}` [{r.role}, {r.unit}] {r.effect:.3f} ({eda2.effect_label(r.effect)})")
    log(f"- Most associated *usable* fields: "
        + ", ".join(f"`{r.field}` ({r.effect:.2f})" for _, r in top_usable.iterrows()))
    log("")

    eda2.plot_association_ranking(summary, save_as="fig16_association_ranking.png")
    eda2.plot_class_mix(meta, CLASS_MIX_FIELDS, save_as="fig17_class_mix_by_field.png")
    eda2.plot_numeric_by_class(meta, "age_approx", save_as="fig18_age_by_class.png")
    plt.close("all")

    # ------------------------------------------------------------------ Part 2
    print(f"Part 2 — colour analysis ({lesions_per_class} lesions per class, both images each)")
    sample = eda2.sample_balanced_lesions(meta, lesions_per_class, seed)
    stats_df, gray_hist, rgb_hist = eda2.image_color_features(sample)
    stats_df.to_csv(TABLES / "session2_color_stats_per_image.csv", index=False)
    by_class = eda2.color_stats_by_class(stats_df)
    by_class.to_csv(TABLES / "session2_color_stats_by_class.csv")
    sep = eda2.color_separability(stats_df)
    sep.to_csv(TABLES / "session2_color_separability.csv")
    clf = eda2.color_only_classifier(stats_df, seed)
    clf.to_csv(TABLES / "session2_color_classifier.csv", index=False)
    conf = eda2.sample_confounders(sample)
    conf.to_csv(TABLES / "session2_color_sample_confounders.csv")

    # How much of the colour variation is modality rather than class?
    modality_eta2 = {f: eda2.numeric_association(stats_df, f, target="image_type")["eta_squared"]
                     for f in ["mean_r", "mean_g", "mean_b", "mean_gray"]}
    means = stats_df.groupby(["image_type", eda2.TARGET])["mean_gray"].mean().unstack().reindex(columns=eda2.CLASSES)
    key["part2"] = {
        "n_per_class_lesions": lesions_per_class, "n_images": len(stats_df),
        "eta2_within": sep.drop(columns="pooled").max(axis=1).round(4).to_dict(),
        "eta2_pooled": sep["pooled"].round(4).to_dict(),
        "max_eta2_feature": sep[eda2.MODALITIES].max(axis=1).idxmax(),
        "max_eta2": float(sep[eda2.MODALITIES].max().max()),
        "modality_eta2": {k: round(v, 3) for k, v in modality_eta2.items()},
        "mean_gray_by_modality_class": means.round(1).to_dict("index"),
        "classifier": clf.round(3).to_dict("records"),
        "confounders": conf.to_dict("index"),
    }
    log("## Part 2 — colour")
    log(f"- Sample: {lesions_per_class} lesions per class x 2 images = {len(stats_df)} images.")
    log(f"- η² of modality (dermoscopic vs clinical) on mean colour: "
        + ", ".join(f"{k} {v:.2f}" for k, v in modality_eta2.items()))
    log(f"- η² of class, within a modality, largest: `{key['part2']['max_eta2_feature']}` "
        f"{key['part2']['max_eta2']:.3f} ({100 * key['part2']['max_eta2']:.1f}% of variance).")
    for r in clf.itertuples():
        log(f"- Colour-only logistic regression, {r.modality}: balanced accuracy {r.balanced_accuracy:.3f} "
            f"± {r.fold_std:.3f} (chance {r.chance:.3f}).")
    log("")

    eda2.plot_gray_histograms(stats_df, gray_hist, save_as="fig19_gray_hist_by_class.png")
    eda2.plot_rgb_histograms(stats_df, rgb_hist, save_as="fig20_rgb_hist_by_class.png")
    eda2.plot_color_boxplots(stats_df, save_as="fig21_color_stats_by_class.png")
    plt.close("all")

    # -------------------------------------------------------------- Parts 3-5
    print("Parts 3-5 — pipeline demos")
    demo_ids = (sample[sample.image_type == "dermoscopic"].groupby(eda2.TARGET).head(1)
                .set_index(eda2.TARGET).reindex(eda2.CLASSES)["isic_id"].tolist())
    viz.plot_raw_vs_processed(demo_ids, labels=eda2.CLASSES, save_as="fig22_raw_vs_processed.png")

    loader = pipeline.BatchLoader(sample, batch_size=batch_size, shuffle=True, seed=seed)
    first = next(iter(loader))
    viz.show_batch(first, n=16, n_cols=8, save_as="fig23_loader_batch.png")
    z_loader = pipeline.BatchLoader(sample, batch_size=batch_size, shuffle=True, seed=seed, normalize="zscore")
    zbatch = next(iter(z_loader))
    viz.plot_batch_summary(zbatch, save_as="fig24_batch_summary.png")
    viz.plot_class_balance(preprocessing.available_subset(meta), group_col="lesion_id",
                           title="Class balance of the images available on disk",
                           save_as="fig25_class_balance.png")
    plt.close("all")

    # Check the pipeline behaves as documented, on real files.
    r = pipeline.process_image(demo_ids[0], channels_first=True)
    assert r.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE) and 0 <= r.value_range[0] <= r.value_range[1] <= 1
    assert first.images.shape[1:] == (3, config.IMAGE_SIZE, config.IMAGE_SIZE) and first.labels.dtype == np.int64
    assert [loader.classes[i] for i in first.labels] == first.label_names
    n_seen = sum(len(b) for b in loader)
    assert n_seen + len(loader.skipped) == loader.n_images == len(sample)
    key["parts3_5"] = {
        "loader_describe": loader.describe(),
        "batch_shape": list(first.images.shape), "batch_dtype": str(first.images.dtype),
        "minmax_range": [round(float(first.images.min()), 3), round(float(first.images.max()), 3)],
        "zscore_mean_std": [round(float(zbatch.images.mean()), 3), round(float(zbatch.images.std()), 3)],
        "steps_example": list(r.steps),
        "demo_ids": demo_ids,
    }
    log("## Parts 3-5 — pipeline")
    log(f"- {loader.describe()}")
    log(f"- First batch: images {tuple(first.images.shape)} {first.images.dtype}, "
        f"range [{first.images.min():.3f}, {first.images.max():.3f}]; z-scored batch mean "
        f"{zbatch.images.mean():.3f}, std {zbatch.images.std():.3f}.")
    log(f"- `process_image` steps: {' → '.join(r.steps)}")

    (TABLES / "session2_key_numbers.json").write_text(json.dumps(key, indent=2, default=str))
    (config.REPORTS_DIR / "session2_summary.md").write_text("\n".join(lines) + "\n")
    n_figs = len(list(config.FIGURES_DIR.glob("fig1[6-9]_*.png"))) + len(list(config.FIGURES_DIR.glob("fig2[0-5]_*.png")))
    print(f"Done in {time.time() - started:.0f}s — {n_figs} Session 2 figures, tables in {TABLES.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lesions-per-class", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=config.SEED)
    a = ap.parse_args()
    main(a.lesions_per_class, a.batch_size, a.seed)
