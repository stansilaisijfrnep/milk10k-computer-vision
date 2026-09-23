"""Milestone 1, B5 — create and verify the train / val / test splits.

Splitting happens at **lesion** level, grouped on ``lesion_id`` and stratified on
the 11-class label; images are assigned afterwards. This is the only script that
writes the committed split files, so re-running it with the same seed reproduces
them exactly.

Also writes the label map (B3), the class weights (B8) and the train-only
channel statistics, because all three are defined *by* the train split and must
never be recomputed from the full dataset.

Usage:
    python scripts/make_splits.py
    python scripts/make_splits.py --seed 7 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from milk10k import config, data, imbalance, labels, preprocessing, splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--val-size", type=float, default=config.VAL_SIZE)
    parser.add_argument("--test-size", type=float, default=config.TEST_SIZE)
    parser.add_argument("--dry-run", action="store_true",
                        help="report the split but write nothing")
    parser.add_argument("--norm-sample", type=int, default=1500,
                        help="images used for the train-only channel statistics")
    args = parser.parse_args()

    config.ensure_project_dirs()
    pd.set_option("display.width", 140)

    print("=" * 72)
    print("B5 — SPLITS")
    print(f"seed={args.seed}  val={args.val_size:.0%}  test={args.test_size:.0%}"
          f"  date={config.SPLIT_DATE}")
    print("=" * 72)

    lesions = data.build_lesion_table()
    image_df = data.load_metadata().merge(
        labels.lesion_labels(), on="lesion_id", how="left", validate="many_to_one")
    print(f"lesions: {len(lesions):,}   images: {len(image_df):,}")

    # --- The split itself --------------------------------------------------
    train_ids, val_ids, test_ids = splits.split_lesions(
        lesions, val_size=args.val_size, test_size=args.test_size, seed=args.seed)

    total = len(lesions)
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        print(f"  {name:5s} {len(ids):5,} lesions ({len(ids) / total:6.2%})"
              f"  -> {2 * len(ids):6,} images")

    # --- Verification (printed, as the brief requires) ---------------------
    print("\n--- verification ---")
    report = splits.verify_splits(lesions, train_ids, val_ids, test_ids)
    for key, value in report.items():
        if isinstance(value, pd.DataFrame):
            print(f"\n{key}:")
            print(value.to_string())
        else:
            print(f"  {key:34s} {value}")

    # Every lesion must carry exactly 2 images into its split.
    print("\n--- images per lesion, per split ---")
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        frame = splits.assign_images(image_df, ids)
        per = frame.groupby("lesion_id").size()
        print(f"  {name:5s} {len(frame):6,} images; "
              f"all lesions have exactly 2 images: {bool((per == 2).all())}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    # --- Artifacts ---------------------------------------------------------
    print("\n--- writing artifacts ---")
    paths = splits.write_splits(train_ids, val_ids, test_ids, image_df=image_df,
                                seed=args.seed)
    for name, path in paths.items():
        print(f"  {name:6s} -> {Path(path).relative_to(config.PROJECT_ROOT)}")

    label_map = labels.write_label_map()
    print(f"  labels -> {config.LABEL_MAP_JSON.relative_to(config.PROJECT_ROOT)}"
          f"  ({len(label_map['primary']['classes'])} primary, "
          f"{len(label_map['stretch']['classes'])} stretch classes)")

    train_lesions = lesions[lesions["lesion_id"].isin(set(train_ids))]
    weights = imbalance.write_class_weights(train_lesions)
    print(f"  weights -> {config.CLASS_WEIGHTS_JSON.relative_to(config.PROJECT_ROOT)}")
    for scheme, block in weights["schemes"].items():
        print(f"      {scheme:8s} imbalance ratio on TRAIN = "
              f"{block['imbalance_ratio']:.1f}x  "
              f"(n={sum(block['counts'].values()):,} lesions)")

    # Channel statistics come from TRAIN images only — computing them over the
    # whole dataset is textbook preprocessing leakage.
    train_images = splits.assign_images(image_df, train_ids)
    sample = train_images["isic_id"].sample(
        n=min(args.norm_sample, len(train_images)), random_state=args.seed)
    stats = preprocessing.compute_channel_stats(sample.tolist())
    preprocessing.write_norm_stats(stats)
    print(f"  norm    -> {config.NORM_STATS_JSON.relative_to(config.PROJECT_ROOT)}")
    print(f"      mean {[round(v, 4) for v in stats['mean']]}  "
          f"std {[round(v, 4) for v in stats['std']]}  (n={stats['n_images']}, TRAIN only)")

    print("\n" + "=" * 72)
    print(f"Splits written with seed {args.seed} on {config.SPLIT_DATE}.")
    print("=" * 72)


if __name__ == "__main__":
    main()
