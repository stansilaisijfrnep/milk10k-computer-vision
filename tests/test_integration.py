"""End-to-end integration test for the whole MILK10k pipeline.

The seven pipeline modules (``labels``, ``splits``, ``transforms``,
``preprocessing``, ``imbalance``, ``datasets``, ``quality``) were written
separately, and each one has its own smoke test. That is not enough: the
interesting failures are the ones *between* modules — a function that returns a
column the next function does not look for, two modules that derive ``dx`` in
two different ways, a label map that disagrees with a weight vector's ordering.
None of those show up until the real path is run end to end.

So this script runs exactly that path, on the real data:

    load metadata + gt
      -> data.build_lesion_table          (5,240 lesions, both views per row)
      -> labels.lesion_labels / label_map (the target and its integer encoding)
      -> splits.split_lesions(seed=42)    (grouped + stratified, full dataset)
      -> splits.verify_splits             (disjoint, complete, representative)
      -> splits.write_splits              (into a TEMP dir, never the repo)
      -> imbalance.compute_class_weights  (TRAIN split only)
      -> imbalance.make_weighted_sampler
      -> datasets.build_dataloaders       (real augmentation + eval transforms)
      -> one real batch from train / val / test
      -> datasets.LesionDataset           (both views, batch_size=8)
      -> datasets.aggregate_predictions   (image probabilities -> lesion)
      -> quality.label_consistency_checks + quality.missing_value_report

Every table-level step runs on the full 5,240 lesions. Only the steps that
decode JPEGs run on a few hundred images, because reading all 10,480 twice
would turn a test you run constantly into one you run never.

Run it with::

    PYTHONPATH=src python tests/test_integration.py

It prints one line per check and exits non-zero if any of them fails.
"""

from __future__ import annotations

import ast
import inspect
import json
import sys
import tempfile
import textwrap
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Make the package importable when this file is run directly from the repo
# root, so that `python tests/test_integration.py` works with or without
# PYTHONPATH=src being set.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milk10k import (  # noqa: E402  (import after the sys.path fix, on purpose)
    config,
    data,
    datasets,
    imbalance,
    labels,
    preprocessing,
    quality,
    splits,
    transforms,
)

# How many lesions from each split are used for the steps that actually decode
# JPEGs. 60 lesions = 120 images per split, so the image path is exercised on
# ~360 real files instead of 10,480.
N_IMAGE_LESIONS = 60

# Batch sizes used for the loader checks.
IMAGE_BATCH = 32
LESION_BATCH = 8


# --- Tiny check harness ----------------------------------------------------
class Results:
    """Collect pass/fail lines so the run ends with one summary, not a stack trace.

    A plain ``assert`` would stop at the first problem, and when you are
    integrating seven modules you want to see *all* the mismatches in one run,
    not to rediscover them one at a time.
    """

    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        self.rows.append((name, bool(passed), detail))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
        return bool(passed)

    def fail(self, name: str, detail: str) -> None:
        self.check(name, False, detail)

    @property
    def n_failed(self) -> int:
        return sum(1 for _, passed, _ in self.rows if not passed)

    def summary(self) -> str:
        n_total = len(self.rows)
        n_passed = n_total - self.n_failed
        lines = [
            "",
            "=" * 78,
            f"INTEGRATION SUMMARY: {n_passed}/{n_total} checks passed",
        ]
        if self.n_failed:
            lines.append("")
            lines += [
                f"  FAILED: {name} — {detail}"
                for name, passed, detail in self.rows
                if not passed
            ]
        lines.append("=" * 78)
        return "\n".join(lines)


def section(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 68 - len(title)))


def has_row_loop(func) -> bool:
    """True if ``func``'s body contains a Python loop over rows.

    Checked on the parsed syntax tree rather than with a substring search, so a
    list comprehension over a handful of column names (which is not a row loop)
    does not count, and ``for _, row in df.iterrows()`` cannot hide behind
    different spacing.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    if any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree)):
        return True
    return any(
        isinstance(node, ast.Attribute) and node.attr in {"iterrows", "itertuples"}
        for node in ast.walk(tree)
    )


def our_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    """Keep only the warnings raised from inside ``src/milk10k``.

    pandas and torch set ``stacklevel`` so that a deprecation warning points at
    the line that called them, which means filtering on the filename is what
    separates "our code is using a removed API" from "a library warned about
    its own internals".
    """
    return [
        f"{Path(w.filename).name}:{w.lineno} {w.category.__name__}: {w.message}"
        for w in caught
        if f"{'/'}milk10k{'/'}" in Path(w.filename).as_posix()
    ]


# --- The run ---------------------------------------------------------------
def run(results: Results) -> None:
    """Execute the whole pipeline once, checking the contract at every seam."""

    # === 1. Raw tables =====================================================
    section("1. raw tables")
    meta = data.load_metadata()
    gt = data.load_training_gt()
    results.check("metadata.csv has 10,480 image rows", len(meta) == 10_480, f"{len(meta):,}")
    results.check("training_gt.csv has 5,240 lesion rows", len(gt) == 5_240, f"{len(gt):,}")

    # === 2. The lesion table ===============================================
    section("2. data.build_lesion_table")
    lesions = data.build_lesion_table(meta, gt)
    expected_cols = ["lesion_id", "derm_id", "clinical_id", "diagnosis_1", "dx",
                     "age", "sex", "site"]
    results.check("lesion table has 5,240 rows", len(lesions) == 5_240, f"{len(lesions):,}")
    results.check(
        "lesion table columns match the contract",
        list(lesions.columns) == expected_cols,
        str(list(lesions.columns)),
    )
    results.check(
        "build_lesion_table uses no Python row loop",
        not has_row_loop(data.build_lesion_table),
        "checked on the AST (no ast.For / ast.While / iterrows / itertuples)",
    )
    results.check(
        "every lesion has both view ids",
        bool(lesions["derm_id"].notna().all() and lesions["clinical_id"].notna().all()),
        f"{lesions['derm_id'].nunique():,} derm + {lesions['clinical_id'].nunique():,} clinical ids",
    )

    # === 3. Labels =========================================================
    section("3. labels")
    lesion_dx = labels.lesion_labels(gt)
    merged = lesions.merge(lesion_dx, on="lesion_id", how="left", suffixes=("", "_from_labels"))
    results.check(
        "labels.lesion_labels agrees with the lesion table's dx",
        bool((merged["dx"] == merged["dx_from_labels"]).all()),
        f"{int((merged['dx'] == merged['dx_from_labels']).sum()):,} / {len(merged):,} rows agree",
    )

    primary_map = labels.label_map("primary")
    stretch_map = labels.label_map("stretch")
    results.check(
        "primary label map is Benign/Indeterminate/Malignant -> 0/1/2",
        primary_map == {"Benign": 0, "Indeterminate": 1, "Malignant": 2},
        str(primary_map),
    )
    results.check("stretch label map covers 11 codes", len(stretch_map) == 11, str(stretch_map))
    results.check(
        "inverse_label_map round-trips",
        labels.inverse_label_map("stretch") == {v: k for k, v in stretch_map.items()},
    )

    encoded = labels.encode(lesions["dx"], scheme="stretch")
    results.check(
        "labels.encode gives int codes in range",
        bool(encoded.dtype == "int64" and encoded.between(0, 10).all()),
        f"dtype={encoded.dtype}, min={encoded.min()}, max={encoded.max()}",
    )

    # The AKIEC finding the brief insists must not be papered over.
    akiec = pd.crosstab(lesions["dx"], lesions["diagnosis_1"]).loc["AKIEC"]
    results.check(
        "AKIEC still straddles Indeterminate and Malignant",
        bool(akiec.get("Indeterminate", 0) > 0 and akiec.get("Malignant", 0) > 0),
        f"Indeterminate {int(akiec.get('Indeterminate', 0))}, Malignant {int(akiec.get('Malignant', 0))}",
    )

    # === 4. Splitting the full 5,240 lesions ===============================
    section("4. splits.split_lesions on all 5,240 lesions")
    train_ids, val_ids, test_ids = splits.split_lesions(lesions, seed=config.SEED)
    train_ids2, val_ids2, test_ids2 = splits.split_lesions(lesions, seed=config.SEED)
    results.check(
        "split_lesions is reproducible with the same seed",
        (train_ids, val_ids, test_ids) == (train_ids2, val_ids2, test_ids2),
        f"train {len(train_ids):,} / val {len(val_ids):,} / test {len(test_ids):,}",
    )
    overlap = (
        len(set(train_ids) & set(val_ids))
        + len(set(train_ids) & set(test_ids))
        + len(set(val_ids) & set(test_ids))
    )
    results.check("the three lesion sets are disjoint", overlap == 0, f"{overlap} shared lesions")
    results.check(
        "the three sets partition every lesion",
        set(train_ids) | set(val_ids) | set(test_ids) == set(lesions["lesion_id"]),
        f"{len(train_ids) + len(val_ids) + len(test_ids):,} of {len(lesions):,}",
    )

    report = splits.verify_splits(lesions, train_ids, val_ids, test_ids)
    results.check(
        "verify_splits passes",
        bool(report["passed"]),
        f"max class deviation {report['max_abs_deviation_pp']:.2f} pp, "
        f"missing in val {report['missing_classes']['val']}, "
        f"missing in test {report['missing_classes']['test']}",
    )
    print(report["sizes"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # === 5. Writing the split artefacts into a temp dir ====================
    section("5. splits.write_splits (temp dir)")
    image_df = splits.image_table(meta, gt)
    with tempfile.TemporaryDirectory(prefix="milk10k_splits_") as tmp:
        paths = splits.write_splits(
            train_ids, val_ids, test_ids, image_df=image_df, out_dir=Path(tmp)
        )
        results.check(
            "write_splits wrote train/val/test + manifest",
            all(p.exists() for p in paths.values()),
            ", ".join(sorted(p.name for p in paths.values())),
        )
        split_tables = {name: pd.read_csv(paths[name]) for name in splits.SPLIT_NAMES}
        n_images = {name: len(t) for name, t in split_tables.items()}
        results.check(
            "each split holds exactly 2 images per lesion",
            all(
                n_images[name] == 2 * len(ids)
                for name, ids in zip(splits.SPLIT_NAMES, (train_ids, val_ids, test_ids))
            ),
            str(n_images),
        )
        lesion_of_split = pd.concat(
            [t.assign(split=name) for name, t in split_tables.items()], ignore_index=True
        )
        crossers = lesion_of_split.groupby("lesion_id")["split"].nunique()
        results.check(
            "no lesion appears in two splits (image level)",
            int((crossers > 1).sum()) == 0,
            f"{int((crossers > 1).sum())} lesions cross a split boundary",
        )

        # The split files are what a later milestone opens without thinking, so
        # a banned column sitting in one is a leak waiting to happen. Written
        # with a deliberately wide frame — the whole of metadata.csv, which is
        # what scripts/make_splits.py passes — because a narrow frame would
        # prove nothing.
        wide = meta.merge(lesion_dx, on="lesion_id", how="left", validate="many_to_one")
        wide_dir = Path(tmp) / "wide"
        wide_paths = splits.write_splits(
            train_ids, val_ids, test_ids, image_df=wide, out_dir=wide_dir
        )
        wide_train = pd.read_csv(wide_paths["train"])
        banned_present = [
            c for c in config.LEAKY_COLUMNS
            if c in wide_train.columns and c not in splits.SPLIT_COLUMNS
        ]
        manifest = json.loads(wide_paths["manifest"].read_text())
        results.check(
            "write_splits strips config.LEAKY_COLUMNS from the split files",
            not banned_present and bool(manifest["dropped_leaky_columns"]),
            f"dropped {manifest['dropped_leaky_columns']}; "
            f"still present: {banned_present or 'none'}",
        )
        results.check(
            "write_splits keeps the grouping key and the target",
            all(c in wide_train.columns for c in splits.SPLIT_COLUMNS),
            f"columns: {list(wide_train.columns)}",
        )

        # === 6. Class weights, on the TRAIN split only =====================
        section("6. imbalance on the train split")
        train_lesions = lesions[lesions["lesion_id"].isin(train_ids)].reset_index(drop=True)
        counts = imbalance.class_counts(train_lesions, config.STRETCH_LABEL_COL)
        ratio = imbalance.imbalance_ratio(train_lesions, config.STRETCH_LABEL_COL)
        results.check(
            "class_counts covers all 11 codes in canonical order",
            list(counts.index) == config.STRETCH_LABELS,
            f"imbalance ratio {ratio:.1f}",
        )
        print("   train lesion counts:", counts.to_dict())

        weights = imbalance.compute_class_weights(
            train_lesions, config.STRETCH_LABEL_COL, "stretch", method="balanced"
        )
        results.check(
            "compute_class_weights returns one weight per class, rarest largest",
            (
                set(weights) == set(config.STRETCH_LABELS)
                and max(weights, key=weights.get) == counts.idxmin()
            ),
            f"largest weight: {max(weights, key=weights.get)} "
            f"({weights[max(weights, key=weights.get)]:.1f})",
        )
        tensor = imbalance.weights_tensor(weights, "stretch")
        results.check(
            "weights_tensor lines up with the label map",
            tuple(tensor.shape) == (11,)
            # float32 storage, so compare with a tolerance rather than ==:
            # at magnitude ~48 a float32 ulp is ~4e-6.
            and abs(
                float(tensor[labels.label_map("stretch")[counts.idxmin()]])
                - max(weights.values())
            )
            < 1e-4 * max(weights.values()),
            f"shape {tuple(tensor.shape)}, rarest weight "
            f"{float(tensor[labels.label_map('stretch')[counts.idxmin()]]):.4f}",
        )

        generator = torch.Generator().manual_seed(config.SEED)
        sampler = imbalance.make_weighted_sampler(
            train_lesions, config.STRETCH_LABEL_COL, "stretch", generator=generator
        )
        drawn = [int(i) for i in sampler]
        drawn_counts = train_lesions[config.STRETCH_LABEL_COL].to_numpy()[drawn]
        share = pd.Series(drawn_counts).value_counts(normalize=True)
        results.check(
            "weighted sampler flattens the class distribution",
            bool(len(drawn) == len(train_lesions) and share.max() < 0.25),
            f"largest sampled share {100 * share.max():.1f}% "
            f"(natural largest {100 * counts.max() / counts.sum():.1f}%)",
        )

        # === 7. DataLoaders over real JPEGs ================================
        section("7. datasets.build_dataloaders on a real image subset")
        subset_ids = {
            name: set(ids[:N_IMAGE_LESIONS])
            for name, ids in zip(splits.SPLIT_NAMES, (train_ids, val_ids, test_ids))
        }
        small = {
            name: table[table["lesion_id"].isin(subset_ids[name])].reset_index(drop=True)
            for name, table in split_tables.items()
        }
        loaders = datasets.build_dataloaders(
            batch_size=IMAGE_BATCH,
            scheme="primary",
            splits=small,
            num_workers=0,
            seed=config.SEED,
        )
        results.check(
            "build_dataloaders returns train/val/test",
            set(loaders) == {"train", "val", "test"},
            ", ".join(f"{k}={len(v.dataset)} images" for k, v in loaders.items()),
        )

        for name in ("train", "val", "test"):
            images, targets, ids = next(iter(loaders[name]))
            results.check(
                f"one real batch from '{name}'",
                (
                    tuple(images.shape) == (IMAGE_BATCH, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
                    and images.dtype == torch.float32
                    and targets.dtype == torch.int64
                    and len(ids) == IMAGE_BATCH
                    and isinstance(ids[0], str)
                ),
                f"images {tuple(images.shape)} {images.dtype}, targets {tuple(targets.shape)} "
                f"{targets.dtype}, ids[0]={ids[0]!r}, "
                f"range [{images.min():+.2f}, {images.max():+.2f}]",
            )

        sampled = datasets.build_dataloaders(
            batch_size=IMAGE_BATCH,
            scheme="stretch",
            splits=small,
            num_workers=0,
            use_sampler=True,
            seed=config.SEED,
        )
        images, targets, _ = next(iter(sampled["train"]))
        results.check(
            "build_dataloaders(use_sampler=True) produces a batch",
            tuple(images.shape) == (IMAGE_BATCH, 3, config.IMAGE_SIZE, config.IMAGE_SIZE),
            f"labels drawn: {sorted(set(targets.tolist()))}",
        )

    # === 8. The two Dataset classes ========================================
    section("8. MilkImageDataset / LesionDataset")
    image_ds = datasets.MilkImageDataset(small["val"], scheme="primary")
    tensor_item, label_item, id_item = image_ds[0]
    results.check(
        "MilkImageDataset returns (tensor, int, isic_id)",
        (
            isinstance(tensor_item, torch.Tensor)
            and tuple(tensor_item.shape) == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)
            and type(label_item) is int
            and isinstance(id_item, str)
        ),
        f"{type(tensor_item).__name__}{tuple(tensor_item.shape)}, "
        f"{type(label_item).__name__}={label_item}, {id_item!r}",
    )

    lesion_subset = lesions[lesions["lesion_id"].isin(subset_ids["val"])].reset_index(drop=True)
    lesion_ds = datasets.LesionDataset(lesion_subset, label_col="dx", scheme="stretch")
    lesion_loader = DataLoader(lesion_ds, batch_size=LESION_BATCH, shuffle=False)
    batch = next(iter(lesion_loader))
    results.check(
        "LesionDataset via DataLoader gives (8, 3, 224, 224) per view",
        (
            tuple(batch["derm"].shape)
            == (LESION_BATCH, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
            and tuple(batch["clinical"].shape)
            == (LESION_BATCH, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
            and tuple(batch["label"].shape) == (LESION_BATCH,)
            and len(batch["lesion_id"]) == LESION_BATCH
        ),
        f"derm {tuple(batch['derm'].shape)}, clinical {tuple(batch['clinical'].shape)}, "
        f"label {tuple(batch['label'].shape)}, lesion_id[0]={batch['lesion_id'][0]!r}",
    )
    results.check(
        "the two views of a lesion are different images",
        not torch.equal(batch["derm"][0], batch["clinical"][0]),
        f"mean abs difference {float((batch['derm'][0] - batch['clinical'][0]).abs().mean()):.3f}",
    )

    # === 9. Transforms =====================================================
    section("9. transforms")
    sample_id = str(small["val"]["isic_id"].iloc[0])
    pil = preprocessing.load_image(sample_id)
    evaluate = transforms.eval_transform()
    first, second = evaluate(pil), evaluate(pil)
    results.check(
        "eval_transform is bit-identical across two applications",
        bool(torch.equal(first, second)),
        f"max abs difference {float((first - second).abs().max()):.1e}",
    )
    transforms.seed_everything(config.SEED)
    augment = transforms.train_transform()
    a_first = augment(pil)
    transforms.seed_everything(config.SEED)
    a_second = transforms.train_transform()(pil)
    results.check(
        "train_transform is random but reproducible after seed_everything",
        bool(torch.equal(a_first, a_second) and not torch.equal(a_first, first)),
        f"augmented vs eval differ by {float((a_first - first).abs().mean()):.3f} on average",
    )
    results.check(
        "AUGMENTATION_TABLE documents every augmentation",
        all(
            set(row) >= {"augmentation", "parameters", "justification"}
            for row in transforms.AUGMENTATION_TABLE
        ),
        f"{len(transforms.AUGMENTATION_TABLE)} rows",
    )

    # === 10. Preprocessing =================================================
    section("10. preprocessing")
    probe = small["val"].head(40)
    available = preprocessing.available_subset(probe)
    results.check(
        "available_subset keeps every row whose file exists",
        len(available) == len(probe),
        f"{len(available)} / {len(probe)} rows have a file on disk",
    )
    array = preprocessing.preprocess_image(sample_id)
    results.check(
        "preprocess_image returns (224, 224, 3) float32 in [0, 1]",
        (
            array.shape == (config.IMAGE_SIZE, config.IMAGE_SIZE, 3)
            and array.dtype == np.float32
            and 0.0 <= float(array.min())
            and float(array.max()) <= 1.0
        ),
        f"{array.shape} {array.dtype} [{array.min():.3f}, {array.max():.3f}]",
    )
    stats = preprocessing.compute_channel_stats(probe["isic_id"].tolist()[:20])
    results.check(
        "compute_channel_stats returns 3 means and 3 stds",
        len(stats["mean"]) == 3 and len(stats["std"]) == 3 and stats["n_images"] == 20,
        f"mean {[round(v, 3) for v in stats['mean']]}, std {[round(v, 3) for v in stats['std']]}",
    )

    # === 11. A missing file must fail loudly ===============================
    section("11. missing image file")
    broken = small["val"].head(1).copy()
    broken.loc[:, "isic_id"] = "ISIC_DOES_NOT_EXIST"
    broken_ds = datasets.MilkImageDataset(broken, scheme="primary")
    try:
        broken_ds[0]
    except FileNotFoundError as exc:
        results.check(
            "a missing image raises FileNotFoundError naming the id",
            "ISIC_DOES_NOT_EXIST" in str(exc),
            str(exc).splitlines()[0],
        )
    except Exception as exc:  # noqa: BLE001 - any other exception is a failure
        results.fail(
            "a missing image raises FileNotFoundError naming the id",
            f"raised {type(exc).__name__} instead: {exc}",
        )
    else:
        results.fail(
            "a missing image raises FileNotFoundError naming the id",
            "no exception was raised at all",
        )

    # === 12. Image predictions -> lesion predictions =======================
    section("12. datasets.aggregate_predictions")
    val_images = small["val"]
    rng = np.random.default_rng(config.SEED)
    raw = rng.random((len(val_images), len(config.PRIMARY_LABELS)))
    synthetic = raw / raw.sum(axis=1, keepdims=True)
    per_lesion = datasets.aggregate_predictions(synthetic, val_images)
    results.check(
        "aggregate_predictions collapses 2 images into 1 lesion row",
        (
            len(per_lesion) == val_images["lesion_id"].nunique()
            and bool((per_lesion["n_views"] == 2).all())
        ),
        f"{len(val_images)} images -> {len(per_lesion)} lesions, "
        f"n_views unique {sorted(per_lesion['n_views'].unique().tolist())}",
    )
    prob_cols = list(config.PRIMARY_LABELS)
    results.check(
        "averaged probabilities still sum to 1 and pred matches argmax",
        (
            np.allclose(per_lesion[prob_cols].to_numpy().sum(axis=1), 1.0)
            and bool(
                (per_lesion["pred"] == per_lesion[prob_cols].idxmax(axis=1)).all()
            )
        ),
        f"columns {list(per_lesion.columns)}",
    )

    # === 13. Quality checks ================================================
    section("13. quality")
    consistency = quality.label_consistency_checks(meta, gt)
    results.check(
        "every label-consistency check passes",
        bool(consistency["passed"].all()),
        f"{int(consistency['passed'].sum())} / {len(consistency)} checks",
    )
    print(consistency.to_string(index=False))

    missing_report = quality.missing_value_report(meta)
    results.check(
        "missing_value_report has a written decision for every NaN column",
        bool((missing_report["decision"] != quality.UNREVIEWED_DECISION).all()),
        f"{len(missing_report)} columns with NaNs",
    )
    print(missing_report[["column", "n_missing", "pct_missing"]].to_string(
        index=False, float_format=lambda v: f"{v:.2f}"
    ))

    # The image-integrity pass runs on a sample here. The full 10,480-file
    # version is what `scripts/run_quality_report.py` does; this only has to
    # prove the two functions still fit together.
    sizes_df, sizes_summary_dict = quality.verify_all_images(meta.head(200))
    size_table = quality.image_size_summary(sizes_df)
    results.check(
        "verify_all_images + image_size_summary agree on a 200-image sample",
        (
            sizes_summary_dict["n_ok"] == 200
            and sizes_summary_dict["n_missing"] == 0
            and sizes_summary_dict["n_unreadable"] == 0
            and int(dict(zip(size_table["metric"], size_table["value"]))["n_ok"]) == 200
        ),
        f"{sizes_summary_dict['n_ok']}/200 readable, "
        f"{sizes_summary_dict['n_distinct_resolutions']} distinct resolution(s), "
        f"{int(sizes_summary_dict['width_median'])}x{int(sizes_summary_dict['height_median'])}",
    )

    crosstabs = quality.shortcut_crosstabs(meta)
    results.check(
        "shortcut_crosstabs returns one row-percentage table per column",
        all(
            set(config.PRIMARY_LABELS).issubset(table.columns) and "n_images" in table.columns
            for table in crosstabs.values()
        ),
        ", ".join(f"{k} ({len(v)} rows)" for k, v in crosstabs.items()),
    )

    # === 14. A scheme that contradicts its label column must be refused =====
    section("14. scheme / label_col guard")
    try:
        datasets.MilkImageDataset(small["val"], label_col="dx", scheme="primary")
    except ValueError as exc:
        results.check(
            "label_col='dx' with scheme='primary' raises a named ValueError",
            "stretch" in str(exc) and "dx" in str(exc),
            str(exc),
        )
    except Exception as exc:  # noqa: BLE001 - any other exception is a failure
        results.fail(
            "label_col='dx' with scheme='primary' raises a named ValueError",
            f"raised {type(exc).__name__} instead: {exc}",
        )
    else:
        results.fail(
            "label_col='dx' with scheme='primary' raises a named ValueError",
            "the contradiction was accepted silently",
        )

    # === 15. The artefact writers ==========================================
    # Every one of these is a Milestone 1 deliverable that a later milestone
    # loads instead of recomputing, so "it writes and it reads back" is part of
    # the contract. They go to a temp dir: a test must never overwrite the
    # committed artifacts/ files.
    section("15. artefact writers (temp dir)")
    with tempfile.TemporaryDirectory(prefix="milk10k_artifacts_") as tmp:
        tmp_dir = Path(tmp)

        label_payload = labels.write_label_map(path=tmp_dir / "label_map.json")
        reloaded = labels.load_label_map(tmp_dir / "label_map.json")
        results.check(
            "label_map.json round-trips and records the AKIEC ambiguity",
            (
                reloaded == label_payload
                and reloaded["primary"]["classes"] == primary_map
                and reloaded["ambiguous_codes"] == ["AKIEC"]
            ),
            f"n_lesions {reloaded['n_lesions']:,}, "
            f"ambiguous {reloaded['ambiguous_codes']}, "
            f"rare {reloaded['stretch']['rare_classes']}",
        )

        weights_payload = imbalance.write_class_weights(
            train_lesions, path=tmp_dir / "class_weights.json"
        )
        weights_back = imbalance.load_class_weights(tmp_dir / "class_weights.json")
        results.check(
            "class_weights.json round-trips and uses the same label map",
            (
                weights_back == weights_payload
                and weights_back["schemes"]["stretch"]["label_map"]
                == labels.label_map("stretch")
                and weights_back["computed_on_split"] == "train"
            ),
            f"stretch imbalance ratio {weights_back['schemes']['stretch']['imbalance_ratio']}, "
            f"n_rows {weights_back['n_rows']:,}",
        )

        stats_path = preprocessing.write_norm_stats(stats, path=tmp_dir / "norm_stats.json")
        results.check(
            "norm_stats.json round-trips",
            preprocessing.load_norm_stats(stats_path) == stats,
            f"written to {stats_path.name}",
        )

        report_path = quality.build_quality_report(
            out_md=tmp_dir / "data_quality_report.md",
            out_csv=tmp_dir / "data_quality_report.csv",
            df=meta,
            gt=gt,
            verify_images=False,
        )
        report_text = report_path.read_text(encoding="utf-8")
        results.check(
            "build_quality_report writes Markdown and CSV",
            (
                report_path.exists()
                and (tmp_dir / "data_quality_report.csv").exists()
                and "diagnosis_confirm_type" in report_text
            ),
            f"{len(report_text.splitlines())} lines of Markdown, "
            f"{len(pd.read_csv(tmp_dir / 'data_quality_report.csv')):,} CSV rows",
        )


def main() -> int:
    """Run everything under a warning recorder and print a PASS/FAIL summary."""
    results = Results()
    print("=" * 78)
    print("MILK10k cross-module integration test")
    print(f"repo   : {config.PROJECT_ROOT}")
    print(f"images : {config.IMG_DIR}")
    print(f"seed   : {config.SEED}")
    print("=" * 78)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            run(results)
        except Exception:  # noqa: BLE001 - a crash is a result, and it is a failure
            print("\n!!! the pipeline raised before finishing:\n")
            traceback.print_exc()
            results.fail("the pipeline runs end to end without raising", "see traceback above")

    section("16. warnings raised from inside src/milk10k")
    from_us = our_warnings(caught)
    results.check(
        "no warnings from our own modules",
        not from_us,
        "; ".join(from_us) if from_us else "none",
    )

    print(results.summary())
    return 1 if results.n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
