"""Milestone 1, B8 — build the DataLoaders and run the sanity checks.

This is the end of the Milestone 1 pipeline: split CSVs + label map in, batches
of tensors out. It prints the checks the brief asks for (batch shape, dtype,
min/max, the label histogram over 20 train batches, and the wall-clock time for
one epoch) and saves the visual check of a transformed batch.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --sampler          # use the WeightedRandomSampler
    python scripts/run_pipeline.py --scheme stretch   # the 11-class target
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch

from milk10k import config, datasets, imbalance, labels, splits, transforms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=["primary", "stretch"], default="primary")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--sampler", action="store_true",
                        help="use the WeightedRandomSampler instead of shuffling")
    parser.add_argument("--n-batches", type=int, default=20)
    parser.add_argument("--no-epoch-timing", action="store_true")
    args = parser.parse_args()

    config.ensure_project_dirs()
    pd.set_option("display.width", 140)
    transforms.seed_everything(config.SEED)

    label_col = (config.PRIMARY_LABEL_COL if args.scheme == "primary"
                 else config.STRETCH_LABEL_COL)

    print("=" * 72)
    print("B8 — DATASET / DATALOADER")
    print(f"scheme={args.scheme}  label_col={label_col}  batch_size={args.batch_size}"
          f"  sampler={args.sampler}  seed={config.SEED}")
    print("=" * 72)

    for path in (config.TRAIN_SPLIT_CSV, config.VAL_SPLIT_CSV, config.TEST_SPLIT_CSV,
                 config.LABEL_MAP_JSON, config.CLASS_WEIGHTS_JSON):
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run scripts/make_splits.py first.")

    loaders = datasets.build_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        scheme=args.scheme,
        label_col=label_col,
        use_sampler=args.sampler,
        seed=config.SEED,
    )
    for name, loader in loaders.items():
        sampler = type(loader.sampler).__name__
        print(f"  {name:5s} {len(loader.dataset):6,} images, "
              f"{len(loader):4,} batches, sampler={sampler}")
    print(f"  num_workers = {loaders['train'].num_workers}")

    # --- The sanity checks the brief lists --------------------------------
    print("\n--- sanity checks ---")
    report = datasets.sanity_check(loaders, n_batches=args.n_batches,
                                   time_epoch=not args.no_epoch_timing)

    # --- Imbalance handling, both mechanisms -------------------------------
    print("\n--- imbalance handling ---")
    weights = imbalance.load_class_weights()
    block = weights["schemes"][args.scheme]
    print(f"  computed on : {weights['computed_on_split']} split only "
          f"({weights['n_rows']:,} lesions, seed {weights['seed']})")
    print(f"  train imbalance ratio : {block['imbalance_ratio']:.1f}x")
    table = pd.DataFrame({
        "train_count": block["counts"],
        "balanced": block["weights"]["balanced"],
        "inverse_sqrt": block["weights"]["inverse_sqrt"],
    })
    print(table.round(3).to_string())

    tensor = imbalance.weights_tensor(block["weights"]["balanced"], args.scheme)
    print(f"  weights tensor for nn.CrossEntropyLoss(weight=...): "
          f"shape {tuple(tensor.shape)}, dtype {tensor.dtype}")

    # --- Determinism of eval ----------------------------------------------
    print("\n--- eval_transform determinism ---")
    first = next(iter(loaders["val"]))[0]
    second = next(iter(loaders["val"]))[0]
    print(f"  two passes over the val loader give identical tensors: "
          f"{torch.equal(first, second)}")

    print("\n" + "=" * 72)
    print("Pipeline OK.")
    print("=" * 72)


if __name__ == "__main__":
    main()
