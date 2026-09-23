"""Milestone 1, B2/B7/B8 — regenerate every figure the brief asks for by name.

Five figures, all written to ``reports/figures/milestone1/``:

* B2 — the class distribution in both label schemes, with the split breakdown
  and the log-scale panel the brief requires;
* B2/B3 — the mapping between the 11-class and 3-class targets, which is where
  AKIEC turns out to straddle two primary classes;
* B2 — a 3x4 gallery, one dermoscopic example per class;
* B7 — one original plus seven augmented views for three classes;
* B8 — a real batch out of the train DataLoader, un-normalised so it is
  viewable, with the integer labels the loss function receives.

Nothing here computes anything. Every figure is drawn by ``milk10k.plots`` from
the committed artifacts, so re-running this script after a pipeline change is
the only step needed to bring the report's images back in line with the data.

Usage:
    python scripts/make_figures.py
    python scripts/make_figures.py --only augmentation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

# Headless on purpose: this script writes PNGs and must never try to open a
# window, so that it behaves identically in a terminal, in CI and over SSH.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from milk10k import config, plots

# --- The figure catalogue --------------------------------------------------
# name -> (output filename, one-line description, builder). Every builder takes
# `save_as` as a keyword and nothing else, which is what lets --only dispatch
# over this dict instead of a chain of if-statements.
FIGURES: dict[str, tuple[str, str, object]] = {
    "class_distribution": (
        "b2_class_distribution.png",
        "B2  class distribution, both schemes, linear + log, per split",
        plots.plot_split_class_distribution,
    ),
    "label_mapping": (
        "b2_label_mapping.png",
        "B2  11-class x diagnosis_1 mapping (A1.1c / B3)",
        plots.plot_label_mapping,
    ),
    "class_gallery": (
        "b2_class_gallery.png",
        "B2  3x4 gallery, one dermoscopic example per class",
        plots.plot_class_gallery_grid,
    ),
    "augmentation": (
        "b7_augmentation_examples.png",
        "B7  original + 7 augmented views, 3 classes",
        plots.plot_augmentation_examples,
    ),
    "batch": (
        "b8_transformed_batch.png",
        "B8  16 transformed images from the train DataLoader",
        plots.plot_transformed_batch,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=sorted(FIGURES),
        help="regenerate a single figure instead of all five",
    )
    args = parser.parse_args()

    config.ensure_project_dirs()
    # Applied once, here, rather than inside each plotting function: the style
    # is a property of the report, not of an individual chart.
    config.apply_style()

    selected = [args.only] if args.only else list(FIGURES)

    print("=" * 72)
    print("B2 / B7 / B8 — MILESTONE 1 FIGURES")
    print(f"out: {config.MILESTONE1_FIGURES_DIR.relative_to(config.PROJECT_ROOT)}")
    print("=" * 72)

    total_kb = 0.0
    for name in selected:
        filename, description, builder = FIGURES[name]
        print(f"\n{name:19s} {description}")

        fig = builder(save_as=filename)
        # Close explicitly: five figures of image grids is a lot of retained
        # memory, and pyplot keeps every one of them alive until told not to.
        plt.close(fig)

        path = config.MILESTONE1_FIGURES_DIR / filename
        size_kb = path.stat().st_size / 1024
        total_kb += size_kb
        print(f"{'':19s} -> {path.relative_to(config.PROJECT_ROOT)}  ({size_kb:,.1f} KB)")

    print("\n" + "=" * 72)
    print(f"{len(selected)} figure(s) written, {total_kb:,.1f} KB total.")
    print("=" * 72)


if __name__ == "__main__":
    main()
