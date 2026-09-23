"""Milestone 1, B1 — full-dataset integrity check.

Verifies that EVERY one of the 10,480 metadata rows resolves to a real image
file, runs ``Image.verify()`` on every one of them (not a sample), and records
the resolution statistics that justify the input size chosen in B6.

Writes ``artifacts/tables/image_size_summary.csv``.

Usage:
    python scripts/run_integrity_check.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from milk10k import config, data, preprocessing, quality


def main() -> None:
    config.ensure_project_dirs()
    pd.set_option("display.width", 130)

    print("=" * 72)
    print("B1 — FULL-DATASET INTEGRITY CHECK")
    print("=" * 72)
    print(f"images directory : {config.IMG_DIR}")
    print(f"exists           : {config.IMG_DIR.exists()}")

    df = data.load_metadata()
    print(f"metadata rows    : {len(df):,}")

    # 1. Does every row resolve to a file on disk?
    missing = preprocessing.missing_images(df)
    print(f"\nrows with no image file : {len(missing)}")
    if missing:
        print("  MISSING (first 20):")
        for iid in missing[:20]:
            print(f"    {iid}")
        print("  -> the loader raises FileNotFoundError on these; it does not skip them.")
    else:
        print("  -> all rows resolve. Nothing is being silently dropped.")

    # 2. Image.verify() on every file, plus size/format.
    print("\nverifying every image (this decodes headers, not full pixels) ...")
    started = time.time()
    sizes, summary = quality.verify_all_images(df)
    print(f"  done in {time.time() - started:.1f}s")

    for key, value in summary.items():
        print(f"    {key:24s} {value}")

    # 3. The saved summary table.
    table = quality.image_size_summary(sizes)
    out = config.IMAGE_SIZE_SUMMARY_CSV
    table.to_csv(out, index=False)
    print(f"\nimage_size_summary -> {out.relative_to(config.PROJECT_ROOT)}")
    print(table.to_string(index=False))

    n_bad = summary.get("n_missing", 0) + summary.get("n_unreadable", 0)
    print("\n" + "=" * 72)
    print("RESULT:", "PASS — every image resolves and decodes." if n_bad == 0
          else f"FAIL — {n_bad} problem file(s), listed above.")
    print("=" * 72)
    raise SystemExit(0 if n_bad == 0 else 1)


if __name__ == "__main__":
    main()
