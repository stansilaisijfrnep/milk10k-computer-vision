"""Milestone 1, B4 — data-quality report.

Extends the Session 1/2 metadata audit rather than repeating it: the handling
decision for every column with missing values, the label-consistency checks, the
shortcut cross-tabs, and the list of columns that must never be model inputs.

Writes ``reports/data_quality_report.md`` and
``artifacts/tables/data_quality_report.csv``.

Usage:
    python scripts/run_quality_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from milk10k import config, quality


def main() -> None:
    config.ensure_project_dirs()
    pd.set_option("display.width", 140)

    print("=" * 72)
    print("B4 — DATA-QUALITY REPORT")
    print("=" * 72)

    path = quality.build_quality_report()

    checks = quality.label_consistency_checks()
    n_failed = int((~checks["passed"]).sum())

    print(f"\nlabel-consistency checks : {len(checks) - n_failed}/{len(checks)} passed")
    if n_failed:
        print(checks[~checks["passed"]].to_string(index=False))

    print(f"\nreport -> {path.relative_to(config.PROJECT_ROOT)}")
    print(f"csv    -> {config.QUALITY_REPORT_CSV.relative_to(config.PROJECT_ROOT)}")
    print("=" * 72)
    raise SystemExit(0 if n_failed == 0 else 1)


if __name__ == "__main__":
    main()
