"""Assemble notebooks/homework_part_a.ipynb from the per-exercise section files.

Each exercise lives in its own file under ``notebooks/parts/`` written in the
standard jupytext "percent" format::

    # %% [markdown]
    # ## A1.1 Cross-check the two label files
    # ...prose...

    # %%
    import pandas as pd
    ...

Keeping one file per exercise means the sections can be written, run and fixed
independently, and the notebook is always rebuildable from source rather than
hand-edited into an unreproducible state.

Usage:
    python scripts/build_part_a.py
    python scripts/build_part_a.py --check      # fail if a section is missing
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "notebooks" / "parts"
OUT = ROOT / "notebooks" / "homework_part_a.ipynb"

# Order is the order of the brief. Every one of these must exist.
SECTIONS = [
    "a1_1", "a1_2", "a1_3", "a1_4",
    "a2_1", "a2_2",
    "a3_1", "a3_2", "a3_3", "a3_4", "a3_5", "a3_6",
]

CELL_RE = re.compile(r"^# %%(?P<kind>[^\n]*)$", re.MULTILINE)

HEADER_MD = """
# Homework — Sessions 1, 2 and 3

**Computer Vision and Speech Recognition · EADA Business School**
Stanislaus Lattorff · MILK10k course project

This notebook is **Part A** of the Sessions 1–3 homework. One section per exercise,
each with the code and the written answer.

Everything reusable lives in the importable package `src/milk10k/` — the same package
Part B's pipeline uses — so nothing is copy-pasted between the notebook and the project
code. The notebook orchestrates and narrates; the package does the work.

**Paths.** There are no absolute paths anywhere. The dataset location comes from a single
place, `milk10k.config`, which reads the environment variable `MILK10K_IMAGES_DIR` if it
is set and otherwise falls back to `data/milk10k/images` inside the repo. See the README.

**Reproducibility.** Every random operation takes its seed from `config.SEED` (= 42).
Restart & Run All reproduces every number below.
""".strip()

SETUP_CODE = '''
import sys
from pathlib import Path

# Make the project package importable no matter where the kernel was started.
PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from milk10k import config, data, labels, plots, preprocessing, splits, transforms

config.apply_style()
config.ensure_project_dirs()

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 40)

print("python  ", sys.version.split()[0])
print("pandas  ", pd.__version__, "| numpy", np.__version__)
def _rel(path):
    """Show a path relative to the repo, so no machine-specific path is printed."""
    try:
        return path.relative_to(config.PROJECT_ROOT)
    except ValueError:
        return path  # MILK10K_IMAGES_DIR points outside the repo

print("data dir", _rel(config.DATA_DIR))
print("images  ", _rel(config.IMG_DIR), "| exists:", config.IMG_DIR.exists())
print("seed    ", config.SEED)
'''.strip()


def parse_percent_file(text: str) -> list[dict]:
    """Split a jupytext percent-format file into notebook cells."""
    marks = list(CELL_RE.finditer(text))
    if not marks:
        raise ValueError("no '# %%' cell markers found")

    cells = []
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[mark.end():end].strip("\n")
        is_md = "markdown" in mark.group("kind")

        if is_md:
            # Markdown cells are written as comments; strip the leading "# ".
            lines = [ln[2:] if ln.startswith("# ") else ln.lstrip("#")
                     for ln in body.split("\n")]
            source = "\n".join(lines).strip()
            if source:
                cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            source = body.strip("\n")
            if source:
                cells.append({"cell_type": "code", "execution_count": None,
                              "metadata": {}, "outputs": [], "source": source})
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="only report which sections are present")
    args = parser.parse_args()

    found, missing = [], []
    for name in SECTIONS:
        path = PARTS / f"{name}.py"
        (found if path.exists() else missing).append(name)

    print(f"sections found  : {len(found)}/{len(SECTIONS)}  {found}")
    if missing:
        print(f"sections MISSING: {missing}")
    if args.check:
        raise SystemExit(1 if missing else 0)
    if missing:
        raise SystemExit(f"Refusing to build an incomplete notebook: {missing}")

    cells: list[dict] = [
        {"cell_type": "markdown", "metadata": {}, "source": HEADER_MD},
        {"cell_type": "markdown", "metadata": {}, "source": "## Setup"},
        {"cell_type": "code", "execution_count": None, "metadata": {},
         "outputs": [], "source": SETUP_CODE},
    ]

    for name in SECTIONS:
        text = (PARTS / f"{name}.py").read_text(encoding="utf-8")
        try:
            section_cells = parse_percent_file(text)
        except ValueError as exc:
            raise SystemExit(f"{name}.py: {exc}") from exc
        cells.extend(section_cells)
        print(f"  {name}: {len(section_cells)} cells")

    for i, cell in enumerate(cells):
        cell["id"] = f"c{i:03d}"

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"\nWrote {OUT.relative_to(ROOT)} — {len(cells)} cells")


if __name__ == "__main__":
    main()
