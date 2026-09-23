"""Execute ONE Part A section exactly as it will run inside the notebook.

The notebook's setup cell imports the package and applies the plotting style;
a section file assumes that has already happened. This harness reproduces that
context so a single exercise can be developed and verified on its own, without
re-running the whole notebook.

Usage:
    python scripts/run_part.py a1_1
    python scripts/run_part.py a1_1 --keep    # keep the generated scratch script
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "notebooks" / "parts"

sys.path.insert(0, str(ROOT / "scripts"))
from build_part_a import SETUP_CODE, parse_percent_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("section", help="e.g. a1_1")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--strict", action="store_true",
                        help="turn FutureWarning/DeprecationWarning into errors")
    args = parser.parse_args()

    path = PARTS / f"{args.section}.py"
    if not path.exists():
        raise SystemExit(f"No such section file: {path}")

    cells = parse_percent_file(path.read_text(encoding="utf-8"))
    code_cells = [c["source"] for c in cells if c["cell_type"] == "code"]
    n_md = len(cells) - len(code_cells)
    print(f"{args.section}: {len(code_cells)} code cells, {n_md} markdown cells\n")

    # Non-interactive backend: the harness must not try to open windows.
    script = "\n\n".join([
        "import matplotlib\nmatplotlib.use('Agg')",
        SETUP_CODE,
        *[f"# ---------- cell {i + 1} ----------\n{c}" for i, c in enumerate(code_cells)],
        "print('\\n=== SECTION OK ===')",
    ])

    # Written into the repo root so the section runs with the same cwd as the
    # notebook; the prefix is git-ignored in case a run is killed mid-way.
    with tempfile.NamedTemporaryFile("w", prefix="_run_part_", suffix=".py",
                                     delete=False, dir=ROOT, encoding="utf-8") as fh:
        fh.write(script)
        tmp = Path(fh.name)

    started = time.time()
    try:
        flags = ["-W", "error::FutureWarning", "-W", "error::DeprecationWarning"] \
            if args.strict else ["-W", "default::FutureWarning"]
        result = subprocess.run([sys.executable, *flags, tmp.name], cwd=ROOT)
    finally:
        if not args.keep:
            tmp.unlink(missing_ok=True)
        else:
            print(f"\n(kept {tmp.relative_to(ROOT)})")

    print(f"\nfinished in {time.time() - started:.1f}s, exit={result.returncode}")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
