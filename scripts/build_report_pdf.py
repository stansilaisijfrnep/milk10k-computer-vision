"""Render reports/milestone1_report.md to a print-ready, two-page PDF.

The brief caps the Milestone 1 report at two pages. Markdown has no notion of a
page, so the Markdown file is the source of truth (it renders on GitHub) and this
script produces the printed version with compact typography, then reports the
page count so the limit is checked rather than assumed.

Figures are embedded as base64, so the HTML/PDF does not depend on where it is
opened from.

Usage:
    python scripts/build_report_pdf.py
"""

from __future__ import annotations

import base64
import mimetypes
import re
import subprocess
import sys
from pathlib import Path

import mistune

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_notebook_pdf import find_chrome  # noqa: E402  (same Chrome lookup)

SRC = ROOT / "reports" / "milestone1_report.md"
OUT_HTML = ROOT / "reports" / "milestone1_report.html"
OUT_PDF = ROOT / "reports" / "milestone1_report.pdf"
PAGE_LIMIT = 2

CSS = """
@page { size: A4; margin: 10mm 11mm; }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
       font-size: 8.4pt; line-height: 1.32; color: #222; }
h1 { font-size: 13pt; margin: 0 0 3px; }
h2 { font-size: 10pt; margin: 7px 0 2px; color: #b35f00;
     border-bottom: 1px solid #e6e6e6; padding-bottom: 1px; }
p, li { margin: 2px 0; }
ul { margin: 2px 0; padding-left: 15px; }
table { border-collapse: collapse; margin: 3px 0; font-size: 7.6pt; width: 100%; }
th, td { border: 1px solid #ddd; padding: 1.5px 4px; text-align: left; vertical-align: top; }
th { background: #f4f4f4; }
code { font-size: 7.6pt; background: #f5f5f5; padding: 0 2px; border-radius: 2px; }
a { color: #1a5fb4; text-decoration: none; }
img { display: block; margin: 3px auto; max-width: 100%; }
img.fig-mapping { width: 58%; }
img.fig-dist    { width: 66%; }
img.fig-aug     { width: 74%; }
"""

# One size per figure: wide grids get more width, the square mapping less.
FIG_CLASS = {
    "b2_label_mapping.png": "fig-mapping",
    "b2_class_distribution.png": "fig-dist",
    "b7_augmentation_examples.png": "fig-aug",
}


def _embed_images(html: str) -> str:
    """Swap relative <img src> paths for base64 data URIs, and size each figure."""
    def repl(match: re.Match) -> str:
        src = match.group(1)
        path = (SRC.parent / src).resolve()
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        data = base64.b64encode(path.read_bytes()).decode()
        cls = FIG_CLASS.get(path.name, "")
        return f'<img class="{cls}" src="data:{mime};base64,{data}"'
    return re.sub(r'<img src="([^"]+)"', repl, html)


def main() -> None:
    body = mistune.html(SRC.read_text(encoding="utf-8"))
    body = _embed_images(body)
    OUT_HTML.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Milestone 1 report</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>",
        encoding="utf-8",
    )

    subprocess.run(
        [find_chrome(), "--headless", "--disable-gpu", "--no-sandbox",
         "--no-pdf-header-footer", f"--print-to-pdf={OUT_PDF}", OUT_HTML.as_uri()],
        check=True, capture_output=True,
    )

    from pypdf import PdfReader
    n_pages = len(PdfReader(str(OUT_PDF)).pages)
    status = "OK" if n_pages <= PAGE_LIMIT else f"OVER THE {PAGE_LIMIT}-PAGE LIMIT"
    print(f"wrote {OUT_PDF.relative_to(ROOT)} — {n_pages} page(s): {status}")
    raise SystemExit(0 if n_pages <= PAGE_LIMIT else 1)


if __name__ == "__main__":
    main()
