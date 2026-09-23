"""Render reports/session2_report.md to the Session 2 PDF (the brief allows 2-4 pages).

Same approach as ``build_report_pdf.py`` for Milestone 1: the Markdown file is
the source of truth (it renders on GitHub), this script prints it with compact
typography via headless Chrome, embeds the figures as base64, and reports the
page count so the limit is checked rather than assumed.

Usage:
    python scripts/run_session2.py        # figures and numbers first
    python scripts/build_session2_pdf.py
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

SRC = ROOT / "reports" / "session2_report.md"
OUT_HTML = ROOT / "reports" / "session2_report.html"
OUT_PDF = ROOT / "reports" / "Session2_MILK10k_EDA_Pipeline_Stanislaus_Lattorff.pdf"
PAGE_LIMIT = 4

CSS = """
@page { size: A4; margin: 9mm 11mm; }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
       font-size: 8.0pt; line-height: 1.27; color: #222; }
h1 { font-size: 13pt; margin: 0 0 3px; }
h2 { font-size: 10pt; margin: 6px 0 2px; color: #b35f00;
     border-bottom: 1px solid #e6e6e6; padding-bottom: 1px; }
h3 { font-size: 8.8pt; margin: 5px 0 1px; color: #333; page-break-after: avoid; }
p, li { margin: 2px 0; }
ul { margin: 2px 0; padding-left: 14px; }
table { border-collapse: collapse; margin: 3px 0; font-size: 7.4pt; width: 100%; }
th, td { border: 1px solid #ddd; padding: 1.2px 4px; text-align: left; vertical-align: top; }
th { background: #f4f4f4; }
code { font-size: 7.4pt; background: #f5f5f5; padding: 0 2px; border-radius: 2px; }
pre { background: #f7f7f7; border: 1px solid #e3e3e3; padding: 3px 6px; margin: 3px 0;
      font-size: 6.9pt; line-height: 1.25; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 6.9pt; }
a { color: #1a5fb4; text-decoration: none; }
img { display: block; margin: 3px auto; max-width: 100%; page-break-inside: avoid; }
"""

# Width per figure, as a share of the text width: wide grids get more room.
FIG_WIDTH = {
    "fig16_association_ranking.png": "48%",
    "fig17_class_mix_by_field.png": "44%",
    "fig18_age_by_class.png": "60%",
    "fig19_gray_hist_by_class.png": "70%",
    "fig20_rgb_hist_by_class.png": "62%",
    "fig21_color_stats_by_class.png": "86%",
    "fig22_raw_vs_processed.png": "40%",
    "fig23_loader_batch.png": "82%",
    "fig24_batch_summary.png": "58%",
    "fig25_class_balance.png": "46%",
}


def _embed_images(html: str) -> str:
    """Swap relative <img src> paths for base64 data URIs, and size each figure."""
    def repl(match: re.Match) -> str:
        path = (SRC.parent / match.group(1)).resolve()
        if not path.is_file():
            raise SystemExit(f"missing figure {path} — run scripts/run_session2.py first")
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        data = base64.b64encode(path.read_bytes()).decode()
        width = FIG_WIDTH.get(path.name, "80%")
        return f'<img style="width:{width}" src="data:{mime};base64,{data}"'
    return re.sub(r'<img src="([^"]+)"', repl, html)


def main() -> None:
    body = _embed_images(mistune.html(SRC.read_text(encoding="utf-8")))
    OUT_HTML.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Session 2 — MILK10k EDA and pipeline</title><style>{CSS}</style></head>"
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
    ok = n_pages <= PAGE_LIMIT
    print(f"wrote {OUT_PDF.relative_to(ROOT)} — {n_pages} page(s): "
          f"{'OK' if ok else f'OVER THE {PAGE_LIMIT}-PAGE LIMIT'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
