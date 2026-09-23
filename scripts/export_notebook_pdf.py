"""Execute a notebook and export it to PDF.

The brief asks for the Part A notebook committed to the repo AND exported to
PDF, with every plot visible and nothing cut off at the page edge. LaTeX is not
installed here, so we take the route the brief explicitly allows: export to HTML
and print that to PDF with headless Chrome.

Usage:
    python scripts/export_notebook_pdf.py notebooks/homework_part_a.ipynb
    python scripts/export_notebook_pdf.py notebooks/homework_part_a.ipynb --no-execute
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]

# Printed page geometry. nbconvert's default HTML is styled for a wide screen;
# without this the code cells run off the right edge of an A4 page, which the
# brief calls out specifically.
PRINT_CSS = """
<style>
@page { size: A4; margin: 14mm 12mm; }
body { font-size: 11px; }
#notebook-container, .jp-Notebook {
    box-shadow: none !important;
    padding: 0 !important;
    max-width: 100% !important;
    width: 100% !important;
}
div.jp-Cell, div.cell { page-break-inside: avoid; }
pre, code, .jp-RenderedText pre, .highlight pre {
    white-space: pre-wrap !important;
    word-break: break-word !important;
    overflow-wrap: anywhere !important;
    font-size: 9.5px !important;
    line-height: 1.35 !important;
}
.jp-OutputArea-output pre, .output_text pre, .jp-RenderedText pre {
    font-size: 8px !important;
}
img, svg, canvas { max-width: 100% !important; height: auto !important; }
table { font-size: 9.5px !important; max-width: 100% !important; }
h1 { page-break-before: always; }
h1:first-of-type { page-break-before: avoid; }
h1, h2, h3 { page-break-after: avoid; }
</style>
"""


def _show(path: Path) -> str:
    """Repo-relative path for messages when possible, absolute otherwise."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def find_chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise SystemExit(
        "No Chrome/Chromium/Edge found for PDF printing. Install one, or export "
        "the HTML by hand and use the browser's Print to PDF."
    )


def execute(nb: Path, timeout: int) -> None:
    """Run the notebook top to bottom, in place, and keep the outputs."""
    print(f"Executing {nb.name} (timeout {timeout}s per cell) ...")
    started = time.time()
    subprocess.run(
        [
            sys.executable, "-m", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            f"--ExecutePreprocessor.timeout={timeout}",
            "--ExecutePreprocessor.kernel_name=python3",
            str(nb),
        ],
        check=True,
        cwd=ROOT,
    )
    print(f"  executed cleanly in {time.time() - started:.0f}s")


def to_html(nb: Path) -> Path:
    """Convert to a single self-contained HTML file with print styling."""
    html = nb.with_suffix(".html")
    subprocess.run(
        [sys.executable, "-m", "nbconvert", "--to", "html", "--embed-images",
         str(nb), "--output", html.name, "--output-dir", str(html.parent)],
        check=True,
        cwd=ROOT,
    )
    # Inject the print stylesheet last so it wins over nbconvert's own CSS.
    text = html.read_text(encoding="utf-8")
    text = text.replace("</head>", PRINT_CSS + "</head>", 1)
    html.write_text(text, encoding="utf-8")
    print(f"  wrote {_show(html)}")
    return html


def to_pdf(html: Path, pdf: Path) -> Path:
    chrome = find_chrome()
    subprocess.run(
        [chrome, "--headless", "--disable-gpu", "--no-sandbox",
         "--no-pdf-header-footer", "--virtual-time-budget=30000",
         f"--print-to-pdf={pdf}", html.as_uri()],
        check=True,
        capture_output=True,
    )
    size_kb = pdf.stat().st_size / 1024
    print(f"  wrote {_show(pdf)} ({size_kb:,.0f} KB)")
    return pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--no-execute", action="store_true",
                        help="export the notebook as it already is on disk")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="per-cell execution timeout in seconds")
    parser.add_argument("--pdf", type=Path, default=None)
    args = parser.parse_args()

    nb = (ROOT / args.notebook).resolve() if not args.notebook.is_absolute() else args.notebook
    if not nb.exists():
        raise SystemExit(f"No such notebook: {nb}")

    if not args.no_execute:
        execute(nb, args.timeout)

    html = to_html(nb)
    pdf = args.pdf or nb.with_suffix(".pdf")
    to_pdf(html, ROOT / pdf if not Path(pdf).is_absolute() else pdf)
    print("Done.")


if __name__ == "__main__":
    main()
