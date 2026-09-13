"""Build the Session 1 homework deliverable PDF.

The brief asks for a PDF with the link to the public GitHub repo holding the
EDA code, and a screenshot of the code editor showing the project structure.
Plus a short explanation of the clinical task. This keeps the text to bullet
points and shows the work through screenshots and plots instead.

Usage:
    python scripts/build_deliverable.py
"""

from __future__ import annotations

import base64
import mimetypes
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
SHOT = ROOT / "reports" / "screenshots"
OUT_HTML = ROOT / "reports" / "session1_deliverable.html"
OUT_PDF = ROOT / "reports" / "Session1_MILK10k_EDA_Stanislaus_Lattorff.pdf"

REPO_URL = "https://github.com/stansilaisijfrnep/milk10k-computer-vision"

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]


def embed(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def figure(path: Path, caption: str, cls: str = "wide") -> str:
    return f"""
<figure class="{cls}">
  <img src="{embed(path)}" alt="{caption}">
  <figcaption>{caption}</figcaption>
</figure>"""


CSS = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 10.5pt; line-height: 1.5; color: #1F1F1F; margin: 0;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
h1 { font-size: 15pt; margin: 0 0 2pt; }
h2 { font-size: 11.5pt; margin: 16pt 0 6pt; }
p  { margin: 0 0 8pt; }
ul { margin: 0 0 8pt; padding-left: 16pt; }
li { margin-bottom: 5pt; }
code { font-family: Menlo, Consolas, monospace; font-size: 9.5pt; }
a { color: #14509B; }

.head { border-bottom: 1pt solid #BBB; padding-bottom: 8pt; margin-bottom: 12pt; }
.head .sub { color: #555; font-size: 10pt; }

.repo { border: 1pt solid #CCC; padding: 9pt 12pt; margin: 4pt 0; background: #FAFAFA; }
.repo .lbl { font-size: 9pt; color: #555; margin-bottom: 3pt; }
.repo a { font-size: 11pt; word-break: break-all; }

figure { margin: 0 0 10pt; page-break-inside: avoid; }
figure img { border: 1pt solid #CCC; display: block; }
figure.wide img  { width: 75%; }
figure.shot img  { width: 76%; }
figure.tall img  { width: 34%; }
figcaption { font-size: 9pt; color: #555; margin-top: 3pt; }

.pagebreak { page-break-before: always; }
"""


def build_html() -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Session 1 homework — MILK10k EDA</title>
<style>{CSS}</style></head><body>

<div class="head">
  <h1>Session 1 homework — MILK10k exploratory data analysis</h1>
  <div class="sub">Computer Vision and Speech Recognition · Stanislaus Lattorff</div>
</div>

<h2>GitHub repository</h2>
<div class="repo">
  <div class="lbl">Public repository (my personal account):</div>
  <a href="{REPO_URL}">{REPO_URL}</a>
</div>
<ul>
  <li>EDA notebook: <code>notebooks/01_eda_milk10k.ipynb</code>, saved with all outputs so the
  plots show on GitHub without running it.</li>
  <li>Code it uses: <code>src/milk10k/</code>. Figures it produces: <code>reports/figures/</code>.</li>
  <li>Dataset is not in the repo (345 MB, CC BY-NC) — <code>scripts/download_data.sh</code>
  pulls it from ISIC.</li>
</ul>

<h2>The clinical task</h2>
<ul>
  <li>Photos of skin lesions from patients who were sent to a dermatologist.</li>
  <li>Each lesion is photographed twice: a normal close-up, and one through a dermatoscope
  (a lens with polarised light that shows structures below the skin surface).</li>
  <li>Almost all lesions were biopsied afterwards, so the labels come from a pathologist and
  not from someone's opinion.</li>
  <li>Task: predict benign / malignant / indeterminate from the image
  (<code>diagnosis_1</code>). There is a finer 11-class label as an optional target.</li>
  <li>It is a triage tool. Skin cancer found early is removed in a small operation; found
  late it is a much bigger problem and melanoma can be fatal.</li>
  <li>So missing a cancer is far worse than doing one unnecessary biopsy. The number that
  matters is recall on the malignant class, not accuracy.</li>
  <li>Important caveat: the dataset only has lesions a doctor already found suspicious enough
  to biopsy. 69% are malignant, nothing like the general population. A model trained here
  answers "given a specialist was already worried, is this cancer?" — not "is this cancer?".</li>
</ul>

<h2>What I did</h2>
<ul>
  <li>Downloaded the full dataset: 10,480 images, 5,240 lesions.</li>
  <li>Loaded <code>metadata.csv</code> and <code>training_gt.csv</code> with pandas.</li>
  <li>Checked the structure instead of assuming it.</li>
  <li>Plotted class distribution, missing values, age, sex, body site, skin tone, image sizes
  and colour statistics.</li>
  <li>Looked at the actual images, per class and per lesion.</li>
</ul>

<div class="pagebreak"></div>

<h2>Code editor with the project structure</h2>
<p>I am using Visual Studio Code.</p>
{figure(SHOT / "vscode_eda_notebook.png",
        "Folder structure on the left, the EDA notebook open.", cls="shot")}
{figure(SHOT / "vscode_project_structure.png",
        "Same project with src/milk10k/data.py open — the loading and structure checks.",
        cls="shot")}

<div class="pagebreak"></div>

<h2>What the EDA showed</h2>
<ul>
  <li>69.4% of lesions are malignant. Always predicting "malignant" gives 69.4% accuracy, so
  accuracy alone says nothing.</li>
  <li>Every lesion has exactly two images. A random split by row can put both on opposite
  sides of train and test, so the split has to group by <code>lesion_id</code>.</li>
  <li>The 11-class target is very unbalanced: BCC has 2,522 lesions, MAL_OTH has 9.</li>
  <li>AKIEC is split between Indeterminate (123) and Malignant (180), so the 11-class label
  cannot be translated into the 3-class one with a lookup table.</li>
  <li>61% of lesions are a single skin tone class.</li>
  <li>Median age is 65 and 60% are men — a specialist clinic population, not the general
  public.</li>
  <li>Images are not all the same size, so they need resizing before training.</li>
</ul>

{figure(FIG / "fig01_class_distribution_primary.png",
        "The 3-class target. Malignant dominates — the opposite of the general population.")}
{figure(FIG / "fig02_class_distribution_11.png",
        "The 11-class target, log scale. 280:1 between the largest and smallest class.")}

<div class="pagebreak"></div>

<h2>The images</h2>
{figure(FIG / "fig10_class_gallery.png",
        "One dermoscopic example per class. BCC, the most common cancer here, is often just "
        "a pale pink patch, while a benign mole looks more dramatic.")}
{figure(FIG / "fig11_lesion_pairs.png",
        "The same lesion in both modalities. This is why the split has to group by lesion_id.",
        cls="tall")}

<h2>Who is in the data</h2>
{figure(FIG / "fig07_skin_tone.png",
        "Skin tone sampling and malignancy rate. One class is 61% of the data, so overall "
        "scores would hide how the model does on the rare tones.")}

</body></html>"""


def main() -> None:
    required = [
        SHOT / "vscode_eda_notebook.png",
        SHOT / "vscode_project_structure.png",
        FIG / "fig01_class_distribution_primary.png",
        FIG / "fig02_class_distribution_11.png",
        FIG / "fig07_skin_tone.png",
        FIG / "fig10_class_gallery.png",
        FIG / "fig11_lesion_pairs.png",
    ]
    for path in required:
        if not path.exists():
            raise SystemExit(f"Missing image: {path}\nRun scripts/run_eda.py first.")

    OUT_HTML.write_text(build_html())
    print(f"Wrote {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.0f} KB)")

    chrome = next((c for c in CHROME_CANDIDATES if Path(c).exists()), None)
    if chrome is None:
        raise SystemExit(
            "No Chrome/Chromium/Edge found to render the PDF.\n"
            f"Open {OUT_HTML} in a browser and print to PDF manually."
        )

    subprocess.run(
        [
            chrome,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={OUT_PDF}",
            OUT_HTML.as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=180,
    )
    print(f"Wrote {OUT_PDF} ({OUT_PDF.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    sys.exit(main())
