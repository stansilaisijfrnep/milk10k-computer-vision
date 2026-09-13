"""Build the Session 1 homework deliverable PDF.

The brief asks for a PDF with two things: the link to the public GitHub repo
with the EDA code, and a screenshot of the code editor showing the project
structure. Plus the short explanation of the clinical task. That is all this
makes -- the EDA itself, the figures and the full write-up live in the repo.

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


CSS = """
@page { size: A4; margin: 20mm 18mm; }
* { box-sizing: border-box; }
body {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 10.5pt; line-height: 1.55; color: #1F1F1F; margin: 0;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
h1 { font-size: 15pt; margin: 0 0 2pt; }
h2 { font-size: 11.5pt; margin: 18pt 0 6pt; }
p  { margin: 0 0 9pt; }
ul { margin: 0 0 9pt; padding-left: 16pt; }
li { margin-bottom: 6pt; }
code { font-family: Menlo, Consolas, monospace; font-size: 9.5pt; }
a { color: #14509B; }

.head { border-bottom: 1pt solid #BBB; padding-bottom: 8pt; margin-bottom: 14pt; }
.head .sub { color: #555; font-size: 10pt; }

.repo { border: 1pt solid #CCC; padding: 10pt 12pt; margin: 4pt 0 4pt; background: #FAFAFA; }
.repo .lbl { font-size: 9pt; color: #555; margin-bottom: 3pt; }
.repo a { font-size: 11pt; word-break: break-all; }

figure { margin: 0 0 12pt; page-break-inside: avoid; }
figure img { width: 82%; border: 1pt solid #CCC; display: block; }
figcaption { font-size: 9pt; color: #555; margin-top: 4pt; }

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

<p>The EDA notebook is at <code>notebooks/01_eda_milk10k.ipynb</code>. It is saved with all
the outputs so the plots are visible on GitHub without running anything. The code it uses is
in <code>src/milk10k/</code> and the figures it produces are in <code>reports/figures/</code>.
The dataset is not in the repo (345 MB and CC BY-NC), so there is a script
<code>scripts/download_data.sh</code> that downloads it from ISIC.</p>

<h2>The clinical task</h2>

<p>MILK10k contains photos of skin lesions from patients who were sent to a dermatologist.
Every lesion was photographed twice: once as a normal close-up photo, and once through a
dermatoscope, which is a lens with polarised light that makes structures below the skin
surface visible. Almost all of these lesions were then biopsied, so the labels come from a
pathologist looking at the tissue under a microscope rather than from someone's opinion.</p>

<p>The task is to predict from the images whether a lesion is benign, malignant or
indeterminate (the <code>diagnosis_1</code> column). There is also a more detailed 11-class
version of the label as an optional target.</p>

<p>Why this matters is mostly about timing. A skin cancer that is found early is usually
removed in a small operation and that is the end of it. The same cancer found late is a much
bigger problem, and for melanoma it can be fatal. So the model is a triage tool. It should
almost never call something benign when it is actually cancer. Missing a melanoma is much
worse than doing one unnecessary biopsy, so the number I care about is recall on the
malignant class, not accuracy.</p>

<p>One thing I think is important to say clearly: this dataset only contains lesions that a
doctor already found suspicious enough to biopsy. 69% of them are malignant, which is nothing
like the general population, where almost everything you look at is a harmless mole. So a
model trained on this data is really answering "given that a specialist was already worried
about this lesion, is it cancer?" and not "is this cancer?". It would not work as a phone app
for the public. A longer version of this is in <code>docs/clinical_task.md</code> in the repo.</p>

<h2>What I did</h2>

<ul>
  <li>Downloaded the full dataset (10,480 images, 5,240 lesions) and loaded
  <code>metadata.csv</code> and <code>training_gt.csv</code> with pandas.</li>
  <li>Checked the structure rather than assuming it. Every lesion does have exactly two
  images, one of each type, and the 11-class labels are single-label.</li>
  <li>Plotted the class distribution for both targets, the missing values, age, sex, body
  site, skin tone, and the image sizes and colour statistics.</li>
  <li>Looked at the actual images: one example per class, and pairs of the same lesion in
  both modalities.</li>
</ul>

<h2>A few things I noticed</h2>

<ul>
  <li>The dataset is 69.4% malignant. Predicting "malignant" for everything already gives
  69.4% accuracy, so accuracy on its own does not tell you anything here.</li>
  <li>Since every lesion has two images, a random split by row can put both images of the
  same lesion on opposite sides of train and test. Then the model has basically already seen
  the test case. The split has to group by <code>lesion_id</code>.</li>
  <li>The 11-class target is very unbalanced. BCC has 2,522 lesions and MAL_OTH has 9.</li>
  <li>AKIEC does not map to one single value of <code>diagnosis_1</code>. It is split between
  Indeterminate (123) and Malignant (180), so I cannot just translate the 11-class label into
  the 3-class one with a lookup table.</li>
  <li>61% of the lesions are one skin tone class. If I only report the overall score I will
  not see how the model does on the tones that are rare in the data.</li>
</ul>

<h2>Editor</h2>

<p>I am using Visual Studio Code. Screenshots of the project on the next page.</p>

<div class="pagebreak"></div>

<h2>Code editor with the project structure</h2>

<figure>
  <img src="{embed(SHOT / 'vscode_eda_notebook.png')}" alt="VS Code with the project structure">
  <figcaption>The project in VS Code. Folder structure on the left, the EDA notebook open.</figcaption>
</figure>

<figure>
  <img src="{embed(SHOT / 'vscode_project_structure.png')}" alt="VS Code with the source code">
  <figcaption>The same project with <code>src/milk10k/data.py</code> open, which is where the
  loading and the structure checks are.</figcaption>
</figure>

</body></html>"""


def main() -> None:
    for required in [SHOT / "vscode_eda_notebook.png", SHOT / "vscode_project_structure.png"]:
        if not required.exists():
            raise SystemExit(f"Missing screenshot: {required}")

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
