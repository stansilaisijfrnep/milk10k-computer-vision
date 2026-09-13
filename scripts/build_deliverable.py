"""Build the Session 1 homework deliverable PDF.

Assembles the repository link, the code-editor screenshot, the clinical task
explanation and the EDA highlights into one PDF, rendered through headless
Chrome so the typography is decent.

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
STUDENT = "Stanislaus Lattorff"

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]


def embed(path: Path) -> str:
    """Inline an image as a data URI so the PDF needs no external files."""
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def fig(name: str, caption: str, width: str = "100%") -> str:
    return f"""
    <figure style="width:{width}">
      <img src="{embed(FIG / name)}" alt="{caption}">
      <figcaption>{caption}</figcaption>
    </figure>"""


CSS = """
@page { size: A4; margin: 16mm 14mm; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
  font-size: 10pt; line-height: 1.5; color: #2B2B2B; margin: 0;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
h1 { font-size: 23pt; margin: 0 0 4pt; letter-spacing: -0.4pt; }
h2 { font-size: 14pt; margin: 0 0 10pt; padding-bottom: 5pt;
     border-bottom: 2.5pt solid #F08200; }
h3 { font-size: 11pt; margin: 16pt 0 5pt; color: #1A1A1A; }
p { margin: 0 0 8pt; }
a { color: #B35F00; text-decoration: none; word-break: break-all; }
.page { page-break-after: always; }
.page:last-child { page-break-after: auto; }

.cover { border-top: 7pt solid #F08200; padding-top: 16pt; }
.eyebrow { font-size: 8.5pt; letter-spacing: 1.6pt; text-transform: uppercase;
           color: #F08200; font-weight: 700; margin-bottom: 10pt; }
.subtitle { font-size: 12.5pt; color: #555; margin: 0 0 22pt; }
.meta { font-size: 9.5pt; color: #444; line-height: 1.9; }
.meta b { display: inline-block; width: 105pt; color: #1A1A1A; }

.linkbox { border: 1.2pt solid #F08200; background: #FFF8F0;
           padding: 12pt 14pt; margin: 20pt 0; border-radius: 3pt; }
.linkbox .lbl { font-size: 8pt; letter-spacing: 1.3pt; text-transform: uppercase;
                color: #B35F00; font-weight: 700; margin-bottom: 5pt; }
.linkbox a { font-size: 11.5pt; font-weight: 600; }

.callout { background: #F5F5F3; border-left: 3.5pt solid #F08200;
           padding: 10pt 13pt; margin: 12pt 0; font-size: 9.5pt; }
.callout b { color: #1A1A1A; }

table { width: 100%; border-collapse: collapse; font-size: 8.8pt; margin: 9pt 0 12pt; }
th { background: #F08200; color: white; text-align: left; padding: 5pt 7pt;
     font-weight: 600; font-size: 8.5pt; }
td { padding: 4.5pt 7pt; border-bottom: 0.6pt solid #E2E2E2; vertical-align: top; }
tr:nth-child(even) td { background: #FAFAF8; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }

figure { margin: 0 0 12pt; page-break-inside: avoid; }
figure img { width: 100%; border: 0.6pt solid #DDD; border-radius: 2pt; display: block; }
figcaption { font-size: 8pt; color: #666; margin-top: 3.5pt; font-style: italic; }

.grid2 { display: flex; gap: 10pt; }
.grid2 > figure { flex: 1; }

ol, ul { margin: 0 0 9pt; padding-left: 15pt; }
li { margin-bottom: 4.5pt; }

.tree { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 7.6pt;
        line-height: 1.45; background: #FAFAF8; border: 0.6pt solid #E2E2E2;
        padding: 9pt 11pt; border-radius: 2pt; white-space: pre; }
.footer { margin-top: 14pt; padding-top: 7pt; border-top: 0.6pt solid #DDD;
          font-size: 7.5pt; color: #888; }
.tag { display:inline-block; background:#F08200; color:#fff; font-size:7.5pt;
       font-weight:700; padding:1.5pt 6pt; border-radius:2pt; margin-right:5pt; }
"""

TREE = """Computer_Vision/
├── README.md                     Project overview + Session 1 findings
├── requirements.txt
├── .gitignore                    Excludes data/ (345 MB) and .venv/
│
├── data/                         Dataset — NOT committed (CC BY-NC)
│   └── milk10k/
│       ├── metadata.csv          10,480 rows — one per IMAGE
│       ├── images/               10,480 JPEGs
│       └── supplements/
│           ├── training_gt.csv   5,240 rows — one per LESION (11-class)
│           ├── training_input.csv  MONET concepts, skin tone, site
│           └── training_supp.csv   Full-text diagnosis
│
├── src/milk10k/                  Reusable package
│   ├── config.py                 Paths, label schemes, plot style
│   ├── data.py                   Loading, joins, integrity checks
│   └── plots.py                  Every figure in the report
│
├── notebooks/
│   └── 01_eda_milk10k.ipynb      ← Session 1 deliverable (executed)
│
├── scripts/
│   ├── download_data.sh          Fetch + unzip the dataset
│   ├── run_eda.py                Regenerate all figures headlessly
│   └── build_notebook.py
│
├── docs/
│   └── clinical_task.md          The clinical problem, explained
│
└── reports/
    ├── eda_summary.md            Generated numeric summary
    ├── figures/                  15 PNGs
    └── tables/                   Summary CSVs"""


def build_html() -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Session 1 — MILK10k EDA</title>
<style>{CSS}</style></head><body>

<!-- ============ PAGE 1 — COVER ============ -->
<section class="page cover">
  <div class="eyebrow">Computer Vision and Speech Recognition · Session 1 Homework</div>
  <h1>MILK10k — Exploratory Data Analysis</h1>
  <p class="subtitle">Skin lesion classification · Individual course project</p>

  <div class="meta">
    <b>Student</b> {STUDENT}<br>
    <b>Deliverable</b> Session 1 — EDA, project structure, clinical task<br>
    <b>Dataset</b> MILK10k (ISIC Archive) — 10,480 images / 5,240 lesions<br>
    <b>Editor / IDE</b> Visual Studio Code<br>
    <b>Environment</b> Python 3.14 · pandas · NumPy · matplotlib · Pillow
  </div>

  <div class="linkbox">
    <div class="lbl">Public GitHub repository</div>
    <a href="{REPO_URL}">{REPO_URL}</a>
  </div>

  <h3>What is in the repository</h3>
  <table>
    <tr><th>Item</th><th>Location</th></tr>
    <tr><td>EDA notebook (executed, all outputs saved)</td><td><code>notebooks/01_eda_milk10k.ipynb</code></td></tr>
    <tr><td>Reusable analysis package</td><td><code>src/milk10k/</code></td></tr>
    <tr><td>Clinical task explanation</td><td><code>docs/clinical_task.md</code></td></tr>
    <tr><td>Generated numeric summary</td><td><code>reports/eda_summary.md</code></td></tr>
    <tr><td>15 figures + summary tables</td><td><code>reports/figures/</code>, <code>reports/tables/</code></td></tr>
    <tr><td>One-command reproduction</td><td><code>scripts/download_data.sh</code>, <code>scripts/run_eda.py</code></td></tr>
  </table>

  <div class="callout">
    <b>Headline result of the EDA.</b> MILK10k is <b>69.4% malignant</b> — the reverse of the
    general population — because it only contains lesions a dermatologist already thought
    were worth biopsying. Every lesion appears <b>exactly twice</b> (one dermoscopic, one
    clinical image), so a random row-level train/test split leaks all 5,240 lesions.
    These two facts dictate the entire pipeline design.
  </div>

  <div class="footer">Dataset: MILK10k via ISIC Archive (CC BY-NC) —
  https://api.isic-archive.com/doi/milk10k/ · The dataset itself is not committed to the repository.</div>
</section>

<!-- ============ PAGE 2 — CLINICAL TASK ============ -->
<section class="page">
  <h2>1 · The clinical task we want to solve</h2>

  <h3>The medical problem</h3>
  <p>Skin cancer outcomes depend almost entirely on <b>when</b> the lesion is found. A melanoma
  caught while still confined to the epidermis is cured by a simple excision; the same tumour
  found later has a dramatically worse prognosis. The bottleneck is not treatment — it is
  <b>triage</b>. Far more lesions look worrying than turn out to be cancer, and the specialists
  who can tell the difference are scarce.</p>

  <p>A dermatologist examining a suspicious lesion decides: reassure, monitor, or excise and send
  to pathology. That decision is made mostly by eye, usually with a <b>dermatoscope</b> — a handheld
  lens pressed against the skin under polarised light, which reveals pigment networks and vessels
  invisible to the naked eye.</p>

  <h3>What MILK10k captures</h3>
  <p>MILK10k is built from exactly that moment. Every one of its <b>5,240 lesions was photographed
  twice</b>, and <b>95.7% were confirmed by histopathology</b> — a pathologist examined the excised
  tissue. The labels are therefore unusually trustworthy.</p>

  <table>
    <tr><th>Modality</th><th>What it is</th><th>Why it matters</th></tr>
    <tr><td>Clinical close-up</td><td>Ordinary photograph, visible light</td><td>Shows size, borders, surrounding skin</td></tr>
    <tr><td>Dermoscopic</td><td>Through a dermatoscope, polarised, magnified</td><td>Shows sub-surface structure — pigment networks, vessels</td></tr>
  </table>

  <h3>The prediction task</h3>
  <p><b>Primary target — <code>diagnosis_1</code>, three classes.</b> Input is one or both images of
  a lesion; output is one label. It is classification: no bounding box, no mask.</p>

  <table>
    <tr><th>Class</th><th class="num">Lesions</th><th class="num">Share</th><th>Clinical meaning</th></tr>
    <tr><td>Malignant</td><td class="num">3,634</td><td class="num">69.4%</td><td>Cancer — needs excision</td></tr>
    <tr><td>Benign</td><td class="num">1,483</td><td class="num">28.3%</td><td>Harmless — reassure or monitor</td></tr>
    <tr><td>Indeterminate</td><td class="num">123</td><td class="num">2.3%</td><td>Pathology itself was inconclusive</td></tr>
  </table>

  <p><b>Stretch target — 11 diagnosis codes</b> (<code>training_gt.csv</code>): BCC, NV, BKL, SCCKA,
  MEL, AKIEC, DF, INF, VASC, BEN_OTH, MAL_OTH.</p>

  <h3>What "good" means here</h3>
  <p>The two errors are not comparable. A <b>false negative</b> sends a patient home with a growing
  cancer. A <b>false positive</b> costs an unnecessary excision — a scar and some anxiety. So the
  metric that matters is <b>recall (sensitivity) on malignant lesions</b>, supported by balanced
  accuracy, macro-F1, per-class recall and per-skin-tone recall.</p>

  <div class="callout">
    <b>The honest framing.</b> Because the dataset only contains biopsy-referred lesions, a model
    trained here does not answer <i>"is this skin cancer?"</i> It answers <i>"given that a
    dermatologist was already worried enough to biopsy this, is it cancer?"</i> Deployed as a
    consumer app, its positive predictive value would collapse. The realistic role is
    <b>decision support</b> — flagging lesions for a specialist — not diagnosis.
  </div>
</section>

<!-- ============ PAGE 3 — EDITOR + STRUCTURE ============ -->
<section class="page">
  <h2>2 · Code editor and project structure</h2>
  <p><span class="tag">IDE</span> Visual Studio Code, with the project open and the Python
  environment (<code>.venv</code>) selected. Screenshot of the working repository:</p>

  <figure>
    <img src="{embed(SHOT / 'vscode_eda_notebook.png')}" alt="VS Code project structure">
    <figcaption>Visual Studio Code — full project tree on the left, the executed EDA notebook
    open in the editor.</figcaption>
  </figure>

  <figure>
    <img src="{embed(SHOT / 'vscode_project_structure.png')}" alt="VS Code source code">
    <figcaption>The reusable analysis package <code>src/milk10k/</code> — loaders, integrity
    checks and plotting live here rather than in notebook cells. Status bar: 0 errors, 0 warnings,
    branch <code>main</code>, interpreter <code>.venv (3.14.3)</code>.</figcaption>
  </figure>
</section>

<!-- ============ PAGE 4 — STRUCTURE TREE ============ -->
<section class="page">
  <h2>3 · Repository layout</h2>
  <p>The design rule: <b>analysis logic lives in <code>src/milk10k/</code>, not in notebook cells</b>,
  so later sessions import the same loaders and the same split logic instead of copy-pasting.</p>
  <div class="tree">{TREE}</div>

  <h3>Reproducing the analysis</h3>
  <div class="tree">git clone {REPO_URL}
cd milk10k-computer-vision
python3 -m venv .venv &amp;&amp; source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_data.sh     # ~345 MB, from the official ISIC source
python scripts/run_eda.py         # regenerates all 15 figures + tables</div>

</section>

<!-- ============ PAGE — DATASET STRUCTURE ============ -->
<section class="page">
  <h2>4 · Dataset structure — two different row units</h2>
  <p>This is the detail that governs everything downstream, and it was verified from the data
  rather than assumed.</p>
  <table>
    <tr><th>File</th><th>Row unit</th><th class="num">Rows</th><th>Contents</th></tr>
    <tr><td><code>metadata.csv</code></td><td>one image</td><td class="num">10,480</td><td>Demographics, 4-level diagnosis hierarchy, image type</td></tr>
    <tr><td><code>training_gt.csv</code></td><td>one lesion</td><td class="num">5,240</td><td>11-class one-hot ground truth</td></tr>
    <tr><td><code>training_input.csv</code></td><td>one image</td><td class="num">10,480</td><td>7 MONET concept scores, skin tone class, body site</td></tr>
    <tr><td><code>training_supp.csv</code></td><td>one image</td><td class="num">10,480</td><td>Full-text diagnosis, confirmation type</td></tr>
  </table>

  <div class="callout">
    <b>Verified structural checks</b> (all passed): every lesion has exactly 2 images —
    5,240 / 5,240; each lesion has exactly one dermoscopic and one clinical image;
    <code>isic_id</code> is unique across all 10,480 rows; diagnosis, age and sex are constant
    within a lesion; and the 11-class ground truth is genuinely single-label
    (0 lesions with anything other than exactly one positive class).
  </div>
</section>

<!-- ============ PAGE 5 — CLASS DISTRIBUTION ============ -->
<section class="page">
  <h2>5 · EDA — class distribution</h2>
  {fig("fig01_class_distribution_primary.png", "Primary target: diagnosis_1 at the lesion level. Malignant dominates — the opposite of the general population.")}
  <p>A model that predicts <b>Malignant</b> for every lesion scores <b>69.4% accuracy</b> while
  catching zero benign lesions. This is why accuracy cannot be the headline metric.</p>

  {fig("fig02_class_distribution_11.png", "Stretch target: the 11-class scheme, log scale. BCC has 2,522 lesions; MAL_OTH has 9.")}
  <p>The 11-class imbalance is <b>280:1</b>. Four classes have fewer than 55 lesions, which after a
  test split leaves only a handful of examples each.</p>

</section>

<!-- ============ PAGE — LABEL HIERARCHY ============ -->
<section class="page">
  <h2>6 · EDA — are the two label schemes consistent?</h2>
  {fig("fig03_diagnosis_hierarchy.png", "The two label schemes are nested — but not perfectly.")}
  <div class="callout">
    <b>A finding worth flagging.</b> Ten of the eleven codes map cleanly to one value of
    <code>diagnosis_1</code>. <b>AKIEC does not</b> — it splits into 123 <i>Indeterminate</i> and
    180 <i>Malignant</i> lesions. Clinically this makes sense (actinic keratosis and
    intraepithelial carcinoma lie on a continuum), but it means <code>diagnosis_1</code>
    <b>cannot be derived from the 11-class code by a lookup table</b>.
  </div>
</section>

<!-- ============ PAGE 6 — POPULATION & FAIRNESS ============ -->
<section class="page">
  <h2>7 · EDA — who and what is in the data</h2>
  {fig("fig05_demographics.png", "Median age 65, 60% male — a specialist skin-cancer clinic population, not the general public.")}
  {fig("fig07_skin_tone.png", "Skin tone sampling and malignancy rate. 61% of lesions are a single skin tone class.")}
  <div class="callout">
    <b>Fairness.</b> Skin tone class III alone is <b>60.6%</b> of lesions; the lightest (2.0%) and
    darkest (7.3%) tones are barely present. Aggregate metrics will be dominated by types III–IV,
    so <b>per-group recall must be reported separately</b>. The apparent rise in malignancy with
    darker tone classes is <b>confounded</b> by referral patterns and body site — it is not a
    causal effect, and with only 105 type-I lesions the leftmost bars are noisy.
  </div>
</section>

<!-- ============ PAGE — MISSING VALUES ============ -->
<section class="page">
  <h2>8 · EDA — missing values</h2>
  {fig("fig04_missing_values.png", "Missingness is structural, not random.")}
  <p><code>melanocytic</code> is blank 77% of the time but is <i>never</i> <code>False</code> — it is
  an annotation, not a boolean, so treating blank as "not melanocytic" would be wrong.</p>
</section>

<!-- ============ PAGE 7 — IMAGES ============ -->
<section class="page">
  <h2>9 · EDA — the images themselves</h2>
  {fig("fig10_class_gallery.png", "One dermoscopic example per diagnosis class (red = malignant, teal = benign, amber = indeterminate).")}
  <p>Note how little separates these visually: BCC, the majority class, is often a barely-pigmented
  pink patch, while a benign nevus is frequently the more dramatic-looking lesion. Colour and size
  heuristics will not work.</p>

  {fig("fig11_lesion_pairs.png", "Two images, one lesion — the most consequential structural fact in the dataset.", "62%")}
  <div class="callout">
    <b>Why the split must group by <code>lesion_id</code>.</b> If a random row split puts the
    dermoscopic image in train and the clinical image of the same lesion in test, the model has
    effectively already seen the test case. Scores rise; real performance does not.
  </div>

</section>

<!-- ============ PAGE — MODALITIES ============ -->
<section class="page">
  <h2>10 · EDA — two imaging modalities</h2>
  {fig("fig09_color_statistics.png", "Dermoscopic and clinical images have measurably different pixel statistics.")}
  <p>They are two modalities, not two samples of one. Images also vary in resolution, so a
  resize/crop step is mandatory before batching.</p>
</section>

<!-- ============ PAGE 8 — CONCLUSIONS ============ -->
<section class="page">
  <h2>11 · What the EDA tells us to do next</h2>
  <table>
    <tr><th>#</th><th>Finding</th><th>Consequence for the pipeline</th></tr>
    <tr><td class="num">1</td><td>Every lesion has exactly 2 images (verified 5,240/5,240)</td><td>Use a <b>grouped split on <code>lesion_id</code></b>. A random row split leaks every lesion.</td></tr>
    <tr><td class="num">2</td><td>69.4% of lesions are Malignant</td><td>Never report bare accuracy. Lead with <b>malignant recall</b>, plus balanced accuracy and macro-F1.</td></tr>
    <tr><td class="num">3</td><td>11-class imbalance is 280:1</td><td>Class weighting, oversampling, or merge the rare tail. Report per-class recall.</td></tr>
    <tr><td class="num">4</td><td>Two modalities with different pixel statistics</td><td>Treat <code>image_type</code> explicitly — stratify, condition, or use two heads.</td></tr>
    <tr><td class="num">5</td><td>AKIEC straddles Indeterminate and Malignant</td><td>Pick one target explicitly; do not derive the 3-class label from the 11-class code.</td></tr>
    <tr><td class="num">6</td><td>Site missing 37%, melanocytic missing 77%</td><td>Metadata needs an explicit "unknown" category, not mode imputation.</td></tr>
    <tr><td class="num">7</td><td>61% of lesions are one skin tone class</td><td>Report per-skin-tone metrics; aggregates hide group failures.</td></tr>
    <tr><td class="num">8</td><td>Images vary in resolution</td><td>Resize/crop is mandatory; document the target resolution.</td></tr>
    <tr><td class="num">9</td><td>Dataset is biopsy-referred, not population-representative</td><td>Every performance claim must state the population it applies to.</td></tr>
  </table>

</section>

<!-- ============ PAGE — MONET ============ -->
<section class="page">
  <h2>12 · MONET concept scores — a sanity check that the data behaves</h2>
  {fig("fig15_monet_by_class.png", "Mean MONET concept score per class, dermoscopic images. Colour = z-score across classes.")}
  <p>Read row by row: VASC scores highest on <i>vasculature/vessels</i>, INF on <i>erythema</i>,
  NV on <i>pigmented</i>. The concepts behave the way a dermatologist would expect, which is good
  evidence they carry real signal. Note also that <i>vasculature</i> averages 0.22 on dermoscopic
  images versus 0.02 on clinical ones — exactly what a dermatoscope is for.</p>

  <h3>Next session</h3>
  <p>Image representation and normalisation, then a first baseline — built on a grouped split,
  with malignant recall as the metric that counts.</p>

  <div class="footer">
    Repository: {REPO_URL}<br>
    Dataset: MILK10k, ISIC Archive (CC BY-NC) · https://api.isic-archive.com/doi/milk10k/ ·
    Description: https://doi.org/10.1016/j.jid.2025.06.1594
  </div>
</section>

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
