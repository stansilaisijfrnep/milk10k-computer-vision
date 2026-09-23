"""Project-wide paths, constants and plotting style for the MILK10k EDA."""

from pathlib import Path

import matplotlib as mpl

# --- Paths -----------------------------------------------------------------
# config.py -> milk10k -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "milk10k"
IMG_DIR = DATA_DIR / "images"
SUPP_DIR = DATA_DIR / "supplements"

METADATA_CSV = DATA_DIR / "metadata.csv"
TRAINING_GT_CSV = SUPP_DIR / "training_gt.csv"
TRAINING_INPUT_CSV = SUPP_DIR / "training_input.csv"
TRAINING_SUPP_CSV = SUPP_DIR / "training_supp.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# --- Label schemes ---------------------------------------------------------
# Primary target (metadata.csv): coarse clinical triage label.
PRIMARY_CLASSES = ["Benign", "Malignant", "Indeterminate"]

# Stretch target (training_gt.csv): 11-class one-hot, one row per lesion.
DIAGNOSIS_CODES = {
    "BCC": "Basal cell carcinoma",
    "NV": "Nevus",
    "BKL": "Benign keratosis-like lesion",
    "SCCKA": "Squamous cell carcinoma / keratoacanthoma",
    "MEL": "Melanoma",
    "AKIEC": "Actinic keratosis / intraepithelial carcinoma",
    "DF": "Dermatofibroma",
    "INF": "Inflammatory / infectious",
    "VASC": "Vascular lesion",
    "BEN_OTH": "Other benign",
    "MAL_OTH": "Other malignant",
}

# How the 11 codes sit inside the 3-class target. Verified against the data in
# `check_label_hierarchy` — AKIEC is genuinely split across Indeterminate and
# Malignant, so it gets its own bucket instead of being forced either way.
MALIGNANT_CODES = {"BCC", "SCCKA", "MEL", "MAL_OTH"}
BENIGN_CODES = {"NV", "BKL", "DF", "INF", "VASC", "BEN_OTH"}
MIXED_CODES = {"AKIEC"}


def code_group(code: str) -> str:
    """Map an 11-class code to Benign / Malignant / Indeterminate for colouring."""
    if code in MALIGNANT_CODES:
        return "Malignant"
    if code in BENIGN_CODES:
        return "Benign"
    return "Indeterminate"

# Fitzpatrick-style skin tone class in training_input.csv is 0-5 in the file.
SKIN_TONE_LABELS = {
    0: "0 (unknown/other)",
    1: "I",
    2: "II",
    3: "III",
    4: "IV",
    5: "V",
}

# --- Plotting style --------------------------------------------------------
ORANGE = "#F08200"  # EADA accent, used as the primary series colour
GREY = "#4D4D4D"
BENIGN_MALIGNANT_COLORS = {
    "Benign": "#2E8B84",
    "Malignant": "#C1272D",
    "Indeterminate": "#F0A202",
}


def apply_style() -> None:
    """Apply a consistent, readable matplotlib style across all figures."""
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#BFBFBF",
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.labelcolor": GREY,
            "text.color": GREY,
            "xtick.color": GREY,
            "ytick.color": GREY,
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def ensure_dirs() -> None:
    """Create the report output folders if they do not exist yet."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Sessions 2-3 homework / Milestone 1 additions
# ===========================================================================
# Everything below is the single configuration point for the pipeline: paths,
# the seed, the input resolution. No module hard-codes any of these.

import os

# --- Data location ---------------------------------------------------------
# The image folder can be overridden without touching code, which is what makes
# the notebook portable (see README). Everything else is derived from the repo.
IMG_DIR = Path(os.environ.get("MILK10K_IMAGES_DIR", IMG_DIR))

# --- Reproducibility -------------------------------------------------------
SEED = 42                 # every split, sampler and torch generator uses this
SPLIT_DATE = "2026-09-22"  # date the committed splits were created

# --- Model input -----------------------------------------------------------
IMAGE_SIZE = 224          # every MILK10k image is 600x450, so this downscales
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# ImageNet statistics: Milestone 2 starts from a pretrained backbone, so the
# inputs must match the distribution those weights were trained on. The
# train-split statistics are computed and stored too (norm_stats.json) so the
# choice can be revisited with evidence rather than by default.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Generated output ------------------------------------------------------
# artifacts/  = machine-readable files that LATER milestones load
# reports/    = files a HUMAN reads
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SPLITS_DIR = ARTIFACTS_DIR / "splits"
ARTIFACT_TABLES_DIR = ARTIFACTS_DIR / "tables"

TRAIN_SPLIT_CSV = SPLITS_DIR / "train.csv"
VAL_SPLIT_CSV = SPLITS_DIR / "val.csv"
TEST_SPLIT_CSV = SPLITS_DIR / "test.csv"

LABEL_MAP_JSON = ARTIFACTS_DIR / "label_map.json"
CLASS_WEIGHTS_JSON = ARTIFACTS_DIR / "class_weights.json"
NORM_STATS_JSON = ARTIFACTS_DIR / "norm_stats.json"
LESION_TABLE_CSV = ARTIFACT_TABLES_DIR / "lesions.csv"
IMAGE_SIZE_SUMMARY_CSV = ARTIFACT_TABLES_DIR / "image_size_summary.csv"

MILESTONE1_FIGURES_DIR = FIGURES_DIR / "milestone1"
QUALITY_REPORT_MD = REPORTS_DIR / "data_quality_report.md"
QUALITY_REPORT_CSV = ARTIFACT_TABLES_DIR / "data_quality_report.csv"

# --- Label strategy (decided in B3, see docs/label_strategy.md) ------------
# Primary target: diagnosis_1 kept as THREE classes. "Indeterminate" is a real
# clinical category (the pathologist could not commit), not noise to be merged.
PRIMARY_LABEL_COL = "diagnosis_1"
PRIMARY_LABELS = ["Benign", "Indeterminate", "Malignant"]   # index = integer label

# Stretch target: all 11 codes kept, rare tail handled with weights + sampler
# rather than by merging, so the confusion matrix stays diagnostically readable.
STRETCH_LABEL_COL = "dx"
STRETCH_LABELS = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
                  "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]

# Columns that must never be model inputs: they encode the label or are only
# knowable after the biopsy that produced the label (B4).
LEAKY_COLUMNS = [
    "diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_4",
    "diagnosis_confirm_type", "diagnosis_full", "invasion_thickness_interval",
    "melanocytic", "concomitant_biopsy", "lesion_id",
]


def ensure_project_dirs() -> None:
    """Create every generated-output folder. Safe to call repeatedly."""
    for d in (FIGURES_DIR, TABLES_DIR, ARTIFACTS_DIR, SPLITS_DIR,
              ARTIFACT_TABLES_DIR, MILESTONE1_FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
