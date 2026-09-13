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
