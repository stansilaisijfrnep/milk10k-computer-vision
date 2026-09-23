"""Augmentation and normalisation pipeline for the MILK10k images.

Milestone 1 (B7) and homework A3.5 both need the same thing: one place that
says how a JPEG on disk becomes a tensor a network can read. It lives in a
module and not in a notebook cell because Milestone 2 fine-tunes a pretrained
backbone with exactly these transforms — an augmentation that only exists in a
notebook cannot be re-used, diffed or defended later.

There are two pipelines and the difference between them is the entire point:

* :func:`train_transform` is deliberately random. Every epoch the network sees a
  slightly different view of the same lesion, which is how a dataset of only
  5,240 lesions avoids being memorised outright.
* :func:`eval_transform` is deliberately deterministic. A validation or test
  number has to be a property of the model, not of the random draw that happened
  during scoring, so the only operations there are a resize and a fixed
  normalisation.

Picking the augmentations is a *medical* decision rather than a default: an
operation is admissible only if a dermatologist would give the transformed image
the same diagnosis as the original. Rotations and flips pass that test, because
a dermatoscope is pressed onto the skin at whatever angle is convenient and skin
has no canonical "up". Note explicitly that the usual vertical-flip caveat does
not apply here: in a chest X-ray a vertical flip moves the heart to the wrong
side and invents situs inversus, whereas a mole is the same mole upside down.
Colour is the opposite case — pigmentation and erythema *are* the diagnosis — so
the colour jitter below is far weaker than the ImageNet defaults everyone copies.
:data:`AUGMENTATION_TABLE` records that reasoning one row per operation, so the
report states why each transform is safe instead of just listing it.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from . import config


# --- Reproducibility -------------------------------------------------------
def seed_everything(seed: int = config.SEED) -> None:
    """Seed every random number generator this project can reach.

    The augmentations, the weighted sampler and the weight initialisation all
    draw from these three generators, so without a single seeding entry point a
    re-run produces different numbers and no claim in the report is checkable.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # Seeds every visible device; harmless on the CPU-only machines we use.
        torch.cuda.manual_seed_all(seed)


# --- Augmentation constants ------------------------------------------------
# Named once here so the pipeline and AUGMENTATION_TABLE below can never drift
# apart: the table is built from these same values.

ROTATION_DEGREES = 20
FLIP_PROB = 0.5

# Rotating a rectangle leaves empty corners. Filling them with black would be a
# bad idea in dermatology: dark, sharply bounded regions at the edge of the
# frame are exactly what a pigmented border or a dermatoscope vignette looks
# like. A mid grey is unambiguously not skin, and after ImageNet normalisation
# it sits close to the channel means, so the corners contribute almost nothing.
ROTATION_FILL = (128, 128, 128)

# Mild on purpose. The lesion border — asymmetry and edge irregularity, the A
# and B of ABCD — is diagnostic, and an aggressive crop can cut it out of frame
# while leaving the label unchanged, which teaches the network a lie.
CROP_SCALE = (0.85, 1.0)

# Every MILK10k image is 600x450, i.e. 4:3, and both pipelines have to end at a
# square 224x224. We squash rather than centre-crop, so that no part of the
# lesion border is ever thrown away. The crop aspect band is therefore centred
# on the native 4:3 and not on 1:1, so a training crop is squashed by the same
# factor as an evaluation frame and the network sees no train/test mismatch.
NATIVE_ASPECT_RATIO = 600 / 450
CROP_ASPECT_RANGE = (NATIVE_ASPECT_RATIO * 0.9, NATIVE_ASPECT_RATIO / 0.9)

# Deliberately conservative. Exercise A3.5 measures the real colour gap between
# benign and malignant lesions; the jitter has to stay well below it, otherwise
# augmentation moves an image across the very boundary the model is learning. A
# hue jitter of 0.5 (180 degrees) turns skin green and is indefensible here.
JITTER_BRIGHTNESS = 0.2
JITTER_CONTRAST = 0.2
JITTER_SATURATION = 0.1
JITTER_HUE = 0.02

# Bilinear resampling everywhere: nearest-neighbour would alias the fine pigment
# network and dotted vessels that dermoscopy grades lesions on.
INTERPOLATION = T.InterpolationMode.BILINEAR


# --- Pipelines -------------------------------------------------------------
def _random_ops(image_size: int = config.IMAGE_SIZE) -> list:
    """The random, still-viewable half of the training pipeline.

    Kept separate from :func:`train_transform` for one practical reason: the B7
    figure and the A3.5 audit need to *look* at augmented images, which is only
    possible before ``ToTensor``/``Normalize`` turn them into normalised floats.
    Defining the random operations once means the figure cannot show a pipeline
    that differs from the one actually used for training.

    The order matters. Rotation comes first, on the full 600x450 frame, so the
    grey corners it creates sit far out at the edge and most crops never touch
    them; cropping first and rotating second would drag those corners into the
    middle of the 224x224 input.
    """
    return [
        T.RandomRotation(
            degrees=ROTATION_DEGREES,
            interpolation=INTERPOLATION,
            fill=ROTATION_FILL,
        ),
        T.RandomResizedCrop(
            image_size,
            scale=CROP_SCALE,
            ratio=CROP_ASPECT_RANGE,
            interpolation=INTERPOLATION,
        ),
        T.RandomHorizontalFlip(p=FLIP_PROB),
        T.RandomVerticalFlip(p=FLIP_PROB),
        T.ColorJitter(
            brightness=JITTER_BRIGHTNESS,
            contrast=JITTER_CONTRAST,
            saturation=JITTER_SATURATION,
            hue=JITTER_HUE,
        ),
    ]


def train_transform(
    image_size: int = config.IMAGE_SIZE,
    mean: tuple[float, float, float] = config.IMAGENET_MEAN,
    std: tuple[float, float, float] = config.IMAGENET_STD,
) -> T.Compose:
    """The training pipeline: geometric and photometric jitter, then normalise.

    Each call returns a fresh ``Compose``; the randomness lives inside the
    transforms and is drawn from torch's global generator, so
    :func:`seed_everything` controls it. Use this for the training split only —
    applying it at evaluation time would make the reported score depend on which
    random crops happened to be drawn.
    """
    return T.Compose([*_random_ops(image_size), T.ToTensor(), T.Normalize(mean, std)])


def eval_transform(
    image_size: int = config.IMAGE_SIZE,
    mean: tuple[float, float, float] = config.IMAGENET_MEAN,
    std: tuple[float, float, float] = config.IMAGENET_STD,
) -> T.Compose:
    """The validation/test pipeline: resize, to tensor, normalise. Nothing else.

    Every operation here is a deterministic function of the input pixels, so the
    same image always yields the same tensor and a score is reproducible to the
    last digit. The resize is to a fixed square rather than a short-side resize
    plus centre crop, because a centre crop of a 4:3 frame discards a quarter of
    the width and can clip the lesion border that carries the diagnosis.
    :func:`assert_eval_deterministic` checks the no-randomness claim rather than
    trusting it.
    """
    return T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=INTERPOLATION),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


# --- The augmentation table (B7 / A3.5 deliverable) ------------------------
# One row per operation in train_transform. Each justification answers the only
# question that matters for a medical augmentation: why can this operation not
# change the correct diagnosis?

AUGMENTATION_TABLE: list[dict] = [
    {
        "augmentation": "RandomRotation",
        "parameters": (
            f"degrees=±{ROTATION_DEGREES}, fill={ROTATION_FILL}, bilinear"
        ),
        "justification": (
            "A dermatoscope is placed on the skin at whatever angle is "
            "convenient, so the rotation of a lesion in the frame is an "
            "artefact of the photographer and carries no diagnostic "
            "information; the grey fill is used instead of black so the empty "
            "corners cannot be mistaken for pigment or lens vignetting."
        ),
    },
    {
        "augmentation": "RandomResizedCrop",
        "parameters": (
            f"size={config.IMAGE_SIZE}, scale={CROP_SCALE}, "
            f"ratio=({CROP_ASPECT_RANGE[0]:.2f}, {CROP_ASPECT_RANGE[1]:.2f})"
        ),
        "justification": (
            "Working distance and zoom vary between clinics, so mild rescaling "
            "is a real-world nuisance rather than a diagnostic feature; the "
            "scale floor is held at 0.85 because a harder crop would cut the "
            "asymmetric, irregular lesion border out of frame while leaving the "
            "malignant label attached to it."
        ),
    },
    {
        "augmentation": "RandomHorizontalFlip",
        "parameters": f"p={FLIP_PROB}",
        "justification": (
            "Skin has no canonical left or right and a lesion mirrored about "
            "the vertical axis is the same lesion with the same ABCD features, "
            "so the label is invariant to the flip."
        ),
    },
    {
        "augmentation": "RandomVerticalFlip",
        "parameters": f"p={FLIP_PROB}",
        "justification": (
            "The caveat that rules vertical flips out for chest X-rays — where "
            "flipping moves the heart to the wrong side and invents a pathology "
            "— does not apply to dermoscopy, because a lesion photographed "
            "upside down is diagnostically identical to the same lesion "
            "photographed the right way up."
        ),
    },
    {
        "augmentation": "ColorJitter",
        "parameters": (
            f"brightness={JITTER_BRIGHTNESS}, contrast={JITTER_CONTRAST}, "
            f"saturation={JITTER_SATURATION}, hue={JITTER_HUE}"
        ),
        "justification": (
            "Illumination, white balance and camera model differ between the "
            "clinics that contributed MILK10k, so small photometric shifts are "
            "acquisition noise; hue is capped at 0.02 (about 7 degrees) because "
            "pigmentation and erythema are themselves the diagnosis and a "
            "larger shift would move a lesion across the benign/malignant "
            "colour gap that exercise A3.5 measures."
        ),
    },
    {
        "augmentation": "ToTensor",
        "parameters": "HWC uint8 PIL -> CHW float32 in [0, 1]",
        "justification": (
            "A pure container and dtype change: dividing every pixel by 255 "
            "rescales the image without altering any ratio between pixels, so "
            "no visual feature is added or destroyed."
        ),
    },
    {
        "augmentation": "Normalize",
        "parameters": (
            f"mean={config.IMAGENET_MEAN}, std={config.IMAGENET_STD} (ImageNet)"
        ),
        "justification": (
            "A fixed per-channel affine rescaling, identical for every image "
            "and therefore diagnosis-preserving by construction; the ImageNet "
            "statistics are used because Milestone 2 fine-tunes a pretrained "
            "backbone whose filters expect inputs in that distribution."
        ),
    },
]


def augmentation_table() -> pd.DataFrame:
    """Return :data:`AUGMENTATION_TABLE` as a frame for the report.

    Building it from the same constants the pipeline uses means the table in the
    report cannot quietly describe a pipeline the code no longer runs.
    """
    return pd.DataFrame(
        AUGMENTATION_TABLE, columns=["augmentation", "parameters", "justification"]
    )


# --- Figure and audit helpers ----------------------------------------------
def augmentation_grid(
    img: Image.Image,
    n: int = 7,
    seed: int = config.SEED,
) -> list[tuple[str, Image.Image]]:
    """The original image plus ``n`` augmented versions, as viewable PIL images.

    This is the B7 figure: it shows what the network is actually trained on, so
    an augmentation that destroys the lesion is visible rather than hypothetical.
    Only the random half of the pipeline is applied — after ``Normalize`` the
    pixels are signed floats and no longer displayable.
    """
    pipeline = T.Compose(_random_ops())

    # fork_rng restores the global torch generator afterwards, so redrawing a
    # report figure cannot shift the random stream of a training run.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        augmented = [(f"augmented {i + 1}", pipeline(img)) for i in range(n)]

    return [("original", img), *augmented]


def colorjitter_probe(
    img: Image.Image,
    hue: float = 0.0,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    seed: int | None = None,
) -> Image.Image:
    """Apply one ColorJitter with the given strengths and return the result.

    Every parameter defaults to zero (i.e. identity) so exercise A3.5 can turn a
    single knob at a time and measure how far that knob alone moves the mean
    colour of a lesion. That measurement is what justifies the conservative
    ``hue=0.02`` above: the jitter has to stay small next to the real
    benign-versus-malignant colour difference, and this is how we check it.
    """
    jitter = T.ColorJitter(
        brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
    )

    if seed is None:
        return jitter(img)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return jitter(img)


# --- Tests -----------------------------------------------------------------
def assert_eval_deterministic(
    img: Image.Image, image_size: int = config.IMAGE_SIZE
) -> bool:
    """Assert that :func:`eval_transform` maps one image to one exact tensor.

    Worth testing rather than assuming: a single stray random operation in the
    evaluation path would make every reported metric depend on the run, and the
    failure is silent because the numbers still look plausible. Raises
    ``AssertionError`` if the two passes differ, otherwise returns ``True``.
    """
    transform = eval_transform(image_size)
    first = transform(img)
    second = transform(img)

    assert torch.equal(first, second), (
        "eval_transform is not deterministic: two passes over the same image "
        "produced different tensors, so some random operation leaked into the "
        "evaluation path."
    )
    return True
