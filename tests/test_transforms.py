"""Guards for the image pipeline in ``milk10k.transforms`` (Milestone 1, B7).

Everything checked here fails *silently* when it breaks, which is the only
reason it is worth spending a test file on.

* A random operation leaking into :func:`~milk10k.transforms.eval_transform`
  does not raise. The model still trains, the validation score still prints,
  and it is simply a different number on every run — so a 2-point improvement
  from a real change becomes indistinguishable from the noise of the scoring
  pass. ``test_eval_transform_is_deterministic`` is the B7 requirement and the
  reason this file exists.
* The mirror image is just as quiet: a training pipeline whose randomness has
  been switched off (or whose seeding is not reproducible) still trains, still
  reports numbers, and only shows up as an overfitting gap nobody can explain.
* A forgotten ``Normalize`` is the third silent one. A pretrained backbone fed
  inputs in [0, 1] instead of ImageNet-standardised values converges to
  something mediocre rather than failing, so it is caught here by checking the
  value range instead of by reading the ``Compose`` and hoping.

These run against a real MILK10k JPEG, not a synthetic array, because the
pipeline's job is to turn *that file* into a tensor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torchvision.transforms as T
from PIL import Image

# Make the package importable when pytest is started without PYTHONPATH=src,
# mirroring what tests/test_integration.py does.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milk10k import config, preprocessing, splits, transforms  # noqa: E402


# --- Fixtures ---------------------------------------------------------------
@pytest.fixture(scope="module")
def lesion_image() -> Image.Image:
    """One real MILK10k image, opened once for the whole module.

    The id is the alphabetically first one in the committed train split rather
    than "row 0", so the test uses the same file whatever order the CSV happens
    to be written in, and a failure is reproducible from the id alone.
    """
    train = splits.load_split("train")
    isic_id = sorted(train["isic_id"].astype(str))[0]
    return preprocessing.load_image(isic_id)


# --- The B7 requirement: evaluation must be deterministic -------------------
def test_eval_transform_is_deterministic(lesion_image: Image.Image) -> None:
    """The same image must map to the exact same tensor, every single time.

    Checked three ways, because "deterministic" can fail at two different
    levels: the pipeline could draw a random number per call (same object,
    two calls), or it could freeze a random choice at construction time (two
    objects built from the same code). Exact equality, not ``allclose`` — a
    resize and an affine rescale are pure functions of the input pixels, so
    there is no floating-point slack to allow for.
    """
    pipeline = transforms.eval_transform()

    first = pipeline(lesion_image)
    second = pipeline(lesion_image)
    assert torch.equal(first, second), (
        "eval_transform() gave two different tensors for the same image: a "
        "random operation has leaked into the evaluation path, which makes "
        "every reported validation and test number depend on the run."
    )

    # A freshly constructed pipeline must agree with the first one too.
    assert torch.equal(first, transforms.eval_transform()(lesion_image)), (
        "two separately constructed eval pipelines disagree on the same image, "
        "so something is being sampled when the Compose is built."
    )

    # The module ships its own assertion helper; the report cites it, so it has
    # to agree with the test rather than merely exist.
    assert transforms.assert_eval_deterministic(lesion_image) is True


def test_eval_transform_output_shape_and_dtype(lesion_image: Image.Image) -> None:
    """The contract with the model: one CHW float32 tensor at the input size."""
    tensor = transforms.eval_transform()(lesion_image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE), (
        f"expected (3, {config.IMAGE_SIZE}, {config.IMAGE_SIZE}), "
        f"got {tuple(tensor.shape)}"
    )
    assert tensor.dtype == torch.float32
    # A NaN here would propagate into the loss and show up only as a dead run.
    assert torch.isfinite(tensor).all()


# --- Training augmentation: random, but reproducibly so ---------------------
def test_train_transform_is_random(lesion_image: Image.Image) -> None:
    """Two passes over one image must differ — that is what augmentation is.

    Seeded first so the test itself is deterministic: the assertion is about
    the pipeline drawing *new* parameters per call, not about which parameters
    it happened to draw.
    """
    pipeline = transforms.train_transform()

    transforms.seed_everything(config.SEED)
    first = pipeline(lesion_image)
    second = pipeline(lesion_image)

    assert not torch.equal(first, second), (
        "train_transform() returned identical tensors for two passes over the "
        "same image, so the augmentation is not actually being applied and "
        "every epoch shows the network the exact same 7,334 pictures."
    )
    assert first.shape == second.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)


def test_train_transform_is_seed_reproducible(lesion_image: Image.Image) -> None:
    """Random, yes; unrepeatable, no. ``seed_everything`` must pin the draw.

    This is what makes a reported number checkable: re-running the pipeline
    from the same seed has to reproduce the same augmented inputs. It also
    proves the randomness comes from torch's global generator rather than from
    per-object state, which is what lets one ``seed_everything`` call at the
    top of a run control the whole epoch.
    """
    pipeline = transforms.train_transform()

    transforms.seed_everything(config.SEED)
    first = pipeline(lesion_image)

    transforms.seed_everything(config.SEED)
    second = pipeline(lesion_image)

    assert torch.equal(first, second), (
        "the same seed produced two different augmented tensors: the training "
        "pipeline draws from a generator seed_everything() does not control."
    )

    # Same seed, a different Compose object: still identical.
    transforms.seed_everything(config.SEED)
    third = transforms.train_transform()(lesion_image)
    assert torch.equal(first, third)


def test_augmentation_table_covers_the_pipeline() -> None:
    """Every operation the pipeline runs must be a row in the reported table.

    The table is the B7 / A3.5 deliverable and is what gets defended out loud,
    so the failure mode it has to be protected from is documentation drift: an
    augmentation added to the code and not to the table, or a row describing an
    operation that was removed months ago.
    """
    pipeline_ops = [type(op).__name__ for op in transforms.train_transform().transforms]
    table = transforms.augmentation_table()

    assert list(table["augmentation"]) == pipeline_ops, (
        "augmentation_table() no longer matches train_transform(): table says "
        f"{list(table['augmentation'])}, pipeline runs {pipeline_ops}"
    )

    for row in table.itertuples(index=False):
        assert isinstance(row.parameters, str) and row.parameters.strip(), (
            f"{row.augmentation}: the table records no parameters"
        )
        # A justification has to be a reason, so a placeholder word is not
        # allowed to pass a test whose whole point is that the reason exists.
        assert isinstance(row.justification, str) and len(row.justification.strip()) > 40, (
            f"{row.augmentation}: justification is empty or too short to be a "
            f"reason — got {row.justification!r}"
        )


# --- Normalisation ----------------------------------------------------------
def test_normalisation_is_applied(lesion_image: Image.Image) -> None:
    """The eval tensor must be ImageNet-standardised, not left in [0, 1].

    Milestone 2 fine-tunes a pretrained backbone, whose first-layer filters
    expect inputs in the distribution ImageNet was standardised to. Skipping
    ``Normalize`` does not crash anything; it just costs accuracy quietly. So
    we assert both halves of the claim: the tensor has left [0, 1], and
    undoing the affine map lands back exactly on the un-normalised pixels.
    """
    tensor = transforms.eval_transform()(lesion_image)

    in_unit_range = float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0
    assert not in_unit_range, (
        "the eval tensor still lies inside [0, 1], so Normalize did not run: "
        f"min {float(tensor.min()):.3f}, max {float(tensor.max()):.3f}"
    )

    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
    recovered = tensor * std + mean

    assert float(recovered.min()) >= -1e-5, float(recovered.min())
    assert float(recovered.max()) <= 1 + 1e-5, float(recovered.max())

    # Stronger than a range check: the recovered tensor must equal the same
    # pipeline with Normalize removed, which pins the exact statistics used.
    unnormalised = T.Compose(
        [
            T.Resize(
                (config.IMAGE_SIZE, config.IMAGE_SIZE),
                interpolation=transforms.INTERPOLATION,
            ),
            T.ToTensor(),
        ]
    )(lesion_image)
    assert torch.allclose(recovered, unnormalised, atol=1e-6), (
        "un-normalising the eval tensor does not reproduce the raw [0, 1] "
        "image, so Normalize is using statistics other than config's."
    )
