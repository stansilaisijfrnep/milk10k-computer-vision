"""Guards for the dataset layer in ``milk10k.datasets`` (homework A3.6).

The dataset layer is where the pipeline's quiet mistakes live. None of the
properties tested here announce themselves when they break:

* A dataset that hands back the wrong label for a lesion still trains happily;
  it just learns a permutation of the classes, and the only symptom is a
  confusion matrix that looks "a bit odd" weeks later.
* ``LesionDataset`` applies the transform to each view separately on purpose.
  If the two views ever shared one random draw, nothing would raise — the
  model would simply see half the augmentation diversity it is documented to
  see, and the augmentation study in the report would describe something that
  never happened.
* A missing JPEG that is skipped instead of raised silently shrinks the
  dataset, so the run trains on a dataset nobody documented.
* ``aggregate_predictions`` turns per-image probabilities into the per-lesion
  answer the clinical question is actually asked in. Get the averaging wrong
  and every reported lesion-level metric is wrong, while every shape still
  matches and no exception is thrown.

So each test asserts the *value*, not just the shape. Everything runs on a
small subset of the real data with ``num_workers=0``: the notebook does the
800-lesion timing run, a test only has to prove the behaviour, and a test slow
enough to skip is a test that does not guard anything.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

# Make the package importable when pytest is started without PYTHONPATH=src,
# mirroring what tests/test_integration.py does.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milk10k import config, data, datasets, labels, transforms  # noqa: E402

# Subset sizes. 120 lesions is enough for the class-count check to be
# meaningful (it spans all three primary classes) and small enough that one
# epoch is a couple of seconds instead of a couple of minutes.
N_EPOCH_LESIONS = 120
N_BATCH_LESIONS = 16
LESION_BATCH = 8


# --- Fixtures ---------------------------------------------------------------
@pytest.fixture(scope="module")
def lesion_table() -> pd.DataFrame:
    """The 5,240-lesion table, loaded once for the whole module.

    The committed ``artifacts/tables/lesions.csv`` is preferred over rebuilding
    it, because that file is what the splits and the class weights were derived
    from: reading it means these tests describe the artefacts the report cites,
    not a freshly built table that merely ought to be identical.
    """
    if config.LESION_TABLE_CSV.exists():
        return pd.read_csv(config.LESION_TABLE_CSV)
    return data.build_lesion_table()


@pytest.fixture(scope="module")
def small_lesions(lesion_table: pd.DataFrame) -> pd.DataFrame:
    """A fixed random subset of lesions, drawn once with the project seed.

    Random rather than the first N rows: the table is sorted by ``lesion_id``,
    and a head slice of a sorted table is not representative of the class mix.
    ``random_state=config.SEED`` keeps it reproducible, so a failure here can
    be reproduced exactly.
    """
    return (
        lesion_table.sample(n=N_EPOCH_LESIONS, random_state=config.SEED)
        .reset_index(drop=True)
    )


@pytest.fixture(scope="module")
def train_images() -> pd.DataFrame:
    """A handful of rows from the committed IMAGE-level train split."""
    return pd.read_csv(config.TRAIN_SPLIT_CSV).head(8)


@pytest.fixture(scope="module")
def primary_map() -> dict[str, int]:
    """The 3-class name -> integer map the datasets are expected to use."""
    return labels.label_map("primary")


def _loader(dataset, batch_size: int = LESION_BATCH) -> DataLoader:
    """A deterministic, single-process loader — the only kind a test should use.

    ``shuffle=False`` so a batch can be compared against the table row by row,
    and ``num_workers=0`` because worker processes on macOS are spawned, which
    would make a fast test slow and a failure hard to read.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# --- LesionDataset: both views of one lesion --------------------------------
def test_lesion_dataset_batch_shapes(small_lesions: pd.DataFrame) -> None:
    """A batch of 8 lesions is (8, 3, 224, 224) *per view*, plus label and id."""
    dataset = datasets.LesionDataset(small_lesions.head(N_BATCH_LESIONS))
    batch = next(iter(_loader(dataset)))

    expected = (LESION_BATCH, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    for view in ("derm", "clinical"):
        assert view in batch, f"the batch has no '{view}' view: keys {list(batch)}"
        assert tuple(batch[view].shape) == expected, (
            f"view '{view}': expected {expected}, got {tuple(batch[view].shape)}"
        )
        assert batch[view].dtype == torch.float32

    # int64 because that is what the classification losses index with.
    assert tuple(batch["label"].shape) == (LESION_BATCH,)
    assert batch["label"].dtype == torch.int64
    assert len(batch["lesion_id"]) == LESION_BATCH


def test_lesion_ids_match_the_label_table(
    small_lesions: pd.DataFrame, primary_map: dict[str, int]
) -> None:
    """The ids and labels coming out of the loader are the table's, in order.

    This is the check that catches an off-by-one between the id array and the
    target array — the failure that trains a model on shuffled labels while
    every shape and count stays exactly right.
    """
    subset = small_lesions.head(N_BATCH_LESIONS)
    dataset = datasets.LesionDataset(subset)

    seen_ids: list[str] = []
    seen_labels: list[int] = []
    for batch in _loader(dataset):
        seen_ids.extend(batch["lesion_id"])
        seen_labels.extend(int(v) for v in batch["label"])

    assert seen_ids == subset["lesion_id"].astype(str).tolist()
    expected_labels = [primary_map[name] for name in subset[config.PRIMARY_LABEL_COL]]
    assert seen_labels == expected_labels

    # And the integers really do mean what label_map says they mean.
    names_back = [dataset.inverse_label_map[i] for i in seen_labels]
    assert names_back == subset[config.PRIMARY_LABEL_COL].tolist()


def test_views_are_transformed_independently(small_lesions: pd.DataFrame) -> None:
    """Each view gets its own random draw, not a shared one.

    Both views of a lesion are pointed at the SAME file on purpose. Any
    difference between the two tensors can then only come from the transform
    drawing separate parameters, which is exactly the documented behaviour;
    comparing the two genuinely different photographs would prove nothing,
    since those differ whatever the transform does. The eval case is the
    control: with a deterministic transform the same file must give the same
    tensor twice.
    """
    same_file = small_lesions.head(4).copy()
    same_file["clinical_id"] = same_file["derm_id"]

    transforms.seed_everything(config.SEED)
    augmented = datasets.LesionDataset(
        same_file, transform=transforms.train_transform()
    )[0]
    assert not torch.equal(augmented["derm"], augmented["clinical"]), (
        "the two views of one lesion came out identical under a random "
        "transform, so both views share a single draw of the augmentation "
        "parameters and the model sees half the diversity it is documented to."
    )

    control = datasets.LesionDataset(
        same_file, transform=transforms.eval_transform()
    )[0]
    assert torch.equal(control["derm"], control["clinical"]), (
        "the control failed: the same file gave two different tensors under "
        "the deterministic transform, so the test above proved nothing."
    )


def test_single_view_subset(small_lesions: pd.DataFrame) -> None:
    """``views=("derm",)`` yields that view only — the single-view ablation."""
    dataset = datasets.LesionDataset(small_lesions.head(N_BATCH_LESIONS), views=("derm",))

    item = dataset[0]
    assert set(item) == {"derm", "label", "lesion_id"}, sorted(item)
    assert tuple(item["derm"].shape) == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)

    batch = next(iter(_loader(dataset)))
    assert "clinical" not in batch
    assert tuple(batch["derm"].shape) == (
        LESION_BATCH, 3, config.IMAGE_SIZE, config.IMAGE_SIZE,
    )

    # An empty view list is a caller error and must not silently yield labels
    # with no pixels attached to them.
    with pytest.raises(ValueError):
        datasets.LesionDataset(small_lesions.head(2), views=())


# --- Missing files: loud, or explicitly opted out of ------------------------
def test_missing_file_raises(train_images: pd.DataFrame, small_lesions: pd.DataFrame) -> None:
    """A missing JPEG raises and names the id; ``allow_missing`` drops and counts.

    The default has to be the crash. A dataset that quietly skips unreadable
    files trains on fewer images than the report claims, and nothing in the
    output ever says so.
    """
    bogus = "ISIC_0000000"
    assert not (config.IMG_DIR / f"{bogus}.jpg").exists(), "pick a different fake id"

    real = train_images.head(4)
    with_bogus = pd.concat([real, real.head(1).assign(isic_id=bogus)], ignore_index=True)

    strict = datasets.MilkImageDataset(with_bogus)
    assert len(strict) == 5, "allow_missing=False must not drop anything up front"
    with pytest.raises(FileNotFoundError, match=re.escape(bogus)):
        strict[4]

    lenient = datasets.MilkImageDataset(with_bogus, allow_missing=True)
    assert len(lenient) == 4, "the row with no file should have been dropped"
    assert lenient.n_dropped == 1
    assert bogus not in set(lenient.isic_ids)

    # LesionDataset has no opt-out at all, and its message names the lesion,
    # the view and the id, so the offending row can be found in the table.
    broken = small_lesions.head(1).copy()
    broken["clinical_id"] = bogus
    with pytest.raises(FileNotFoundError, match=re.escape(bogus)):
        datasets.LesionDataset(broken)[0]


# --- MilkImageDataset: one image per item -----------------------------------
def test_image_dataset_returns_id(
    train_images: pd.DataFrame, primary_map: dict[str, int]
) -> None:
    """Items are ``(tensor, int, isic_id)`` — the id is what error analysis needs."""
    dataset = datasets.MilkImageDataset(train_images)
    item = dataset[0]

    assert isinstance(item, tuple) and len(item) == 3
    image, label, isic_id = item

    assert isinstance(image, torch.Tensor)
    assert tuple(image.shape) == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert image.dtype == torch.float32

    # A plain Python int, not a 0-d tensor or a numpy scalar: the default
    # collate turns these into the int64 target tensor the loss expects.
    assert isinstance(label, int) and not isinstance(label, bool)
    assert label == primary_map[train_images[config.PRIMARY_LABEL_COL].iloc[0]]

    assert isinstance(isic_id, str)
    assert isic_id == str(train_images["isic_id"].iloc[0])


def test_class_counts_over_one_epoch(
    small_lesions: pd.DataFrame, primary_map: dict[str, int]
) -> None:
    """One full epoch returns each class exactly as often as the subset holds it.

    Counting what actually comes out of the loader — rather than trusting the
    table — is what catches a dataset that drops the short last batch, skips a
    row, or encodes a class it does not know about.
    """
    dataset = datasets.LesionDataset(small_lesions)

    seen: dict[int, int] = {}
    n_items = 0
    for batch in _loader(dataset, batch_size=16):
        for label in batch["label"].tolist():
            seen[label] = seen.get(label, 0) + 1
        n_items += len(batch["lesion_id"])

    assert n_items == N_EPOCH_LESIONS, "one epoch did not return every lesion once"

    expected = {
        primary_map[name]: int(count)
        for name, count in small_lesions[config.PRIMARY_LABEL_COL]
        .value_counts()
        .items()
    }
    assert seen == expected, f"loader saw {seen}, subset holds {expected}"

    # The subset has to be worth counting: a single-class subset would pass the
    # assertion above while proving nothing about the encoding.
    assert len(expected) >= 2, f"subset spans only {len(expected)} class(es)"


# --- Image predictions -> lesion predictions --------------------------------
def test_aggregate_predictions() -> None:
    """Averaging two views into one lesion prediction, on hand-computed numbers.

    Three cases, each chosen because it distinguishes probability averaging
    from something that would look correct on easy inputs:

    1. Both views agree -> the lesion keeps that class.
    2. The views disagree, and the average picks a class NEITHER view called.
       Benign 0.50 and Malignant 0.55 are each one view's argmax, but
       Indeterminate is second on both, so averaging carries it to 0.425 and it
       wins. A hard vote over the two views could never produce this answer,
       which is exactly why the module averages probabilities instead.
    3. A lesion with a single view -> its probabilities pass through unchanged
       and ``n_views`` records that only one image was behind the row.
    """
    # Columns are config.PRIMARY_LABELS: Benign, Indeterminate, Malignant.
    image_probs = np.array(
        [
            [0.10, 0.10, 0.80],   # lesion A, derm      -> Malignant
            [0.20, 0.10, 0.70],   # lesion A, clinical  -> Malignant
            [0.50, 0.40, 0.10],   # lesion B, derm      -> Benign
            [0.00, 0.45, 0.55],   # lesion B, clinical  -> Malignant
            [0.05, 0.15, 0.80],   # lesion C, derm only -> Malignant
        ]
    )
    image_table = pd.DataFrame(
        {
            "isic_id": ["ISIC_A1", "ISIC_A2", "ISIC_B1", "ISIC_B2", "ISIC_C1"],
            "lesion_id": ["IL_A", "IL_A", "IL_B", "IL_B", "IL_C"],
        }
    )

    out = datasets.aggregate_predictions(image_probs, image_table)

    assert list(out.columns) == [
        "lesion_id", "n_views", "Benign", "Indeterminate", "Malignant",
        "pred_index", "pred",
    ]
    assert out["lesion_id"].tolist() == ["IL_A", "IL_B", "IL_C"]
    assert out["n_views"].tolist() == [2, 2, 1]

    expected_probs = np.array(
        [
            [0.15, 0.100, 0.750],   # (0.10+0.20)/2, (0.10+0.10)/2, (0.80+0.70)/2
            [0.25, 0.425, 0.325],   # (0.50+0.00)/2, (0.40+0.45)/2, (0.10+0.55)/2
            [0.05, 0.150, 0.800],   # one view: passed through
        ]
    )
    np.testing.assert_allclose(
        out[list(config.PRIMARY_LABELS)].to_numpy(), expected_probs, atol=1e-12
    )

    assert out["pred"].tolist() == ["Malignant", "Indeterminate", "Malignant"]
    assert out["pred_index"].tolist() == [2, 1, 2]

    # pred_index must index the same class list the label map does, otherwise
    # predictions and targets would be compared in two different orderings.
    assert [labels.inverse_label_map("primary")[i] for i in out["pred_index"]] == (
        out["pred"].tolist()
    )
