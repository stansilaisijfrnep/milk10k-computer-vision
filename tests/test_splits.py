"""Property tests for :func:`milk10k.splits.split_lesions` (exercise A3.2).

A split is not the kind of thing one example can validate. ``seed=42`` producing
3,667 / 786 / 787 disjoint lesions proves that *seed 42* works; it says nothing
about whether the function is correct. So these tests state the three properties
the rest of the pipeline actually relies on and check them across ten seeds:

1. the three splits are pairwise disjoint and together cover every lesion,
2. each split's size is within +-1 percentage point of what was requested,
3. the same seed twice gives identical output.

The same three checks, with the same seeds, run inside the notebook section
``notebooks/parts/a3_2.py``; this file is the version that runs in CI without a
kernel. Run it with::

    PYTHONPATH=src python -m pytest tests/test_splits.py -v

Note on the sklearn ``UserWarning`` about the least populated class having 9
members: that is true and expected — ``MAL_OTH`` has 9 lesions and stage A asks
for 10 folds — and it is deliberately not silenced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Importable when run directly from the repo root, with or without PYTHONPATH=src.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from milk10k import config, data, splits  # noqa: E402  (after the sys.path fix)

# The project seed plus nine others. Ten is enough to catch a seed-dependent bug
# while keeping the file fast enough that people actually run it.
SEEDS = [config.SEED, *range(9)]

# One percentage point. The splitter works in whole folds, so it cannot hit a
# requested proportion exactly; anything beyond a point would mean the fold
# arithmetic in splits.fold_plan is wrong, not merely rounded.
TOLERANCE_PP = 1.0


@pytest.fixture(scope="module")
def lesions():
    """The A1.3 lesion table: 5,240 rows, one per lesion. Built once for the file."""
    table = data.build_lesion_table()
    assert table["lesion_id"].is_unique, "lesion table is not one row per lesion"
    return table


@pytest.fixture(scope="module")
def splits_by_seed(lesions):
    """``{seed: (train_ids, val_ids, test_ids)}`` — computed once, shared by the tests."""
    return {seed: splits.split_lesions(lesions, seed=seed) for seed in SEEDS}


@pytest.mark.parametrize("seed", SEEDS)
def test_splits_are_disjoint(lesions, splits_by_seed, seed):
    """No lesion may appear in two splits, and none may be dropped.

    Disjointness alone is not enough: a function that returned three empty lists
    would pass it. The partition check — union equals the input, and the sizes
    add up — is what makes the property meaningful.
    """
    train_ids, val_ids, test_ids = splits_by_seed[seed]
    train, val, test = set(train_ids), set(val_ids), set(test_ids)

    assert not train & val, f"seed {seed}: {len(train & val)} lesions in both train and val"
    assert not train & test, f"seed {seed}: {len(train & test)} lesions in both train and test"
    assert not val & test, f"seed {seed}: {len(val & test)} lesions in both val and test"

    assert train | val | test == set(lesions["lesion_id"]), (
        f"seed {seed}: the three splits do not cover every lesion"
    )
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(lesions), (
        f"seed {seed}: a lesion is listed twice inside a single split"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_split_sizes_within_tolerance(lesions, splits_by_seed, seed):
    """Every split must land within +-1 pp of the requested proportion."""
    train_ids, val_ids, test_ids = splits_by_seed[seed]
    n_total = len(lesions)
    requested = {
        "train": 1.0 - config.VAL_SIZE - config.TEST_SIZE,
        "val": config.VAL_SIZE,
        "test": config.TEST_SIZE,
    }
    realised = {"train": train_ids, "val": val_ids, "test": test_ids}

    for name, ids in realised.items():
        deviation_pp = abs(len(ids) / n_total - requested[name]) * 100
        assert deviation_pp <= TOLERANCE_PP, (
            f"seed {seed}: {name} is {len(ids) / n_total:.2%}, "
            f"requested {requested[name]:.2%} ({deviation_pp:.2f} pp off)"
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_split_is_reproducible(lesions, splits_by_seed, seed):
    """The same seed must give the identical three lists, in the same order.

    Order matters as well as membership: the split files are committed, and a
    function that returned the same lesions in a different order every run would
    produce a spurious diff and destroy the provenance the manifest claims.
    """
    assert splits.split_lesions(lesions, seed=seed) == splits_by_seed[seed], (
        f"seed {seed}: two calls with the same seed disagreed"
    )


@pytest.mark.parametrize("seed", SEEDS[:3])
def test_different_seeds_give_different_splits(lesions, splits_by_seed, seed):
    """Reproducibility must not be reproducibility by accident.

    ``StratifiedGroupKFold`` ignores ``random_state`` unless ``shuffle=True``. If
    someone dropped that flag, every test above would still pass while the
    multi-seed study silently became one split repeated ten times. This is the
    test that would catch it.
    """
    others = [s for s in SEEDS if s != seed]
    test_sets = {frozenset(splits_by_seed[s][2]) for s in others}
    assert frozenset(splits_by_seed[seed][2]) not in test_sets, (
        f"seed {seed} produced the same test set as another seed — is shuffle=True set?"
    )
