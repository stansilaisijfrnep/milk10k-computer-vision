"""Tests for the Session 2 modules: pipeline (Parts 3-4), viz (Part 5), eda2 statistics (Part 1).

Fast by design — a handful of real images, no full-dataset passes.

    PYTHONPATH=src python -m pytest tests/test_session2.py -q
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy import stats  # noqa: E402

from milk10k import config, data, eda2, pipeline, viz  # noqa: E402

pytestmark = pytest.mark.skipif(not config.METADATA_CSV.exists(), reason="dataset not downloaded")


@pytest.fixture(scope="module")
def meta() -> pd.DataFrame:
    return data.load_metadata()


@pytest.fixture(scope="module")
def some_ids(meta) -> list[str]:
    return meta.groupby(config.PRIMARY_LABEL_COL).head(2)["isic_id"].tolist()


# --- Part 3 ----------------------------------------------------------------
def test_process_image_minmax_shape_range_and_record(some_ids):
    r = pipeline.process_image(some_ids[0], channels_first=True)
    assert r.image.shape == r.shape == (3, 224, 224)
    assert r.image.dtype == np.float32
    assert 0.0 <= r.value_range[0] <= r.value_range[1] <= 1.0
    assert r.original_size == (600, 450)
    assert any("resize 600x450 -> 224x224" in s for s in r.steps)
    assert r.source == some_ids[0]


def test_grayscale_matches_opencv(some_ids):
    gray = pipeline.process_image(some_ids[0], color_space="gray", normalize="none").image[..., 0]
    rgb = pipeline.process_image(some_ids[0], normalize="none").image.astype(np.uint8)
    assert np.abs(gray - cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)).max() <= 1.0


def test_zscore_with_own_stats_is_standardised(some_ids):
    raw = pipeline.process_image(some_ids[0], normalize="minmax").image
    mean, std = raw.reshape(-1, 3).mean(0), raw.reshape(-1, 3).std(0)
    z = pipeline.process_image(some_ids[0], normalize="zscore", mean=mean, std=std).image
    np.testing.assert_allclose(z.reshape(-1, 3).mean(0), 0, atol=1e-3)  # float32 summation
    np.testing.assert_allclose(z.reshape(-1, 3).std(0), 1, atol=1e-3)


def test_same_pixels_as_training_resize(some_ids):
    """The pipeline must not drift from preprocessing.preprocess_image."""
    from milk10k import preprocessing
    ours = pipeline.process_image(some_ids[0]).image
    theirs = preprocessing.preprocess_image(some_ids[0])
    np.testing.assert_array_equal(ours, theirs)


def test_array_input_roundtrip(some_ids):
    r = pipeline.process_image(some_ids[0], size=None, normalize="none")
    again = pipeline.process_image(r.image.astype(np.uint8), size=None, normalize="none")
    np.testing.assert_array_equal(r.image, again.image)


def test_bad_config_raises_instead_of_skipping(some_ids):
    with pytest.raises(ValueError):
        pipeline.process_batch(some_ids, color_space="cmyk")
    with pytest.raises(ValueError):
        pipeline.process_image(some_ids[0], normalize="zscore", mean=[0.5], std=[0.2])  # 1 value, 3 channels


def test_batch_skips_and_reports_bad_files(some_ids, tmp_path):
    truncated = tmp_path / "ISIC_TRUNC01.jpg"
    truncated.write_bytes((config.IMG_DIR / f"{some_ids[0]}.jpg").read_bytes()[:2000])
    not_image = tmp_path / "ISIC_TEXT001.jpg"
    not_image.write_text("hello")
    res = pipeline.process_batch([some_ids[0], truncated, not_image, "ISIC_0000000", some_ids[1]])
    assert res.ids == [some_ids[0], some_ids[1]]
    assert res.images.shape == (2, 224, 224, 3)
    assert {name for name, _ in res.skipped} == {"ISIC_TRUNC01", "ISIC_TEXT001", "ISIC_0000000"}
    assert all(":" in reason for _, reason in res.skipped)
    assert res.n_requested == 5


def test_batch_accepts_dataframe(meta):
    res = pipeline.process_batch(meta.head(3), color_space="gray")
    assert res.images.shape == (3, 224, 224, 1) and res.ids == meta.head(3)["isic_id"].tolist()


# --- Part 4 ----------------------------------------------------------------
def test_loader_filters_missing_and_covers_every_image(meta):
    table = pd.concat([meta.head(37), pd.DataFrame([{"isic_id": "ISIC_0000000",
                                                     config.PRIMARY_LABEL_COL: "Benign"}])])
    loader = pipeline.BatchLoader(table, batch_size=10)
    assert loader.n_missing == 1 and loader.missing_ids == ["ISIC_0000000"]
    batches = list(loader)
    assert len(batches) == len(loader) == 4
    assert [len(b) for b in batches] == [10, 10, 10, 7]
    assert sorted(i for b in batches for i in b.ids) == sorted(meta.head(37)["isic_id"])
    b = batches[0]
    assert b.images.shape == (10, 3, 224, 224) and b.labels.dtype == np.int64
    assert [loader.classes[i] for i in b.labels] == b.label_names


def test_loader_drop_last_and_shuffle_reproducible(meta):
    a = pipeline.BatchLoader(meta.head(25), batch_size=10, shuffle=True, seed=1, drop_last=True)
    b = pipeline.BatchLoader(meta.head(25), batch_size=10, shuffle=True, seed=1, drop_last=True)
    assert len(a) == 2
    first_a, first_b = [x.ids for x in a], [x.ids for x in b]
    assert first_a == first_b                       # same seed, same order
    assert [x.ids for x in a] != first_a            # next epoch reshuffles


def test_loader_rejects_unknown_labels(meta):
    bad = meta.head(3).assign(**{config.PRIMARY_LABEL_COL: "Unknown"})
    with pytest.raises(ValueError):
        pipeline.BatchLoader(bad)


# --- Part 5 ----------------------------------------------------------------
def test_to_displayable_handles_every_layout():
    rng = np.random.default_rng(0)
    u8 = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    chw_z = rng.normal(size=(3, 8, 8)).astype(np.float32)
    gray = rng.random((1, 8, 8)).astype(np.float32)
    for arr in (u8, chw_z, gray):
        out = viz.to_displayable(arr)
        assert out.min() >= 0 and out.max() <= 1
    assert viz.to_displayable(chw_z).shape == (8, 8, 3)
    assert viz.to_displayable(gray).shape == (8, 8)
    np.testing.assert_allclose(viz.to_displayable(u8), u8 / 255.0, atol=1e-6)


def test_to_displayable_undoes_zscore(some_ids):
    mm = pipeline.process_image(some_ids[0]).image
    mean, std = pipeline.default_zscore_stats()
    z = pipeline.process_image(some_ids[0], normalize="zscore").image
    np.testing.assert_allclose(viz.to_displayable(z, mean, std), np.clip(mm, 0, 1), atol=1e-5)


def test_visualisers_accept_a_loader_batch(meta):
    batch = next(iter(pipeline.BatchLoader(meta.head(6), batch_size=6)))
    for fig in (viz.show_image_grid(batch), viz.show_batch(batch, n=4),
                viz.plot_batch_summary(batch), viz.plot_class_balance(meta.head(50), group_col="lesion_id")):
        assert fig.axes
    matplotlib.pyplot.close("all")


# --- Part 1 statistics -----------------------------------------------------
def test_cramers_v_matches_hand_computation():
    df = pd.DataFrame({"x": ["a"] * 30 + ["b"] * 30,
                       config.PRIMARY_LABEL_COL: ["Benign"] * 20 + ["Malignant"] * 10
                       + ["Benign"] * 5 + ["Malignant"] * 25})
    res = eda2.categorical_association(df, "x")
    chi2 = stats.chi2_contingency([[20, 10], [5, 25]], correction=False)[0]
    assert res["statistic"] == pytest.approx(chi2)
    assert res["effect"] == pytest.approx(np.sqrt(chi2 / 60))


def test_eta_squared_matches_definition():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"v": np.r_[rng.normal(0, 1, 50), rng.normal(1, 1, 50)],
                       config.PRIMARY_LABEL_COL: ["Benign"] * 50 + ["Malignant"] * 50})
    res = eda2.numeric_association(df, "v")
    g = df.groupby(config.PRIMARY_LABEL_COL)["v"]
    ss_b = (g.size() * (g.mean() - df.v.mean()) ** 2).sum()
    assert res["eta_squared"] == pytest.approx(ss_b / ((df.v - df.v.mean()) ** 2).sum())


def test_image_type_is_independent_of_class_by_construction(meta):
    assert eda2.categorical_association(meta, "image_type")["effect"] == pytest.approx(0, abs=1e-12)


def test_every_metadata_column_has_a_reviewed_role(meta):
    inv = eda2.metadata_inventory(meta)
    assert "unreviewed" not in set(inv["role"])
    assert set(inv.query("unit == 'image'")["column"]) == {"image_type", "image_manipulation"}
