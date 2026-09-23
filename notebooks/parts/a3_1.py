# %% [markdown]
# ## A3.1 Measure the leak
#
# - Counting leaked lesions is easy; I want to see what the leak does to an actual metric.
# - Setup:
#   - Random Forest on **metadata only** (age, sex, site, image type) → `diagnosis_1`
#   - `min_samples_leaf=1`, so it *can* memorise rows, which is exactly what a leak rewards
#   - everything at **image** level, because that's where the leak happens (two photos = two rows)
#   - imputation and encoding inside a `Pipeline`, so they're fitted on train only

# %%
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = config.PRIMARY_LABEL_COL                       # "diagnosis_1"
NUMERIC = ["age_approx"]
CATEGORICAL = ["sex", "anatom_site_general", "image_type"]
FEATURES = NUMERIC + CATEGORICAL

N_SPLITS = 5                     # grouped scheme: 1 fold out of 5 is the test set
TEST_SIZE = 1 / N_SPLITS         # naive scheme holds out the same 20%
SEEDS = [0, 1, 2, 3, 4]

meta = data.load_metadata()
images = meta[["isic_id", "lesion_id", *FEATURES, TARGET]].copy()

# The structural facts the rest of the section depends on. If any of these ever
# stopped holding, the leak story would be a different story.
assert len(images) == 10_480, f"expected 10,480 images, got {len(images):,}"
assert images["lesion_id"].nunique() == 5_240
assert (images.groupby("lesion_id").size() == 2).all(), "not 2 images per lesion"
# The oracle below only works because the label does not vary within a lesion:
# the sibling image literally carries the answer.
assert (images.groupby("lesion_id")[TARGET].nunique() == 1).all()

# derm_id / clinical_id side by side is exactly the sibling lookup we need.
lesion_table = data.build_lesion_table()
_sib = dict(zip(lesion_table["derm_id"], lesion_table["clinical_id"]))
_sib.update(dict(zip(lesion_table["clinical_id"], lesion_table["derm_id"])))
sibling_of = images["isic_id"].map(_sib)
assert sibling_of.notna().all(), "some image has no sibling"

X = images[FEATURES]
y = images[TARGET]
label_of = images.set_index("isic_id")[TARGET]

print(f"images {len(images):,} | lesions {images['lesion_id'].nunique():,}")
print(f"features {FEATURES} -> target {TARGET!r}")
print(f"missing: age {int(X['age_approx'].isna().sum())}, "
      f"site {int(X['anatom_site_general'].isna().sum())}")
print(f"held out per scheme: {TEST_SIZE:.0%} | seeds {SEEDS}")

# %% [markdown]
# ### Two splits, three measurements
#
# - **(a) naive:** `train_test_split` on the 10,480 images, stratified on `diagnosis_1`, ignoring `lesion_id`
# - **(b) grouped:** `StratifiedGroupKFold` with `groups=lesion_id`; first fold = test (same 20%)
#
# On each I measure:
# 1. the Random Forest's balanced accuracy
# 2. how many test lesions also appear in train
# 3. a **sibling oracle**: predict each test image with the label of the lesion's other photo if it's in train, otherwise the majority class. It's not a model, just a way to measure how much free information the split gives away.

# %%
def make_model(seed: int) -> Pipeline:
    """Metadata-only Random Forest, with all fitting done inside the pipeline."""
    preprocess = ColumnTransformer([
        ("age", SimpleImputer(strategy="median"), NUMERIC),
        # A missing body site is itself informative (3,912 images have none), so
        # it becomes its own level instead of being imputed towards a real site.
        ("cat", Pipeline([
            ("fill", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])
    return Pipeline([
        ("pre", preprocess),
        # min_samples_leaf=1: the forest is allowed to memorise. If the leak can
        # be exploited by this feature set, this model is the one that would.
        ("rf", RandomForestClassifier(n_estimators=200, min_samples_leaf=1,
                                      random_state=seed, n_jobs=-1)),
    ])


def naive_split(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Row-level stratified split — the mistake, reproduced faithfully."""
    positions = np.arange(len(images))
    return train_test_split(positions, test_size=TEST_SIZE,
                            random_state=seed, stratify=y)


def grouped_split(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Lesion-grouped stratified split — both photos always move together."""
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    return next(sgkf.split(np.zeros((len(images), 1)), y, images["lesion_id"]))


def sibling_oracle(train_pos: np.ndarray, test_pos: np.ndarray) -> tuple[np.ndarray, float]:
    """Answer each test image from its sibling in train; else guess the majority.

    Returns the predictions and the share of test images whose sibling is in
    train — the fraction of the test set that is answerable for free.
    """
    train_ids = set(images["isic_id"].to_numpy()[train_pos])
    majority = y.iloc[train_pos].value_counts().idxmax()

    siblings = sibling_of.iloc[test_pos]
    available = siblings.isin(train_ids)
    predictions = np.where(available, siblings.map(label_of).to_numpy(), majority)
    return predictions, float(available.mean())


records = []
for scheme, splitter in [("naive image-level", naive_split),
                         ("grouped by lesion", grouped_split)]:
    for seed in SEEDS:
        train_pos, test_pos = splitter(seed)

        model = make_model(seed).fit(X.iloc[train_pos], y.iloc[train_pos])
        rf_ba = balanced_accuracy_score(y.iloc[test_pos],
                                        model.predict(X.iloc[test_pos]))

        train_lesions = set(images["lesion_id"].to_numpy()[train_pos])
        test_lesions = set(images["lesion_id"].to_numpy()[test_pos])
        shared = train_lesions & test_lesions

        oracle_pred, sibling_rate = sibling_oracle(train_pos, test_pos)
        oracle_ba = balanced_accuracy_score(y.iloc[test_pos], oracle_pred)

        records.append({
            "scheme": scheme, "seed": seed,
            "n_test_images": len(test_pos),
            "n_test_lesions": len(test_lesions),
            "leaked_lesions": len(shared),
            "leaked_pct": 100 * len(shared) / len(test_lesions),
            "sibling_in_train_pct": 100 * sibling_rate,
            "rf_balanced_acc": rf_ba,
            "oracle_balanced_acc": oracle_ba,
        })

per_seed = pd.DataFrame(records)
print(per_seed.round(4).to_string(index=False))

# %% [markdown]
# ### Mean ± std over five seeds

# %%
def mean_sd(frame: pd.DataFrame, column: str, digits: int = 3) -> pd.Series:
    """Collapse the five seeds of a column into one 'mean ± sd' string."""
    stats = frame.groupby("scheme", sort=False)[column].agg(["mean", "std"])
    return stats.apply(
        lambda r: f"{r['mean']:.{digits}f} ± {r['std']:.{digits}f}", axis=1
    )


summary = pd.DataFrame({
    "test images": per_seed.groupby("scheme", sort=False)["n_test_images"].mean()
        .map(lambda v: f"{v:,.0f}"),
    "test lesions": per_seed.groupby("scheme", sort=False)["n_test_lesions"].mean()
        .map(lambda v: f"{v:,.0f}"),
    "leaked lesions": mean_sd(per_seed, "leaked_lesions", digits=1),
    "leaked %": mean_sd(per_seed, "leaked_pct", digits=1),
    "sibling in train %": mean_sd(per_seed, "sibling_in_train_pct", digits=1),
    "RF balanced acc": mean_sd(per_seed, "rf_balanced_acc"),
    "oracle balanced acc": mean_sd(per_seed, "oracle_balanced_acc"),
})
print(summary.to_string())

naive = per_seed[per_seed["scheme"] == "naive image-level"]
grouped = per_seed[per_seed["scheme"] == "grouped by lesion"]

rf_gap = naive["rf_balanced_acc"].mean() - grouped["rf_balanced_acc"].mean()
oracle_gap = naive["oracle_balanced_acc"].mean() - grouped["oracle_balanced_acc"].mean()
# Four decimals, because the whole point is that the first number is tiny and
# rounding it to three would hide even its sign.
print(f"\nRandom Forest, naive minus grouped : {rf_gap:+.4f} balanced-accuracy points")
print(f"Sibling oracle, naive minus grouped: {oracle_gap:+.4f} balanced-accuracy points")
print(f"chance level (3 classes)           :  {1 / y.nunique():.4f}")

# The grouped split must leak nothing at all — that is the whole point of it.
assert grouped["leaked_lesions"].max() == 0
assert grouped["sibling_in_train_pct"].max() == 0
# With no sibling ever available the oracle predicts one constant class, so its
# balanced accuracy is exactly 1/len(classes). Verify rather than assert-by-prose.
assert np.allclose(grouped["oracle_balanced_acc"], 1 / y.nunique())
print(f"grouped-split oracle collapses to the constant "
      f"{y.iloc[grouped_split(0)[0]].value_counts().idxmax()!r} "
      f"-> balanced accuracy exactly 1/{y.nunique()} = {1 / y.nunique():.3f}")

# %% [markdown]
# ### What the leak does to a weak model vs. what it makes possible

# %%
schemes = ["naive image-level", "grouped by lesion"]
means = {
    "Random Forest (metadata only)": [
        per_seed.loc[per_seed["scheme"] == s, "rf_balanced_acc"].mean() for s in schemes],
    "Sibling oracle": [
        per_seed.loc[per_seed["scheme"] == s, "oracle_balanced_acc"].mean() for s in schemes],
}
errors = {
    "Random Forest (metadata only)": [
        per_seed.loc[per_seed["scheme"] == s, "rf_balanced_acc"].std() for s in schemes],
    "Sibling oracle": [
        per_seed.loc[per_seed["scheme"] == s, "oracle_balanced_acc"].std() for s in schemes],
}

fig, ax = plt.subplots(figsize=(7.2, 4.0))
positions = np.arange(len(schemes))
width = 0.36
for offset, (name, colour) in zip(
    (-width / 2, width / 2),
    [("Random Forest (metadata only)", config.GREY),
     ("Sibling oracle", config.ORANGE)],
):
    ax.bar(positions + offset, means[name], width, yerr=errors[name], capsize=4,
           color=colour, label=name)
    # Put each value label above the top of its error bar, not on it.
    for x, value, err in zip(positions + offset, means[name], errors[name]):
        ax.text(x, value + err + 0.02, f"{value:.3f}", ha="center", fontsize=9)

# The chance line goes in the legend: a text label next to it collides with the bars.
ax.axhline(1 / y.nunique(), color="#C1272D", linestyle="--", linewidth=1,
           label="chance (1/3)")
ax.set_xticks(positions, schemes)
ax.set_ylabel("balanced accuracy on the test part")
ax.set_ylim(0, 1.05)
ax.set_title("The leak barely moves a weak model — and hands an oracle the answer")
ax.legend(loc="upper right")
plt.show()

# %% [markdown]
# **My answer**
#
# - **Did the leak inflate the Random Forest? No.** 0.426 ± 0.004 naive vs 0.426 ± 0.003 grouped (difference −0.0001), even though 88.8% ± 0.7 of naive test lesions are also in train.
# - **Why not, I think:** age, sex and site are the same on both photos of a lesion, so the leaked sibling adds nothing the forest couldn't learn from similar patients. And at 0.426 vs 0.333 chance, there's hardly anything to memorise anyway.
# - **The oracle shows the real potential:** 79.8% ± 1.2 of naive test images have their sibling in train, and just copying its label gives **0.847 ± 0.022**, vs **0.333** with the grouped split (sibling rate 0%, it always says "Malignant").
# - **What that means for a model that can recognise the lesion** (e.g. a CNN seeing two photos of the same skin): from what I understood, it could turn the leak into almost free correct answers. The leak is there; this weak model just can't use it.
# - **Why "no inflation measured" is not a reason to skip the grouped split:** I only tested one weak model. The split belongs to the experiment, not the model, and it has to hold for the Milestone 2 CNN too — the model most able to exploit the leak. And I only get one honest look at the test set.

