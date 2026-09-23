# %% [markdown]
# ## A3.3 Why StratifiedGroupKFold?
#
# Three 5-fold strategies, scored on the same data: `GroupKFold` (groups, no
# stratification), `StratifiedKFold` (stratification, no groups) and
# `StratifiedGroupKFold` (both). Each is judged on three things — how many lesions leak
# across the fold boundary, how far any class's share drifts between validation folds,
# and how many folds see no `MAL_OTH` at all (9 lesions in the whole dataset).
#
# **Read the grain before reading the table.** (i) and (iii) cut the *lesion* table, where
# a lesion is one row, so a lesion cannot be on both sides of a fold: their leak count is
# 0 *by construction*, not by luck. (ii) cuts the *image* table, where each lesion is two
# rows — that is exactly where the leak appears, and refusing to run it on images would
# hide the failure we are testing for. To keep the comparison apples-to-apples, all three
# are then scored at **image** level: each lesion-level fold assignment is pushed down to
# the images that inherit it, so every strategy is measured on the same 10,480 rows a
# model would actually train on.

# %%
# The two grains, side by side, so the row counts in the comparison are unambiguous.
lesions = data.build_lesion_table()
image_df = splits.image_table()

per_lesion_images = image_df.groupby("lesion_id").size()
assert len(lesions) == 5240 and len(image_df) == 10480
assert (per_lesion_images == 2).all(), "the 2-images-per-lesion assumption has broken"

print(f"lesion table : {len(lesions):,} rows  (one per lesion)")
print(f"image table  : {len(image_df):,} rows  (exactly 2 per lesion)")

comparison = splits.compare_strategies(
    lesions, image_df, n_splits=5, seed=config.SEED, rare_class="MAL_OTH"
)
print()
print(comparison.to_string(index=False))

# %% [markdown]
# The two lesion-level strategies are tied at 0 leaks, so the leak column cannot separate
# them — the class-proportion spread and the rare-class coverage have to. Below, the folds
# are recomputed from scratch and the per-fold composition is printed, so every number in
# the summary table can be checked by hand instead of taken on trust.

# %%
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

# Recomputed independently of splits.compare_strategies: if the per-fold tables below
# reproduce its summary numbers (asserted further down), both are right.
X_LESION = np.zeros((len(lesions), 1))
X_IMAGE = np.zeros((len(image_df), 1))


def image_folds_from_lesions(splitter):
    """Fold index per image, inherited from the image's lesion."""
    fold_of_lesion = pd.Series(-1, index=lesions["lesion_id"].to_numpy())
    for k, (_, held) in enumerate(
        splitter.split(X_LESION, lesions["dx"], lesions["lesion_id"])
    ):
        fold_of_lesion.iloc[held] = k
    return image_df["lesion_id"].map(fold_of_lesion).rename("fold")


def image_folds_from_images(splitter):
    """Fold index per image, assigned directly to the image."""
    fold_of_image = pd.Series(-1, index=image_df.index, name="fold")
    for k, (_, held) in enumerate(splitter.split(X_IMAGE, image_df["dx"])):
        fold_of_image.iloc[held] = k
    return fold_of_image


def fold_composition(fold_of_image):
    """(counts, shares in %) of every class in every validation fold."""
    counts = pd.crosstab(fold_of_image, image_df["dx"])
    return counts, counts.div(counts.sum(axis=1), axis=0) * 100


group_folds = image_folds_from_lesions(
    GroupKFold(n_splits=5, shuffle=True, random_state=config.SEED)
)
group_counts, group_shares = fold_composition(group_folds)

print("GroupKFold — images per class in each validation fold")
print(group_counts.to_string())
print("\nsame thing as a share of the fold (%)")
print(group_shares.round(2).to_string())

group_spread = (group_shares.max(axis=0) - group_shares.min(axis=0)).sort_values(
    ascending=False
)
print("\nspread = max fold share - min fold share, percentage points")
print(group_spread.round(3).to_string())

reported = comparison.set_index("strategy").loc["GroupKFold", "max_class_spread_pp"]
assert np.isclose(group_spread.max(), reported), "detail disagrees with the summary table"
print(f"\nworst class: {group_spread.idxmax()} at {group_spread.max():.2f} pp "
      f"({group_shares[group_spread.idxmax()].min():.2f}% to "
      f"{group_shares[group_spread.idxmax()].max():.2f}% of a fold)")
print(f"MAL_OTH images per fold: {group_counts['MAL_OTH'].tolist()} "
      f"-> absent from {(group_counts['MAL_OTH'] == 0).sum()} of 5 folds")

# %%
# Strategy (ii): the leak, shown rather than asserted.
strat_folds = image_folds_from_images(
    StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
)
folds_per_lesion = (
    image_df.assign(fold=strat_folds.to_numpy()).groupby("lesion_id")["fold"].nunique()
)
leaked = folds_per_lesion[folds_per_lesion > 1]

print(f"StratifiedKFold: {len(leaked):,} of {len(folds_per_lesion):,} lesions "
      f"({100 * len(leaked) / len(folds_per_lesion):.1f}%) have their two images "
      f"in different folds")
# A lesion survives only if its second image lands in the same fold as the first:
# chance 1/5, so ~80% leakage is the expected outcome, not a bug in the splitter.
print(f"expected under random assignment: {100 * (1 - 1 / 5):.1f}%")

example = image_df[image_df["lesion_id"] == leaked.index[0]].assign(
    fold=strat_folds[image_df["lesion_id"] == leaked.index[0]].to_numpy()
)
print("\none leaked lesion — same skin, same day, two folds:")
print(example[["isic_id", "lesion_id", "image_type", "dx", "fold"]].to_string(index=False))

# %%
# Strategy (iii), and why its spread cannot be driven to zero.
sgk_folds = image_folds_from_lesions(
    StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.SEED)
)
sgk_counts, sgk_shares = fold_composition(sgk_folds)
sgk_spread = (sgk_shares.max(axis=0) - sgk_shares.min(axis=0)).max()

print("StratifiedGroupKFold — images per class in each validation fold")
print(sgk_counts.to_string())
print(f"\nMAL_OTH lesions per fold: {(sgk_counts['MAL_OTH'] // 2).tolist()} "
      f"(9 in total) -> absent from {(sgk_counts['MAL_OTH'] == 0).sum()} of 5 folds")

# The smallest move a splitter can make sets a floor on the spread: one image for an
# image-level splitter, one whole lesion (= 2 images) for a lesion-level one.
fold_size = int(sgk_counts.sum(axis=1).min())
print(f"\nfold size: {fold_size:,} images")
print(f"floor at image grain  (1 image  / fold): {100 / fold_size:.3f} pp")
print(f"floor at lesion grain (1 lesion / fold): {200 / fold_size:.3f} pp")
print(comparison[["strategy", "split_unit", "max_class_spread_pp"]].to_string(index=False))

# %% [markdown]
# Both stratified strategies sit exactly on their granularity floor: `StratifiedKFold`
# reaches 0.048 pp, which is one image out of a 2,096-image fold, and
# `StratifiedGroupKFold` reaches 0.095 pp, which is one *lesion* — the smallest unit it is
# allowed to move. Neither is balancing better than the other; they are both perfect, and
# the factor of two is the price of keeping a lesion whole. `GroupKFold` is 39x worse than
# that floor at 3.72 pp, because nothing in it looks at the label.
#
# The honest caveat: `StratifiedKFold`'s 0.048 pp and its clean `MAL_OTH` coverage are
# real, and taken alone they look like the best row in the table. They are worthless
# anyway, because 79.4% of lesions have a near-duplicate photo on the other side of the
# boundary — the validation score it produces measures memorisation of lesions.

# %% [markdown]
# **Answer (3 lines).**
#
# 1. I would use **StratifiedGroupKFold**: it is the only strategy with 0 leaked lesions *and* balanced folds (worst class spread 0.095 pp, the one-lesion floor) *and* `MAL_OTH` in all 5 validation folds.
# 2. `GroupKFold` is also leak-free but ignores the label, so class shares drift by up to 3.72 pp (BKL 8.40% to 12.12%) and `MAL_OTH` is missing from 1 of 5 folds.
# 3. `StratifiedKFold` balances best (0.048 pp) but puts the two images of 4,158 of 5,240 lesions (79.4%) in different folds, so its validation score rewards memorising lesions and is optimistic.
