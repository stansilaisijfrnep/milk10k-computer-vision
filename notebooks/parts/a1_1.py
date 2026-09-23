# %% [markdown]
# ## A1.1 Cross-check the two label files
#
# `metadata.csv` (one row per **image**) and `training_gt.csv` (one row per **lesion**)
# come from different parts of the MILK study pipeline, and every later step assumes they
# describe the same universe and agree with each other. Below, five of those assumptions
# are written as executable checks, run against the real data, and then re-run against
# deliberately corrupted copies to show that each check can actually fail. A check that
# can only pass is decoration.

# %%
meta = data.load_metadata()
gt = data.load_training_gt()

ONEHOT_COLS = [c for c in gt.columns if c != "lesion_id"]

print(f"metadata.csv     {meta.shape[0]:>6,} rows x {meta.shape[1]:>2} cols   "
      f"{meta['isic_id'].nunique():,} images, {meta['lesion_id'].nunique():,} lesions")
print(f"training_gt.csv  {gt.shape[0]:>6,} rows x {gt.shape[1]:>2} cols   "
      f"{gt['lesion_id'].nunique():,} lesions, {len(ONEHOT_COLS)} one-hot columns")
print(f"one-hot columns: {ONEHOT_COLS}")

# The grain is the thing to get right before anything else: a per-image row count of
# 10,480 next to a per-lesion row count of 5,240 is the whole reason this cross-check
# exists. Assert the identifiers are what we think they are, not just the shapes.
assert meta["isic_id"].is_unique, "isic_id repeats — metadata is not one row per image"
assert gt["lesion_id"].is_unique, "lesion_id repeats — gt is not one row per lesion"
assert set(ONEHOT_COLS) == set(config.STRETCH_LABELS), "unexpected one-hot columns"

# %% [markdown]
# ### The five checks
#
# Each check is a function of `(meta, gt)` returning `(passed, detail)`. Writing them as
# functions rather than inline assertions is what makes the corruption test in the last
# cell possible — the same code has to be runnable on data we know is broken.

# %%
# Fields that describe the lesion, not the photograph, and must therefore be identical
# on both of a lesion's two rows. image_type is deliberately absent: it is the one field
# that is *supposed* to differ.
LESION_CONSTANT_FIELDS = [
    "age_approx", "sex", "anatom_site_general",
    "diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_4",
]


def lesion_view(meta_df, gt_df):
    """One row per lesion: 11-class code from gt, diagnosis_1 from metadata.

    An inner join, so a lesion-id mismatch shows up as a shorter table here rather
    than as a NaN that quietly propagates into the crosstab.
    """
    dx = pd.DataFrame({
        "lesion_id": gt_df["lesion_id"],
        "dx": gt_df[ONEHOT_COLS].idxmax(axis=1),
    })
    primary = meta_df[["lesion_id", "diagnosis_1"]].drop_duplicates(subset="lesion_id")
    return dx.merge(primary, on="lesion_id", how="inner", validate="one_to_one")


def code_crosstab(meta_df, gt_df):
    """11-class code (rows) x diagnosis_1 (columns), counted in lesions."""
    view = lesion_view(meta_df, gt_df)
    ct = pd.crosstab(view["dx"], view["diagnosis_1"])
    present = [c for c in config.PRIMARY_LABELS if c in ct.columns]
    assert set(ct.columns) == set(present), f"unexpected diagnosis_1 values: {list(ct.columns)}"
    return ct.reindex(columns=config.PRIMARY_LABELS, fill_value=0)


def parents_per_child(meta_df, child, parent):
    """How many distinct `parent` values each non-null `child` value appears under."""
    pairs = meta_df[[child, parent]].dropna(subset=[child])
    return pairs.groupby(child, dropna=True)[parent].nunique(dropna=False)


def check_a_lesion_ids_match(meta_df, gt_df):
    in_meta, in_gt = set(meta_df["lesion_id"]), set(gt_df["lesion_id"])
    only_meta, only_gt = in_meta - in_gt, in_gt - in_meta
    return (not only_meta and not only_gt,
            f"{len(in_meta):,} in metadata, {len(in_gt):,} in gt; "
            f"{len(only_meta)} only in metadata, {len(only_gt)} only in gt")


def check_b_one_positive(meta_df, gt_df):
    positives = gt_df[ONEHOT_COLS].sum(axis=1)
    counts = positives.value_counts().sort_index()
    bad = int((positives != 1).sum())
    return (bad == 0,
            f"{len(gt_df) - bad:,}/{len(gt_df):,} rows have exactly one positive; "
            "positives-per-row seen: "
            + ", ".join(f"{int(k)} positive x {int(v):,} rows" for k, v in counts.items()))


def check_c_code_to_diagnosis_1(meta_df, gt_df):
    ct = code_crosstab(meta_df, gt_df)
    ambiguous = ct.index[ct.gt(0).sum(axis=1) > 1].tolist()
    n_affected = int(ct.loc[ambiguous].to_numpy().sum()) if ambiguous else 0
    return (not ambiguous,
            f"{len(ct) - len(ambiguous)}/{len(ct)} codes map to one diagnosis_1; "
            f"straddling: {ambiguous or 'none'} ({n_affected:,} lesions)")


def check_d_taxonomy_nests(meta_df, gt_df):
    details, n_bad_total = [], 0
    for child, parent in [("diagnosis_3", "diagnosis_2"), ("diagnosis_2", "diagnosis_1")]:
        n_parents = parents_per_child(meta_df, child, parent)
        n_bad = int((n_parents > 1).sum())
        n_bad_total += n_bad
        details.append(f"{n_bad}/{len(n_parents)} {child} values have >1 {parent}")
    return n_bad_total == 0, "; ".join(details)


def check_e_pair_agreement(meta_df, gt_df):
    varying = {
        col: int((meta_df.groupby("lesion_id")[col].nunique(dropna=False) > 1).sum())
        for col in LESION_CONSTANT_FIELDS
    }
    offenders = {c: n for c, n in varying.items() if n > 0}
    return (not offenders,
            f"{len(LESION_CONSTANT_FIELDS)} fields checked across "
            f"{meta_df['lesion_id'].nunique():,} lesions; disagreeing: {offenders or 'none'}")


CHECKS = [
    ("a", "lesion_id sets identical in metadata.csv and training_gt.csv", check_a_lesion_ids_match),
    ("b", "exactly one positive class per lesion in the one-hot", check_b_one_positive),
    ("c", "each 11-class code maps to exactly one diagnosis_1", check_c_code_to_diagnosis_1),
    ("d", "diagnosis_3 -> diagnosis_2 -> diagnosis_1 each nest cleanly", check_d_taxonomy_nests),
    ("e", "a lesion's two images agree on age, sex, site and diagnosis", check_e_pair_agreement),
]


def run_checks(meta_df, gt_df):
    rows = [(cid, question, *fn(meta_df, gt_df)) for cid, question, fn in CHECKS]
    out = pd.DataFrame(rows, columns=["id", "check", "passed", "detail"])
    out.insert(3, "result", np.where(out["passed"], "PASS", "VIOLATION"))
    return out.drop(columns="passed")


# %%
summary = run_checks(meta, gt)
print(summary.to_string(index=False))

n_violations = int((summary["result"] == "VIOLATION").sum())
print(f"\n{len(summary) - n_violations}/{len(summary)} checks pass, {n_violations} violation(s).")

# (a), (b), (d) and (e) are hard requirements: if any of them failed the dataset would be
# unusable as it stands, so they are asserted rather than merely reported.
must_pass = summary[summary["id"].isin(["a", "b", "d", "e"])]
assert (must_pass["result"] == "PASS").all(), f"structural failure:\n{must_pass}"

# (c) is different. It is *expected* to report a violation, and the violation is the
# finding of this exercise — so pin down exactly which code straddles, and fail loudly
# if that ever changes.
assert summary.loc[summary["id"] == "c", "result"].item() == "VIOLATION"

# %% [markdown]
# ### (c) in detail — where the 11-class code does not determine `diagnosis_1`

# %%
ct = code_crosstab(meta, gt)
n_lesions = int(ct.to_numpy().sum())

detail = ct.copy()
detail["total"] = ct.sum(axis=1)
detail["maps_to"] = ct.gt(0).apply(lambda row: " + ".join(ct.columns[row.to_numpy()]), axis=1)
print(detail.sort_values("total", ascending=False).to_string())

ambiguous = ct.index[ct.gt(0).sum(axis=1) > 1].tolist()
assert ambiguous == ["AKIEC"], f"the set of straddling codes changed: {ambiguous}"

print(f"\nCodes that do NOT map to exactly one diagnosis_1: {ambiguous}")
n_affected = 0
for code in ambiguous:
    row = ct.loc[code]
    n_affected += int(row.sum())
    parts = ", ".join(f"{cls} {int(n):,}" for cls, n in row.items() if n > 0)
    print(f"  {code} ({config.DIAGNOSIS_CODES[code]}): {parts}")
print(f"  affected lesions: {n_affected:,} of {n_lesions:,} "
      f"({n_affected / n_lesions * 100:.1f}% of the dataset)")

# The other direction is the more alarming one: which diagnosis_1 values can only be
# reached from a single code? If a class has exactly one source code, the two label
# schemes are not independent views of the lesion — one is nearly a relabelling of the other.
print("\nReachability, diagnosis_1 <- 11-class code:")
for cls in ct.columns:
    sources = ct.index[ct[cls] > 0].tolist()
    print(f"  {cls:<14} {ct[cls].sum():>5,} lesions from {len(sources)} code(s): {sources}")

# %% [markdown]
# ### (d) in detail — does the diagnosis taxonomy nest?

# %%
for child, parent in [("diagnosis_3", "diagnosis_2"), ("diagnosis_2", "diagnosis_1")]:
    n_parents = parents_per_child(meta, child, parent)
    bad = n_parents[n_parents > 1]
    print(f"{child:<12} -> {parent:<12} "
          f"{len(n_parents):>3} distinct {child} values, "
          f"{len(bad)} with more than one {parent}")
    if len(bad):
        print(bad.to_string())

# diagnosis_4 is not part of the brief's check but is the same taxonomy one level deeper,
# so it costs nothing to confirm the pattern holds all the way down.
n_parents_4 = parents_per_child(meta, "diagnosis_4", "diagnosis_3")
print(f"{'diagnosis_4':<12} -> {'diagnosis_3':<12} "
      f"{len(n_parents_4):>3} distinct diagnosis_4 values, "
      f"{int((n_parents_4 > 1).sum())} with more than one diagnosis_3")

# %% [markdown]
# ### Would these checks notice if the data were broken?
#
# Each check is now re-run on a copy of the data with one targeted defect injected. A
# check that does not change its verdict on the corrupted copy is not testing anything.

# %%
def corrupt_a(meta_df, gt_df):
    """Drop one lesion from the ground truth — labels and images no longer line up."""
    return meta_df, gt_df.drop(index=gt_df.index[0])


def corrupt_b(meta_df, gt_df):
    """Add a second positive to one one-hot row — the lesion becomes multi-label."""
    broken = gt_df.copy()
    first = broken.index[0]
    already = broken.loc[first, ONEHOT_COLS].idxmax()
    other = next(c for c in ONEHOT_COLS if c != already)
    broken.loc[first, other] = 1
    return meta_df, broken


def corrupt_c(meta_df, gt_df):
    """Flip one NV lesion to Malignant — a second code starts straddling diagnosis_1."""
    broken = meta_df.copy()
    nv = lesion_view(meta_df, gt_df).query("dx == 'NV'")["lesion_id"].iloc[0]
    broken.loc[broken["lesion_id"] == nv, "diagnosis_1"] = "Malignant"
    return broken, gt_df


def corrupt_d(meta_df, gt_df):
    """Re-parent one row — a diagnosis_3 value now sits under two diagnosis_2 values."""
    broken = meta_df.copy()
    row = broken.index[broken["diagnosis_3"].notna()][0]
    broken.loc[row, "diagnosis_2"] = "Wrong parent"
    return broken, gt_df


def corrupt_e(meta_df, gt_df):
    """Change the age on one image only — a lesion now disagrees with itself."""
    broken = meta_df.copy()
    row = broken.index[0]
    broken.loc[row, "age_approx"] = broken.loc[row, "age_approx"] + 20
    return broken, gt_df


CORRUPTIONS = {"a": corrupt_a, "b": corrupt_b, "c": corrupt_c, "d": corrupt_d, "e": corrupt_e}

rows = []
for cid, question, fn in CHECKS:
    real_passed, real_detail = fn(meta, gt)
    broken_meta, broken_gt = CORRUPTIONS[cid](meta, gt)
    broken_passed, broken_detail = fn(broken_meta, broken_gt)

    # Two ways a check can prove it is alive: flip from pass to fail, or - for (c),
    # which already reports a violation - change what it says about the violation.
    reacted = (not broken_passed) and (broken_detail != real_detail)
    rows.append((cid, CORRUPTIONS[cid].__doc__.splitlines()[0],
                 "PASS" if real_passed else "VIOLATION",
                 "PASS" if broken_passed else "VIOLATION",
                 "yes" if reacted else "NO — CHECK IS BLIND"))
    assert reacted, f"check ({cid}) did not react to its corruption: {broken_detail}"

control = pd.DataFrame(rows, columns=["id", "injected defect", "on real data",
                                      "on corrupted copy", "detected"])
print(control.to_string(index=False))

# (c) already reports a violation on the real data, so "it still says VIOLATION" would
# prove nothing. What it has to show is that the verdict tracks the data.
print("\n(c) is the one check that already fails, so its evidence is the change in detail:")
print(f"   real data     : {check_c_code_to_diagnosis_1(meta, gt)[1]}")
print(f"   corrupted copy: {check_c_code_to_diagnosis_1(*corrupt_c(meta, gt))[1]}")

# %% [markdown]
# ### Cross-check against the project package
#
# The same structural claims are implemented inside `milk10k.quality` and
# `milk10k.labels`, which is what the pipeline and the data-quality report use. Two
# independent implementations agreeing is worth more than one implementation asserting.

# %%
from milk10k import quality

pkg_checks = quality.label_consistency_checks(df=meta, gt=gt)
print(pkg_checks.to_string(index=False))

pkg_table = labels.code_to_primary_table(gt=gt, meta=meta)
pkg_ambiguous = pkg_table.loc[pkg_table["ambiguous"], "dx"].tolist()
print(f"\nmilk10k.labels says the straddling codes are: {pkg_ambiguous}")
assert pkg_ambiguous == ambiguous, "the notebook and the package disagree about ambiguity"

# The package's crosstab must contain the same numbers this section computed by hand.
pkg_counts = pkg_table.set_index("dx")[config.PRIMARY_LABELS].reindex(index=ct.index)
assert (pkg_counts.to_numpy() == ct.to_numpy()).all(), "counts differ from the package"
print("Package and notebook agree on all "
      f"{ct.shape[0]} x {ct.shape[1]} code x diagnosis_1 counts.")

# %% [markdown]
# **Answer.** Ten of the eleven codes map to a single `diagnosis_1`; AKIEC is the sole
# exception, splitting 123 Indeterminate against 180 Malignant, and — the sharper half of
# the finding — all 123 Indeterminate lesions in the dataset are AKIEC, so that class has
# exactly one source code. `diagnosis_1` therefore cannot be derived from an 11-class
# prediction: for those 303 lesions, 5.8% of the 5,240, the code leaves the coarse label
# undetermined, and they are the hardest cases precisely because AKIEC bundles actinic
# keratosis with squamous cell carcinoma in situ, a genuine biological continuum rather
# than a data-entry bug, which is why this project trains the two targets separately
# instead of deriving one from the other. On any new medical dataset I would repeat all
# five checks before training: ID-set equality between the label file and the image file,
# one label per modelling unit, the fine-label-to-coarse-label crosstab, the nesting of
# every taxonomy level, and the within-group constancy of every attribute I intend to
# aggregate over. The most dangerous failure is the silent label corruption of (b) and
# (e): a one-hot row with two positives, or a lesion whose two photographs carry
# different diagnoses, trains the network on a flat contradiction with no error message
# anywhere — `idxmax` simply returns the alphabetically first class, the loss still goes
# down, and the damage only surfaces as unexplained ceiling error much later. A mismatch
# in (a) is the next worst, because a lesion present in one file and missing from the
# other changes the grouping key rather than crashing, and a broken grouping key puts one
# photograph of a lesion in train and the other in test — leakage that inflates every
# number in the report.
