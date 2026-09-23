# %% [markdown]
# ## A3.2 A reusable three-way split + property test
#
# - The split lives in `milk10k.splits.split_lesions`, because the same function writes the `train/val/test.csv` files Part B uses.
# - I run it with the project seed, then test it with **ten** seeds, not just one lucky run.

# %%
import warnings

# Stage A asks StratifiedGroupKFold for 10 folds while MAL_OTH has only 9 lesions, so
# sklearn warns once per call. That is expected and is exactly what the rare-class
# study at the end of this section measures, so the repeated warning is silenced here
# instead of being printed 30 times.
warnings.filterwarnings("ignore", message="The least populated class in y has only")

lesions = data.build_lesion_table()

# The split unit is the lesion, so the table it is given must be one row per lesion.
assert lesions["lesion_id"].is_unique, "build_lesion_table is not one row per lesion"

# The fold arithmetic behind the two stages, printed rather than asserted in prose.
holdout_fraction = config.VAL_SIZE + config.TEST_SIZE
k_a, m_a = splits.fold_plan(holdout_fraction)
k_b, m_b = splits.fold_plan(config.TEST_SIZE / holdout_fraction)
print(f"stage A  take {m_a} of {k_a} folds = {m_a / k_a:.2%}  "
      f"(requested val+test = {holdout_fraction:.2%})")
print(f"stage B  take {m_b} of {k_b} folds = {m_b / k_b:.2%}  "
      f"(test's share of that block)")

train_ids, val_ids, test_ids = splits.split_lesions(lesions, seed=config.SEED)

# verify_splits is the package's own audit of a split; reuse it instead of
# recomputing sizes and overlaps by hand.
report = splits.verify_splits(lesions, train_ids, val_ids, test_ids)
sizes = report["sizes"].copy()
sizes["pct"] = (sizes["fraction"] * 100).round(2)
sizes["requested_pct"] = (sizes["requested"] * 100).round(2)
sizes["deviation_pp"] = sizes["deviation_pp"].round(3)

print(f"\nseed = {config.SEED}, {len(lesions):,} lesions")
print(sizes[["split", "n_lesions", "pct", "requested_pct", "deviation_pp"]]
      .to_string(index=False))
print("\npairwise overlaps:", report["overlaps"])
print("every lesion assigned exactly once:", report["partition_ok"])

# Stratification check: lesions per class in each split, and the largest gap between a
# class's share in one split and its share in the whole table.
print("\nlesions per class and split:")
print(report["class_counts"].to_string())
worst = report["worst_cell"]
print(f"\nmax class-proportion deviation from the full table: "
      f"{report['max_abs_deviation_pp']:.2f} pp ({worst['class']} in {worst['split']})")

# %% [markdown]
# **How I turn `(val_size, test_size)` into folds**
#
# - `StratifiedGroupKFold` wants a number of folds, not a percentage.
# - `splits.fold_plan(fraction)` tries every `k` from 2 to 20 and picks the `m` of `k` folds closest to the fraction I want.
# - Two stages:
#   - **A:** hold out val + test together = 0.30 → 3 of 10 folds; the other 7 are train
#   - **B:** cut that block in half → 1 of 2 folds is test, the other is val
# - The simple version (1 fold of 7 for test) would give 14.29% blocks and 71.4% train, 1.4 pp off. Mine is within 0.02 pp for seed 42 and 0.04 pp for all ten seeds.
#
# Two details:
#
# - `shuffle=True` is needed, otherwise `random_state` is ignored and every seed gives the same split.
# - Here `lesion_id` is unique per row, so the grouping doesn't really do anything. From what I understood, the real no-leak guarantee comes in `splits.assign_images`, where both photos of a lesion get the lesion's split. I still pass `groups=` so it stays correct if a lesion ever had a third image.

# %% [markdown]
# ### The property test
#
# - For ten seeds I check:
#   - the three splits don't overlap and together cover every lesion
#   - each size is within ±1 pp of what I asked for
#   - same seed twice → identical lists
# - The per-seed table comes first, then the asserts.

# %%
SEEDS = [config.SEED, *range(9)]   # the project seed plus nine others
TOLERANCE_PP = 1.0

rows = []
split_by_seed = {}
for seed in SEEDS:
    first = splits.split_lesions(lesions, seed=seed)
    repeat = splits.split_lesions(lesions, seed=seed)  # reproducibility: same in, same out
    check = splits.verify_splits(lesions, *first)

    split_by_seed[seed] = first
    rows.append({
        "seed": seed,
        "n_train": len(first[0]),
        "n_val": len(first[1]),
        "n_test": len(first[2]),
        "max_dev_pp": round(float(check["sizes"]["deviation_pp"].abs().max()), 3),
        "overlaps": sum(check["overlaps"].values()),
        "partition_ok": check["partition_ok"],
        "reproducible": repeat == first,
    })

prop = pd.DataFrame(rows)
print(prop.to_string(index=False))

# %%
assert (prop["overlaps"] == 0).all(), \
    f"overlapping splits at seeds {prop.loc[prop['overlaps'] > 0, 'seed'].tolist()}"
assert prop["partition_ok"].all(), \
    f"not a partition at seeds {prop.loc[~prop['partition_ok'], 'seed'].tolist()}"
assert (prop["max_dev_pp"] <= TOLERANCE_PP).all(), \
    f"size off by more than {TOLERANCE_PP} pp, worst = {prop['max_dev_pp'].max()} pp"
assert prop["reproducible"].all(), \
    f"not reproducible at seeds {prop.loc[~prop['reproducible'], 'seed'].tolist()}"

print(f"{len(SEEDS)} seeds: disjoint, complete, reproducible; "
      f"worst size deviation {prop['max_dev_pp'].max():.3f} pp "
      f"(tolerance {TOLERANCE_PP} pp)")
print("the same three assertions run as pytest in tests/test_splits.py")

# %% [markdown]
# ### Do the rare classes survive the split?
#
# - If a class isn't in test, there's no recall to report.
# - `splits.rare_class_coverage` counts, over the same ten seeds, how often each rare class lands in val, in test, and in both.

# %%
RARE = ("MAL_OTH", "DF", "INF", "VASC", "BEN_OTH")
coverage = splits.rare_class_coverage(lesions, seeds=SEEDS, rare=RARE)
print(coverage.to_string(index=False))

# Presence is not the same as evaluability, so look at the actual head counts of the
# rarest class in each run rather than at the tick in the "runs_in_both" column.
dx_of = lesions.set_index("lesion_id")["dx"]
rarest = coverage.loc[coverage["n_lesions"].idxmin(), "dx"]
per_seed = pd.DataFrame([
    {"seed": seed,
     "n_val": int((dx_of.loc[val].to_numpy() == rarest).sum()),
     "n_test": int((dx_of.loc[test].to_numpy() == rarest).sum())}
    for seed, (_, val, test) in split_by_seed.items()
])
print(f"\n{rarest} head count per run "
      f"({int(coverage.loc[coverage['dx'] == rarest, 'n_lesions'].iloc[0])} lesions in total):")
print(per_seed.to_string(index=False))
print(f"\ntest-set {rarest} lesions: min {per_seed['n_test'].min()}, "
      f"max {per_seed['n_test'].max()}, pooled over {len(SEEDS)} seeds "
      f"{per_seed['n_test'].sum()}")

# %%
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% CI for a proportion. Wilson, not Wald: at k == n Wald gives [1.0, 1.0],
    which would claim certainty from a single observation."""
    centre = (k + z**2 / 2) / (n + z**2)
    half = z / (n + z**2) * np.sqrt(k * (n - k) / n + z**2 / 4)
    return max(0.0, centre - half), min(1.0, centre + half)


# What a per-class recall can actually say at these denominators.
for n in sorted(per_seed["n_test"].unique()):
    for k in range(n + 1):
        low, high = wilson_interval(k, int(n))
        print(f"recall {k}/{n} = {k / n:.2f}   95% CI [{low:.2f}, {high:.2f}]"
              f"   width {high - low:.2f}")

macro_share_pp = 100 / len(config.STRETCH_LABELS)
print(f"\none of {len(config.STRETCH_LABELS)} classes is {macro_share_pp:.1f}% of a "
      f"macro-average, so a 0.00 -> 1.00 swing on {rarest} moves it by "
      f"{macro_share_pp:.1f} pp")

# %% [markdown]
# **My answer**
#
# - All five rare classes are in both val and test in **10 of 10** runs, so stratification works.
# - But `MAL_OTH` only gets 1–2 lesions in val and 1–2 in test (9 in the whole dataset). Being present doesn't mean I can evaluate it.
# - **(i) I don't think `MAL_OTH` can be evaluated at any split size.** Recall with 1 or 2 test lesions can only be 0.00, 0.50 or 1.00. Even a perfect 1/1 has a 95% interval of [0.21, 1.00]. A bigger test set (30%) would still only give 2–3.
# - **(ii) Keep it out of the headline number.** One class is 9.1% of a macro-average, so one lesion could move macro-recall by 9.1 pp. I'd report it separately, marked *n<5, indicative only*.
# - **(iii) If I really needed a number:** repeated stratified CV over many seeds, reporting the spread. Even then it's the same 9 lesions every time (13 test appearances over ten seeds), so it helps less than it seems.
# - **(iv) Overall:** one split is the wrong tool for a class this rare. Stratification can't fix a denominator of 1, and reporting "recall 1.00" on one lesion would be misleading.

