# %% [markdown]
# ## A1.3 Build a lesion-level table
#
# The dataset ships on two different row grains. `metadata.csv` has one row per
# **image**; `training_gt.csv` has one row per **lesion**. Every lesion was
# photographed twice — once dermoscopically, once as a clinical close-up — so the
# image table is exactly twice as long as the label table.
#
# This exercise collapses the image grain onto the lesion grain and puts both image
# ids of a lesion side by side in one row:
#
# `lesion_id, derm_id, clinical_id, diagnosis_1, dx, age, sex, site`
#
# **Why the lesion grain matters.** The lesion is the unit of *prediction* — the
# clinical question is "is this mole malignant", not "is this photograph malignant".
# It is therefore also the unit of *evaluation*: two predictions on the same lesion
# are one decision, not two independent test cases, and scoring them separately
# inflates the effective test-set size. Most importantly it is the unit the
# train/val/test split must **group** on. The two photos of a lesion show the same
# piece of skin on the same patient; putting one in train and the other in test
# leaks the answer and buys accuracy that will not survive contact with a new
# patient. Holding `derm_id` and `clinical_id` in the same row makes that mistake
# structurally impossible — the smallest thing you can move between splits is a
# whole lesion.

# %% [markdown]
# ### The table
#
# `data.build_lesion_table()` does the reshape. The logic lives in the package, not
# in this notebook, so the split code and the dataloaders in Part B build the table
# the same way rather than re-deriving it.

# %%
lesions = data.build_lesion_table()

print("shape:", lesions.shape)
print("columns:", list(lesions.columns))
print()
print(lesions.head().to_string(index=False))

# %% [markdown]
# ### The assertions, shown rather than hidden
#
# `build_lesion_table()` asserts these internally, but a reader of the notebook
# should see them pass on this data rather than take the function's word for it.

# %%
checks = {
    "one row per lesion (5,240)": len(lesions) == 5_240,
    "lesion_id is unique": bool(lesions["lesion_id"].is_unique),
    "no missing derm_id": bool(lesions["derm_id"].notna().all()),
    "no missing clinical_id": bool(lesions["clinical_id"].notna().all()),
    # The two id columns must be disjoint and together account for every image:
    # if one photo were filed under both types the pivot would have duplicated it.
    "derm_id and clinical_id disjoint": not set(lesions["derm_id"]) & set(lesions["clinical_id"]),
    "2 x lesions == all images": 2 * len(lesions) == len(data.load_metadata()),
}

for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}  {name}")

assert all(checks.values()), [k for k, v in checks.items() if not v]

# %% [markdown]
# ### The reshape itself — no loop over rows
#
# The brief requires the two image ids to be brought together with
# `pivot`/`unstack`/`groupby`, not with a Python loop. That requirement is the whole
# point of the exercise, so it is worth seeing rather than trusting to a function
# call. `image_type` is the only thing that distinguishes a lesion's two rows, so it
# becomes the **column** axis and `isic_id` becomes the value.
#
# `unstack` is also a correctness check in disguise: it requires
# `(lesion_id, image_type)` to be unique, so a lesion with two dermoscopic photos
# would raise instead of silently keeping one of them.

# %%
meta = data.load_metadata()

# One line does the whole reshape: 10,480 image rows -> 5,240 lesion rows, two id
# columns. No iteration, and no assumption about the order the rows arrive in.
wide = meta.set_index(["lesion_id", "image_type"])["isic_id"].unstack("image_type")
print(wide.columns.tolist())
print(wide.head().to_string())

# %%
# Rebuild the full table inline from that reshape, then prove it is the same object
# the package produces. groupby(...).first() is legitimate for the attribute columns
# because age/sex/site/diagnosis do not vary within a lesion (verified by check (e) in A1.1).
inline = wide[["dermoscopic", "clinical: close-up"]]
inline.columns = ["derm_id", "clinical_id"]
inline = inline.rename_axis(columns=None).reset_index()

attrs = (
    meta.groupby("lesion_id", as_index=False)[
        ["diagnosis_1", "age_approx", "sex", "anatom_site_general"]
    ]
    .first()
    .rename(columns={"age_approx": "age", "anatom_site_general": "site"})
)

# The 11-class code comes from the one-hot block in training_gt.csv; collapsing it
# is package logic (it also checks the one-positive-per-row assumption), so reuse it.
dx = data.gt_to_label(data.load_training_gt())[["lesion_id", "label_11"]].rename(
    columns={"label_11": "dx"}
)

inline = (
    inline.merge(attrs, on="lesion_id", how="left", validate="one_to_one")
    .merge(dx, on="lesion_id", how="left", validate="one_to_one")
)[data.LESION_TABLE_COLUMNS]

matches = inline.equals(lesions)
print("inline pivot == data.build_lesion_table():", matches)
assert matches, "the inline reshape disagrees with the package implementation"

# %% [markdown]
# ### Save it
#
# The table is written to CSV so that later work loads one fixed set of 5,240
# lesions instead of rebuilding it and hoping the result is identical.

# %%
path = data.save_lesion_table(lesions)
print("written to:", path.relative_to(config.PROJECT_ROOT))
print("rows on disk:", len(pd.read_csv(path)))

# A3 calls the same data.build_lesion_table() that produced this file's
# contents, so every section splits exactly these 5,240 rows.

# %% [markdown]
# **Answer.** The lesion table has 5,240 rows — one per lesion, exactly half the
# 10,480 image rows — with columns `lesion_id, derm_id, clinical_id, diagnosis_1,
# dx, age, sex, site`. It is built with a single `unstack` on `image_type` plus a
# `groupby(...).first()` for the lesion-constant attributes: no Python loop over
# rows anywhere. Every lesion has both a dermoscopic and a clinical close-up id
# (no missing values in either column), `lesion_id` is unique, and the two id
# columns are disjoint and together cover all 10,480 images. The inline reshape
# above reproduces `data.build_lesion_table()` exactly (`.equals` is `True`), so
# the package version is doing what the notebook claims. The table is saved to
# `artifacts/tables/lesions.csv`; A3 obtains the identical table through the same
# `data.build_lesion_table()` call and splits it — the split groups on `lesion_id`, which is only safe because both of a lesion's
# images live in the same row.
