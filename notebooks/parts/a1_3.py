# %% [markdown]
# ## A1.3 Build a lesion-level table
#
# - `metadata.csv` = one row per **image**, `training_gt.csv` = one row per **lesion**.
# - Each lesion has 2 photos (dermoscopic + clinical), so I put both image ids in one row:
#   `lesion_id, derm_id, clinical_id, diagnosis_1, dx, age, sex, site`
#
# **Why I think the lesion level matters**
#
# - It's what we predict: "is this lesion malignant", not "is this photo malignant".
# - It's what we evaluate on: two photos of one lesion are one decision, not two test cases.
# - It's what the split has to group on: one photo in train and the other in test would leak the answer.
# - With both ids in the same row, the smallest thing I can move between splits is a whole lesion.

# %% [markdown]
# ### The table
#
# - `data.build_lesion_table()` does the reshape.
# - It lives in the package so Part B builds the table exactly the same way.

# %%
lesions = data.build_lesion_table()

print("shape:", lesions.shape)
print("columns:", list(lesions.columns))
print()
print(lesions.head().to_string(index=False))

# %% [markdown]
# ### The asserts
#
# - The function checks these internally, but I want them visible here too.

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
# ### The reshape itself, without looping over rows
#
# - `image_type` is the only thing that differs between a lesion's two rows, so it becomes the columns and `isic_id` the value.
# - `unstack` also acts as a check: if a lesion had two dermoscopic photos it would raise an error instead of silently keeping one.

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
# - I save it to CSV so later work loads the same fixed 5,240 lesions.

# %%
path = data.save_lesion_table(lesions)
print("written to:", path.relative_to(config.PROJECT_ROOT))
print("rows on disk:", len(pd.read_csv(path)))

# A3 calls the same data.build_lesion_table() that produced this file's
# contents, so every section splits exactly these 5,240 rows.

# %% [markdown]
# **My answer**
#
# - 5,240 rows, one per lesion (exactly half of the 10,480 images), with columns `lesion_id, derm_id, clinical_id, diagnosis_1, dx, age, sex, site`.
# - Built with one `unstack` on `image_type` + `groupby(...).first()`, no loops.
# - No missing `derm_id` / `clinical_id`, no duplicate `lesion_id`, and the two id columns together cover all 10,480 images.
# - My inline version matches `data.build_lesion_table()` exactly (`.equals` is `True`).
# - Saved to `artifacts/tables/lesions.csv`; A3 gets the same table from the same function and splits it on `lesion_id`.

