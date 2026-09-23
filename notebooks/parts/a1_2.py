# %% [markdown]
# ## A1.2 Ask the data five questions
#
# - Five questions, each answered with a few lines of pandas.
# - **Grain matters**, because mixing images and lesions doubles or halves counts:
#   - Q1, Q3, Q4 are about lesions/patients → **lesion** table (5,240 rows)
#   - Q2 asks for a "percentage of images" → **image** table (10,480 rows)
#   - Q5 → **lesion** table, since the confirmation type describes the biopsy of a lesion

# %%
# Both grains, built once from the package rather than re-derived per question.
meta = data.load_metadata()                 # one row per IMAGE  (10,480)
lesions = data.build_lesion_table()         # one row per LESION (5,240)

# The 11-class code lives at lesion grain; broadcasting it onto the images is what
# makes an image-grain question like Q2 answerable at all.
images = meta.merge(labels.lesion_labels(), on="lesion_id",
                    how="left", validate="many_to_one")

assert len(meta) == 10_480 and len(lesions) == 5_240
assert images["dx"].notna().all(), "an image was left without an 11-class code"
assert (meta.groupby("lesion_id")["image_type"].nunique() == 2).all()

print(f"image grain : {len(images):,} rows")
print(f"lesion grain: {len(lesions):,} rows")

# %% [markdown]
# ### Q1 — BCC lesions in patients aged 70+ on the head/neck
#
# - Lesion grain: one BCC is one lesion, however many photos it has.

# %%
q1 = lesions[(lesions["dx"] == "BCC")
             & (lesions["age"] >= 70)
             & (lesions["site"] == "head/neck")]

n_bcc = int((lesions["dx"] == "BCC").sum())
print(f"BCC lesions, age >= 70, head/neck: {len(q1):,} "
      f"({100 * len(q1) / n_bcc:.1f}% of all {n_bcc:,} BCC lesions)")
print(f"excluded by the filter because the field is missing: "
      f"{int(lesions['age'].isna().sum())} lesions without age, "
      f"{int(lesions['site'].isna().sum()):,} without site")

# %% [markdown]
# **My answer:** 380 lesions (15.1% of the 2,522 BCCs).
#
# - I think this is a lower bound: the filter drops the 20 lesions with no age and the 1,956 with no site.

# %% [markdown]
# ### Q2 — which class has the highest share of "altered" images?
#
# - Image grain, since `image_manipulation` describes a photo, not a lesion.

# %%
images["is_altered"] = images["image_manipulation"].eq("altered")

altered = images.groupby("dx")["is_altered"].agg(n_images="size", n_altered="sum")
altered["pct_altered"] = 100 * altered["n_altered"] / altered["n_images"]
altered = altered.sort_values("pct_altered", ascending=False)

# Counts stay next to the rate: a rate over a tiny denominator is not a finding.
print(altered.round(2).to_string())
top = altered.index[0]
print(f"\nHighest: {top} at {altered.loc[top, 'pct_altered']:.2f}% "
      f"({altered.loc[top, 'n_altered']} of {altered.loc[top, 'n_images']} images)")

# %% [markdown]
# **My answer:** BEN_OTH, with 18.18%.
#
# - But that is only 16 of 88 images, so I wouldn't read too much into it.
# - Next are INF 8.00% (8/100), AKIEC 7.10% (43/606), MEL 6.33% (57/900).
# - From what I understood, the top two are small classes whose rates jump around a lot (MAL_OTH is at 0/18 for the same reason), not classes that are really retouched more.

# %% [markdown]
# ### Q3 — youngest and oldest melanoma patient
#
# - Lesion grain: the question is about patients.

# %%
mel = lesions[lesions["dx"] == "MEL"]
youngest, oldest = mel["age"].min(), mel["age"].max()

print(f"MEL lesions: {len(mel):,} | missing age: {int(mel['age'].isna().sum())}")
print(f"youngest {youngest:.0f}  oldest {oldest:.0f}  gap {oldest - youngest:.0f} years")

# %% [markdown]
# **My answer:** youngest 20, oldest 85, so a 65-year gap (no missing ages among the 450 MEL lesions).
#
# - `age_approx` is binned in fives, so these are bins, not exact ages.
# - I think age alone says little about melanoma here, since it covers almost the whole adult range.

# %% [markdown]
# ### Q4 — is a missing `anatom_site_general` informative?
#
# - 3,912 of 10,480 image rows have no site.
# - If that were random, the class mix with and without a site would look the same.

# %%
has_site = lesions["site"].notna()

site_cmp = pd.DataFrame({
    "with_site_pct": lesions.loc[has_site, "dx"].value_counts(normalize=True) * 100,
    "without_site_pct": lesions.loc[~has_site, "dx"].value_counts(normalize=True) * 100,
}).fillna(0)
site_cmp["delta_pp"] = site_cmp["without_site_pct"] - site_cmp["with_site_pct"]
site_cmp = site_cmp.reindex(site_cmp["delta_pp"].abs().sort_values(ascending=False).index)

print(f"n = {int(has_site.sum()):,} with a site / {int((~has_site).sum()):,} without\n")
print(site_cmp.round(2).to_string())

benign_by_site = pd.crosstab(has_site.map({True: "with site", False: "without site"}),
                             lesions["diagnosis_1"], normalize="index") * 100
print("\ndiagnosis_1 mix (% of lesions):")
print(benign_by_site.round(1).to_string())

# %% [markdown]
# **My answer:** no, it's not random.
#
# - NV changes the most: 10.14% with a site vs 21.11% without → **+10.97 pp**.
# - The skin cancers go the other way: SCCKA −9.83 pp, AKIEC −5.96 pp; MEL +3.84 pp.
# - My guess: sites get written down more for sun-damage cancers on the head/neck, less for nevi.
# - What I do about it:
#   - keep "unknown" as its own category instead of imputing a site that was never recorded
#   - never treat it as a neutral default, because it leans benign (34.5% Benign without a site vs 24.6% with one)

# %% [markdown]
# ### Q5 — malignancy rate by `diagnosis_confirm_type`
#
# - Lesion grain (I assert below that it's the same on both images of a lesion).

# %%
confirm_varies = meta.groupby("lesion_id")["diagnosis_confirm_type"].nunique(dropna=False)
assert (confirm_varies == 1).all(), "confirmation type disagrees within a lesion"

lesions_c = lesions.merge(
    meta.groupby("lesion_id", as_index=False)["diagnosis_confirm_type"].first(),
    on="lesion_id", how="left", validate="one_to_one",
)

confirm = pd.crosstab(lesions_c["diagnosis_confirm_type"], lesions_c["diagnosis_1"],
                      normalize="index") * 100
confirm["n_lesions"] = lesions_c["diagnosis_confirm_type"].value_counts()

print(confirm.round(1).to_string())
print(f"\nwhole dataset: {100 * (lesions['diagnosis_1'] == 'Malignant').mean():.1f}% Malignant")

# %% [markdown]
# **My answer:** histopathology → 72.3% Malignant; single-contributor clinical assessment → 2.2% Malignant (87.1% Benign, 10.7% Indeterminate). That's 5,016 vs 224 lesions.
#
# - From what I understood, this column records the **decision to biopsy**, which depends on how worried the doctor was.
# - **How the dataset was built:** it's mostly biopsied lesions, which is why it's 69.4% Malignant, far above a real clinic, so accuracy here doesn't describe clinic performance.
# - **Which lesions get in:** the suspicious ones, and that makes `diagnosis_confirm_type` a leaky feature: it comes after the label and doesn't exist at prediction time. It's on my `config.LEAKY_COLUMNS` list.

