# %% [markdown]
# ## A1.2 Ask the data five questions
#
# Five questions that a clinician or a modeller would actually ask, each answered with a
# few lines of pandas. The point is not the pandas — it is that the answers change what
# the model gets built on. Two of them (Q4, Q5) turn into design decisions later.
#
# **Grain matters.** The dataset has two row units and mixing them silently doubles or
# halves every count. Q1, Q3 and Q4 are about lesions and patients, so they use the
# **lesion** table (5,240 rows). Q2 asks for a "percentage of images", so it uses the
# **image** table (10,480 rows). Q5 is stated at **lesion** grain, because
# `diagnosis_confirm_type` describes the biopsy of a lesion, not a photograph.

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
# Lesion grain: one basal cell carcinoma is one lesion, however many photos of it exist.

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
# **Answer.** 380 lesions — 15.1% of the 2,522 BCC lesions in the dataset. The filter is
# deliberately strict: `age >= 70` drops the 20 lesions with no recorded age, and
# `site == "head/neck"` drops the 1,956 with no recorded site (both printed above), so 380 is a floor, not an
# estimate of how many such lesions exist.

# %% [markdown]
# ### Q2 — which class has the highest share of "altered" images?
#
# Image grain, because `image_manipulation` is a property of a photograph, not of a lesion.

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
# **Answer.** BEN_OTH, at 18.18% — but that is 16 altered images out of only 88, so the
# rate is built on 16 observations and would swing by ~1 pp if a single image were
# reclassified. The runners-up are INF 8.00% (8/100), AKIEC 7.10% (43/606) and MEL 6.33%
# (57/900); MEL is the first entry with a denominator large enough to take at face value.
# The honest reading is that the top two (BEN_OTH, INF, both under 50 lesions) sit there
# mostly because small denominators are volatile — the rarest class of all, MAL_OTH, sits
# at 0/18 for the same reason — not because those lesions are systematically retouched.

# %% [markdown]
# ### Q3 — youngest and oldest melanoma patient
#
# Lesion grain: the question is about patients, and ages are recorded per lesion.

# %%
mel = lesions[lesions["dx"] == "MEL"]
youngest, oldest = mel["age"].min(), mel["age"].max()

print(f"MEL lesions: {len(mel):,} | missing age: {int(mel['age'].isna().sum())}")
print(f"youngest {youngest:.0f}  oldest {oldest:.0f}  gap {oldest - youngest:.0f} years")

# %% [markdown]
# **Answer.** Youngest 20, oldest 85, a 65-year gap, with zero missing ages among the 450
# melanoma lesions. `age_approx` is binned in fives, so these are bin labels rather than
# exact ages. A 65-year span means age carries little discriminative power for melanoma on
# its own — it spreads across essentially the whole adult range of the dataset.

# %% [markdown]
# ### Q4 — is a missing `anatom_site_general` informative?
#
# 3,912 of 10,480 image rows have no recorded site. If that missingness were random, the
# class mix of lesions with and without a site would look the same. Lesion grain.

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
# **Answer.** Not random. NV moves the most: 10.14% of lesions with a recorded site versus
# 21.11% of those without, **+10.97 pp** — nevi are more than twice as common in the
# no-site group. The keratinocyte cancers move the other way (SCCKA −9.83 pp, AKIEC
# −5.96 pp), and MEL is +3.84 pp. That pattern is clinically coherent: the lesions that
# get a site recorded are disproportionately the sun-damage cancers that cluster on the
# head and neck, while a nevus is photographed and filed without the site being typed in.
#
# Two consequences for the pipeline. First, "unknown site" is itself a weak predictor, so
# it must be kept as an explicit category — imputing it to the modal site would destroy
# real signal and invent a body location that was never observed. Second, it must never be
# treated as a neutral default, because it is not: it leans benign (34.5% Benign without a
# site versus 24.6% with one, printed above).

# %% [markdown]
# ### Q5 — malignancy rate by `diagnosis_confirm_type`
#
# Lesion grain. The confirmation type is constant across a lesion's two images, which is
# asserted below before collapsing.

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
# **Answer.** Histopathology-confirmed lesions are 72.3% Malignant; lesions confirmed by a
# single contributor's clinical assessment are 2.2% Malignant (87.1% Benign, 10.7%
# Indeterminate). That is a 70-point gap across 5,016 versus 224 lesions.
#
# The confirmation type is not a property of the skin — it records the decision to biopsy,
# and that decision was driven by the clinician's suspicion. Anything alarming enough to
# cut out went to pathology; anything obviously benign was signed off by eye. So the field
# is a direct readout of prior suspicion, which explains two things:
#
# - **Why the dataset is 69.4% Malignant.** It is a collection of biopsied lesions, not a
#   sample of skin. Real-world prevalence in a dermatology clinic is nothing like this, so
#   any accuracy or calibration number from this data describes a pre-filtered population
#   and cannot be quoted as clinic performance.
# - **Why `diagnosis_confirm_type` is a textbook leaky feature.** It sits downstream of the
#   label and does not exist at prediction time — a model that sees it is reading the
#   clinician's answer, not the image. It is on the `config.LEAKY_COLUMNS` list for exactly
#   this reason.
