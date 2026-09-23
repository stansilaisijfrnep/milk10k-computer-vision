# %% [markdown]
# ## A1.4 Task framing
#
# Before any modelling decision, the task has to be stated as a problem a clinic would
# actually pose. The numbers the framing rests on are printed first, so the specification
# below quotes the dataset rather than asserting things about it.

# %%
from PIL import Image

from milk10k import imbalance

meta = data.load_metadata()
lesions = data.build_lesion_table()

# The modelling unit is the lesion, not the image — this is what makes that legitimate.
assert len(meta) == 2 * len(lesions), "the two-images-per-lesion assumption broke"

print(f"images : {len(meta):,}")
print(f"lesions: {len(lesions):,}  ({len(meta) // len(lesions)} images each: "
      f"{', '.join(sorted(meta['image_type'].unique()))})")

counts = lesions["diagnosis_1"].value_counts().reindex(config.PRIMARY_LABELS)
share = 100 * counts / len(lesions)
print("\ndiagnosis_1, one row per lesion")
for cls in config.PRIMARY_LABELS:
    print(f"  {cls:<14}{counts[cls]:>6,}{share[cls]:>8.1f}%")

# The majority-class share IS the accuracy of a model that has learned nothing, which is
# the whole argument for not optimising accuracy.
print(f"\nmajority-class ('Malignant') accuracy : {share['Malignant']:.1f}%")
print(f"imbalance ratio, 3-class target      : "
      f"{imbalance.imbalance_ratio(lesions, 'diagnosis_1'):.1f}x")
print(f"imbalance ratio, 11-class target     : "
      f"{imbalance.imbalance_ratio(lesions, 'dx'):.1f}x")

# Two numbers for the deployment-gap claims: how narrow the population is, and how
# uniform the acquisition is. Only the JPEG headers are read, so the sample costs nothing.
tone = data.load_training_input().groupby("lesion_id")["skin_tone_class"].first()
top_tone = tone.value_counts(normalize=True)
print(f"\ncommonest skin-tone class            : "
      f"{config.SKIN_TONE_LABELS[int(top_tone.index[0])]} "
      f"-> {100 * top_tone.iloc[0]:.1f}% of lesions")

# Header-only reads (no decoding): about two seconds for all 10,480 files.
sizes = set()
for isic_id in meta["isic_id"]:
    with Image.open(data.image_path(isic_id)) as im:
        sizes.add(im.size)
print(f"distinct image sizes, all {len(meta):,} images: {sorted(sizes)}")

# %% [markdown]
# **Answer — problem specification.**
#
# **Input.** Both views of one lesion: dermoscopic image and clinical close-up. Metadata
# is excluded in Milestone 1; several columns encode the label or postdate the biopsy.
#
# **Output.** One prediction per lesion: Benign / Indeterminate / Malignant.
#
# **User and moment.** A dermatologist examining a suspicious lesion and deciding whether
# to biopsy: a triage aid, not population screening or autonomous diagnosis.
#
# **Metric.** Malignant sensitivity at a fixed, clinically acceptable specificity, with
# balanced accuracy and per-class recall alongside. With 69.4% of lesions Malignant, a
# constant "Malignant" predictor scores 69.4% accuracy, so accuracy measures nothing.
#
# **Costlier error.** The false negative: a missed melanoma can be fatal; a false
# positive costs a biopsy and a scar.
#
# **Two gaps to deployment.** (i) Biopsy enrichment: 69.4% malignant here versus a small
# minority in a real clinic, so the learned prior is wrong and precision collapses.
# (ii) Acquisition and population: every image is a uniform 600x450 study capture and
# 60.6% of lesions are skin-tone class III; phone photos, other dermatoscopes or darker
# skin are out of distribution.
#
# (173 words including headings, counted with `wc -w`; limit 200)
