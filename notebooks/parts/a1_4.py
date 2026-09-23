# %% [markdown]
# ## A1.4 Task framing
#
# - I print the few numbers my framing relies on first, so it's based on the data.

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
# **My problem specification**
#
# - **Input:** both photos of a lesion (dermoscopic + clinical close-up). I leave metadata out for now, since some columns give away the label.
# - **Output:** one prediction per lesion: Benign / Indeterminate / Malignant.
# - **Who and when:** a dermatologist looking at a suspicious lesion and deciding whether to biopsy. I see it as a triage aid, not screening or an automatic diagnosis.
# - **Metric:** sensitivity on Malignant at a fixed specificity, plus balanced accuracy and per-class recall. With 69.4% Malignant, always saying "Malignant" already gets 69.4% accuracy, so accuracy tells me nothing.
# - **Costlier error:** a false negative. Missing a melanoma can kill; a false positive costs a biopsy and a scar.
# - **Two differences from real use:**
#   1. The data is biopsy-enriched (69.4% malignant, much higher than in a real clinic), so the model learns the wrong prior.
#   2. All images are uniform 600x450 study captures and 60.6% of lesions are skin type III, so phone photos, other dermatoscopes or darker skin are outside what it saw.
#
# (173 words, limit 200)

