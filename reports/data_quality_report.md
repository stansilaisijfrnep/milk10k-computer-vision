# MILK10k — data quality report

Generated 2026-09-22 from `metadata.csv` and `training_gt.csv`.

## 1. Scope

- 10,480 image rows, 5,240 lesions (2 images per lesion).
- 5,240 ground-truth rows, one per lesion, 11 diagnosis codes.
- Modelling unit is the **lesion**, so every split is grouped on `lesion_id`.

## 2. Image integrity (B1)

All 10,480 files were opened and verified (header parse only, no pixel decode). Every referenced file exists and its header parses.

| metric | value | note |
|---|---|---|
| n_images | 10,480 | rows in the metadata, i.e. files expected |
| n_missing | 0 | referenced file not on disk |
| n_unreadable | 0 | file present but the header failed to parse |
| n_ok | 10,480 | opened and verified successfully |
| width_min | 600.0 | pixels |
| width_median | 600.0 | pixels |
| width_max | 600.0 | pixels |
| height_min | 450.0 | pixels |
| height_median | 450.0 | pixels |
| height_max | 450.0 | pixels |
| n_distinct_resolutions | 1 | 1 means the dataset is dimensionally uniform |
| most_common_resolution | 600x450 | 100.0% of readable images |
| frac_at_least_224 | 1.0 | share with both sides >= 224px (the configured input size) |
| frac_at_least_256 | 1.0 | share with both sides >= 256px (the next common backbone resolution) |

The dataset is dimensionally uniform (1 distinct resolution, 600x450), and 100.0% of images have both sides at least 224px. Resizing to 224x224 therefore only ever discards detail, never invents it — which is the argument for that input size.

## 3. Missing values and handling decisions (B4)

| column | n_missing | pct_missing | dtype | decision |
|---|---|---|---|---|
| anatom_site_special | 10,274 | 98.0 | str | Drop the column. 2% coverage cannot support a category level, and the few populated values duplicate anatom_site_general. |
| diagnosis_4 | 8,958 | 85.5 | str | Leave as-is. Same reasoning as diagnosis_3: absence is information about the taxonomy depth, and it is a label column, never an input. |
| melanocytic | 8,088 | 77.2 | object | Leave the NaN as-is and DO NOT USE the column as a model input: it is derived from the diagnosis, so it leaks the label (see the leakage section). Imputing it would only make the leak look tidier. |
| anatom_site_general | 3,912 | 37.3 | str | Encode the NaN as an explicit 'unknown' category. At 37% the values are clearly not missing at random (some contributors simply never recorded a site), so mode-imputation would invent 37% of a feature. |
| diagnosis_3 | 158 | 1.5 | str | Leave as-is. NaN here means 'no finer level exists for this lesion' — structurally absent, not missing — and it is a label column, never an input. |
| age_approx | 40 | 0.4 | float64 | Impute with the TRAIN-split median (never the full-data median, that would leak); add no missingness indicator — at 0.4% the flag would be almost constant and carries no usable signal. |

## 4. Label and structural consistency (B4)

12 of 12 checks pass.

| check | passed | detail |
|---|---|---|
| Every lesion has exactly 2 images | PASS | 5,240 / 5,240 lesions |
| Each lesion has exactly one image of each type | PASS | 5,240 / 5,240 lesions; types present: ['clinical: close-up', 'dermoscopic'] |
| lesion_id sets identical in metadata and training_gt | PASS | 5,240 metadata / 5,240 ground-truth; 0 only in metadata, 0 only in gt |
| Exactly one positive class per lesion in the one-hot | PASS | 5,240 / 5,240 rows; positives per row: [1] |
| 'diagnosis_1' is constant within a lesion | PASS | 0 lesions with more than one value |
| 'age_approx' is constant within a lesion | PASS | 0 lesions with more than one value |
| 'sex' is constant within a lesion | PASS | 0 lesions with more than one value |
| 'anatom_site_general' is constant within a lesion | PASS | 0 lesions with more than one value |
| isic_id is unique | PASS | 10,480 unique / 10,480 rows |
| Every diagnosis_2 value maps to exactly one diagnosis_1 | PASS | 0 of 19 distinct diagnosis_2 values have >1 parent |
| Every diagnosis_3 value maps to exactly one diagnosis_2 | PASS | 0 of 27 distinct diagnosis_3 values have >1 parent |
| Every diagnosis_4 value maps to exactly one diagnosis_3 | PASS | 0 of 15 distinct diagnosis_4 values have >1 parent |

## 5. Shortcuts, confounds and leakage (B4)

### 5.1 Image manipulation x diagnosis_1 (row %)

| image_manipulation | Benign | Indeterminate | Malignant | n_images |
|---|---|---|---|---|
| altered | 39.7 | 12.8 | 47.5 | 335 |
| instrument only | 27.9 | 2.0 | 70.1 | 10,145 |

### 5.2 Image type x diagnosis_1 (row %)

| image_type | Benign | Indeterminate | Malignant | n_images |
|---|---|---|---|---|
| clinical: close-up | 28.3 | 2.3 | 69.4 | 5,240 |
| dermoscopic | 28.3 | 2.3 | 69.4 | 5,240 |

### 5.3 Diagnosis confirmation type x diagnosis_1 (row %)

| diagnosis_confirm_type | Benign | Indeterminate | Malignant | n_images |
|---|---|---|---|---|
| histopathology | 25.7 | 2.0 | 72.3 | 10,032 |
| single contributor clinical assessment | 87.1 | 10.7 | 2.2 | 448 |

### Conclusion

**`diagnosis_confirm_type` is the loudest leak in the dataset.** `histopathology` images are 72.3% Malignant (10,032 images), while `single contributor clinical assessment` images are 87.1% Benign (448 images). This is not a quirk of the data: the decision to send a lesion to histopathology *is* the clinical suspicion of malignancy, so the column encodes the answer. It must never be a model input, and it must not be used to filter the test set either, or the reported accuracy will be measured on an artificially easy subset.

**`image_type` is not a shortcut, and that is the useful result.** Both modalities show an identical class mix (Benign 28.3%, Indeterminate 2.3%, Malignant 69.4%), which follows by construction: every lesion contributes exactly one dermoscopic and one clinical close-up image, so the two rows are the same lesions counted twice. Modality therefore carries no label information on its own, and both views can be used as training images (and later fused per lesion) without introducing a confound.

**`image_manipulation` is a mild confound, not a shortcut.** `altered` images are 12.8% Indeterminate against 2.0% for `instrument only`, but `altered` covers only 335 of 10,480 images. The skew is real and worth reporting, yet too small a slice to drive a classifier. We keep those images (dropping them would bias the rare Indeterminate class further) and simply never expose the column to the model.

## 6. Columns excluded from model inputs

Defined once in `config.LEAKY_COLUMNS`; the reason each one is banned:

| column | reason |
|---|---|
| diagnosis_1 | It is the primary target itself. |
| diagnosis_2 | Finer level of the same label taxonomy as the target. |
| diagnosis_3 | Finer level of the same label taxonomy as the target. |
| diagnosis_4 | Finer level of the same label taxonomy as the target. |
| diagnosis_confirm_type | Records how the diagnosis was confirmed. A lesion goes to histopathology because a clinician already suspected malignancy, so this column encodes the clinical decision, not the image. |
| diagnosis_full | Free-text form of the label. |
| invasion_thickness_interval | Tumour thickness measured on the excised lesion — it exists only for lesions already diagnosed as invasive malignancies. |
| melanocytic | Derived from the diagnosis (whether the lesion is of melanocytic origin), so it is a coarse label in disguise. |
| concomitant_biopsy | Records that a biopsy was performed, which is again the clinician's suspicion rather than anything visible in the image. |
| lesion_id | An identifier, not a feature. It is the grouping key for the split; as an input it would let the model memorise individual lesions. |
