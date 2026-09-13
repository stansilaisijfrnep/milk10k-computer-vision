# Data directory

The MILK10k dataset is **not** committed to this repository (~345 MB, CC-BY-NC licence).
Download it yourself with:

```bash
bash scripts/download_data.sh
```

Expected layout after download:

```
data/milk10k/
├── metadata.csv                      # 10,480 rows — one per IMAGE
├── attribution.txt
├── licenses/CC-BY-NC.txt
├── images/                           # 10,480 JPEGs
└── supplements/
    ├── training_gt.csv               # 5,240 rows — one per LESION, 11-class one-hot
    ├── training_input.csv            # 10,480 rows — MONET concept scores, skin tone, site
    └── training_supp.csv             # 10,480 rows — full-text diagnosis
```

Source: <https://api.isic-archive.com/doi/milk10k/>
