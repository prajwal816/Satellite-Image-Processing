# Data Directory

This directory is **gitignored** — raw datasets should never be committed to Git.

## Obtaining the EuroSAT Dataset

The project uses the **EuroSAT** dataset (Sentinel-2 satellite images, all 13 spectral bands in TIFF format).

### Option 1: Download from the official source

1. Visit <https://github.com/phelber/eurosat>
2. Download the **EuroSAT (all bands)** archive (`EuroSATallBands.zip`).
3. Extract into this directory so the folder structure looks like:

```
data/
└── EuroSATallBands/
    └── ds/
        └── images/
            └── remote_sensing/
                └── otherDatasets/
                    └── sentinel_2/
                        └── tif/
                            ├── AnnualCrop/
                            ├── Forest/
                            ├── ...
                            └── SeaLake/
```

### Option 2: Kaggle

The dataset is also available on Kaggle:
<https://www.kaggle.com/datasets/apollo2506/eurosat-dataset>

After downloading, extract and place the contents under `data/` following the same layout above.

> **Note:** Do not commit the downloaded data to Git. The `.gitignore` is configured to ignore everything inside `data/` except this `README.md`.
