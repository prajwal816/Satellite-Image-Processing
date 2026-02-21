# Models Directory

This directory is **gitignored** — trained model binaries should not be committed to Git.

## Trained Models

After running the training script, the trained model will be saved here as `eurosat_cnn_model.h5` (or as configured in `configs/default.yaml`).

### Training a Model

```bash
python -m satellite_image_processing.train --config configs/default.yaml
```

### Using Git LFS (optional)

If you need to share a trained model via Git, consider using [Git LFS](https://git-lfs.github.com/):

```bash
git lfs install
git lfs track "models/*.h5"
git add .gitattributes
git add models/eurosat_cnn_model.h5
git commit -m "Add trained model via LFS"
```

> **Note:** Do not commit model files to Git without LFS. The `.gitignore` is configured to ignore everything inside `models/` except this `README.md`.
