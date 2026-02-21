# Satellite Image Processing — EuroSAT CNN Classifier

A convolutional neural network (CNN) for classifying Sentinel-2 satellite images into 10 land-use categories using the [EuroSAT](https://github.com/phelber/eurosat) dataset.

## Classes

| # | Category |
|---|---|
| 0 | AnnualCrop |
| 1 | Forest |
| 2 | HerbaceousVegetation |
| 3 | Highway |
| 4 | Industrial |
| 5 | Pasture |
| 6 | PermanentCrop |
| 7 | Residential |
| 8 | River |
| 9 | SeaLake |

## Project Structure

```
├── README.md
├── .gitignore
├── requirements.txt
├── configs/
│   └── default.yaml            # Training / eval / prediction config
├── src/
│   └── satellite_image_processing/
│       ├── __init__.py
│       ├── __main__.py
│       ├── data.py              # Dataset loading & preprocessing
│       ├── model.py             # CNN architecture
│       ├── train.py             # Training CLI
│       ├── evaluate.py          # Evaluation CLI
│       └── predict.py           # Inference CLI
├── tests/
│   └── test_smoke.py            # Smoke tests
├── data/
│   └── README.md                # How to obtain the dataset
├── models/
│   └── README.md                # Trained model storage (gitignored)
└── reports/
    └── figures/                 # Generated plots (gitignored)
```

## Quick Start

### 1. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Obtain the dataset

Download the **EuroSAT all-bands** dataset and extract it into `data/`. See [`data/README.md`](data/README.md) for detailed instructions.

### 4. Train

```bash
python -m satellite_image_processing.train --config configs/default.yaml
```

You can override training parameters via the CLI:

```bash
python -m satellite_image_processing.train \
    --config configs/default.yaml \
    --epochs 10 \
    --batch-size 64 \
    --data-dir path/to/custom/data
```

### 5. Evaluate

```bash
python -m satellite_image_processing.evaluate --config configs/default.yaml
```

This prints a classification report and saves a confusion matrix to `reports/figures/confusion_matrix.png`.

### 6. Predict on a single image

```bash
python -m satellite_image_processing.predict \
    --config configs/default.yaml \
    --image path/to/sample.tif \
    --save-vis reports/figures/prediction.png
```

### 7. Run tests

```bash
python -m pytest tests/ -v
```

## Model Architecture

```
Input (64 × 64 × 3)
  → Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
  → Conv2D(64, 3×3, ReLU) → MaxPool(2×2)
  → Conv2D(128, 3×3, ReLU) → MaxPool(2×2)
  → Flatten
  → Dense(128, ReLU) → Dropout(0.5)
  → Dense(10, Softmax)
```

Compiled with Adam optimiser and sparse categorical cross-entropy loss.

## Configuration

All hyperparameters and paths are stored in `configs/default.yaml`. Edit this file or pass CLI flags to override values. No hardcoded absolute paths exist in the source code.

## License

This project is provided for educational purposes.
