"""Dataset loading and preprocessing for EuroSAT satellite images (PyTorch)."""

import logging
import os

import cv2
import numpy as np
import tifffile as tiff
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def load_tiff_image(img_path: str, image_size: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Load a single TIFF satellite image, extract RGB bands, resize and normalise.

    Parameters
    ----------
    img_path : str
        Absolute or relative path to a ``.tif`` file.
    image_size : tuple[int, int]
        Target (height, width) after resizing.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(3, *image_size)`` with values in ``[0, 1]``.
    """
    img = tiff.imread(img_path)

    # Handle varying band counts
    if len(img.shape) == 2:
        # Single-band (greyscale) → duplicate to 3 channels
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] >= 3:
        img = img[:, :, :3]  # Take first 3 bands (R, G, B)
    else:
        img = np.stack([img[:, :, 0]] * 3, axis=-1)

    img = cv2.resize(img, image_size)
    img = np.array(img, dtype=np.float32) / 255.0
    # Convert from HWC to CHW (PyTorch convention)
    img = np.transpose(img, (2, 0, 1))
    return img


def load_dataset(
    dataset_path: str,
    image_size: tuple[int, int] = (64, 64),
    test_split: float = 0.2,
    seed: int = 42,
    batch_size: int = 32,
):
    """Load the EuroSAT dataset and return PyTorch DataLoaders.

    The directory is expected to contain one sub-folder per class (e.g.
    ``AnnualCrop/``, ``Forest/``, …), each holding ``.tif`` image files.

    Parameters
    ----------
    dataset_path : str
        Path to the root of the extracted EuroSAT TIFF dataset.
    image_size : tuple[int, int]
        Target (height, width) for every image.
    test_split : float
        Fraction of data used for testing (default 0.2).
    seed : int
        Random state for reproducible train/test splitting.
    batch_size : int
        Batch size for the DataLoaders.

    Returns
    -------
    tuple
        ``(train_loader, test_loader, categories, x_test_tensor, y_test_tensor)``
    """
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_path}\n"
            "Please download the EuroSAT dataset first.  "
            "See data/README.md for instructions."
        )

    categories = sorted(os.listdir(dataset_path))
    categories = [c for c in categories if os.path.isdir(os.path.join(dataset_path, c))]
    num_classes = len(categories)

    logger.info("Found %d classes: %s", num_classes, categories)

    x_data, y_data = [], []

    for label, category in enumerate(categories):
        category_path = os.path.join(dataset_path, category)
        files = [f for f in os.listdir(category_path) if f.lower().endswith(".tif")]
        logger.info("  %s: %d images", category, len(files))

        for img_name in files:
            img_path = os.path.join(category_path, img_name)
            try:
                img = load_tiff_image(img_path, image_size)
                x_data.append(img)
                y_data.append(label)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping %s: %s", img_path, exc)

    x_data = np.array(x_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.int64)

    logger.info("Total samples loaded: %d", len(x_data))

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=test_split, random_state=seed,
    )

    logger.info("Train samples: %d | Test samples: %d", len(x_train), len(x_test))

    # Convert to PyTorch tensors
    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    x_test_t = torch.from_numpy(x_test)
    y_test_t = torch.from_numpy(y_test)

    # Build DataLoaders
    train_ds = TensorDataset(x_train_t, y_train_t)
    test_ds = TensorDataset(x_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, categories, x_test_t, y_test_t
