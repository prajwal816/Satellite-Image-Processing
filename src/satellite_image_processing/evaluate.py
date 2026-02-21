"""Evaluation script for a trained EuroSAT CNN model."""

import argparse
import logging
import os

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main(args: list[str] | None = None) -> None:
    """Evaluate a saved model on the test split."""
    parser = argparse.ArgumentParser(description="Evaluate EuroSAT CNN classifier")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--model", type=str, default=None, help="Override model path")
    parser.add_argument("--data-dir", type=str, default=None, help="Override dataset path")
    parser.add_argument("--figures-dir", type=str, default=None, help="Override figures output dir")
    args = parser.parse_args(args=args)

    # Load config
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    output_cfg = cfg.get("output", {})

    dataset_path = args.data_dir or data_cfg.get("dataset_path", "data/EuroSATallBands")
    image_size = tuple(data_cfg.get("image_size", [64, 64]))
    test_split = data_cfg.get("test_split", 0.2)
    seed = data_cfg.get("seed", 42)
    model_path = args.model or model_cfg.get("save_path", "models/eurosat_cnn_model.h5")
    figures_dir = args.figures_dir or output_cfg.get("figures_dir", "reports/figures")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load data (only the test split is used)
    from satellite_image_processing.data import load_dataset

    logger.info("Loading dataset from: %s", dataset_path)
    _, x_test, _, y_test, categories = load_dataset(
        dataset_path, image_size=image_size, test_split=test_split, seed=seed,
    )

    # Load model
    logger.info("Loading model from: %s", model_path)
    model = keras.models.load_model(model_path)

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    logger.info("Test Loss: %.4f | Test Accuracy: %.4f", test_loss, test_acc)

    # Classification report
    y_pred = np.argmax(model.predict(x_test), axis=1)
    report = classification_report(y_test, y_pred, target_names=categories)
    logger.info("Classification Report:\n%s", report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=categories, yticklabels=categories)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    os.makedirs(figures_dir, exist_ok=True)
    cm_path = os.path.join(figures_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    logger.info("Confusion matrix saved to: %s", cm_path)

    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()
