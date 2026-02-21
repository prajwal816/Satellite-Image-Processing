"""Training entrypoint for EuroSAT satellite image classification."""

import argparse
import logging
import os
import random

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def load_config(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main(args: argparse.Namespace | None = None) -> None:
    """Run the training pipeline."""
    parser = argparse.ArgumentParser(description="Train EuroSAT CNN classifier")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--data-dir", type=str, default=None, help="Override dataset path")
    parser.add_argument("--save-path", type=str, default=None, help="Override model save path")
    args = parser.parse_args(args=args)

    # Load config
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    # Apply CLI overrides
    dataset_path = args.data_dir or data_cfg.get("dataset_path", "data/EuroSATallBands")
    image_size = tuple(data_cfg.get("image_size", [64, 64]))
    test_split = data_cfg.get("test_split", 0.2)
    seed = data_cfg.get("seed", 42)
    epochs = args.epochs or train_cfg.get("epochs", 20)
    batch_size = args.batch_size or train_cfg.get("batch_size", 32)
    save_path = args.save_path or model_cfg.get("save_path", "models/eurosat_cnn_model.h5")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Set seeds
    set_seeds(seed)
    logger.info("Random seed set to %d", seed)

    # Import heavy modules lazily (so --help is fast)
    from satellite_image_processing.data import load_dataset
    from satellite_image_processing.model import build_model

    # Load data
    logger.info("Loading dataset from: %s", dataset_path)
    x_train, x_test, y_train, y_test, categories = load_dataset(
        dataset_path, image_size=image_size, test_split=test_split, seed=seed,
    )
    num_classes = len(categories)
    logger.info("Number of classes: %d", num_classes)

    # Build model
    model = build_model(image_size=image_size, num_classes=num_classes)
    model.summary()

    # Train
    logger.info("Training for %d epochs, batch_size=%d", epochs, batch_size)
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test)
    logger.info("Test Loss: %.4f | Test Accuracy: %.4f", test_loss, test_acc)

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    logger.info("Model saved to: %s", save_path)


if __name__ == "__main__":
    main()
