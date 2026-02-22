"""Training entrypoint for EuroSAT satellite image classification (PyTorch + CUDA)."""

import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    save_path = args.save_path or model_cfg.get("save_path", "models/eurosat_cnn_model.pth")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Force CUDA — abort if no GPU is visible
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA-capable GPU detected by PyTorch. "
            "Make sure you have a CUDA-compatible GPU, the correct "
            "NVIDIA drivers, and a CUDA-enabled PyTorch installation."
        )
    device = torch.device("cuda")
    logger.info("Using GPU: %s", torch.cuda.get_device_name(0))

    # Set seeds
    set_seeds(seed)
    logger.info("Random seed set to %d", seed)

    # Import heavy modules lazily (so --help is fast)
    from satellite_image_processing.data import load_dataset
    from satellite_image_processing.model import build_model

    # Load data
    logger.info("Loading dataset from: %s", dataset_path)
    train_loader, test_loader, categories, _, _ = load_dataset(
        dataset_path, image_size=image_size, test_split=test_split,
        seed=seed, batch_size=batch_size,
    )
    num_classes = len(categories)
    logger.info("Number of classes: %d", num_classes)

    # Build model
    model = build_model(image_size=image_size, num_classes=num_classes, device=device)
    logger.info("Model architecture:\n%s", model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # ── Training loop with tqdm ──────────────────────────────────────────
    logger.info("Training for %d epochs, batch_size=%d", epochs, batch_size)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # tqdm progress bar for each epoch
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs}",
            unit="batch",
            leave=True,
            colour="green",
        )

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar with live metrics
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.0 * correct / total:.2f}%",
            )

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total

        # ── Validation ───────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total

        logger.info(
            "Epoch %d/%d — train_loss: %.4f  train_acc: %.2f%%  |  val_loss: %.4f  val_acc: %.2f%%",
            epoch, epochs, epoch_loss, epoch_acc, val_loss, val_acc,
        )

    # ── Final evaluation ─────────────────────────────────────────────────
    logger.info("Test Loss: %.4f | Test Accuracy: %.2f%%", val_loss, val_acc)

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info("Model saved to: %s", save_path)


if __name__ == "__main__":
    main()
