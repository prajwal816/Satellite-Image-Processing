"""Single-image or batch inference for EuroSAT CNN classifier (PyTorch)."""

import argparse
import logging
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def predict_single(
    model: torch.nn.Module,
    img_path: str,
    categories: list[str],
    image_size: tuple[int, int] = (64, 64),
    device: torch.device | None = None,
    save_vis: str | None = None,
) -> str:
    """Run inference on a single .tif image and optionally save visualisation.

    Parameters
    ----------
    model : torch.nn.Module
        Trained classification model (already on *device*).
    img_path : str
        Path to a ``.tif`` satellite image.
    categories : list[str]
        Ordered list of class names.
    image_size : tuple[int, int]
        Input size expected by the model.
    device : torch.device, optional
        Device to run inference on.
    save_vis : str | None
        If given, save a visualisation (original + edge maps) to this path.

    Returns
    -------
    str
        Predicted class name.
    """
    from satellite_image_processing.data import load_tiff_image

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = load_tiff_image(img_path, image_size=image_size)  # shape: (3, H, W)
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)  # (1, 3, H, W)

    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        predicted_idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0][predicted_idx].item())

    predicted_class = categories[predicted_idx] if predicted_idx < len(categories) else str(predicted_idx)

    logger.info(
        "%s → %s (confidence: %.2f%%)", img_path, predicted_class, confidence * 100,
    )

    if save_vis:
        # Convert CHW → HWC for visualisation
        img_hwc = np.transpose(img, (1, 2, 0))
        img_uint8 = (img_hwc * 255).astype(np.uint8)
        img_gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        canny = cv2.Canny(img_gray, 100, 200)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_hwc)
        axes[0].set_title(f"Predicted: {predicted_class}")
        axes[0].axis("off")
        axes[1].imshow(laplacian, cmap="gray")
        axes[1].set_title("Laplacian Edge Detection")
        axes[1].axis("off")
        axes[2].imshow(canny, cmap="gray")
        axes[2].set_title("Canny Edge Detection")
        axes[2].axis("off")
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_vis) or ".", exist_ok=True)
        plt.savefig(save_vis, dpi=150)
        plt.close(fig)
        logger.info("Visualisation saved to: %s", save_vis)

    return predicted_class


def main(args: list[str] | None = None) -> None:
    """CLI entrypoint for single-image inference."""
    parser = argparse.ArgumentParser(description="EuroSAT CNN inference")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--model", type=str, default=None, help="Override model path")
    parser.add_argument("--image", type=str, required=True, help="Path to .tif image for prediction")
    parser.add_argument("--save-vis", type=str, default=None, help="Save visualisation to this path")
    parser.add_argument(
        "--categories", type=str, nargs="*", default=None,
        help="Class names (auto-detected from data dir if omitted)",
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset path for auto-detecting categories")
    parsed = parser.parse_args(args=args)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config
    cfg = load_config(parsed.config)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    model_path = parsed.model or model_cfg.get("save_path", "models/eurosat_cnn_model.pth")
    image_size = tuple(data_cfg.get("image_size", [64, 64]))

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Determine categories
    dataset_path = parsed.data_dir or data_cfg.get("dataset_path", "data/EuroSATallBands")
    if parsed.categories:
        categories = parsed.categories
    elif os.path.isdir(dataset_path):
        categories = sorted(
            d for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        )
    else:
        categories = [str(i) for i in range(10)]
        logger.warning("Could not detect categories; using numeric labels.")

    num_classes = len(categories)

    # Load model
    from satellite_image_processing.model import build_model

    logger.info("Loading model from: %s", model_path)
    model = build_model(image_size=image_size, num_classes=num_classes, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Predict
    predicted = predict_single(
        model, parsed.image, categories,
        image_size=image_size, device=device, save_vis=parsed.save_vis,
    )
    print(f"Predicted class: {predicted}")


if __name__ == "__main__":
    main()
