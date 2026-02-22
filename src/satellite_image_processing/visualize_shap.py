"""SHAP explainability visualizations for EuroSAT CNN classifier.

Uses SHAP GradientExplainer to produce pixel-level importance maps
and mean absolute SHAP value bar charts.
"""

import argparse
import logging
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def plot_shap_images(
    shap_values: np.ndarray,
    images: np.ndarray,
    categories: list[str],
    true_labels: np.ndarray,
    save_path: str,
    n_samples: int = 4,
) -> None:
    """Plot original images alongside their SHAP attribution heatmaps."""
    n = min(n_samples, len(images))
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))

    for i in range(n):
        img = images[i]
        true_cls = categories[true_labels[i]] if true_labels[i] < len(categories) else str(true_labels[i])

        # SHAP values for the predicted class — sum across RGB channels
        shap_for_pred = shap_values[true_labels[i]][i]  # (H, W, 3)
        shap_abs = np.abs(shap_for_pred).sum(axis=-1)   # (H, W)
        shap_signed = shap_for_pred.sum(axis=-1)         # (H, W)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"True: {true_cls}", fontsize=11, fontweight="bold")
        axes[i, 0].axis("off")

        im1 = axes[i, 1].imshow(shap_abs, cmap="hot")
        axes[i, 1].set_title("|SHAP| Attribution", fontsize=11)
        axes[i, 1].axis("off")
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Overlay
        axes[i, 2].imshow(img)
        axes[i, 2].imshow(shap_abs, cmap="jet", alpha=0.5)
        axes[i, 2].set_title("SHAP Overlay", fontsize=11)
        axes[i, 2].axis("off")

    fig.suptitle("SHAP Feature Attribution — EuroSAT CNN", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("SHAP image plot saved to: %s", save_path)


def plot_shap_mean_values(
    shap_values: np.ndarray,
    categories: list[str],
    save_path: str,
) -> None:
    """Bar chart of mean |SHAP| value per class."""
    mean_shap = []
    for cls_idx in range(len(categories)):
        vals = shap_values[cls_idx]  # (n_samples, H, W, 3)
        mean_shap.append(np.abs(vals).mean())

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(categories)))
    bars = ax.barh(categories, mean_shap, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean |SHAP| Value", fontsize=12)
    ax.set_title("Mean Absolute SHAP Values per Class", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Annotate bar values
    for bar, val in zip(bars, mean_shap):
        ax.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("SHAP mean values plot saved to: %s", save_path)


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SHAP explainability visualizations")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--figures-dir", type=str, default=None)
    parser.add_argument("--n-background", type=int, default=50, help="Number of background samples for SHAP")
    parser.add_argument("--n-explain", type=int, default=4, help="Number of images to explain")
    parsed = parser.parse_args(args=args)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = load_config(parsed.config)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    output_cfg = cfg.get("output", {})

    dataset_path = parsed.data_dir or data_cfg.get("dataset_path", "data/EuroSATallBands")
    image_size = tuple(data_cfg.get("image_size", [64, 64]))
    model_path = parsed.model or model_cfg.get("save_path", "models/eurosat_cnn_model.pth")
    figures_dir = parsed.figures_dir or output_cfg.get("figures_dir", "reports/figures")
    os.makedirs(figures_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ── Load data ────────────────────────────────────────────────────────
    from satellite_image_processing.data import load_dataset

    logger.info("Loading dataset from: %s", dataset_path)
    _, _, categories, x_test, y_test = load_dataset(
        dataset_path, image_size=image_size, test_split=0.2, seed=42,
    )

    # ── Load model ───────────────────────────────────────────────────────
    from satellite_image_processing.model import build_model

    logger.info("Loading model from: %s", model_path)
    model = build_model(image_size=image_size, num_classes=len(categories), device=device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # ── Prepare samples ──────────────────────────────────────────────────
    n_bg = min(parsed.n_background, len(x_test))
    n_explain = min(parsed.n_explain, len(x_test))

    # Pick diverse explain samples — one per class where possible
    explain_indices = []
    for cls_idx in range(len(categories)):
        matches = (y_test == cls_idx).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            explain_indices.append(matches[0].item())
        if len(explain_indices) >= n_explain:
            break
    # Pad with random indices if needed
    while len(explain_indices) < n_explain:
        idx = np.random.randint(0, len(x_test))
        if idx not in explain_indices:
            explain_indices.append(idx)

    background = x_test[:n_bg].to(device)
    explain_images = x_test[explain_indices].to(device)
    explain_labels = y_test[explain_indices].numpy()

    # ── SHAP GradientExplainer ───────────────────────────────────────────
    import shap

    logger.info("Creating SHAP GradientExplainer with %d background samples...", n_bg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.GradientExplainer(model, background)

        logger.info("Computing SHAP values for %d samples...", n_explain)
        shap_values = explainer.shap_values(explain_images)

    # shap_values is a list of arrays (one per class): each (n_explain, C, H, W)
    # Convert to (n_classes, n_explain, H, W, C) for plotting
    shap_values_nhwc = []
    for cls_shap in shap_values:
        # cls_shap: (n_explain, 3, H, W) → (n_explain, H, W, 3)
        shap_values_nhwc.append(np.transpose(cls_shap.cpu().numpy() if isinstance(cls_shap, torch.Tensor) else cls_shap, (0, 2, 3, 1)))
    shap_values_nhwc = np.array(shap_values_nhwc)

    # Images for display: CHW → HWC
    explain_imgs_np = explain_images.cpu().numpy().transpose(0, 2, 3, 1)
    explain_imgs_np = np.clip(explain_imgs_np, 0, 1)

    # ── Plot ─────────────────────────────────────────────────────────────
    plot_shap_images(
        shap_values_nhwc, explain_imgs_np, categories, explain_labels,
        os.path.join(figures_dir, "shap_explanation.png"),
        n_samples=n_explain,
    )

    plot_shap_mean_values(
        shap_values_nhwc, categories,
        os.path.join(figures_dir, "shap_mean_values.png"),
    )

    logger.info("All SHAP visualizations complete.")


if __name__ == "__main__":
    main()
