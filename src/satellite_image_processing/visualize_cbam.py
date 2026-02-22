"""CBAM (Convolutional Block Attention Module) visualization for EuroSAT CNN.

Generates channel-attention bar charts, spatial-attention heatmap overlays,
and a multi-sample attention grid for the trained model.
"""

import argparse
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)


# ── CBAM modules ────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """Channel attention sub-module (squeeze-and-excitation style)."""

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(in_channels // reduction, 1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, mid),
            nn.ReLU(),
            nn.Linear(mid, in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_pool = x.mean(dim=[2, 3])                               # (B, C)
        max_pool = x.amax(dim=[2, 3])                               # (B, C)
        att = torch.sigmoid(self.shared_mlp(avg_pool) + self.shared_mlp(max_pool))  # (B, C)
        return att.view(b, c, 1, 1) * x


class SpatialAttention(nn.Module):
    """Spatial attention sub-module."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)                       # (B, 1, H, W)
        max_out = x.amax(dim=1, keepdim=True)                       # (B, 1, H, W)
        combined = torch.cat([avg_out, max_out], dim=1)              # (B, 2, H, W)
        att_map = torch.sigmoid(self.conv(combined))                 # (B, 1, H, W)
        return att_map, att_map * x


class CBAM(nn.Module):
    """Full CBAM block: channel attention → spatial attention."""

    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor):
        x = self.channel_att(x)
        spatial_map, x = self.spatial_att(x)
        return x, spatial_map


# ── Helper: extract feature maps from a Conv2d layer ────────────────────────

def get_feature_maps(model, layer_idx: int, input_tensor: torch.Tensor):
    """Run a forward pass and capture the output of a specific conv layer."""
    model.eval()
    feature_maps = []

    def hook_fn(module, inp, out):
        feature_maps.append(out.detach())

    # Register hook on the target layer
    target_layer = None
    conv_count = 0
    for module in model.features:
        if isinstance(module, nn.Conv2d):
            if conv_count == layer_idx:
                target_layer = module
                break
            conv_count += 1

    if target_layer is None:
        raise ValueError(f"Conv layer index {layer_idx} not found in model.features")

    handle = target_layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(input_tensor)
    handle.remove()

    return feature_maps[0]


# ── Plotting helpers ────────────────────────────────────────────────────────

def plot_channel_attention(channel_weights: np.ndarray, save_path: str) -> None:
    """Bar chart of per-channel attention weights."""
    fig, ax = plt.subplots(figsize=(12, 5))
    channels = np.arange(len(channel_weights))
    colors = plt.cm.viridis(channel_weights / channel_weights.max())
    ax.bar(channels, channel_weights, color=colors, edgecolor="none")
    ax.set_xlabel("Channel Index", fontsize=12)
    ax.set_ylabel("Attention Weight", fontsize=12)
    ax.set_title("CBAM Channel Attention Weights", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Channel attention plot saved to: %s", save_path)


def plot_spatial_attention(
    original_img: np.ndarray,
    spatial_map: np.ndarray,
    save_path: str,
    title: str = "CBAM Spatial Attention",
) -> None:
    """Overlay spatial attention heatmap on the original image."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    im = axes[1].imshow(spatial_map, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Spatial Attention Map", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(original_img)
    axes[2].imshow(spatial_map, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title("Attention Overlay", fontsize=12)
    axes[2].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Spatial attention plot saved to: %s", save_path)


def plot_attention_grid(
    images: list[np.ndarray],
    spatial_maps: list[np.ndarray],
    labels: list[str],
    save_path: str,
) -> None:
    """3×3 grid of images with attention overlays."""
    n = min(len(images), 9)
    rows = (n + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if n == 1 else axes

    for i in range(n):
        axes[i].imshow(images[i])
        axes[i].imshow(spatial_maps[i], cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[i].set_title(labels[i], fontsize=11, fontweight="bold")
        axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle("CBAM Attention Maps — Sample Images", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Attention grid saved to: %s", save_path)


# ── Main ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CBAM attention visualizations")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--figures-dir", type=str, default=None)
    parser.add_argument("--layer", type=int, default=2, help="Conv layer index (0-based) to visualize")
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

    # ── Pick one representative sample per class ─────────────────────────
    sample_indices = []
    for cls_idx in range(len(categories)):
        matches = (y_test == cls_idx).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            sample_indices.append(matches[0].item())
    sample_indices = sample_indices[:9]  # max 9 for grid

    # ── Compute CBAM attention ───────────────────────────────────────────
    layer_idx = parsed.layer
    # Determine the number of channels at the target conv layer
    conv_channels = [32, 64, 128]
    in_ch = conv_channels[min(layer_idx, len(conv_channels) - 1)]
    cbam = CBAM(in_channels=in_ch).to(device)
    cbam.eval()

    images_for_grid = []
    spatial_maps_for_grid = []
    labels_for_grid = []

    for i, idx in enumerate(sample_indices):
        img_tensor = x_test[idx].unsqueeze(0).to(device)  # (1, 3, H, W)

        # Get feature maps from the selected conv layer
        feat = get_feature_maps(model, layer_idx, img_tensor)

        # Apply CBAM
        with torch.no_grad():
            _, spatial_map = cbam(feat)  # spatial_map: (1, 1, h, w)

        spatial_np = spatial_map[0, 0].cpu().numpy()
        # Resize spatial map to original image size
        import cv2
        spatial_resized = cv2.resize(spatial_np, image_size, interpolation=cv2.INTER_LINEAR)

        # Original image in HWC for display
        img_hwc = x_test[idx].numpy().transpose(1, 2, 0)
        img_hwc = np.clip(img_hwc, 0, 1)

        images_for_grid.append(img_hwc)
        spatial_maps_for_grid.append(spatial_resized)
        labels_for_grid.append(categories[y_test[idx].item()])

        # Save the first sample as the main spatial attention plot
        if i == 0:
            plot_spatial_attention(
                img_hwc, spatial_resized,
                os.path.join(figures_dir, "cbam_spatial_attention.png"),
                title=f"CBAM Spatial Attention — {categories[y_test[idx].item()]}",
            )

            # Channel attention: get weights from channel attention sub-module
            ch_att = cbam.channel_att
            avg_pool = feat.mean(dim=[2, 3])
            max_pool = feat.amax(dim=[2, 3])
            with torch.no_grad():
                ch_weights = torch.sigmoid(ch_att.shared_mlp(avg_pool) + ch_att.shared_mlp(max_pool))
            ch_np = ch_weights[0].cpu().numpy()
            plot_channel_attention(ch_np, os.path.join(figures_dir, "cbam_channel_attention.png"))

    # ── Attention grid ───────────────────────────────────────────────────
    plot_attention_grid(
        images_for_grid, spatial_maps_for_grid, labels_for_grid,
        os.path.join(figures_dir, "cbam_attention_grid.png"),
    )

    logger.info("All CBAM visualizations complete.")


if __name__ == "__main__":
    main()
