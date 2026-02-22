"""CNN model architecture for EuroSAT satellite image classification (PyTorch)."""

import torch
import torch.nn as nn


class EuroSATCNN(nn.Module):
    """CNN classifier for EuroSAT satellite images.

    Architecture
    ------------
    Conv2d(32) → MaxPool → Conv2d(64) → MaxPool → Conv2d(128) → MaxPool →
    Flatten → Dense(128, relu) → Dropout(0.5) → Dense(num_classes)

    Parameters
    ----------
    image_size : tuple[int, int]
        Spatial dimensions (height, width) of the input images.
    num_classes : int
        Number of output classes.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (64, 64),
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        # Compute the flattened size after the conv layers
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *image_size)
            flat_size = self.features(dummy).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(
    image_size: tuple[int, int] = (64, 64),
    num_classes: int = 10,
    device: torch.device | None = None,
) -> EuroSATCNN:
    """Build the EuroSAT classification CNN and move it to *device*.

    Parameters
    ----------
    image_size : tuple[int, int]
        Spatial dimensions (height, width) of the input images.
    num_classes : int
        Number of output classes.
    device : torch.device, optional
        Target device (defaults to CUDA if available, else CPU).

    Returns
    -------
    EuroSATCNN
        A PyTorch model ready for training.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EuroSATCNN(image_size=image_size, num_classes=num_classes)
    return model.to(device)
