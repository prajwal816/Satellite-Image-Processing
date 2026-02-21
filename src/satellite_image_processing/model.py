"""CNN model architecture for EuroSAT satellite image classification."""

from tensorflow import keras
from tensorflow.keras import layers


def build_model(
    image_size: tuple[int, int] = (64, 64),
    num_classes: int = 10,
) -> keras.Sequential:
    """Build and compile the EuroSAT classification CNN.

    Architecture
    ------------
    Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool →
    Flatten → Dense(128, relu) → Dropout(0.5) → Dense(num_classes, softmax)

    Parameters
    ----------
    image_size : tuple[int, int]
        Spatial dimensions (height, width) of the input images.
    num_classes : int
        Number of output classes.

    Returns
    -------
    keras.Sequential
        A compiled Keras model ready for training.
    """
    model = keras.Sequential([
        layers.Input(shape=(*image_size, 3)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
