"""Smoke tests for the satellite_image_processing package."""

import numpy as np
import pytest


def test_imports():
    """All package modules should import without errors."""
    import satellite_image_processing
    import satellite_image_processing.data
    import satellite_image_processing.model
    import satellite_image_processing.train
    import satellite_image_processing.evaluate
    import satellite_image_processing.predict


def test_build_model_default():
    """build_model() should return a compiled Keras model with the correct output shape."""
    from satellite_image_processing.model import build_model

    model = build_model(image_size=(64, 64), num_classes=10)
    # Output layer should have 10 units
    assert model.output_shape == (None, 10)
    # Model should already be compiled (has an optimizer)
    assert model.optimizer is not None


def test_forward_pass():
    """A forward pass with random data should produce valid softmax predictions."""
    from satellite_image_processing.model import build_model

    model = build_model(image_size=(64, 64), num_classes=10)

    # Create a tiny random batch
    batch = np.random.rand(2, 64, 64, 3).astype(np.float32)
    predictions = model.predict(batch, verbose=0)

    assert predictions.shape == (2, 10)
    # Each row should sum to ~1 (softmax output)
    for row in predictions:
        assert abs(float(np.sum(row)) - 1.0) < 1e-5
