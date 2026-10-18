
import pytest
import torch
from PIL import Image

from app import (
    CLASSES,
    CONFIDENCE_THRESHOLD,
    classify_image,
    load_model,
    transform,
)


def test_class_names():
    assert CLASSES == ["buildings", "forest", "sea"]


def test_preprocessing_shape():
    image = Image.new("RGB", (300, 200), color="green")
    tensor = transform(image)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_model_output_shape():
    model = load_model()
    dummy = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy)
    assert output.shape == (1, 3)


def test_missing_image():
    scores, message = classify_image(None)
    assert scores == {}
    assert "upload" in message.lower()


def test_tiny_image_rejected():
    image = Image.new("RGB", (10, 10), color="green")
    scores, message = classify_image(image)
    assert scores == {}
    assert "too small" in message.lower()


def test_probabilities_sum_to_one():
    image = Image.effect_noise((224, 224), 80).convert("RGB")
    scores, message = classify_image(image)
    assert len(scores) == 3
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-5)


def test_confidence_threshold_valid():
    assert 0 < CONFIDENCE_THRESHOLD < 1
