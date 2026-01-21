""" Code to test the predictions part of the API """
from __future__ import annotations

from io import BytesIO
from typing import List

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.inference import InferenceArtifacts


class DummyModel:
    """Returns fixed logits so we can assert deterministic output."""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # batch_size x num_classes
        # Make class index 1 the winner.
        batch = x.shape[0]
        return torch.tensor([[0.1, 2.0, -1.0]]).repeat(batch, 1)


def make_test_image_bytes() -> bytes:
    img = Image.new("RGB", (300, 300), color=(120, 200, 80))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def client() -> TestClient:
    # Inject dummy artifacts before creating client
    app.state.artifacts = InferenceArtifacts(
        model=DummyModel(),
        class_names=["class0", "class1", "class2"],
        device="cpu",
    )
    return TestClient(app)


def test_predict_happy_path(client: TestClient):
    img_bytes = make_test_image_bytes()
    resp = client.post(
        "/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["class_name"] == "class1"
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_rejects_non_image_content_type(client: TestClient):
    resp = client.post(
        "/predict",
        files={"file": ("test.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 415


def test_predict_rejects_invalid_image_bytes(client: TestClient):
    resp = client.post(
        "/predict",
        files={"file": ("bad.jpg", b"not an image", "image/jpeg")},
    )
    assert resp.status_code == 400
    assert "not a valid image" in resp.text.lower()
