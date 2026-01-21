from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Protocol, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from Models.models import ModelFactory
from Models.data_loader import get_class_names


class TorchModel(Protocol):
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class InferenceArtifacts:
    model: TorchModel
    class_names: List[str]
    device: str


def load_inference_model(
    model_name: str,
    num_classes: int,
    model_path: str,
    device: str = "cpu",
) -> torch.nn.Module:
    """
    Instantiate architecture and load trained weights.
    Mirrors your predict.py logic. :contentReference[oaicite:2]{index=2}
    """
    model = ModelFactory.create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
    )

    checkpoint: Any = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


_PREPROCESS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _load_pil_image(image_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as e:
        raise ValueError("Uploaded file is not a valid image.") from e
    return img


@torch.inference_mode()
def predict_image_bytes(
    artifacts: InferenceArtifacts,
    image_bytes: bytes,
) -> Tuple[str, float]:
    """
    Returns (class_name, confidence).
    Confidence is softmax probability of the predicted class.
    """
    img = _load_pil_image(image_bytes)
    img_tensor = _PREPROCESS(img).unsqueeze(0).to(artifacts.device)

    output = artifacts.model(img_tensor)
    probabilities = F.softmax(output, dim=1)
    confidence, predicted_idx = torch.max(probabilities, dim=1)

    class_name = artifacts.class_names[int(predicted_idx.item())]
    conf = float(confidence.item())
    return class_name, conf


def load_artifacts(
    *,
    data_dir: str,
    model_file: str,
    model_name: str,
    device: str,
) -> InferenceArtifacts:
    class_names = get_class_names(data_dir)
    model = load_inference_model(model_name, len(class_names), model_file, device=device)
    return InferenceArtifacts(model=model, class_names=class_names, device=device)
