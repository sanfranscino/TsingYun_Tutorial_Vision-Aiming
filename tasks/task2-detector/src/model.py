"""MNIST digit model scaffold for Task 2.

Detector code should call the inference function in this module. Training code
lives in train.py so detector.py stays focused on board detection, corner
geometry, and PnP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

import torch
import torch.nn.functional as F

RgbPixel = tuple[int, int, int]
ImageLike = np.ndarray

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "mnist_classifier.npz"

def preprocess_mnist_crop(board_crop: ImageLike) -> np.ndarray:
    # 1. 转成 numpy 数组
    crop = np.asarray(board_crop)

    if crop.dtype != np.uint8:
        crop = np.clip(crop, 0, 255).astype(np.uint8)

    # 2. 转灰度图
    if crop.ndim == 3:
        # board_crop 一般是 RGB 图
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    elif crop.ndim == 2:
        gray = crop
    else:
        raise ValueError(f"Unsupported crop shape: {crop.shape}")

    # 3. 调整成 MNIST 的 28x28
    gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    # 4. 归一化到 [0, 1]
    normalized = gray.astype(np.float32) / 255.0

    # 5. 转成 CNN 输入格式: [batch, channel, height, width]
    model_input = normalized[np.newaxis, np.newaxis, :, :]

    return model_input


def load_mnist_model(model_path: Path = DEFAULT_MODEL_PATH) -> object:
    from train import MNISTClassifier

    if not model_path.exists():
        raise FileNotFoundError(
            f"MNIST model file not found: {model_path}. "
            "Please run: uv run python tasks/task2-detector/src/train.py"
        )

    model = MNISTClassifier()

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def predict_mnist_digit(model: object, model_input: np.ndarray) -> tuple[int, float]:
    # 1. numpy 转 torch tensor
    if isinstance(model_input, np.ndarray):
        input_tensor = torch.from_numpy(model_input).float()
    else:
        input_tensor = model_input.float()

    # 2. 推理，不计算梯度
    with torch.no_grad():
        logits = model(input_tensor)

        # 3. logits 转概率
        probabilities = F.softmax(logits, dim=1)

        # 4. 找概率最大的数字
        confidence_tensor, digit_tensor = torch.max(probabilities, dim=1)

    digit = int(digit_tensor.item())
    confidence = float(confidence_tensor.item())

    return digit, confidence


def classify_mnist_digit(board_crop: ImageLike, model_path: Path = DEFAULT_MODEL_PATH) -> tuple[int, float]:
    model_input = preprocess_mnist_crop(board_crop)

    model = load_mnist_model(model_path)

    digit, confidence = predict_mnist_digit(model, model_input)

    return digit, confidence
