"""Task 2 MNIST-board detector helpers with student TODO extension points.

This file belongs to Task 2. The simulator runner imports it so that a Task 2
implementation can be tested both offline and inside the Unity simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from simulator_client.protocol import Matrix3x3
from model import classify_mnist_digit

Point2D = tuple[float, float]
CornerSet = tuple[Point2D, Point2D, Point2D, Point2D]
RgbPixel = tuple[int, int, int]
ImageLike = np.ndarray
WARP_OUTPUT_SIZE = 128
MNIST_INNER_RATIO = 0.69


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> Point2D:
        return (self.x + self.width * 0.5, self.y + self.height * 0.5)


@dataclass
class Detection:
    class_id: int
    confidence: float
    bbox: BoundingBox
    corners: CornerSet
    rvec: object | None = None
    tvec: object | None = None


def _bbox_from_corners(corners: Sequence[Point2D]) -> BoundingBox:
    if len(corners) != 4:
        raise ValueError(f"Expected 4 corners, got {len(corners)}")

    xs = [float(point[0]) for point in corners]
    ys = [float(point[1]) for point in corners]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    return BoundingBox(
        x=min_x,
        y=min_y,
        width=max_x - min_x + 1.0,
        height=max_y - min_y + 1.0,
    )


def _crop_bounds(corners: Sequence[Point2D], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    bbox = _bbox_from_corners(corners)
    x0 = max(0, min(image_width, int(np.floor(bbox.x))))
    y0 = max(0, min(image_height, int(np.floor(bbox.y))))
    x1 = max(0, min(image_width, int(np.ceil(bbox.x + bbox.width))))
    y1 = max(0, min(image_height, int(np.ceil(bbox.y + bbox.height))))
    return x0, y0, x1, y1


def crop_bbox(image: np.ndarray, corner_candidates: Sequence[Sequence[Point2D]]) -> list[np.ndarray]:
    crops: list[np.ndarray] = []
    for corners in corner_candidates:
        if len(corners) != 4:
            continue

        # `corners` are expected in LU, RU, RD, LD order.
        src = np.array(corners, dtype=np.float32)

        # Shrink the source quad toward its center so the warp removes the outer red border.
        center = np.mean(src, axis=0)
        src = center + (src - center) * MNIST_INNER_RATIO

        dst = np.array(
            [
                [0, 0],
                [WARP_OUTPUT_SIZE - 1, 0],
                [WARP_OUTPUT_SIZE - 1, WARP_OUTPUT_SIZE - 1],
                [0, WARP_OUTPUT_SIZE - 1],
            ],
            dtype=np.float32,
        )

        perspective = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            image,
            perspective,
            (WARP_OUTPUT_SIZE, WARP_OUTPUT_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        crops.append(warped)
    return crops


def order_corners(corners: Sequence[Point2D]) -> CornerSet:
    pts = np.array(corners, dtype=np.float32)

    if pts.shape != (4, 2):
        raise ValueError(f"Expected four 2D corners, got shape {pts.shape}")

    s = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmax(diff)]
    bottom_left = pts[np.argmin(diff)]

    return (
        (float(top_left[0]), float(top_left[1])),
        (float(top_right[0]), float(top_right[1])),
        (float(bottom_right[0]), float(bottom_right[1])),
        (float(bottom_left[0]), float(bottom_left[1])),
    )


def detect_bbox(image: ImageLike, threshold: int = 200) -> list[CornerSet]:
    # 1. 转成 OpenCV 能处理的 uint8 图像
    image_array = np.asarray(image)

    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    # 2. 提取 RGB 三个通道
    # 注意：这里默认输入 image 是 RGB 格式
    r = image_array[:, :, 0]
    g = image_array[:, :, 1]
    b = image_array[:, :, 2]

    # 3. 找红色区域
    # 红色特点：R 很大，G/B 相对较小
    red_mask = (
        (r > threshold) &
        (g < 120) &
        (b < 120)
    ).astype(np.uint8) * 255

    # 4. 形态学处理，去掉小噪点，让红框更连续
    kernel = np.ones((5, 5), dtype=np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    # 5. 找轮廓
    contours, _ = cv2.findContours(
        red_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    corner_candidates: list[CornerSet] = []

    for contour in contours:
        # 6. 面积太小的不要，基本是噪声
        area = cv2.contourArea(contour)
        if area < 500:
            continue

        # 7. 把轮廓近似成多边形
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True,
        )

        # 8. 只要四边形
        if len(polygon) != 4:
            continue

        # 9. 必须是凸四边形
        if not cv2.isContourConvex(polygon):
            continue

        # 10. 检查宽高比，太扁/太细的一般不是目标板
        x, y, w, h = cv2.boundingRect(polygon)
        if w <= 0 or h <= 0:
            continue

        aspect_ratio = w / h
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            continue

        # 11. 整理四个角点格式
        points = polygon.reshape(4, 2)
        corners = order_corners(points)

        corner_candidates.append(corners)

    return corner_candidates


def detect_mnist_board(image: ImageLike, threshold: int = 200) -> list[Detection]:
    # 1. 找出所有红色目标板的四个角点
    corner_candidates = detect_bbox(image, threshold=threshold)

    if not corner_candidates:
        return []

    # 2. 根据角点裁剪出数字区域
    crops = crop_bbox(image, corner_candidates)

    detections: list[Detection] = []

    # 3. 对每个裁剪出来的小图识别数字
    for corners, crop in zip(corner_candidates, crops):
        class_id, confidence = classify_mnist_digit(crop)

        # 4. 过滤掉置信度太低的识别结果
        if confidence < 0.5:
            continue

        # 5. 根据角点生成 bbox
        bbox = _bbox_from_corners(corners)

        # 6. 打包成 Detection 对象
        detection = Detection(
            class_id=int(class_id),
            confidence=float(confidence),
            bbox=bbox,
            corners=corners,
        )

        detections.append(detection)

    return detections


def solve_pnp(
    detections: Sequence[Detection],
    camera_matrix: Matrix3x3,
    board_width_meters: float,
    board_height_meters: float,
    dist_coeffs: Sequence[float] | None = None,
) -> list[Detection]:
    # 1. 目标板真实世界里的四个角点坐标，单位是米
    half_width = board_width_meters / 2.0
    half_height = board_height_meters / 2.0

    object_points = np.array(
        [
            [-half_width, -half_height, 0.0],  # 左上
            [ half_width, -half_height, 0.0],  # 右上
            [ half_width,  half_height, 0.0],  # 右下
            [-half_width,  half_height, 0.0],  # 左下
        ],
        dtype=np.float32,
    )

    # 2. 相机内参矩阵
    camera_array = np.array(camera_matrix, dtype=np.float64)

    # 3. 畸变参数，没有就默认无畸变
    if dist_coeffs is None:
        dist_array = np.zeros((5, 1), dtype=np.float64)
    else:
        dist_array = np.array(dist_coeffs, dtype=np.float64)

    result: list[Detection] = []

    for detection in detections:
        # 4. 图像里的四个角点
        image_points = np.array(detection.corners, dtype=np.float32)

        if image_points.shape != (4, 2):
            continue

        # 5. PnP 解算
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_array,
            dist_array,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            continue

        # 6. 填回 detection 里面
        detection.rvec = rvec
        detection.tvec = tvec

        result.append(detection)

    return result
