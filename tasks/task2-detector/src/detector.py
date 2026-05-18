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
    # TODO(student): Sort the four target corners into a stable order.
    # Input: four 2D corners in arbitrary order.
    # Output: corners ordered as top-left, top-right, bottom-right, bottom-left.
    # Compute a stable ordering rule that works for the target board geometry.
    raise NotImplementedError("order_corners is not implemented")


def detect_bbox(image: ImageLike, threshold: int = 200) -> list[CornerSet]:
    # TODO(student): Detect board candidates.
    # image_array = convert image to an OpenCV-compatible uint8 array
    # red_mask = threshold reddish pixels into a binary image
    # optionally clean red_mask with morphology so small noisy blobs disappear
    # contours = cv2.findContours(red_mask)
    # corner_candidates = []
    # for each contour:
    #     if contour area is too small:
    #         continue
    #     polygon = cv2.approxPolyDP(contour, epsilon, closed=true)
    #     if polygon does not have exactly 4 edges/corners:
    #         continue
    #     if polygon is not convex or has unreasonable aspect ratio:
    #         continue
    #     corners = order_corners(the four polygon vertices)
    #     append corners to corner_candidates
    # return corner_candidates
    raise NotImplementedError("detect_bbox is not implemented")


def detect_mnist_board(image: ImageLike, threshold: int = 200) -> list[Detection]:
    # TODO(student): Classify detected MNIST-board candidates and filter them.
    # Input: one RGB image and a threshold parameter.
    # Output: a list of Detection objects.
    # Step 1: call detect_bbox(...) to get board candidates.
    # Step 2: call crop_bbox(...) to extract candidate crops.
    # Step 3: call classify_mnist_digit(...) on each crop and filter low-confidence results.
    # Step 4: package the remaining results as Detection objects.
    raise NotImplementedError("detect_mnist_board is not implemented")


def solve_pnp(
    detections: Sequence[Detection],
    camera_matrix: Matrix3x3,
    board_width_meters: float,
    board_height_meters: float,
    dist_coeffs: Sequence[float] | None = None,
) -> list[Detection]:
    # TODO(student): Fill rvec and tvec for every valid Detection.
    # half_width = board_width_meters / 2
    # half_height = board_height_meters / 2
    # object_points = four physical board corners as float32:
    #     (-half_width, -half_height, 0)
    #     (half_width, -half_height, 0)
    #     (half_width, half_height, 0)
    #     (-half_width, half_height, 0)
    # camera_array = camera_matrix as a 3x3 float64 array
    # dist_array = zero distortion if dist_coeffs is not provided
    # result = []
    # for each detection:
    #     image_points = detection.corners as a float32 4x2 array
    #     call cv2.solvePnP with object_points, image_points, camera_array, and dist_array
    #     if OpenCV reports failure:
    #         skip this detection or raise a clear error
    #     fill detection.rvec and detection.tvec with the OpenCV result
    #     append detection to result
    # return result
    raise NotImplementedError("solve_pnp is not implemented")
