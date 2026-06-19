"""Shared statistics for binary masks."""

import cv2
import numpy as np


def mask_integral(mask: np.ndarray) -> np.ndarray:
    return cv2.integral((mask > 0).astype(np.uint8))


def integral_support(
    integral: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
) -> float:
    x2 = x + width
    y2 = y + height
    total = (
        integral[y2, x2]
        - integral[y, x2]
        - integral[y2, x]
        + integral[y, x]
    )
    return float(total) / float(max(1, width * height))
