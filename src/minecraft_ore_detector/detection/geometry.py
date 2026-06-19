"""Shared bounding-box geometry."""

from typing import Iterable, Tuple

Box = Tuple[int, int, int, int]


def clip_box(box: Box, image_shape: Tuple[int, ...]) -> Box:
    x, y, width, height = box
    image_height, image_width = image_shape[:2]
    x = max(0, min(int(x), image_width - 1))
    y = max(0, min(int(y), image_height - 1))
    width = max(1, min(int(width), image_width - x))
    height = max(1, min(int(height), image_height - y))
    return x, y, width, height


def box_iou(box_a: Box, box_b: Box) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    intersection_x1 = max(ax, bx)
    intersection_y1 = max(ay, by)
    intersection_x2 = min(ax + aw, bx + bw)
    intersection_y2 = min(ay + ah, by + bh)
    intersection = (
        max(0, intersection_x2 - intersection_x1)
        * max(0, intersection_y2 - intersection_y1)
    )
    union = aw * ah + bw * bh - intersection
    return intersection / float(union) if union > 0 else 0.0


def containment_ratio(box_a: Box, box_b: Box) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    intersection_x1 = max(ax, bx)
    intersection_y1 = max(ay, by)
    intersection_x2 = min(ax + aw, bx + bw)
    intersection_y2 = min(ay + ah, by + bh)
    intersection = (
        max(0, intersection_x2 - intersection_x1)
        * max(0, intersection_y2 - intersection_y1)
    )
    smaller_area = min(aw * ah, bw * bh)
    return intersection / float(smaller_area) if smaller_area > 0 else 0.0


def center_distance(box_a: Box, box_b: Box) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    center_a = (ax + aw / 2.0, ay + ah / 2.0)
    center_b = (bx + bw / 2.0, by + bh / 2.0)
    return (
        (center_a[0] - center_b[0]) ** 2
        + (center_a[1] - center_b[1]) ** 2
    ) ** 0.5


def overlaps_any_box(
    box: Box,
    other_boxes: Iterable[Box],
    iou_threshold: float,
) -> bool:
    return any(
        box_iou(box, other_box) > iou_threshold
        for other_box in other_boxes
    )
