# -*- coding: utf-8 -*-
"""Contour-based candidate box extraction."""

from typing import List, Optional, Tuple

import cv2
import numpy as np

def _merge_nearby_boxes(
    boxes: List[Tuple[int, int, int, int]],
    gap: int = 12
) -> List[Tuple[int, int, int, int]]:
    """
    Führt nahe oder überlappende Bounding Boxes zusammen.

    Das ist wichtig, weil ein einzelner Erzblock oft aus mehreren kleinen farbigen
    Bereichen besteht, die sonst als einzelne Kandidaten behandelt würden.

    Parameters
    ----------
    boxes : List[Tuple[int, int, int, int]]
        Liste von Boxen im Format (x, y, w, h).
    gap : int
        Maximale Lücke zwischen Boxen, damit sie zusammengeführt werden.

    Returns
    -------
    List[Tuple[int, int, int, int]]
        Zusammengeführte Boxen.
    """

    if not boxes:
        return []

    work = [
        [x, y, x + w, y + h]
        for (x, y, w, h) in boxes
    ]

    changed = True

    while changed:
        changed = False
        new_boxes = []
        used = [False] * len(work)

        for i in range(len(work)):
            if used[i]:
                continue

            x1, y1, x2, y2 = work[i]
            used[i] = True

            merged_this_round = True

            while merged_this_round:
                merged_this_round = False

                for j in range(len(work)):
                    if used[j]:
                        continue

                    a1, b1, a2, b2 = work[j]

                    overlap_or_close = not (
                        x2 + gap < a1
                        or a2 + gap < x1
                        or y2 + gap < b1
                        or b2 + gap < y1
                    )

                    if overlap_or_close:
                        x1 = min(x1, a1)
                        y1 = min(y1, b1)
                        x2 = max(x2, a2)
                        y2 = max(y2, b2)

                        used[j] = True
                        merged_this_round = True
                        changed = True

            new_boxes.append([x1, y1, x2, y2])

        work = new_boxes

    return [
        (x1, y1, x2 - x1, y2 - y1)
        for (x1, y1, x2, y2) in work
    ]


def find_candidates(
    mask: np.ndarray,
    color_mask: Optional[np.ndarray] = None,
    ore: Optional[str] = None,
) -> List[Tuple[int, int, int, int]]:
    """
    Sucht Kandidatenregionen in einer binären Maske.

    GEÄNDERT:
    Die Filterung ist stabil, aber nicht zu aggressiv.
    Sehr große Wandflächen, extrem lange dünne Linien und Mini-Rauschen
    werden entfernt.

    Parameters
    ----------
    mask : np.ndarray
        Binärmaske.
    color_mask : Optional[np.ndarray]
        Reine Farbmaske des jeweiligen Erzes.

    Returns
    -------
    List[Tuple[int, int, int, int]]
        Kandidatenboxen im Format (x, y, w, h).
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    img_h, img_w = mask.shape[:2]
    img_area = img_h * img_w

    min_area = max(
        30,
        int(img_area * 0.000035)
    )

    max_area = int(img_area * (0.045 if ore and ore.lower() == "copper" else 0.012))

    max_w = int(img_w * (0.35 if ore and ore.lower() == "copper" else 0.18))
    max_h = int(img_h * (0.35 if ore and ore.lower() == "copper" else 0.18))

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        area = w * h

        if area < min_area:
            continue

        if area > max_area:
            continue

        if w > max_w or h > max_h:
            continue

        ratio = max(w / float(h), h / float(w))

        if ratio > 4.8:
            continue

        if color_mask is not None:
            crop = color_mask[y:y + h, x:x + w]

            if crop.size == 0:
                continue

            white_pixels = int((crop > 0).sum())
            density = white_pixels / float(crop.size)

            if ore and ore.lower() == "copper":
                if density < 0.030 and white_pixels < 36:
                    continue
            elif density < 0.015 and white_pixels < 18:
                continue

        candidates.append((x, y, w, h))

    merged = _merge_nearby_boxes(
        candidates,
        gap=12
    )

    final = []

    for (x, y, w, h) in merged:
        area = w * h

        if area < max(55, int(img_area * 0.000045)):
            continue

        if area > max_area:
            continue

        if w > max_w or h > max_h:
            continue

        ratio = max(w / float(h), h / float(w))

        if ratio > 5.0:
            continue

        if color_mask is not None:
            crop = color_mask[y:y + h, x:x + w]

            if crop.size == 0:
                continue

            white_pixels = int((crop > 0).sum())
            density = white_pixels / float(crop.size)

            if density < 0.010 and white_pixels < 14:
                continue

        final.append((x, y, w, h))

    return final
