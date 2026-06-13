# -*- coding: utf-8 -*-
"""
Maskenfilter fuer stoerende Bildbereiche wie HUD, Wasser und grosse Flaechen.
"""

import cv2
import numpy as np


class MaskRegionFilter:
    """
    Entfernt Regionen, die fuer die Erzerkennung systematisch problematisch sind.
    """

    def remove_hud_regions(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape[:2]
        out = mask.copy()

        out[int(0.82 * h):, :] = 0
        out[int(0.55 * h):, int(0.72 * w):] = 0

        cx = w // 2
        cy = h // 2
        cross_w = max(12, int(0.015 * w))
        cross_h = max(12, int(0.020 * h))

        x0 = max(0, cx - cross_w)
        x1 = min(w, cx + cross_w)
        y0 = max(0, cy - cross_h)
        y1 = min(h, cy + cross_h)

        out[y0:y1, x0:x1] = 0
        return out

    def remove_water_regions(self, mask: np.ndarray, hsv: np.ndarray) -> np.ndarray:
        h, w = mask.shape[:2]
        out = mask.copy()

        lower_water = np.array([85, 25, 20], dtype=np.uint8)
        upper_water = np.array([125, 255, 255], dtype=np.uint8)
        water_mask = cv2.inRange(hsv, lower_water, upper_water)

        kernel = np.ones((7, 7), np.uint8)
        water_mask = cv2.dilate(water_mask, kernel, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            water_mask,
            connectivity=8
        )
        image_area = h * w

        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area > int(image_area * 0.0035):
                out[labels == i] = 0

        return out

    def remove_large_mask_regions(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape[:2]
        out = mask.copy()

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            out,
            connectivity=8
        )
        image_area = h * w

        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])

            if area > int(image_area * 0.020):
                out[labels == i] = 0
                continue

            if bw > int(0.25 * w) or bh > int(0.25 * h):
                out[labels == i] = 0

        return out

    def clean_runtime_mask(self, mask: np.ndarray, hsv: np.ndarray) -> np.ndarray:
        out = self.remove_hud_regions(mask)
        out = self.remove_water_regions(out, hsv)
        out = self.remove_large_mask_regions(out)
        return out

