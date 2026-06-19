# -*- coding: utf-8 -*-
"""
Maskenfilter fuer stoerende Bildbereiche wie HUD, Wasser und grosse Flaechen.
"""

import cv2
import numpy as np


class RuntimeMaskFilter:
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

    def remove_large_mask_regions(
        self,
        mask: np.ndarray,
        hsv: np.ndarray,
        ore: str | None = None,
    ) -> np.ndarray:
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
            if ore == "copper" and area > int(image_area * 0.020):
                component = np.zeros_like(out)
                component[labels == i] = 255

                # Copper darf in Hoehlen grossflaechig sein, aber nur wenn
                # die Region wirklich copper-aehnlich und nicht nur Wandlicht ist.
                if self._keeps_large_copper_region(component, hsv):
                    continue

            if area > int(image_area * 0.020):
                out[labels == i] = 0
                continue

            if bw > int(0.25 * w) or bh > int(0.25 * h):
                out[labels == i] = 0

        return out

    def _keeps_large_copper_region(self, mask: np.ndarray, hsv: np.ndarray) -> bool:
        """
        True, wenn eine grosse Copper-Region trotz Flaeche erhalten bleiben sollte.
        """

        if mask.size == 0:
            return False

        mask_area = float(cv2.countNonZero(mask))
        if mask_area <= 0:
            return False

        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            return False

        x0 = int(xs.min())
        x1 = int(xs.max()) + 1
        y0 = int(ys.min())
        y1 = int(ys.max()) + 1

        roi = hsv[y0:y1, x0:x1]
        if roi.size == 0:
            return False

        roi_mask = mask[y0:y1, x0:x1]
        support = float(cv2.countNonZero(roi_mask)) / float(roi_mask.shape[0] * roi_mask.shape[1])
        mean_s = float(roi[:, :, 1].mean())
        mean_v = float(roi[:, :, 2].mean())
        std_v = float(roi[:, :, 2].std())

        return support >= 0.35 and mean_s >= 75.0 and mean_v >= 60.0 and std_v >= 12.0

    def filter_mask(self, mask: np.ndarray, hsv: np.ndarray, ore: str | None = None) -> np.ndarray:
        out = self.remove_hud_regions(mask)

        # GEÄNDERT:
        # Diamond und Lapis liegen farblich direkt im frueheren Wasserbereich.
        # Der Wasserfilter hat dadurch in test1/test2 echte Erzpixel entfernt.
        if ore not in {"diamond", "lapis"}:
            out = self.remove_water_regions(out, hsv)

        out = self.remove_large_mask_regions(out, hsv, ore=ore)
        return out
