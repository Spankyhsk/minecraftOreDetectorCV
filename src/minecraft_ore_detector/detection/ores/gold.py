# -*- coding: utf-8 -*-
"""Gold-spezifische Zusatzstrategien."""

from typing import Dict, List

import cv2
import numpy as np

from minecraft_ore_detector.app.config import OreDetectorConfig
from minecraft_ore_detector.detection import (
    _color_compatibility,
    _color_support_mask,
    match_template_multiscale,
    non_max_suppression,
)
from minecraft_ore_detector.detection.mask_statistics import (
    integral_support,
    mask_integral,
)
from minecraft_ore_detector.imaging.runtime_mask_filter import RuntimeMaskFilter
from minecraft_ore_detector.imaging.segmentation import color_mask


class GoldDetector:
    """Erkennt Gold-Fälle, die von der Standard-Kandidatensuche verpasst werden."""

    def __init__(
        self,
        config: OreDetectorConfig,
        mask_filter: RuntimeMaskFilter,
    ):
        self.config = config
        self.mask_filter = mask_filter

    def detect_large_mask_windows(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        hsv: np.ndarray,
        template_bank: Dict[str, np.ndarray],
    ) -> List[Dict]:
        """
        Sucht blockartige Fenster in grossen, stark maskierten Goldbereichen.
        """

        if not template_bank:
            return []

        raw_mask = color_mask(hsv, "gold")
        raw_mask = self.mask_filter.remove_hud_regions(raw_mask)

        if cv2.countNonZero(raw_mask) < 1800:
            return []

        raw_integral = mask_integral(raw_mask)
        orig_integral = mask_integral(
            _color_support_mask("gold", img)
        )
        pre_integral = mask_integral(
            _color_support_mask("gold", img_preprocessed)
        )

        img_h, img_w = img.shape[:2]
        candidates = []

        for side in (112, 128, 144, 160):
            if side > img_w or side > img_h:
                continue

            step = max(28, int(side * 0.40))
            max_y = min(img_h - side, int(img_h * 0.82) - side)

            if max_y < 0:
                continue

            for wy in range(0, max_y + 1, step):
                for wx in range(0, img_w - side + 1, step):
                    raw_support = integral_support(
                        raw_integral,
                        wx,
                        wy,
                        side,
                        side,
                    )

                    if raw_support < 0.55:
                        continue

                    gold_support = integral_support(
                        orig_integral,
                        wx,
                        wy,
                        side,
                        side,
                    )
                    gold_pre_support = integral_support(
                        pre_integral,
                        wx,
                        wy,
                        side,
                        side,
                    )

                    if gold_support < 0.42 or gold_pre_support < 0.55:
                        continue

                    roi = img[wy:wy + side, wx:wx + side]
                    pre_roi = img_preprocessed[wy:wy + side, wx:wx + side]

                    if roi.size == 0 or pre_roi.size == 0:
                        continue

                    gold_compatibility = max(
                        _color_compatibility("gold", roi),
                        _color_compatibility("gold", pre_roi),
                    )

                    if gold_compatibility < 0.82:
                        continue

                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    edge_density = float(np.mean(edges > 0))
                    texture_strength = float(gray.std())
                    mean_value = float(gray.mean())

                    if not (0.035 <= edge_density <= 0.125):
                        continue
                    if not (8.0 <= texture_strength <= 28.0):
                        continue
                    if mean_value < 30.0:
                        continue

                    best_score = 0.0
                    best_name = None

                    for name, template in template_bank.items():
                        score = match_template_multiscale(roi, template)
                        if score > best_score:
                            best_score = score
                            best_name = name

                    if best_name is None or best_score < 0.82:
                        continue

                    final_score = min(
                        0.96,
                        0.50
                        + 0.30 * best_score
                        + 0.08 * gold_compatibility
                        + 0.08 * gold_support
                        + 0.04 * min(1.0, edge_density * 12.0),
                    )

                    candidates.append({
                        "label": "Gold",
                        "variant": best_name,
                        "score": float(final_score),
                        "box": (wx, wy, side, side),
                        "source": "gold_large_window",
                        "template_score": float(best_score),
                        "gold_support": float(gold_support),
                        "gold_pre_support": float(gold_pre_support),
                        "gold_compatibility": float(gold_compatibility),
                        "edge_density": float(edge_density),
                        "texture_strength": float(texture_strength),
                        "mean_value": float(mean_value),
                        "raw_support": float(raw_support),
                    })

        if not candidates:
            return []

        candidates.sort(key=lambda detection: detection["score"], reverse=True)
        return non_max_suppression(
            candidates[:6],
            iou_threshold=self.config.nms_iou_threshold,
        )[:3]
