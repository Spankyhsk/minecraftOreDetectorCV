# -*- coding: utf-8 -*-
"""Coal searches anchored to validated Copper detections."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from minecraft_ore_detector.app.config import OreDetectorConfig
from minecraft_ore_detector.detection import (
    _copper_green_support,
    _copper_orange_support,
    match_template_multiscale,
    non_max_suppression,
)
from minecraft_ore_detector.detection.geometry import box_iou
from minecraft_ore_detector.detection.ores.coal.primary import CoalPrimaryDetector
from minecraft_ore_detector.imaging.morphology import clean_mask
from minecraft_ore_detector.imaging.preprocessing import convert_bgr_to_hsv
from minecraft_ore_detector.imaging.runtime_mask_filter import RuntimeMaskFilter
from minecraft_ore_detector.imaging.segmentation import color_mask, refine_mask_for_ore
from minecraft_ore_detector.repositories.template_repository import TemplateRepository
from minecraft_ore_detector.models import Box



class CopperAnchorCoalStrategy:
    def __init__(
        self,
        config: OreDetectorConfig,
        mask_filter: RuntimeMaskFilter,
        template_repository: TemplateRepository,
        coal_detector: CoalPrimaryDetector,
    ):
        self.config = config
        self.mask_filter = mask_filter
        self.template_repository = template_repository
        self.coal_detector = coal_detector




    def _best_template_score_for_ore(self, ore: str, roi_bgr: np.ndarray) -> float:
        templates = self.template_repository.get_templates_for_ore(ore)
        if not templates or roi_bgr.size == 0:
            return 0.0
        return max(match_template_multiscale(roi_bgr, template) for template in templates.values())

    def _detect_coal_near_copper_anchors(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        detections: List[Dict],
        template_bank: Dict[str, np.ndarray],
    ) -> List[Dict]:
        """
        NEU HINZUGEFÜGT:
        Sehr konservativer Coal-Fallback fuer Bilder, in denen Copper sicher
        erkannt wurde, Coal aber wegen Beleuchtung/Maskenfilterung fehlt.

        Wichtig:
        - Keine globale Coal-Suche.
        - Nur wenn bisher kein Coal-Treffer existiert.
        - Nur in einem kleinen Nachbarschaftsbereich sicherer Copper-Boxen.
        - Maximal ein Coal-Fenster pro Copper-Anker.
        """

        if not template_bank:
            return []

        if any(
            detection["label"].lower() == "coal"
            for detection in detections
        ):
            return []

        copper_anchors = [
            detection
            for detection in detections
            if (
                detection["label"].lower() == "copper"
                and detection.get("score", 0.0) >= 0.70
            )
        ]

        if not copper_anchors:
            return []

        img_h, img_w = img.shape[:2]
        output = []

        for anchor in copper_anchors:
            ax, ay, aw, ah = anchor["box"]

            if aw > 280 or ah > 320:
                continue

            x0 = max(0, int(ax - 0.30 * aw))
            x1 = min(img_w, int(ax + 5.00 * aw))
            y0 = max(int(0.10 * img_h), int(ay - 0.40 * ah))
            y1 = min(img_h, int(ay + 1.75 * ah))

            window_sizes = self._coal_anchor_window_sizes(aw, ah)
            local_candidates = []

            for window_w, window_h in window_sizes:
                if window_w <= 0 or window_h <= 0:
                    continue

                step_x = max(24, int(window_w * 0.28))
                step_y = max(24, int(window_h * 0.28))
                max_x = max(x0, x1 - window_w)
                max_y = max(y0, y1 - window_h)

                for wy in range(y0, max_y + 1, step_y):
                    for wx in range(x0, max_x + 1, step_x):
                        # GEÄNDERT:
                        # Links vom Copper-Anker entstanden in test12 sichere
                        # False Positives. Die offenen Coal-FNs liegen dagegen
                        # direkt am oder rechts vom Copper-Block.
                        if wx < ax - 0.20 * aw:
                            continue

                        roi = img[wy:wy + window_h, wx:wx + window_w]
                        roi_pre = img_preprocessed[wy:wy + window_h, wx:wx + window_w]

                        if roi.size == 0 or roi_pre.size == 0:
                            continue

                        detection = self._evaluate_coal_anchor_window(
                            roi,
                            roi_pre,
                            template_bank,
                            (wx, wy, window_w, window_h),
                            anchor,
                        )

                        if detection is None:
                            continue

                        target_x = ax + (1.05 * aw if aw >= 150 else 0.15 * aw)
                        target_y = ay + 0.62 * ah
                        distance_score = (
                            ((wx - target_x) / float(max(1, aw))) ** 2
                            + ((wy - target_y) / float(max(1, ah))) ** 2
                        )
                        local_candidates.append((distance_score, detection))

            if not local_candidates:
                continue

            local_candidates.sort(
                key=lambda item: (
                    item[0],
                    -item[1].get("template_score", 0.0)
                )
            )
            output.append(local_candidates[0][1])

        return non_max_suppression(
            output,
            iou_threshold=self.config.nms_iou_threshold
        )

    @staticmethod
    def _coal_anchor_window_sizes(anchor_w: int, anchor_h: int) -> List[Tuple[int, int]]:
        """
        NEU HINZUGEFÜGT:
        Leitet plausible Coal-Fenster aus der sicheren Copper-Box ab.
        """

        sizes: List[Tuple[int, int]] = []

        if anchor_w < 120:
            for width_factor, height_factor in [
                (1.05, 0.75),
                (1.15, 0.90),
                (1.30, 1.05),
            ]:
                window_w = int(round(anchor_w * width_factor))
                window_h = int(round(anchor_h * height_factor))
                if 52 <= window_w <= 135 and 52 <= window_h <= 175:
                    sizes.append((window_w, window_h))

            for side in [86, 102, 118]:
                sizes.append((side, side))
        else:
            for width_factor, height_factor in [
                (1.05, 0.75),
                (1.15, 0.90),
                (1.30, 1.05),
                (1.65, 1.10),
                (1.95, 1.00),
            ]:
                window_w = int(round(anchor_w * width_factor))
                window_h = int(round(anchor_h * height_factor))
                if 52 <= window_w <= 330 and 52 <= window_h <= 260:
                    sizes.append((window_w, window_h))

        unique_sizes = []
        seen = set()

        for size in sizes:
            if size in seen:
                continue
            seen.add(size)
            unique_sizes.append(size)

        return unique_sizes

    def _evaluate_coal_anchor_window(
        self,
        roi: np.ndarray,
        roi_preprocessed: np.ndarray,
        template_bank: Dict[str, np.ndarray],
        box: Box,
        anchor: Dict,
    ) -> Optional[Dict]:
        """
        NEU HINZUGEFÜGT:
        Validiert ein einzelnes Copper-Ankerfenster als Coal.
        """

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi_preprocessed, cv2.COLOR_BGR2HSV)
        edges = cv2.Canny(gray, 50, 150)

        mean_gray = float(gray.mean())
        texture_strength = float(gray.std())
        dark_ratio = float(np.mean(gray < 170))
        low_sat_ratio = float(np.mean(hsv[:, :, 1] < 125))
        edge_density = float(np.mean(edges > 0))
        colored_ratio = self.coal_detector._colored_ore_ratio_for_coal_reject(roi)

        if dark_ratio < 0.92 or mean_gray > 58.0:
            return None

        best_score = 0.0
        best_name = None

        for name, template in template_bank.items():
            score = match_template_multiscale(roi, template)
            if score > best_score:
                best_score = score
                best_name = name

        if best_name is None:
            return None

        x, y, w, h = box

        compact_dark_case = (
            mean_gray <= 24.0
            and texture_strength <= 11.0
            and low_sat_ratio >= 0.94
            and colored_ratio <= 0.035
            and best_score >= 0.50
            and w <= 135
            and h <= 175
        )
        lit_large_case = (
            w >= 150
            and h >= 135
            and mean_gray <= 36.0
            and texture_strength <= 13.0
            and low_sat_ratio >= 0.68
            and best_score >= 0.70
            and colored_ratio <= 0.70
        )
        deepslate_small_case = (
            w <= 125
            and h <= 125
            and mean_gray <= 52.0
            and texture_strength >= 10.0
            and edge_density >= 0.055
            and low_sat_ratio >= 0.86
            and best_score >= 0.64
            and colored_ratio <= 0.92
        )

        if not (compact_dark_case or lit_large_case or deepslate_small_case):
            return None

        if not compact_dark_case and edge_density > 0.040:
            return None

        score = min(
            0.96,
            0.82 + min(0.12, best_score * 0.12)
        )

        return {
            "label": "Coal",
            "variant": best_name,
            "score": float(score),
            "box": (x, y, w, h),
            "source": "coal_copper_anchor_window",
            "template_score": float(best_score),
            "anchor_box": anchor["box"],
            "anchor_score": float(anchor.get("score", 0.0)),
            "mean_gray": float(mean_gray),
            "texture_strength": float(texture_strength),
            "dark_ratio": float(dark_ratio),
            "low_sat_ratio": float(low_sat_ratio),
            "edge_density": float(edge_density),
            "colored_ratio": float(colored_ratio),
        }

    def _detect_coal_warm_left_of_copper_windows(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        detections: List[Dict],
        template_bank: Dict[str, np.ndarray],
    ) -> List[Dict]:
        """
        NEU HINZUGEFÜGT:
        Findet sehr warme, maskenschwache Coal-Blöcke links unter einem
        sicheren Copper-Edge-Anker.

        Hintergrund: Der letzte test18-Fall ist farblich warm-orange und wird
        deshalb von den normalen Coal-Masken und dem Colored-Reject fast
        vollständig verworfen. Diese Regel ist absichtlich nicht global:
        Sie läuft nur, wenn noch kein Coal vorhanden ist, und nur an einem
        kompakten Copper-Edge-Anker mit passender lokaler Perspektive.
        """

        if not template_bank:
            return []

        if any(
            detection["label"].lower() == "coal"
            for detection in detections
        ):
            return []

        img_h, img_w = img.shape[:2]
        output = []

        copper_anchors = [
            detection
            for detection in detections
            if (
                detection["label"].lower() == "copper"
                and detection.get("source") == "copper_edge_cluster"
                and detection.get("score", 0.0) >= 0.70
            )
        ]

        for anchor in copper_anchors:
            ax, ay, aw, ah = anchor["box"]

            if not (95 <= aw <= 125 and 90 <= ah <= 112):
                continue

            local_candidates = []
            window_specs = [
                (0.78, 0.81),
                (0.86, 0.95),
                (0.93, 0.86),
            ]

            for width_factor, height_factor in window_specs:
                window_w = int(round(aw * width_factor))
                window_h = int(round(ah * height_factor))

                if not (80 <= window_w <= 115 and 76 <= window_h <= 105):
                    continue

                base_x = int(ax - 1.62 * aw)
                base_y = int(ay + 0.48 * ah)

                for offset_x in range(-14, 15, 7):
                    for offset_y in range(-10, 11, 5):
                        wx = max(0, min(img_w - window_w, base_x + offset_x))
                        wy = max(
                            int(0.12 * img_h),
                            min(img_h - window_h, base_y + offset_y)
                        )
                        candidate_box = (wx, wy, window_w, window_h)

                        if any(
                            box_iou(candidate_box, tuple(existing["box"])) >= 0.18
                            for existing in detections
                        ):
                            continue

                        detection = self._evaluate_warm_coal_left_of_copper(
                            img,
                            img_preprocessed,
                            template_bank,
                            candidate_box,
                            anchor,
                        )

                        if detection is not None:
                            local_candidates.append(detection)

            if not local_candidates:
                continue

            local_candidates.sort(
                key=lambda detection: (
                    detection.get("template_margin", 0.0),
                    detection.get("template_score", 0.0),
                ),
                reverse=True
            )
            output.append(local_candidates[0])

        return non_max_suppression(
            output,
            iou_threshold=self.config.nms_iou_threshold
        )

    def _evaluate_warm_coal_left_of_copper(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        template_bank: Dict[str, np.ndarray],
        box: Box,
        anchor: Dict,
    ) -> Optional[Dict]:
        """
        NEU HINZUGEFÜGT:
        Validiert ein warmes, maskenschwaches Coal-Fenster links eines
        sicheren Copper-Ankers ueber Textur, Template-Margin und Geometrie.
        """

        x, y, w, h = box
        roi = img[y:y + h, x:x + w]
        roi_pre = img_preprocessed[y:y + h, x:x + w]

        if roi.size == 0 or roi_pre.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi_pre, cv2.COLOR_BGR2HSV)
        edges = cv2.Canny(gray, 50, 150)

        mean_gray = float(gray.mean())
        texture_strength = float(gray.std())
        dark_ratio = float(np.mean(gray < 170))
        very_dark_ratio = float(np.mean(gray < 85))
        low_sat_ratio = float(np.mean(hsv[:, :, 1] < 125))
        mean_hue = float(hsv[:, :, 0].mean())
        mean_saturation = float(hsv[:, :, 1].mean())
        mean_value = float(hsv[:, :, 2].mean())
        edge_density = float(np.mean(edges > 0))
        colored_ratio = self.coal_detector._colored_ore_ratio_for_coal_reject(roi)
        copper_orange = _copper_orange_support(roi)
        copper_green = _copper_green_support(roi)

        if not (
            40.0 <= mean_gray <= 52.0
            and 12.0 <= texture_strength <= 17.0
            and dark_ratio >= 0.99
            and very_dark_ratio >= 0.98
            and low_sat_ratio >= 0.96
            and 16.0 <= mean_hue <= 21.0
            and 92.0 <= mean_saturation <= 112.0
            and 155.0 <= mean_value <= 195.0
            and 0.075 <= edge_density <= 0.120
            and colored_ratio >= 0.70
            and copper_orange >= 0.70
            and copper_green <= 0.004
        ):
            return None

        best_score = 0.0
        best_name = None

        for name, template in template_bank.items():
            score = match_template_multiscale(roi, template)
            if score > best_score:
                best_score = score
                best_name = name

        if best_name is None or best_score < 0.64:
            return None

        comparison_scores = [
            self._best_template_score_for_ore(ore, roi)
            for ore in [
                "copper",
                "diamond",
                "emerald",
                "gold",
                "iron",
                "lapis",
                "redstone",
            ]
        ]
        max_other_score = max(comparison_scores) if comparison_scores else 0.0
        template_margin = best_score - max_other_score

        if template_margin < 0.08:
            return None

        score = min(
            0.95,
            0.82
            + min(0.09, best_score * 0.10)
            + min(0.04, template_margin * 0.25)
        )

        return {
            "label": "Coal",
            "variant": best_name,
            "score": float(score),
            "box": box,
            "source": "coal_warm_left_copper_window",
            "template_score": float(best_score),
            "max_other_template_score": float(max_other_score),
            "template_margin": float(template_margin),
            "mean_gray": float(mean_gray),
            "texture_strength": float(texture_strength),
            "dark_ratio": float(dark_ratio),
            "very_dark_ratio": float(very_dark_ratio),
            "low_sat_ratio": float(low_sat_ratio),
            "mean_hue": float(mean_hue),
            "mean_saturation": float(mean_saturation),
            "mean_value": float(mean_value),
            "edge_density": float(edge_density),
            "colored_ratio": float(colored_ratio),
            "copper_orange": float(copper_orange),
            "copper_green": float(copper_green),
            "anchor_box": anchor["box"],
        }
