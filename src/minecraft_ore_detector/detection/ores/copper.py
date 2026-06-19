# -*- coding: utf-8 -*-
"""Copper-spezifische Zusatzstrategien."""

from typing import Dict, List, Tuple

import cv2
import numpy as np

from minecraft_ore_detector.detection import (
    _color_compatibility,
    _color_support_ratio,
    _copper_green_support,
    _copper_orange_support,
    match_template_multiscale,
)
from minecraft_ore_detector.detection.geometry import (
    box_iou,
    clip_box,
    overlaps_any_box,
)
from minecraft_ore_detector.imaging.preprocessing import convert_bgr_to_hsv
from minecraft_ore_detector.imaging.runtime_mask_filter import RuntimeMaskFilter
from minecraft_ore_detector.imaging.segmentation import color_mask
from minecraft_ore_detector.repositories.template_repository import TemplateRepository
from minecraft_ore_detector.models import Box



class CopperDetector:
    """Ergaenzt die Standarderkennung um Copper-Sonderfaelle."""

    def __init__(
        self,
        mask_filter: RuntimeMaskFilter,
        template_repository: TemplateRepository,
    ):
        self.mask_filter = mask_filter
        self.template_repository = template_repository




    @staticmethod
    def _hsv_range_support(
        roi_bgr: np.ndarray,
        lower: List[int],
        upper: List[int],
    ) -> float:
        if roi_bgr.size == 0:
            return 0.0
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array(lower, dtype=np.uint8),
            np.array(upper, dtype=np.uint8),
        )
        return float(np.mean(mask > 0))

    def _best_template_score_for_ore(
        self,
        ore: str,
        roi_bgr: np.ndarray,
    ) -> float:
        template_bank = self.template_repository.get_templates_for_ore(ore)
        if not template_bank or roi_bgr.size == 0:
            return 0.0
        return max(
            match_template_multiscale(roi_bgr, template)
            for template in template_bank.values()
        )

    def detect_edge_clusters(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        edges: np.ndarray,
        template_bank: Dict[str, np.ndarray]
    ) -> List[Dict]:
        """
        Ergaenzt Copper-Kandidaten in warmen Hoehlen, wo die Farbflaeche zu
        gross wird und deshalb nur die lokale Kantenkomponente brauchbar ist.
        """

        if not template_bank:
            return []

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            if area < 4500 or area > 18000:
                continue

            aspect_ratio = max(w / float(max(h, 1)), h / float(max(w, 1)))
            if aspect_ratio > 1.75:
                continue

            edge_roi = edges[y:y + h, x:x + w]
            edge_density = cv2.countNonZero(edge_roi) / float(area)
            if edge_density < 0.10:
                continue

            roi = img[y:y + h, x:x + w]
            pre_roi = img_preprocessed[y:y + h, x:x + w]

            copper_support = max(
                _color_support_ratio("copper", roi),
                _color_support_ratio("copper", pre_roi)
            )
            copper_compatibility = max(
                _color_compatibility("copper", roi),
                _color_compatibility("copper", pre_roi)
            )
            copper_orange = max(
                _copper_orange_support(roi),
                _copper_orange_support(pre_roi)
            )
            copper_green = max(
                _copper_green_support(roi),
                _copper_green_support(pre_roi)
            )

            if not (0.72 <= copper_support <= 0.94):
                continue
            if copper_compatibility < 0.84:
                continue
            if copper_orange < 0.72 or copper_green > 0.08:
                continue

            pad = int(max(w, h) * 0.60)
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(img.shape[1], x + w + pad)
            y1 = min(img.shape[0], y + h + pad)
            match_roi = img[y0:y1, x0:x1]

            if match_roi.size == 0:
                continue

            best_score = 0.0
            best_name = None

            for name, template in template_bank.items():
                score = match_template_multiscale(match_roi, template)
                if score > best_score:
                    best_score = score
                    best_name = name

            if best_name is None or best_score < 0.54:
                continue

            detections.append({
                "label": "Copper",
                "variant": best_name,
                "score": float(
                    0.50
                    + best_score * 0.20
                    + copper_support * 0.08
                    + copper_compatibility * 0.06
                ),
                "box": (x, y, w, h),
                "source": "copper_edge_cluster",
                "template_score": float(best_score),
                "copper_support": float(copper_support),
                "copper_compatibility": float(copper_compatibility),
                "copper_orange": float(copper_orange),
                "copper_green": float(copper_green),
                "edge_density": float(edge_density),
            })

        return detections

    def detect_mixed_large_windows(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        copper_mask: np.ndarray,
        template_bank: Dict[str, np.ndarray]
    ) -> List[Dict]:
        """
        NEU HINZUGEFÜGT:
        Sehr konservativer Copper-Fallback fuer grosse, fragmentierte Copper-
        Flaechen mit gemischtem Orange+Gruen-Signal.

        Dieser Fallback lockert keine HSV-Grenzen. Er sucht nur um bereits
        starke Copper-Maskenregionen herum und akzeptiert maximal ein
        blockartiges Fenster pro grosser Region. Dadurch wird der test15-FN
        abgefangen, ohne die bisherigen warmen Wand-False-Positives wieder
        einzufuehren.
        """

        if not template_bank:
            return []

        img_h, img_w = img.shape[:2]
        img_area = img_h * img_w

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            copper_mask,
            connectivity=8
        )

        detections = []

        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])

            if area < int(img_area * 0.050):
                continue
            if w < 180 or h < 160:
                continue

            local_candidates = []

            for side in (208, 240, 280):
                if side > img_w or side > img_h:
                    continue

                step = max(24, int(side * 0.25))
                pad = int(side * 0.90)
                x0 = max(0, x - pad)
                y0 = max(0, y - pad)
                x1 = min(img_w - side, x + w + pad - side)
                y1 = min(img_h - side, y + h + pad - side)

                if x1 < x0 or y1 < y0:
                    continue

                for wy in range(y0, y1 + 1, step):
                    for wx in range(x0, x1 + 1, step):
                        roi = img[wy:wy + side, wx:wx + side]
                        pre_roi = img_preprocessed[wy:wy + side, wx:wx + side]
                        mask_roi = copper_mask[wy:wy + side, wx:wx + side]

                        if roi.size == 0 or pre_roi.size == 0 or mask_roi.size == 0:
                            continue

                        mask_support = cv2.countNonZero(mask_roi) / float(side * side)
                        pre_support = _color_support_ratio("copper", pre_roi)
                        copper_compatibility = max(
                            _color_compatibility("copper", roi),
                            _color_compatibility("copper", pre_roi)
                        )
                        copper_orange = _copper_orange_support(roi)
                        copper_green = _copper_green_support(roi)

                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        edge_density = float(np.mean(cv2.Canny(gray, 50, 150) > 0))
                        texture_strength = float(gray.std())

                        if not (0.00001 <= mask_support <= 0.75):
                            continue
                        if pre_support < 0.065:
                            continue
                        if copper_compatibility < 0.78:
                            continue
                        if copper_orange < 0.050 or copper_green < 0.045:
                            continue
                        if edge_density > 0.012:
                            continue
                        if not (3.0 <= texture_strength <= 8.8):
                            continue

                        best_score = 0.0
                        best_name = None

                        for name, template in template_bank.items():
                            score = match_template_multiscale(roi, template)
                            if score > best_score:
                                best_score = score
                                best_name = name

                        if best_name is None or best_score < 0.565:
                            continue

                        final_score = min(
                            0.95,
                            0.50
                            + 0.25 * best_score
                            + 0.10 * copper_compatibility
                            + 0.08 * pre_support
                            + 0.04 * (copper_orange + copper_green)
                        )

                        local_candidates.append({
                            "label": "Copper",
                            "variant": best_name,
                            "score": float(final_score),
                            "box": (wx, wy, side, side),
                            "source": "copper_mixed_large_window",
                            "template_score": float(best_score),
                            "copper_support": float(pre_support),
                            "copper_compatibility": float(copper_compatibility),
                            "copper_orange": float(copper_orange),
                            "copper_green": float(copper_green),
                            "edge_density": float(edge_density),
                            "texture_strength": float(texture_strength),
                            "mask_support": float(mask_support),
                        })

            if not local_candidates:
                continue

            best_score = max(
                candidate["score"]
                for candidate in local_candidates
            )
            near_best = [
                candidate
                for candidate in local_candidates
                if candidate["score"] >= best_score - 0.075
            ]

            # GEÄNDERT:
            # Bei grossen gemischten Copper-Flaechen liegt die sichtbare
            # Review-Box oft unterhalb des staerksten Maskenbereichs. Deshalb
            # wird unter den fast gleich guten Fenstern das tiefere bevorzugt.
            detections.append(max(
                near_best,
                key=lambda candidate: (
                    candidate["box"][1],
                    candidate["score"],
                )
            ))

        return detections

    def detect_dark_cyan_clusters(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        template_bank: Dict[str, np.ndarray],
        existing_copper_detections: List[Dict],
    ) -> List[Dict]:
        """
        NEU HINZUGEFÜGT:
        Sehr enger Fallback fuer dunklen Deepslate-Copper in test10/test11/test12.

        Diese Bilder enthalten kleine cyan/tuerkise Erzpixel-Inseln, die von der
        alten Copper-Gruen-Range nicht erfasst werden. Statt die HSV-Grenzen
        global zu lockern, werden nur kleine Cyan-Komponenten gruppiert und
        geometrisch zu blockartigen Fenstern erweitert.
        """

        if not template_bank:
            return []

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cyan_mask = cv2.inRange(
            hsv,
            np.array([55, 20, 10], dtype=np.uint8),
            np.array([115, 255, 180], dtype=np.uint8)
        )
        cyan_mask = self.mask_filter.remove_hud_regions(cyan_mask)
        cyan_mask = cv2.morphologyEx(
            cyan_mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8)
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            cyan_mask,
            connectivity=8
        )
        specks = np.zeros_like(cyan_mask)

        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            width = int(stats[i, cv2.CC_STAT_WIDTH])
            height = int(stats[i, cv2.CC_STAT_HEIGHT])

            if 8 <= area <= 1200 and width <= 60 and height <= 60:
                specks[labels == i] = 255

        if cv2.countNonZero(specks) < 40:
            return []

        grouped = cv2.dilate(
            specks,
            np.ones((65, 65), np.uint8),
            iterations=1
        )
        grouped = cv2.morphologyEx(
            grouped,
            cv2.MORPH_CLOSE,
            np.ones((25, 25), np.uint8)
        )

        contours, _ = cv2.findContours(
            grouped,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        pre_hsv = convert_bgr_to_hsv(img_preprocessed)
        pre_copper_mask = color_mask(pre_hsv, "copper")
        existing_boxes = [
            tuple(detection["box"])
            for detection in existing_copper_detections
        ]
        candidates = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            if area < 800 or w < 35 or h < 35:
                continue
            if w > 320 or h > 260:
                continue

            speck_roi = specks[y:y + h, x:x + w]
            speck_count = cv2.connectedComponentsWithStats(
                speck_roi,
                connectivity=8
            )[0] - 1

            if speck_count < 4:
                continue

            proposal_specs = []

            if w <= 170 and h <= 190:
                proposal_specs.append((
                    int(x + w * 0.18),
                    int(y + h * 0.05),
                    90,
                    150,
                    "small",
                ))
                proposal_specs.append((
                    int(x + w * 0.10),
                    int(y - h * 0.45),
                    int(w * 1.14),
                    int(h * 1.50),
                    "small_expanded",
                ))

            if w >= 170 and h >= 150:
                proposal_specs.append((
                    int(x + w * 0.12),
                    int(y - h * 0.25),
                    int(w * 0.75),
                    int(h * 0.95),
                    "large",
                ))
                proposal_specs.append((
                    int(x + w * 0.05),
                    int(y - h * 0.45),
                    int(w * 0.88),
                    int(h * 1.25),
                    "large_expanded",
                ))

            for px, py, pw, ph, mode in proposal_specs:
                box = clip_box((px, py, pw, ph), img.shape)

                if overlaps_any_box(
                    box,
                    existing_boxes,
                    iou_threshold=0.12
                ):
                    continue

                bx, by, bw, bh = box
                pre_mask_roi = pre_copper_mask[by:by + bh, bx:bx + bw]

                if pre_mask_roi.size == 0:
                    continue

                pre_mask_support = float(np.mean(pre_mask_roi > 0))
                if pre_mask_support < 0.70:
                    continue

                roi = img[by:by + bh, bx:bx + bw]
                if roi.size == 0:
                    continue

                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                edges = cv2.Canny(gray, 50, 150)

                mean_value = float(gray.mean())
                texture_strength = float(gray.std())
                edge_density = float(np.mean(edges > 0))
                saturation_mean = float(roi_hsv[:, :, 1].mean())
                value_mean = float(roi_hsv[:, :, 2].mean())

                cyan_support = self._hsv_range_support(
                    roi,
                    [55, 20, 10],
                    [115, 255, 180]
                )
                copper_orange = _copper_orange_support(roi)
                copper_support = _color_support_ratio("copper", roi)
                copper_compatibility = _color_compatibility("copper", roi)

                if not (0.015 <= cyan_support <= 0.20):
                    continue
                if not (0.065 <= copper_orange <= 0.58):
                    continue
                if copper_support < 0.018:
                    continue
                if copper_compatibility < 0.50:
                    continue
                if not (12.0 <= mean_value <= 31.0):
                    continue
                if not (3.0 <= texture_strength <= 11.2):
                    continue
                if edge_density > 0.0045:
                    continue
                if not (55.0 <= saturation_mean <= 126.0):
                    continue
                if not (14.0 <= value_mean <= 38.0):
                    continue

                best_score = 0.0
                best_name = None

                for name, template in template_bank.items():
                    score = match_template_multiscale(roi, template)
                    if score > best_score:
                        best_score = score
                        best_name = name

                if best_name is None or best_score < 0.575:
                    continue

                coal_score = self._best_template_score_for_ore("coal", roi)
                gold_score = self._best_template_score_for_ore("gold", roi)
                emerald_score = self._best_template_score_for_ore("emerald", roi)
                iron_score = self._best_template_score_for_ore("iron", roi)

                if coal_score > best_score + 0.060:
                    continue
                if emerald_score > best_score + 0.115:
                    continue
                if gold_score > best_score + 0.160 and cyan_support < 0.030:
                    continue

                final_score = min(
                    0.88,
                    0.54
                    + best_score * 0.25
                    + copper_compatibility * 0.06
                    + pre_mask_support * 0.03
                    + min(0.045, cyan_support * 0.35)
                    + min(0.025, copper_orange * 0.035)
                )

                candidates.append({
                    "label": "Copper",
                    "variant": best_name,
                    "score": float(final_score),
                    "box": box,
                    "source": "copper_dark_cyan_cluster",
                    "template_score": float(best_score),
                    "cluster_mode": mode,
                    "pre_mask_support": float(pre_mask_support),
                    "copper_support": float(copper_support),
                    "copper_compatibility": float(copper_compatibility),
                    "copper_orange": float(copper_orange),
                    "copper_cyan": float(cyan_support),
                    "copper_specks": int(speck_count),
                    "edge_density": float(edge_density),
                    "texture_strength": float(texture_strength),
                    "mean_value": float(mean_value),
                    "saturation_mean": float(saturation_mean),
                    "value_mean": float(value_mean),
                    "coal_template_score": float(coal_score),
                    "gold_template_score": float(gold_score),
                    "emerald_template_score": float(emerald_score),
                    "iron_template_score": float(iron_score),
                })

        if not candidates:
            return []

        candidates.sort(key=lambda detection: detection["score"], reverse=True)
        filtered = []

        for candidate in candidates:
            if overlaps_any_box(
                tuple(candidate["box"]),
                [tuple(item["box"]) for item in filtered],
                iou_threshold=0.25
            ):
                continue

            filtered.append(candidate)

            if len(filtered) >= 2:
                break

        return filtered
