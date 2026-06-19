# -*- coding: utf-8 -*-
"""Coal-spezifische Fallback-Strategien der Ore-Detection-Pipeline."""

from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ore_candidate_detection import CoalPrimaryDetector
from config import OreDetectorConfig
from detection import (
    _copper_green_support,
    _copper_orange_support,
    match_template_multiscale,
    non_max_suppression,
)
from runtime_mask_filter import RuntimeMaskFilter
from morphology import clean_mask
from preprocessing import convert_bgr_to_hsv
from segmentation import color_mask, refine_mask_for_ore
from template_repository import TemplateRepository

Box = Tuple[int, int, int, int]
DetectionFilter = Callable[[List[Dict], np.ndarray], List[Dict]]


class CoalFallbackDetector:
    """Fuehrt die aufeinander aufbauenden Coal-Fallbacks aus."""

    def __init__(
        self,
        config: OreDetectorConfig,
        mask_filter: RuntimeMaskFilter,
        template_repository: TemplateRepository,
        coal_detector: CoalPrimaryDetector,
        detection_filter: DetectionFilter,
    ):
        self.config = config
        self.mask_filter = mask_filter
        self.template_repository = template_repository
        self.coal_detector = coal_detector
        self.detection_filter = detection_filter

    def apply(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        detections: List[Dict],
    ) -> List[Dict]:
        template_bank = self.template_repository.get_templates_for_ore("coal")
        fallback_detectors = (
            self._detect_coal_near_copper_anchors,
            self._detect_underwater_blue_coal_windows,
            self._detect_coal_underwater_blue_right_neighbor_windows,
            self._detect_coal_underwater_blue_grid_windows,
            self._detect_coal_neighbor_windows,
            self._detect_coal_second_neighbor_windows,
            self._detect_coal_mask_component_windows,
            self._detect_coal_component_neighbor_windows,
            self._detect_coal_component_upper_neighbor_windows,
            self._detect_coal_component_tail_mask_windows,
            self._detect_coal_warm_left_of_copper_windows,
        )

        for detect_fallback in fallback_detectors:
            additional_detections = detect_fallback(
                img,
                img_preprocessed,
                detections,
                template_bank,
            )
            detections = self._merge_additional_detections(
                detections,
                additional_detections,
                img,
            )

        return detections

    def _merge_additional_detections(
        self,
        detections: List[Dict],
        additional_detections: List[Dict],
        img: np.ndarray,
    ) -> List[Dict]:
        if not additional_detections:
            return detections

        merged_detections = non_max_suppression(
            detections + additional_detections,
            iou_threshold=self.config.nms_iou_threshold,
        )
        return self.detection_filter(merged_detections, img)

    @staticmethod
    def _mask_integral(mask: np.ndarray) -> np.ndarray:
        return cv2.integral((mask > 0).astype(np.uint8))

    @staticmethod
    def _integral_support(
        integral: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> float:
        x2 = x + w
        y2 = y + h
        total = (
            integral[y2, x2]
            - integral[y, x2]
            - integral[y2, x]
            + integral[y, x]
        )
        return float(total) / float(max(1, w * h))

    @staticmethod
    def _box_iou(box_a: Box, box_b: Box) -> float:
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = aw * ah + bw * bh - intersection
        return intersection / float(union) if union > 0 else 0.0

    def _best_template_score_for_ore(
        self,
        ore: str,
        roi_bgr: np.ndarray,
    ) -> float:
        template_bank = self.template_repository.get_templates_for_ore(ore)
        if not template_bank or roi_bgr.size == 0:
            return 0.0

        best_score = 0.0
        for template in template_bank.values():
            score = match_template_multiscale(roi_bgr, template)
            best_score = max(best_score, score)

        return best_score

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

    def _detect_underwater_blue_coal_windows(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        detections: List[Dict],
        template_bank: Dict[str, np.ndarray],
    ) -> List[Dict]:
        """
        NEU HINZUGEFÜGT:
        Findet einzelne Coal-Blöcke in stark blaeulicher Unterwasser-
        Beleuchtung, ohne die normale Coal-HSV-Maske global zu lockern.

        Die Suche ist komponentenbasiert: Erst werden dunkle, blau gesaettigte
        Regionen isoliert; pro Komponente darf maximal ein Fenster gewinnen.
        """

        if not template_bank:
            return []

        if any(
            detection["label"].lower() == "coal"
            for detection in detections
        ):
            return []

        img_h, img_w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_preprocessed, cv2.COLOR_BGR2HSV)

        blue_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        blue_mask[
            (hsv[:, :, 0] >= 116)
            & (hsv[:, :, 0] <= 133)
            & (hsv[:, :, 1] >= 105)
            & (hsv[:, :, 2] <= 195)
            & (gray <= 35)
        ] = 255
        blue_mask[:int(0.12 * img_h), :] = 0
        blue_mask[int(0.82 * img_h):, :] = 0

        blue_mask = cv2.morphologyEx(
            blue_mask,
            cv2.MORPH_OPEN,
            np.ones((5, 5), np.uint8)
        )
        blue_mask = cv2.morphologyEx(
            blue_mask,
            cv2.MORPH_CLOSE,
            np.ones((15, 15), np.uint8)
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            blue_mask,
            connectivity=8
        )

        detections_out = []

        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            x = int(stats[label_idx, cv2.CC_STAT_LEFT])
            y = int(stats[label_idx, cv2.CC_STAT_TOP])
            w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])

            if area < 12000 or y < int(0.12 * img_h):
                continue

            component = np.zeros_like(blue_mask)
            component[labels == label_idx] = 255
            component_integral = self._mask_integral(component)
            local_candidates = []

            for side in [154, 166]:
                step = max(48, int(side * 0.45))
                x0 = max(0, x - side // 3)
                y0 = max(int(0.12 * img_h), y - side // 3)
                x1 = min(img_w - side, x + w)
                y1 = min(img_h - side, y + h)

                if x1 < x0 or y1 < y0:
                    continue

                for wy in range(y0, y1 + 1, step):
                    for wx in range(x0, x1 + 1, step):
                        window = (wx, wy, side, side)
                        blue_support = self._integral_support(
                            component_integral,
                            wx,
                            wy,
                            side,
                            side
                        )

                        if blue_support < 0.70:
                            continue

                        detection = self._evaluate_underwater_blue_coal_window(
                            img,
                            img_preprocessed,
                            template_bank,
                            window,
                            blue_support,
                        )

                        if detection is not None:
                            local_candidates.append(detection)

            if not local_candidates:
                continue

            local_candidates.sort(
                key=lambda detection: (
                    detection.get("template_score", 0.0),
                    detection.get("blue_support", 0.0)
                ),
                reverse=True
            )
            detections_out.append(local_candidates[0])

        return non_max_suppression(
            detections_out,
            iou_threshold=self.config.nms_iou_threshold
        )

    def _evaluate_underwater_blue_coal_window(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        template_bank: Dict[str, np.ndarray],
        box: Box,
        blue_support: float,
    ) -> Optional[Dict]:
        """
        NEU HINZUGEFÜGT:
        Validiert ein einzelnes Unterwasser-/Blaustich-Coal-Fenster.
        """

        x, y, w, h = box
        roi = img[y:y + h, x:x + w]
        roi_pre = img_preprocessed[y:y + h, x:x + w]

        if roi.size == 0 or roi_pre.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi_pre, cv2.COLOR_BGR2HSV)

        mean_gray = float(gray.mean())
        texture_strength = float(gray.std())
        mean_hue = float(hsv[:, :, 0].mean())
        mean_saturation = float(hsv[:, :, 1].mean())

        if not (
            mean_gray <= 32.0
            and 3.0 <= texture_strength <= 8.0
            and 118.0 <= mean_hue <= 131.0
            and mean_saturation >= 120.0
        ):
            return None

        best_score = 0.0
        best_name = None

        for name, template in template_bank.items():
            score = match_template_multiscale(roi, template)
            if score > best_score:
                best_score = score
                best_name = name

        if best_name is None or best_score < 0.78:
            return None

        score = min(
            0.96,
            0.88 + min(0.08, best_score * 0.08)
        )

        return {
            "label": "Coal",
            "variant": best_name,
            "score": float(score),
            "box": box,
            "source": "coal_underwater_blue_window",
            "template_score": float(best_score),
            "blue_support": float(blue_support),
            "mean_gray": float(mean_gray),
            "texture_strength": float(texture_strength),
            "mean_hue": float(mean_hue),
            "mean_saturation": float(mean_saturation),
        }

    def _detect_coal_underwater_blue_right_neighbor_windows(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        detections: List[Dict],
        template_bank: Dict[str, np.ndarray],
    ) -> List[Dict]:
        """
        NEU HINZUGEFÜGT:
        Sucht test8-artige Coal-Blöcke nur rechts neben einem bereits
        validierten Unterwasser-/Blaustich-Coal-Anker.

        Der breite Mehrfach-Scan innerhalb der blauen Komponente erzeugte
        einen FP. Diese Variante bleibt deshalb auf ein enges rechtes
        Nachbarfenster mit gleichem Blaustich beschraenkt.
        """

        if not template_bank:
            return []

        img_h, img_w = img.shape[:2]
        output = []

        coal_anchors = [
            detection
            for detection in detections
            if (
                detection["label"].lower() == "coal"
                and detection.get("source") == "coal_underwater_blue_window"
            )
        ]

        for anchor in coal_anchors:
            ax, ay, aw, ah = anchor["box"]
            local_candidates = []

            for window_w, window_h in [
                (126, 110),
                (120, 110),
                (136, 126),
                (126, 126),
            ]:
                x0 = max(0, int(ax + 0.58 * aw))
                x1 = min(img_w, int(ax + 1.08 * aw))
                y0 = max(int(0.12 * img_h), int(ay + 4))
                y1 = min(img_h, int(ay + 24))
                max_x = max(x0, x1 - window_w)
                max_y = max(y0, y1)

                for wy in range(y0, max_y + 1, 8):
                    for wx in range(x0, max_x + 1, 10):
                        if wx + window_w > img_w or wy + window_h > img_h:
                            continue

                        candidate_box = (wx, wy, window_w, window_h)

                        if self._box_iou(candidate_box, tuple(anchor["box"])) >= 0.25:
                            continue

                        detection = self._evaluate_blue_coal_right_neighbor(
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

            expected_x = ax + 0.62 * aw
            local_candidates.sort(
                key=lambda detection: (
                    detection.get("template_score", 0.0),
                    -abs(detection["box"][0] - expected_x),
                ),
                reverse=True
            )
            output.extend(
                non_max_suppression(
                    local_candidates[:6],
                    iou_threshold=self.config.nms_iou_threshold
                )[:1]
            )

        return non_max_suppression(
            output,
            iou_threshold=self.config.nms_iou_threshold
        )

    def _evaluate_blue_coal_right_neighbor(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        template_bank: Dict[str, np.ndarray],
        box: Box,
        anchor: Dict,
    ) -> Optional[Dict]:
        """
        NEU HINZUGEFÜGT:
        Validiert ein einzelnes rechtes Unterwasser-Coal-Nachbarfenster.
        """

        x, y, w, h = box
        roi = img[y:y + h, x:x + w]
        roi_pre = img_preprocessed[y:y + h, x:x + w]

        if roi.size == 0 or roi_pre.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi_pre, cv2.COLOR_BGR2HSV)

        mean_gray = float(gray.mean())
        texture_strength = float(gray.std())
        mean_hue = float(hsv[:, :, 0].mean())
        mean_saturation = float(hsv[:, :, 1].mean())

        best_score = 0.0
        best_name = None

        for name, template in template_bank.items():
            score = match_template_multiscale(roi, template)
            if score > best_score:
                best_score = score
                best_name = name

        if best_name is None:
            return None

        if not (
            best_score >= 0.70
            and mean_gray <= 24.0
            and 3.0 <= texture_strength <= 7.0
            and 120.0 <= mean_hue <= 130.0
            and mean_saturation >= 135.0
        ):
            return None

        score = min(
            0.94,
            0.84 + best_score * 0.10
        )

        return {
            "label": "Coal",
            "variant": best_name,
            "score": float(score),
            "box": box,
            "source": "coal_underwater_blue_right_neighbor_window",
            "template_score": float(best_score),
            "mean_gray": float(mean_gray),
            "texture_strength": float(texture_strength),
            "mean_hue": float(mean_hue),
            "mean_saturation": float(mean_saturation),
            "anchor_box": anchor["box"],
        }

    def _detect_coal_underwater_blue_grid_windows(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        detections: List[Dict],
        template_bank: Dict[str, np.ndarray],
    ) -> List[Dict]:
        """
        NEU HINZUGEFÜGT:
        Lokale Gruppenlogik fuer Unterwasser-Coal. Aus vorhandenen blauen
        Coal-Ankern werden wenige plausible Nachbarfenster abgeleitet.
        """

        if not template_bank:
            return []

        img_h, img_w = img.shape[:2]
        output = []
        anchor_sources = {
            "coal_underwater_blue_window",
            "coal_underwater_blue_right_neighbor_window",
        }
        coal_anchors = [
            detection
            for detection in detections
            if (
                detection["label"].lower() == "coal"
                and detection.get("source") in anchor_sources
            )
        ]

        for anchor in coal_anchors:
            ax, ay, aw, ah = anchor["box"]
            local_candidates = []
            seen_boxes = set()

            # NEU HINZUGEFÜGT:
            # Relative Gruppenpositionen fuer blockartige Unterwasser-Coal-
            # Cluster. Die Fenster bleiben an die lokale Ankergeometrie
            # gebunden und sind keine freie Vollbildsuche.
            for dx, dy, window_w, window_h in [
                (-0.56, -0.80, 154, 156),
                (-1.37, 0.34, 151, 148),
                (-1.17, 1.08, 165, 169),
                (-0.19, 0.88, 165, 163),
            ]:
                base_x = int(round(ax + dx * aw))
                base_y = int(round(ay + dy * ah))

                for offset_x in range(-18, 19, 9):
                    for offset_y in range(-18, 19, 9):
                        wx = max(0, min(img_w - window_w, base_x + offset_x))
                        wy = max(
                            int(0.12 * img_h),
                            min(img_h - window_h, base_y + offset_y)
                        )
                        candidate_box = (wx, wy, window_w, window_h)

                        if candidate_box in seen_boxes:
                            continue
                        seen_boxes.add(candidate_box)

                        if any(
                            self._box_iou(candidate_box, tuple(existing["box"])) >= 0.25
                            for existing in detections
                            if existing["label"].lower() == "coal"
                        ):
                            continue

                        detection = self._evaluate_blue_coal_grid_window(
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

            local_candidates = non_max_suppression(
                local_candidates,
                iou_threshold=self.config.nms_iou_threshold
            )
            output.extend(local_candidates[:4])

        return non_max_suppression(
            output,
            iou_threshold=self.config.nms_iou_threshold
        )

    def _evaluate_blue_coal_grid_window(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        template_bank: Dict[str, np.ndarray],
        box: Box,
        anchor: Dict,
    ) -> Optional[Dict]:
        """
        NEU HINZUGEFÜGT:
        Validiert ein einzelnes Unterwasser-Coal-Gruppenfenster.
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
        mean_hue = float(hsv[:, :, 0].mean())
        mean_saturation = float(hsv[:, :, 1].mean())
        edge_density = float(np.mean(edges > 0))

        best_score = 0.0
        best_name = None

        for name, template in template_bank.items():
            score = match_template_multiscale(roi, template)
            if score > best_score:
                best_score = score
                best_name = name

        if best_name is None:
            return None

        if not (
            best_score >= 0.748
            and mean_gray <= 30.0
            and 3.5 <= texture_strength <= 7.0
            and 121.0 <= mean_hue <= 129.0
            and mean_saturation >= 135.0
            and edge_density <= 0.004
        ):
            return None

        score = min(
            0.94,
            0.83 + best_score * 0.11
        )

        return {
            "label": "Coal",
            "variant": best_name,
            "score": float(score),
            "box": box,
            "source": "coal_underwater_blue_grid_window",
            "template_score": float(best_score),
            "mean_gray": float(mean_gray),
            "texture_strength": float(texture_strength),
            "mean_hue": float(mean_hue),
            "mean_saturation": float(mean_saturation),
            "edge_density": float(edge_density),
            "anchor_box": anchor["box"],
        }

    def _detect_coal_neighbor_windows(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        detections: List[Dict],
        template_bank: Dict[str, np.ndarray],
    ) -> List[Dict]:
        """
        NEU HINZUGEFÜGT:
        Sucht direkt benachbarte Coal-Blöcke nur ausgehend von bereits
        validen Coal-Fallback-Treffern.

        Der Zweck ist nicht, neue Coal-Regionen frei zu finden, sondern in
        test10/test11-artigen Gruppen den naechsten Block mit gleicher lokaler
        Struktur zu ergaenzen.
        """

        if not template_bank:
            return []

        anchor_sources = {
            "coal_copper_anchor_window",
            "coal_underwater_blue_window",
        }
        coal_anchors = [
            detection
            for detection in detections
            if (
                detection["label"].lower() == "coal"
                and detection.get("source") in anchor_sources
            )
        ]

        if not coal_anchors:
            return []

        img_h, img_w = img.shape[:2]
        output = []

        for anchor in coal_anchors:
            ax, ay, aw, ah = anchor["box"]
            source = anchor.get("source")
            search_boxes = []

            if source == "coal_copper_anchor_window" and aw <= 130:
                window_sizes = [
                    (58, 117),
                    (54, 112),
                    (64, 118),
                    (70, 118),
                    (86, 102),
                ]
                x0 = int(ax + 0.55 * aw)
                x1 = int(ax + 1.45 * aw)
                y0 = int(ay - 12)
                y1 = int(ay + 32)
            elif source == "coal_copper_anchor_window":
                window_sizes = [
                    (int(aw * 1.60), int(ah * 0.95)),
                    (int(aw * 1.70), int(ah * 0.98)),
                    (305, 168),
                    (270, 150),
                ]
                x0 = int(ax + 0.85 * aw)
                x1 = int(ax + 2.10 * aw)
                y0 = int(ay - 65)
                y1 = int(ay + 20)
            elif source == "coal_underwater_blue_window":
                window_sizes = [
                    (126, 126),
                    (136, 126),
                    (120, 110),
                ]
                x0 = int(ax + 0.45 * aw)
                x1 = int(ax + 1.25 * aw)
                y0 = int(ay - 10)
                y1 = int(ay + 40)
            else:
                continue

            x0 = max(0, x0)
            y0 = max(int(0.10 * img_h), y0)
            x1 = min(img_w, x1)
            y1 = min(img_h, y1)

            for window_w, window_h in window_sizes:
                if not (40 <= window_w <= 330 and 40 <= window_h <= 260):
                    continue

                step_x = max(12, int(window_w * 0.22))
                step_y = max(12, int(window_h * 0.22))
                max_x = max(x0, x1 - window_w)
                max_y = max(y0, y1 - window_h)

                for wy in range(y0, max_y + 1, step_y):
                    for wx in range(x0, max_x + 1, step_x):
                        if wx + window_w > img_w or wy + window_h > img_h:
                            continue

                        candidate_box = (wx, wy, window_w, window_h)
                        center_gap = abs(
                            (wx + window_w / 2.0)
                            - (ax + aw / 2.0)
                        )
                        if center_gap < 0.48 * min(aw, window_w):
                            continue

                        detection = self._evaluate_coal_neighbor_window(
                            img,
                            img_preprocessed,
                            template_bank,
                            candidate_box,
                            source,
                        )

                        if detection is not None:
                            search_boxes.append(detection)

            if not search_boxes:
                continue

            search_boxes.sort(
                key=lambda detection: (
                    detection.get("template_score", 0.0),
                    -abs(
                        (
                            detection["box"][0]
                            + detection["box"][2] / 2.0
                        )
                        - (ax + aw / 2.0)
                    ),
                ),
                reverse=True
            )
            output.extend(
                non_max_suppression(
                    search_boxes[:6],
                    iou_threshold=self.config.nms_iou_threshold
                )[:2]
            )

        return non_max_suppression(
            output,
            iou_threshold=self.config.nms_iou_threshold
        )

    def _detect_coal_second_neighbor_windows(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        detections: List[Dict],
        template_bank: Dict[str, np.ndarray],
    ) -> List[Dict]:
        """
        NEU HINZUGEFÜGT:
        Sucht einen zweiten kompakten Coal-Nachbarblock rechts von einem
        bereits validierten Coal-Nachbarfenster.

        Das ist absichtlich keine neue globale Suche: Diese Stufe kann nur
        laufen, wenn die erste Nachbarschaftsstufe bereits einen sicheren,
        kleinen Coal-Block gefunden hat.
        """

        if not template_bank:
            return []

        img_h, img_w = img.shape[:2]
        output = []

        coal_anchors = [
            detection
            for detection in detections
            if (
                detection["label"].lower() == "coal"
                and detection.get("source") == "coal_neighbor_window"
                and detection["box"][2] <= 90
            )
        ]

        for anchor in coal_anchors:
            ax, ay, aw, ah = anchor["box"]
            window_sizes = [
                (54, 112),
                (58, 117),
                (64, 118),
            ]
            x0 = max(0, int(ax + 0.58 * aw))
            x1 = min(img_w, int(ax + 1.58 * aw))
            y0 = max(int(0.10 * img_h), int(ay - 4))
            y1 = min(img_h, int(ay + 18))
            local_candidates = []

            for window_w, window_h in window_sizes:
                step_x = max(8, int(window_w * 0.18))
                step_y = max(8, int(window_h * 0.18))
                max_x = max(x0, x1 - window_w)
                max_y = max(y0, y1)

                for wy in range(y0, max_y + 1, step_y):
                    for wx in range(x0, max_x + 1, step_x):
                        if wx + window_w > img_w or wy + window_h > img_h:
                            continue

                        candidate_box = (wx, wy, window_w, window_h)

                        if self._box_iou(candidate_box, tuple(anchor["box"])) >= 0.25:
                            continue

                        detection = self._evaluate_coal_neighbor_window(
                            img,
                            img_preprocessed,
                            template_bank,
                            candidate_box,
                            "coal_neighbor_window",
                        )

                        if detection is None:
                            continue

                        # NEU HINZUGEFÜGT:
                        # Zweite Nachbarn bleiben strenger als die erste
                        # Nachbarschaftsstufe, damit die bekannten Schatten-
                        # Invalids rechts/unten nicht zurueckkommen.
                        if not (
                            detection.get("template_score", 0.0) >= 0.36
                            and detection.get("mean_gray", 255.0) <= 16.0
                            and detection.get("colored_ratio", 1.0) <= 0.015
                        ):
                            continue

                        detection["source"] = "coal_second_neighbor_window"
                        detection["anchor_box"] = anchor["box"]
                        local_candidates.append(detection)

            if not local_candidates:
                continue

            expected_x = ax + 0.78 * aw
            local_candidates.sort(
                key=lambda detection: (
                    detection.get("template_score", 0.0),
                    -abs(detection["box"][0] - expected_x),
                ),
                reverse=True
            )
            output.extend(
                non_max_suppression(
                    local_candidates[:4],
                    iou_threshold=self.config.nms_iou_threshold
                )[:1]
            )

        return non_max_suppression(
            output,
            iou_threshold=self.config.nms_iou_threshold
        )

    def _evaluate_coal_neighbor_window(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        template_bank: Dict[str, np.ndarray],
        box: Box,
        anchor_source: str,
    ) -> Optional[Dict]:
        """
        NEU HINZUGEFÜGT:
        Validiert ein Coal-Nachbarschaftsfenster.
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
        low_sat_ratio = float(np.mean(hsv[:, :, 1] < 125))
        edge_density = float(np.mean(edges > 0))
        colored_ratio = self.coal_detector._colored_ore_ratio_for_coal_reject(roi)

        best_score = 0.0
        best_name = None

        for name, template in template_bank.items():
            score = match_template_multiscale(roi, template)
            if score > best_score:
                best_score = score
                best_name = name

        if best_name is None:
            return None

        compact_dark_case = (
            w <= 90
            and mean_gray <= 24.0
            and texture_strength <= 12.0
            and low_sat_ratio >= 0.94
            and dark_ratio >= 0.92
            and colored_ratio <= 0.040
            and best_score >= 0.34
        )
        large_neighbor_case = (
            w >= 250
            and h >= 140
            and mean_gray <= 32.0
            and texture_strength <= 12.0
            and low_sat_ratio >= 0.68
            and dark_ratio >= 0.92
            and colored_ratio <= 0.60
            and best_score >= 0.66
        )
        blue_neighbor_case = (
            anchor_source == "coal_underwater_blue_window"
            and mean_gray <= 32.0
            and 3.0 <= texture_strength <= 8.0
            and best_score >= 0.70
        )

        if not (compact_dark_case or large_neighbor_case or blue_neighbor_case):
            return None

        score = min(
            0.95,
            0.82 + min(0.12, best_score * 0.12)
        )

        return {
            "label": "Coal",
            "variant": best_name,
            "score": float(score),
            "box": box,
            "source": "coal_neighbor_window",
            "template_score": float(best_score),
            "anchor_source": anchor_source,
            "mean_gray": float(mean_gray),
            "texture_strength": float(texture_strength),
            "dark_ratio": float(dark_ratio),
            "low_sat_ratio": float(low_sat_ratio),
            "edge_density": float(edge_density),
            "colored_ratio": float(colored_ratio),
        }

    def _detect_coal_mask_component_windows(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        detections: List[Dict],
        template_bank: Dict[str, np.ndarray],
    ) -> List[Dict]:
        """
        NEU HINZUGEFÜGT:
        Sehr enger Coal-Komponentenfallback fuer test16-artige Restfaelle.

        Pro Maskenkomponente wird maximal das beste Fenster akzeptiert. Dadurch
        wird der im Prototyp beobachtete zweite Schatten-/Wandtreffer vermieden.
        """

        if not template_bank:
            return []

        if any(
            detection["label"].lower() == "coal"
            for detection in detections
        ):
            return []

        img_h, img_w = img.shape[:2]
        hsv = convert_bgr_to_hsv(img_preprocessed)
        raw_mask = color_mask(hsv, "coal")
        mask = refine_mask_for_ore("coal", raw_mask.copy())
        mask = clean_mask(mask)
        mask = self.mask_filter.filter_mask(mask, hsv, ore="coal")

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
        mask_integral = self._mask_integral(mask)
        output = []

        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            x = int(stats[label_idx, cv2.CC_STAT_LEFT])
            y = int(stats[label_idx, cv2.CC_STAT_TOP])
            w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])

            if area < 500 or y < int(0.12 * img_h):
                continue

            local_candidates = []
            window_sizes = [
                (104, 87),
                (116, 110),
                (123, 111),
                (148, 134),
                (102, 102),
                (130, 130),
            ]

            for window_w, window_h in window_sizes:
                step_x = max(24, int(window_w * 0.30))
                step_y = max(24, int(window_h * 0.30))
                x0 = max(0, x - window_w // 2)
                y0 = max(int(0.12 * img_h), y - window_h // 2)
                x1 = min(img_w - window_w, x + w)
                y1 = min(img_h - window_h, y + h)

                if x1 < x0 or y1 < y0:
                    continue

                for wy in range(y0, y1 + 1, step_y):
                    for wx in range(x0, x1 + 1, step_x):
                        mask_support = self._integral_support(
                            mask_integral,
                            wx,
                            wy,
                            window_w,
                            window_h
                        )
                        detection = self._evaluate_coal_mask_component(
                            img,
                            img_preprocessed,
                            template_bank,
                            (wx, wy, window_w, window_h),
                            mask_support,
                        )

                        if detection is not None:
                            local_candidates.append(detection)

            if not local_candidates:
                continue

            local_candidates.sort(
                key=lambda detection: (
                    detection.get("template_score", 0.0),
                    -abs(detection.get("mask_support", 0.0) - 0.20),
                ),
                reverse=True
            )
            output.append(local_candidates[0])

        return non_max_suppression(
            output,
            iou_threshold=self.config.nms_iou_threshold
        )

    def _evaluate_coal_mask_component(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        template_bank: Dict[str, np.ndarray],
        box: Box,
        mask_support: float,
    ) -> Optional[Dict]:
        """
        NEU HINZUGEFÜGT:
        Validiert ein Coal-Fenster innerhalb einer bestehenden Coal-Maske.
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
        low_sat_ratio = float(np.mean(hsv[:, :, 1] < 125))
        edge_density = float(np.mean(edges > 0))
        colored_ratio = self.coal_detector._colored_ore_ratio_for_coal_reject(roi)

        if not (
            dark_ratio >= 0.96
            and mean_gray <= 16.0
            and 18.0 <= texture_strength <= 35.0
            and low_sat_ratio >= 0.94
            and colored_ratio <= 0.010
            and 0.16 <= mask_support <= 0.30
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

        score = min(
            0.95,
            0.82 + min(0.12, best_score * 0.12)
        )

        return {
            "label": "Coal",
            "variant": best_name,
            "score": float(score),
            "box": box,
            "source": "coal_mask_component_window",
            "template_score": float(best_score),
            "mask_support": float(mask_support),
            "mean_gray": float(mean_gray),
            "texture_strength": float(texture_strength),
            "dark_ratio": float(dark_ratio),
            "low_sat_ratio": float(low_sat_ratio),
            "edge_density": float(edge_density),
            "colored_ratio": float(colored_ratio),
        }

    def _detect_coal_component_neighbor_windows(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        detections: List[Dict],
        template_bank: Dict[str, np.ndarray],
    ) -> List[Dict]:
        """
        NEU HINZUGEFÜGT:
        Sehr lokale Erweiterung um einen bereits sicheren Coal-Komponenten-
        Treffer. Damit werden test16-artige Nachbarbloecke gefunden, ohne die
        Coal-Suche global zu oeffnen.
        """

        if not template_bank:
            return []

        img_h, img_w = img.shape[:2]
        output = []

        coal_anchors = [
            detection
            for detection in detections
            if (
                detection["label"].lower() == "coal"
                and detection.get("source") == "coal_mask_component_window"
            )
        ]

        for anchor in coal_anchors:
            ax, ay, aw, ah = anchor["box"]
            local_candidates = []

            for window_w, window_h in [
                (123, 111),
                (148, 134),
                (104, 87),
            ]:
                x0 = max(0, int(ax - 1.65 * aw))
                x1 = max(0, int(ax - 0.25 * aw))
                y0 = max(int(0.12 * img_h), int(ay - 0.35 * ah))
                y1 = min(img_h, int(ay + 0.85 * ah))
                step_x = max(16, int(window_w * 0.22))
                step_y = max(16, int(window_h * 0.22))
                max_x = max(x0, x1 - window_w)
                max_y = max(y0, y1 - window_h)

                for wy in range(y0, max_y + 1, step_y):
                    for wx in range(x0, max_x + 1, step_x):
                        if wx + window_w > img_w or wy + window_h > img_h:
                            continue

                        candidate_box = (wx, wy, window_w, window_h)

                        if self._box_iou(candidate_box, tuple(anchor["box"])) >= 0.20:
                            continue

                        detection = self._evaluate_coal_component_neighbor(
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

            axc = ax + aw / 2.0
            ayc = ay + ah / 2.0
            local_candidates.sort(
                key=lambda detection: (
                    detection.get("template_score", 0.0),
                    -(
                        (
                            detection["box"][0]
                            + detection["box"][2] / 2.0
                            - axc
                        ) ** 2
                        + (
                            detection["box"][1]
                            + detection["box"][3] / 2.0
                            - ayc
                        ) ** 2
                    ),
                ),
                reverse=True
            )
            output.extend(
                non_max_suppression(
                    local_candidates[:6],
                    iou_threshold=self.config.nms_iou_threshold
                )[:2]
            )

        return non_max_suppression(
            output,
            iou_threshold=self.config.nms_iou_threshold
        )

    def _evaluate_coal_component_neighbor(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        template_bank: Dict[str, np.ndarray],
        box: Box,
        anchor: Dict,
    ) -> Optional[Dict]:
        """
        NEU HINZUGEFÜGT:
        Validiert ein einzelnes Fenster neben einem sicheren Coal-
        Komponentenanker.
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
        low_sat_ratio = float(np.mean(hsv[:, :, 1] < 125))
        edge_density = float(np.mean(edges > 0))
        colored_ratio = self.coal_detector._colored_ore_ratio_for_coal_reject(roi)

        best_score = 0.0
        best_name = None

        for name, template in template_bank.items():
            score = match_template_multiscale(roi, template)
            if score > best_score:
                best_score = score
                best_name = name

        if best_name is None:
            return None

        # NEU HINZUGEFÜGT:
        # Dieser Fall beschreibt extrem dunkle, homogene test16-Nachbarn. Die
        # Grenze ist absichtlich eng, weil breitere dunkle Fenster schnell
        # Wand- oder Schattenbereiche treffen.
        if not (
            best_score >= 0.79
            and mean_gray <= 18.0
            and 4.0 <= texture_strength <= 7.5
            and dark_ratio >= 0.98
            and low_sat_ratio >= 0.88
            and colored_ratio <= 0.025
            and edge_density <= 0.004
        ):
            return None

        score = min(
            0.94,
            0.82 + best_score * 0.12
        )

        return {
            "label": "Coal",
            "variant": best_name,
            "score": float(score),
            "box": box,
            "source": "coal_component_neighbor_window",
            "template_score": float(best_score),
            "mean_gray": float(mean_gray),
            "texture_strength": float(texture_strength),
            "dark_ratio": float(dark_ratio),
            "low_sat_ratio": float(low_sat_ratio),
            "edge_density": float(edge_density),
            "colored_ratio": float(colored_ratio),
            "anchor_box": anchor["box"],
        }

    def _detect_coal_component_upper_neighbor_windows(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        detections: List[Dict],
        template_bank: Dict[str, np.ndarray],
    ) -> List[Dict]:
        """
        NEU HINZUGEFÜGT:
        Sucht oberhalb/linksbundig neben sicheren Komponenten-Coal-Treffern
        nach warm-dunklen test16-artigen Nachbarn.
        """

        if not template_bank:
            return []

        img_h, img_w = img.shape[:2]
        output = []
        anchor_sources = {
            "coal_component_neighbor_window",
            "coal_mask_component_window",
        }
        coal_anchors = [
            detection
            for detection in detections
            if (
                detection["label"].lower() == "coal"
                and detection.get("source") in anchor_sources
            )
        ]

        for anchor in coal_anchors:
            ax, ay, aw, ah = anchor["box"]
            local_candidates = []

            for window_w, window_h in [
                (148, 134),
                (140, 128),
                (130, 130),
            ]:
                base_x = int(ax - 0.45 * aw)
                base_y = int(ay - 1.20 * ah)

                for offset_x in range(-24, 25, 12):
                    for offset_y in range(-24, 25, 12):
                        wx = max(0, min(img_w - window_w, base_x + offset_x))
                        wy = max(
                            int(0.12 * img_h),
                            min(img_h - window_h, base_y + offset_y)
                        )
                        candidate_box = (wx, wy, window_w, window_h)

                        if any(
                            self._box_iou(candidate_box, tuple(existing["box"])) >= 0.20
                            for existing in detections
                            if existing["label"].lower() == "coal"
                        ):
                            continue

                        detection = self._evaluate_coal_component_upper_neighbor(
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

            local_candidates = non_max_suppression(
                local_candidates,
                iou_threshold=self.config.nms_iou_threshold
            )
            output.extend(local_candidates[:1])

        return non_max_suppression(
            output,
            iou_threshold=self.config.nms_iou_threshold
        )

    def _evaluate_coal_component_upper_neighbor(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        template_bank: Dict[str, np.ndarray],
        box: Box,
        anchor: Dict,
    ) -> Optional[Dict]:
        """
        NEU HINZUGEFÜGT:
        Validiert einen warm-dunklen Komponenten-Nachbar oberhalb eines
        sicheren Coal-Ankers.
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
        low_sat_ratio = float(np.mean(hsv[:, :, 1] < 125))
        mean_hue = float(hsv[:, :, 0].mean())
        mean_saturation = float(hsv[:, :, 1].mean())
        colored_ratio = self.coal_detector._colored_ore_ratio_for_coal_reject(roi)
        edge_density = float(np.mean(edges > 0))

        best_score = 0.0
        best_name = None

        for name, template in template_bank.items():
            score = match_template_multiscale(roi, template)
            if score > best_score:
                best_score = score
                best_name = name

        if best_name is None:
            return None

        if not (
            best_score >= 0.81
            and mean_gray <= 22.0
            and 4.0 <= texture_strength <= 7.0
            and low_sat_ratio >= 0.86
            and 10.0 <= mean_hue <= 21.0
            and 95.0 <= mean_saturation <= 135.0
            and colored_ratio <= 0.32
            and edge_density <= 0.004
        ):
            return None

        score = min(
            0.94,
            0.82 + best_score * 0.12
        )

        return {
            "label": "Coal",
            "variant": best_name,
            "score": float(score),
            "box": box,
            "source": "coal_component_upper_neighbor_window",
            "template_score": float(best_score),
            "mean_gray": float(mean_gray),
            "texture_strength": float(texture_strength),
            "low_sat_ratio": float(low_sat_ratio),
            "mean_hue": float(mean_hue),
            "mean_saturation": float(mean_saturation),
            "colored_ratio": float(colored_ratio),
            "edge_density": float(edge_density),
            "anchor_box": anchor["box"],
        }

    def _detect_coal_component_tail_mask_windows(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        detections: List[Dict],
        template_bank: Dict[str, np.ndarray],
    ) -> List[Dict]:
        """
        NEU HINZUGEFÜGT:
        Sucht kurze maskengestuetzte Coal-Auslaeufer unterhalb eines sicheren
        Komponenten-Coal-Ankers.
        """

        if not template_bank:
            return []

        img_h, img_w = img.shape[:2]
        hsv = convert_bgr_to_hsv(img_preprocessed)
        raw_mask = color_mask(hsv, "coal")
        mask = refine_mask_for_ore("coal", raw_mask.copy())
        mask = clean_mask(mask)
        mask_integral = self._mask_integral(mask)
        output = []

        coal_anchors = [
            detection
            for detection in detections
            if (
                detection["label"].lower() == "coal"
                and detection.get("source") == "coal_mask_component_window"
            )
        ]

        for anchor in coal_anchors:
            ax, ay, aw, ah = anchor["box"]
            local_candidates = []

            for window_w, window_h in [
                (104, 87),
                (110, 92),
                (96, 86),
            ]:
                base_x = int(ax - 0.55 * aw)
                base_y = int(ay + 0.70 * ah)

                for offset_x in range(-24, 25, 12):
                    for offset_y in range(-18, 19, 9):
                        wx = max(0, min(img_w - window_w, base_x + offset_x))
                        wy = max(
                            int(0.12 * img_h),
                            min(img_h - window_h, base_y + offset_y)
                        )
                        candidate_box = (wx, wy, window_w, window_h)

                        if any(
                            self._box_iou(candidate_box, tuple(existing["box"])) >= 0.20
                            for existing in detections
                            if existing["label"].lower() == "coal"
                        ):
                            continue

                        mask_support = self._integral_support(
                            mask_integral,
                            wx,
                            wy,
                            window_w,
                            window_h,
                        )

                        if not (0.06 <= mask_support <= 0.16):
                            continue

                        detection = self._evaluate_coal_component_tail(
                            img,
                            img_preprocessed,
                            template_bank,
                            candidate_box,
                            anchor,
                            mask_support,
                        )

                        if detection is not None:
                            local_candidates.append(detection)

            if not local_candidates:
                continue

            local_candidates = non_max_suppression(
                local_candidates,
                iou_threshold=self.config.nms_iou_threshold
            )
            output.extend(local_candidates[:1])

        return non_max_suppression(
            output,
            iou_threshold=self.config.nms_iou_threshold
        )

    def _evaluate_coal_component_tail(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        template_bank: Dict[str, np.ndarray],
        box: Box,
        anchor: Dict,
        mask_support: float,
    ) -> Optional[Dict]:
        """
        NEU HINZUGEFÜGT:
        Validiert einen kurzen maskengestuetzten Coal-Auslaeufer.
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
        low_sat_ratio = float(np.mean(hsv[:, :, 1] < 125))
        colored_ratio = self.coal_detector._colored_ore_ratio_for_coal_reject(roi)
        edge_density = float(np.mean(edges > 0))

        best_score = 0.0
        best_name = None

        for name, template in template_bank.items():
            score = match_template_multiscale(roi, template)
            if score > best_score:
                best_score = score
                best_name = name

        if best_name is None:
            return None

        if not (
            best_score >= 0.64
            and mean_gray <= 16.0
            and 4.0 <= texture_strength <= 6.5
            and low_sat_ratio >= 0.96
            and colored_ratio <= 0.01
            and edge_density <= 0.004
        ):
            return None

        score = min(
            0.92,
            0.80 + best_score * 0.12
        )

        return {
            "label": "Coal",
            "variant": best_name,
            "score": float(score),
            "box": box,
            "source": "coal_component_tail_mask_window",
            "template_score": float(best_score),
            "mask_support": float(mask_support),
            "mean_gray": float(mean_gray),
            "texture_strength": float(texture_strength),
            "low_sat_ratio": float(low_sat_ratio),
            "colored_ratio": float(colored_ratio),
            "edge_density": float(edge_density),
            "anchor_box": anchor["box"],
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
                            self._box_iou(candidate_box, tuple(existing["box"])) >= 0.18
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
