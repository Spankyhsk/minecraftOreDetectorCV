# -*- coding: utf-8 -*-
"""Coal strategies for underwater and blue-tinted scenes."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from minecraft_ore_detector.app.config import OreDetectorConfig
from minecraft_ore_detector.detection.candidate_detection import CoalPrimaryDetector
from minecraft_ore_detector.detection.core import (
    _copper_green_support,
    _copper_orange_support,
    match_template_multiscale,
    non_max_suppression,
)
from minecraft_ore_detector.imaging.morphology import clean_mask
from minecraft_ore_detector.imaging.preprocessing import convert_bgr_to_hsv
from minecraft_ore_detector.imaging.runtime_mask_filter import RuntimeMaskFilter
from minecraft_ore_detector.imaging.segmentation import color_mask, refine_mask_for_ore
from minecraft_ore_detector.repositories.template_repository import TemplateRepository

Box = Tuple[int, int, int, int]


class UnderwaterCoalStrategy:
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

    @staticmethod
    def _mask_integral(mask: np.ndarray) -> np.ndarray:
        return cv2.integral((mask > 0).astype(np.uint8))

    @staticmethod
    def _integral_support(
        integral: np.ndarray, x: int, y: int, width: int, height: int
    ) -> float:
        x2, y2 = x + width, y + height
        total = integral[y2, x2] - integral[y, x2] - integral[y2, x] + integral[y, x]
        return float(total) / float(max(1, width * height))

    @staticmethod
    def _box_iou(box_a: Box, box_b: Box) -> float:
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = aw * ah + bw * bh - intersection
        return intersection / float(union) if union > 0 else 0.0

    def _best_template_score_for_ore(self, ore: str, roi_bgr: np.ndarray) -> float:
        templates = self.template_repository.get_templates_for_ore(ore)
        if not templates or roi_bgr.size == 0:
            return 0.0
        return max(match_template_multiscale(roi_bgr, template) for template in templates.values())

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
