# -*- coding: utf-8 -*-
"""Coal strategies that expand validated neighboring detections."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from minecraft_ore_detector.app.config import OreDetectorConfig
from minecraft_ore_detector.detection.candidate_detection import CoalPrimaryDetector
from minecraft_ore_detector.detection import (
    _copper_green_support,
    _copper_orange_support,
    match_template_multiscale,
    non_max_suppression,
)
from minecraft_ore_detector.detection.geometry import box_iou
from minecraft_ore_detector.imaging.morphology import clean_mask
from minecraft_ore_detector.imaging.preprocessing import convert_bgr_to_hsv
from minecraft_ore_detector.imaging.runtime_mask_filter import RuntimeMaskFilter
from minecraft_ore_detector.imaging.segmentation import color_mask, refine_mask_for_ore
from minecraft_ore_detector.repositories.template_repository import TemplateRepository

Box = Tuple[int, int, int, int]


class NeighborCoalStrategy:
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

                        if box_iou(candidate_box, tuple(anchor["box"])) >= 0.25:
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
