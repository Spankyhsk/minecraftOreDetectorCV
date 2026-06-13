# -*- coding: utf-8 -*-
"""
Zentrale Ore-Detection-Pipeline.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from candidate_filters import CoalDetector, DiamondCandidateExpander
from config import OreDetectorConfig
from detection import (
    _color_support_mask,
    _color_support_ratio,
    detect_with_template_bank,
    find_candidates,
    non_max_suppression,
)
from mask_filters import MaskRegionFilter
from morphology import clean_mask
from preprocessing import match_scene_brightness, to_hsv
from segmentation import (
    color_mask,
    edge_mask,
    hybrid_mask,
    refine_mask_for_ore,
    supported_ores,
    use_edges_for_ore,
)
from template_repository import TemplateRepository
from visualization import draw, draw_debug

Box = Tuple[int, int, int, int]


@dataclass
class OreDetectionResult:
    """
    Ergebnisdaten einer Pipeline-Ausfuehrung.
    """

    image: np.ndarray
    detections: List[Dict]
    candidates: List[Box]


class DebugMaskWriter:
    """
    Speichert Zwischenmasken, wenn Debug-Ausgabe aktiviert ist.
    """

    def __init__(self, output_dir: str, enabled: bool):
        self.output_dir = output_dir
        self.enabled = enabled

    def save(self, name: str, mask: np.ndarray) -> None:
        if not self.enabled:
            return

        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"{name}.png")
        cv2.imwrite(path, mask)


class OreDetector:
    """
    Orchestriert Preprocessing, Segmentierung, Kandidatenbildung und Matching.
    """

    def __init__(self, config: Optional[OreDetectorConfig] = None):
        self.config = config or OreDetectorConfig()
        self.mask_filter = MaskRegionFilter()
        self.template_repository = TemplateRepository(self.config.templates_dir)
        self.debug_masks = DebugMaskWriter(
            self.config.debug_mask_dir,
            self.config.save_debug_masks
        )
        self.coal_detector = CoalDetector(self.mask_filter)
        self.diamond_expander = DiamondCandidateExpander()

    def run(self, img: np.ndarray) -> np.ndarray:
        """
        Fuehrt die Pipeline aus und gibt das annotierte Bild zurueck.
        """

        result = self.detect(img)

        if self.config.debug:
            return draw_debug(img, result.candidates, result.detections)

        return draw(img, result.detections)

    def detect(self, img: np.ndarray) -> OreDetectionResult:
        """
        Fuehrt die Pipeline aus und gibt strukturierte Zwischenergebnisse zurueck.
        """

        img_preprocessed = match_scene_brightness(img)
        hsv = to_hsv(img_preprocessed)

        edges = edge_mask(img_preprocessed)
        edges = self.mask_filter.clean_runtime_mask(edges, hsv)
        self.debug_masks.save("00_edges_cleaned", edges)

        all_raw_detections = []
        all_candidates: List[Box] = []

        for ore in supported_ores():
            color = color_mask(hsv, ore)
            color = self.mask_filter.clean_runtime_mask(color, hsv)
            self.debug_masks.save(f"01_color_{ore}", color)

            mask = hybrid_mask(color, edges) if use_edges_for_ore(ore) else color
            mask = refine_mask_for_ore(ore, mask)
            mask = clean_mask(mask)
            mask = self.mask_filter.clean_runtime_mask(mask, hsv)
            self.debug_masks.save(f"02_mask_{ore}", mask)

            if ore == "coal":
                candidates = self.coal_detector.find_candidates(img, color)
                all_candidates.extend(candidates)

                if candidates:
                    all_raw_detections.extend(
                        self.coal_detector.detect_direct(img, candidates)
                    )
                continue

            candidates = find_candidates(mask, color)

            if ore == "diamond":
                candidates = self.diamond_expander.expand(candidates, img.shape)

            all_candidates.extend(candidates)

            if not candidates:
                continue

            template_bank = self.template_repository.get_for_ore(ore)
            if not template_bank:
                continue

            raw = detect_with_template_bank(
                img,
                candidates,
                template_bank,
                label=self._ore_label(ore),
                threshold=self.config.ore_match_thresholds.get(ore, 0.8),
                brightness_split=None
            )
            all_raw_detections.extend(raw)

        detections = non_max_suppression(
            all_raw_detections,
            iou_threshold=self.config.nms_iou_threshold
        )
        detections = self._filter_low_confidence_outputs(detections, img)
        detections = self._merge_close_diamond_detections(detections)

        return OreDetectionResult(
            image=img,
            detections=detections,
            candidates=all_candidates
        )

    def _ore_label(self, ore_key: str) -> str:
        return ore_key.capitalize()

    def _filter_low_confidence_outputs(self, detections: List[Dict], img: np.ndarray) -> List[Dict]:
        """
        Entfernt erzspezifische Low-Confidence-Ausgaben nach NMS.

        Diese Schwellen sind bewusst nur fuer die aktuell review-basiert
        auffaelligen False-Positive-Treiber gesetzt.
        """

        filtered = []

        for detection in detections:
            label = detection["label"].lower()
            min_score = self.config.min_detection_scores.get(label, 0.0)

            if detection.get("score", 0.0) < min_score:
                continue
            if not self._passes_roi_plausibility(detection, img):
                continue

            filtered.append(detection)

        return filtered

    def _passes_roi_plausibility(self, detection: Dict, img: np.ndarray) -> bool:
        """
        Prueft einfache klassische ROI-Merkmale fuer bekannte FP-Muster.
        """

        label = detection["label"].lower()
        x, y, w, h = detection["box"]
        roi = img[y:y + h, x:x + w]

        if roi.size == 0:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        s_mean = float(hsv[:, :, 1].mean())
        v_mean = float(hsv[:, :, 2].mean())
        texture_strength = float(gray.std())
        edge_density = float(np.mean(edges > 0))
        aspect_ratio = max(w / float(h), h / float(w))

        if label == "copper":
            return edge_density >= 0.12 and v_mean >= 75.0 and s_mean <= 85.0

        if label == "diamond":
            bright_textured_case = (
                edge_density >= 0.08
                and v_mean >= 80.0
                and texture_strength >= 16.0
            )
            dark_cave_case = (
                s_mean >= 100.0
                and v_mean >= 35.0
                and texture_strength >= 10.0
                and edge_density >= 0.020
                and _color_support_ratio("diamond", roi) >= 0.025
            )

            return bright_textured_case or dark_cave_case

        if label == "iron":
            return edge_density >= 0.10 and v_mean >= 80.0 and s_mean <= 80.0

        if label == "lapis":
            return edge_density >= 0.08 and v_mean >= 80.0 and aspect_ratio <= 1.50

        if label == "redstone":
            color_mask = _color_support_mask("redstone", roi)
            selected = hsv[color_mask > 0]

            if selected.size == 0:
                return False

            red_s_mean = float(selected[:, 1].mean())
            red_v_mean = float(selected[:, 2].mean())

            return red_s_mean >= 135.0 and red_v_mean >= 75.0

        return True

    def _merge_close_diamond_detections(self, detections: List[Dict]) -> List[Dict]:
        diamonds = [d for d in detections if d["label"].lower() == "diamond"]
        others = [d for d in detections if d["label"].lower() != "diamond"]

        if len(diamonds) <= 1:
            return detections

        work = [dict(d) for d in diamonds]
        changed = True

        while changed:
            changed = False
            merged = []
            used = [False] * len(work)

            for i, det in enumerate(work):
                if used[i]:
                    continue

                current = dict(det)
                used[i] = True

                for j in range(i + 1, len(work)):
                    if used[j]:
                        continue

                    if self._boxes_close(current["box"], work[j]["box"], gap=45):
                        current = self._merge_detection_pair(current, work[j])
                        used[j] = True
                        changed = True

                merged.append(current)

            work = merged

        return others + work

    def _boxes_close(self, box_a: Box, box_b: Box, gap: int) -> bool:
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b

        return not (
            ax + aw + gap < bx
            or bx + bw + gap < ax
            or ay + ah + gap < by
            or by + bh + gap < ay
        )

    def _merge_detection_pair(self, det_a: Dict, det_b: Dict) -> Dict:
        ax, ay, aw, ah = det_a["box"]
        bx, by, bw, bh = det_b["box"]

        x1 = min(ax, bx)
        y1 = min(ay, by)
        x2 = max(ax + aw, bx + bw)
        y2 = max(ay + ah, by + bh)

        keep = det_a if det_a.get("score", 0.0) >= det_b.get("score", 0.0) else det_b
        merged = dict(keep)
        merged["box"] = (x1, y1, x2 - x1, y2 - y1)
        merged["score"] = max(det_a.get("score", 0.0), det_b.get("score", 0.0))
        return merged
