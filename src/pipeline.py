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
from detection import detect_with_template_bank, find_candidates, non_max_suppression
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
        detections = self._filter_low_confidence_outputs(detections)

        return OreDetectionResult(
            image=img,
            detections=detections,
            candidates=all_candidates
        )

    def _ore_label(self, ore_key: str) -> str:
        return ore_key.capitalize()

    def _filter_low_confidence_outputs(self, detections: List[Dict]) -> List[Dict]:
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

            filtered.append(detection)

        return filtered
