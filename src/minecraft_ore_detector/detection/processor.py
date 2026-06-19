# -*- coding: utf-8 -*-
"""Interne Komponenten und Erkennungsstrategien der Ore-Detection-Pipeline."""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from minecraft_ore_detector.detection.candidate_detection import (
    CoalPrimaryDetector,
    DiamondCandidateExpander,
)
from minecraft_ore_detector.detection.ores.coal.fallbacks import CoalFallbackDetector
from minecraft_ore_detector.app.config import OreDetectorConfig
from minecraft_ore_detector.detection.ores.copper import CopperDetector
from minecraft_ore_detector.detection import (
    _color_compatibility,
    _copper_green_support,
    _copper_orange_support,
    _expand_box,
    _color_support_mask,
    _color_support_ratio,
    detect_with_template_bank,
    find_candidates,
    match_template_multiscale,
    non_max_suppression,
)
from minecraft_ore_detector.detection.ores.diamond import DiamondPostprocessor
from minecraft_ore_detector.detection.ores.gold import GoldDetector
from minecraft_ore_detector.detection.ores.iron import IronDetector
from minecraft_ore_detector.detection.plausibility import DetectionPlausibilityFilter
from minecraft_ore_detector.imaging.runtime_mask_filter import RuntimeMaskFilter
from minecraft_ore_detector.imaging.morphology import clean_mask
from minecraft_ore_detector.imaging.preprocessing import normalize_scene_brightness, convert_bgr_to_hsv
from minecraft_ore_detector.imaging.segmentation import (
    color_mask,
    edge_mask,
    hybrid_mask,
    refine_mask_for_ore,
    supported_ores,
    use_edges_for_ore,
)
from minecraft_ore_detector.repositories.template_repository import TemplateRepository
from minecraft_ore_detector.models import Box, Detection

@dataclass
class OreDetectionResult:
    """
    Ergebnisdaten einer Pipeline-Ausfuehrung.
    """

    image: np.ndarray
    detections: List[Detection]
    candidates: List[Box]


class DebugMaskWriter:
    """
    Speichert Zwischenmasken, wenn Debug-Ausgabe aktiviert ist.
    """

    def __init__(self, output_dir: str, enabled: bool):
        self.output_dir = output_dir
        self.enabled = enabled

    def save_mask(self, name: str, mask: np.ndarray) -> None:
        if not self.enabled:
            return

        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"{name}.png")
        cv2.imwrite(path, mask)


class OreDetectionProcessor:
    """
    Buendelt die internen Verarbeitungsschritte der Ore-Detection-Pipeline.
    """

    def __init__(self, config: OreDetectorConfig):
        self.config = config
        self.mask_filter = RuntimeMaskFilter()
        self.template_repository = TemplateRepository(self.config.templates_dir)
        self.plausibility_filter = DetectionPlausibilityFilter(self.config)
        self.debug_masks = DebugMaskWriter(
            self.config.debug_mask_dir,
            self.config.save_debug_masks
        )
        self.coal_detector = CoalPrimaryDetector(self.mask_filter)
        self.coal_fallback_detector = CoalFallbackDetector(
            config=self.config,
            mask_filter=self.mask_filter,
            template_repository=self.template_repository,
            coal_detector=self.coal_detector,
            detection_filter=self.plausibility_filter.filter,
        )
        self.diamond_expander = DiamondCandidateExpander()
        self.diamond_postprocessor = DiamondPostprocessor()
        self.copper_detector = CopperDetector(
            self.mask_filter,
            self.template_repository,
        )
        self.gold_detector = GoldDetector(self.config, self.mask_filter)
        self.iron_detector = IronDetector(self.config, self.mask_filter)

    def preprocess_image(
        self,
        img: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        img_preprocessed = normalize_scene_brightness(img)
        hsv = convert_bgr_to_hsv(img_preprocessed)

        edges = edge_mask(img_preprocessed)
        edges = self.mask_filter.filter_mask(edges, hsv)
        self.debug_masks.save_mask("00_edges_cleaned", edges)

        return img_preprocessed, hsv, edges

    def detect_primary_ores(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        hsv: np.ndarray,
        edges: np.ndarray,
    ) -> Tuple[List[Dict], List[Box]]:
        all_raw_detections = []
        all_candidates: List[Box] = []

        for ore in supported_ores():
            color = color_mask(hsv, ore)
            color = self.mask_filter.filter_mask(color, hsv, ore=ore)
            self.debug_masks.save_mask(f"01_color_{ore}", color)

            mask = hybrid_mask(color, edges) if use_edges_for_ore(ore) else color
            mask = refine_mask_for_ore(ore, mask)
            mask = clean_mask(mask)
            mask = self.mask_filter.filter_mask(mask, hsv, ore=ore)
            self.debug_masks.save_mask(f"02_mask_{ore}", mask)

            if ore == "coal":
                candidates = self.coal_detector.find_candidates(img, color)
                all_candidates.extend(candidates)

                if candidates:
                    all_raw_detections.extend(
                        self.coal_detector.detect_from_candidates(img, candidates)
                    )
                else:
                    # NEU HINZUGEFÜGT:
                    # Sehr konservativer Coal-Fallback: nur wenn der direkte
                    # Coal-Detector leer bleibt, wird blockweise mit der
                    # Template-Bank nach einem sehr sicheren Treffer gesucht.
                    template_bank = self.template_repository.get_templates_for_ore(ore)
                    all_raw_detections.extend(
                        self.coal_detector.detect_template_fallback(
                            img,
                            color,
                            template_bank
                        )
                    )
                continue

            candidates = find_candidates(mask, color, ore=ore)

            if ore == "diamond":
                candidates = self.diamond_expander.expand_candidates(candidates, img.shape)

            template_bank = self.template_repository.get_templates_for_ore(ore)
            if not template_bank:
                all_candidates.extend(candidates)
                continue

            all_candidates.extend(candidates)

            if ore == "copper":
                all_raw_detections.extend(
                    self.copper_detector.detect_mixed_large_windows(
                        img,
                        img_preprocessed,
                        mask,
                        template_bank
                    )
                )

            if ore == "gold":
                all_raw_detections.extend(
                    self.gold_detector.detect_large_mask_windows(
                        img,
                        img_preprocessed,
                        hsv,
                        template_bank
                    )
                )

            if ore == "iron":
                all_raw_detections.extend(
                    self.iron_detector.detect_compact_windows(
                        img,
                        img_preprocessed,
                        hsv,
                        template_bank
                    )
                )
                all_raw_detections.extend(
                    self.iron_detector.detect_dense_wide_split(
                        img,
                        img_preprocessed,
                        hsv,
                        template_bank
                    )
                )
                all_raw_detections.extend(
                    self.iron_detector.detect_large_region_windows(
                        img,
                        img_preprocessed,
                        hsv,
                        template_bank
                    )
                )
                all_raw_detections.extend(
                    self.iron_detector.detect_dark_top_windows(
                        img,
                        img_preprocessed,
                        hsv,
                        template_bank
                    )
                )

            if not candidates:
                continue

            raw = detect_with_template_bank(
                img,
                candidates,
                template_bank,
                label=self._format_ore_label(ore),
                threshold=self.config.ore_match_thresholds.get(ore, 0.8),
                brightness_split=None
            )
            all_raw_detections.extend(raw)

            if ore == "copper":
                all_raw_detections.extend(
                    self.copper_detector.detect_edge_clusters(
                        img,
                        img_preprocessed,
                        edges,
                        template_bank
                    )
                )
                all_raw_detections.extend(
                    self.copper_detector.detect_dark_cyan_clusters(
                        img,
                        img_preprocessed,
                        template_bank,
                        [
                            detection
                            for detection in all_raw_detections
                            if detection["label"].lower() == "copper"
                        ]
                    )
                )

            if ore == "iron":
                iron_color_cluster_detections = self.iron_detector.detect_color_clusters(
                    img,
                    img_preprocessed,
                    color,
                    template_bank
                )
                all_raw_detections.extend(iron_color_cluster_detections)
                all_raw_detections.extend(
                    self.iron_detector.detect_pre_mask_tail_windows(
                        img,
                        hsv,
                        template_bank,
                        iron_color_cluster_detections
                    )
                )

        return all_raw_detections, all_candidates

    def postprocess_detections(
        self,
        raw_detections: List[Dict],
        img: np.ndarray,
    ) -> List[Dict]:
        detections = non_max_suppression(
            raw_detections,
            iou_threshold=self.config.nms_iou_threshold
        )
        detections = self.plausibility_filter.filter(detections, img)
        return self.diamond_postprocessor.postprocess(detections, img)

    def apply_coal_fallbacks(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        detections: List[Dict],
    ) -> List[Dict]:
        return self.coal_fallback_detector.apply(
            img,
            img_preprocessed,
            detections,
        )

    def _format_ore_label(self, ore_key: str) -> str:
        return ore_key.capitalize()
