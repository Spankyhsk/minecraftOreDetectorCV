# -*- coding: utf-8 -*-
"""Orchestrates the ordered Coal fallback strategies."""

from typing import Callable, Dict, List

import numpy as np

from minecraft_ore_detector.app.config import OreDetectorConfig
from minecraft_ore_detector.detection import non_max_suppression
from minecraft_ore_detector.detection.ores.coal.components import ComponentCoalStrategy
from minecraft_ore_detector.detection.ores.coal.copper_anchors import CopperAnchorCoalStrategy
from minecraft_ore_detector.detection.ores.coal.neighbors import NeighborCoalStrategy
from minecraft_ore_detector.detection.ores.coal.underwater import UnderwaterCoalStrategy
from minecraft_ore_detector.detection.ores.coal.primary import CoalPrimaryDetector
from minecraft_ore_detector.imaging.runtime_mask_filter import RuntimeMaskFilter
from minecraft_ore_detector.repositories.template_repository import TemplateRepository

DetectionFilter = Callable[[List[Dict], np.ndarray], List[Dict]]


class CoalFallbackDetector:
    """Runs Coal fallbacks in the order required by their anchor dependencies."""

    def __init__(
        self,
        config: OreDetectorConfig,
        mask_filter: RuntimeMaskFilter,
        template_repository: TemplateRepository,
        coal_detector: CoalPrimaryDetector,
        detection_filter: DetectionFilter,
    ):
        self.config = config
        self.template_repository = template_repository
        self.detection_filter = detection_filter

        strategy_args = (
            config,
            mask_filter,
            template_repository,
            coal_detector,
        )
        self.copper_anchors = CopperAnchorCoalStrategy(*strategy_args)
        self.underwater = UnderwaterCoalStrategy(*strategy_args)
        self.neighbors = NeighborCoalStrategy(*strategy_args)
        self.components = ComponentCoalStrategy(*strategy_args)

    def apply(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        detections: List[Dict],
    ) -> List[Dict]:
        template_bank = self.template_repository.get_templates_for_ore("coal")
        fallback_detectors = (
            self.copper_anchors._detect_coal_near_copper_anchors,
            self.underwater._detect_underwater_blue_coal_windows,
            self.underwater._detect_coal_underwater_blue_right_neighbor_windows,
            self.underwater._detect_coal_underwater_blue_grid_windows,
            self.neighbors._detect_coal_neighbor_windows,
            self.neighbors._detect_coal_second_neighbor_windows,
            self.components._detect_coal_mask_component_windows,
            self.components._detect_coal_component_neighbor_windows,
            self.components._detect_coal_component_upper_neighbor_windows,
            self.components._detect_coal_component_tail_mask_windows,
            self.copper_anchors._detect_coal_warm_left_of_copper_windows,
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

        merged = non_max_suppression(
            detections + additional_detections,
            iou_threshold=self.config.nms_iou_threshold,
        )
        return self.detection_filter(merged, img)
