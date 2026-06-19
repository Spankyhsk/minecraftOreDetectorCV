# -*- coding: utf-8 -*-
"""Uebersichtlicher Ablauf der Ore-Detection-Pipeline."""

import numpy as np

from minecraft_ore_detector.app.config import OreDetectorConfig
from minecraft_ore_detector.detection.processor import OreDetectionResult, OreDetectionProcessor
from minecraft_ore_detector.presentation.visualization import draw_debug_overlay, draw_detection_boxes


class OreDetector:
    """Orchestriert die Verarbeitungsschritte der Erz-Erkennung."""

    def __init__(self, config: OreDetectorConfig | None = None):
        self.config = config or OreDetectorConfig()
        self.processor = OreDetectionProcessor(self.config)

    def detect_and_render(self, img: np.ndarray) -> np.ndarray:
        result = self.detect(img)
        return self._render_detection_result(img, result)

    def detect(self, img: np.ndarray) -> OreDetectionResult:
        img_preprocessed, hsv, edges = self.processor.preprocess_image(img)

        raw_detections, candidates = self.processor.detect_primary_ores(
            img,
            img_preprocessed,
            hsv,
            edges,
        )

        detections = self.processor.postprocess_detections(raw_detections, img)
        detections = self.processor.apply_coal_fallbacks(
            img,
            img_preprocessed,
            detections,
        )

        return OreDetectionResult(
            image=img,
            detections=detections,
            candidates=candidates,
        )

    def _render_detection_result(
        self,
        img: np.ndarray,
        result: OreDetectionResult,
    ) -> np.ndarray:
        if self.config.debug:
            return draw_debug_overlay(
                img,
                result.candidates,
                result.detections,
            )

        return draw_detection_boxes(img, result.detections)
