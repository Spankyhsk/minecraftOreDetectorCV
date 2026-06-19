# -*- coding: utf-8 -*-
"""
Zentrale Konfiguration für die Ore-Detection-Pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import Dict

from minecraft_ore_detector.detection.rules import min_detection_scores, ore_match_thresholds
from minecraft_ore_detector.paths import DATA_DIR as PROJECT_DATA_DIR


DATA_DIR = str(PROJECT_DATA_DIR)


@dataclass(frozen=True)
class OreDetectorConfig:
    """
    Bündelt Pfade, Debug-Schalter und Matching-Thresholds.
    """

    data_dir: str = DATA_DIR
    image_path: str = os.path.join(DATA_DIR, "screenshots", "test18.png")
    templates_dir: str = os.path.join(DATA_DIR, "templates")
    debug_mask_dir: str = os.path.join(DATA_DIR, "debug_masks")
    debug: bool = False
    save_debug_masks: bool = True
    nms_iou_threshold: float = 0.25
    min_detection_scores: Dict[str, float] = field(default_factory=min_detection_scores)
    ore_match_thresholds: Dict[str, float] = field(default_factory=ore_match_thresholds)
