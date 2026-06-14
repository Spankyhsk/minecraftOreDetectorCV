# -*- coding: utf-8 -*-
"""
Zentrale Konfiguration fuer die Ore-Detection-Pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import Dict

from ore_rules import min_detection_scores, ore_match_thresholds


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")


@dataclass(frozen=True)
class OreDetectorConfig:
    """
    Buendelt Pfade, Debug-Schalter und Matching-Thresholds.
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
