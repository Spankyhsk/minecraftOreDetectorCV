# -*- coding: utf-8 -*-
"""
Zentrale Konfiguration fuer die Ore-Detection-Pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import Dict


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")


@dataclass(frozen=True)
class OreDetectorConfig:
    """
    Buendelt Pfade, Debug-Schalter und Matching-Thresholds.
    """

    data_dir: str = DATA_DIR
    image_path: str = os.path.join(DATA_DIR, "screenshots", "test6.png")
    templates_dir: str = os.path.join(DATA_DIR, "templates")
    debug_mask_dir: str = os.path.join(DATA_DIR, "debug_masks")
    debug: bool = False
    save_debug_masks: bool = True
    nms_iou_threshold: float = 0.25
    min_detection_scores: Dict[str, float] = field(default_factory=lambda: {
        "gold": 0.82,
        "redstone": 0.68,
    })
    ore_match_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "coal": 0.30,
        "copper": 0.56,
        "diamond": 0.55,
        "emerald": 0.55,
        "gold": 0.58,
        "iron": 0.61,
        "lapis": 0.50,
        "redstone": 0.53,
    })
