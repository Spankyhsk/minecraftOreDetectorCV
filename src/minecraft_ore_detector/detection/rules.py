# -*- coding: utf-8 -*-
"""
Zentrale Regeln fuer die klassische Erz-Erkennung.

Die Werte in diesem Modul sind bewusst keine ML-Parameter, sondern feste
Schwellen fuer Farbe, Template-Matching und Plausibilitaetspruefungen.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

HsvRange = Tuple[List[int], List[int]]


@dataclass(frozen=True)
class OreRule:
    segmentation_ranges: List[HsvRange]
    plausibility_ranges: List[HsvRange]
    match_threshold: float
    min_color_support: float
    good_color_support: float
    min_compatibility: float
    min_detection_score: float = 0.0
    use_edges: bool = True
    connect_fragments: bool = False


ORE_RULES: Dict[str, OreRule] = {
    "coal": OreRule(
        segmentation_ranges=[([0, 0, 0], [179, 85, 155])],
        plausibility_ranges=[([0, 0, 0], [179, 75, 105])],
        match_threshold=0.30,
        min_color_support=0.10,
        good_color_support=0.16,
        min_compatibility=0.75,
        use_edges=False,
    ),
    "copper": OreRule(
        segmentation_ranges=[
            ([5, 45, 30], [30, 255, 255]),
            ([70, 25, 25], [98, 255, 255]),
        ],
        plausibility_ranges=[
            ([5, 65, 35], [25, 255, 255]),
            ([70, 25, 25], [98, 255, 255]),
        ],
        match_threshold=0.56,
        min_color_support=0.018,
        good_color_support=0.050,
        min_compatibility=0.32,
    ),
    "diamond": OreRule(
        segmentation_ranges=[
            ([75, 18, 14], [112, 255, 255]),
            ([68, 10, 8], [118, 255, 125]),
        ],
        plausibility_ranges=[([75, 24, 18], [112, 255, 255])],
        match_threshold=0.55,
        min_color_support=0.003,
        good_color_support=0.014,
        min_compatibility=0.10,
        connect_fragments=True,
    ),
    "emerald": OreRule(
        segmentation_ranges=[([45, 25, 18], [85, 255, 255])],
        plausibility_ranges=[([45, 28, 18], [85, 255, 255])],
        match_threshold=0.55,
        min_color_support=0.003,
        good_color_support=0.014,
        min_compatibility=0.10,
    ),
    "gold": OreRule(
        segmentation_ranges=[([15, 40, 40], [42, 255, 255])],
        plausibility_ranges=[([15, 55, 40], [42, 255, 255])],
        match_threshold=0.58,
        min_color_support=0.006,
        good_color_support=0.025,
        min_compatibility=0.14,
        min_detection_score=0.82,
    ),
    "iron": OreRule(
        segmentation_ranges=[
            ([0, 20, 45], [12, 115, 255]),
            ([8, 12, 45], [28, 140, 255]),
            ([165, 20, 45], [179, 115, 255]),
        ],
        plausibility_ranges=[
            ([0, 25, 45], [12, 125, 255]),
            ([8, 25, 45], [30, 165, 255]),
            ([165, 25, 45], [179, 125, 255]),
        ],
        match_threshold=0.54,
        min_color_support=0.014,
        good_color_support=0.045,
        min_compatibility=0.26,
    ),
    "lapis": OreRule(
        segmentation_ranges=[([95, 45, 30], [135, 255, 255])],
        plausibility_ranges=[([95, 50, 30], [135, 255, 255])],
        match_threshold=0.50,
        min_color_support=0.004,
        good_color_support=0.020,
        min_compatibility=0.10,
    ),
    "redstone": OreRule(
        segmentation_ranges=[
            ([0, 55, 35], [10, 255, 255]),
            ([165, 55, 35], [179, 255, 255]),
        ],
        plausibility_ranges=[
            ([0, 65, 35], [10, 255, 255]),
            ([165, 65, 35], [179, 255, 255]),
        ],
        match_threshold=0.53,
        min_color_support=0.004,
        good_color_support=0.020,
        min_compatibility=0.10,
        min_detection_score=0.67,
    ),
}


def supported_ores() -> List[str]:
    return list(ORE_RULES.keys())


def get_ore_rule(ore: str) -> OreRule:
    ore_key = ore.lower()
    if ore_key not in ORE_RULES:
        raise ValueError(f"Unbekanntes Erz '{ore}'. Unterstuetzt werden: {supported_ores()}")
    return ORE_RULES[ore_key]


def ore_match_thresholds() -> Dict[str, float]:
    return {ore: rule.match_threshold for ore, rule in ORE_RULES.items()}


def min_detection_scores() -> Dict[str, float]:
    return {
        ore: rule.min_detection_score
        for ore, rule in ORE_RULES.items()
        if rule.min_detection_score > 0.0
    }
