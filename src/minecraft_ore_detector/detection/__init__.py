"""Public facade for core detection algorithms."""

from minecraft_ore_detector.detection.candidate_finder import find_candidates
from minecraft_ore_detector.detection.color_validation import (
    _color_compatibility,
    _color_support_mask,
    _color_support_ratio,
    _copper_green_support,
    _copper_orange_support,
)
from minecraft_ore_detector.detection.suppression import non_max_suppression
from minecraft_ore_detector.detection.template_detection import (
    _expand_box,
    detect_diamond,
    detect_with_template_bank,
)
from minecraft_ore_detector.detection.template_matching import (
    load_template,
    match_template,
    match_template_multiscale,
)

__all__ = [
    "detect_diamond",
    "detect_with_template_bank",
    "find_candidates",
    "load_template",
    "match_template",
    "match_template_multiscale",
    "non_max_suppression",
]
