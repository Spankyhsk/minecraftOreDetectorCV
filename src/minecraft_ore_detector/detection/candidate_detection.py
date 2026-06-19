# -*- coding: utf-8 -*-
"""Compatibility facade for ore-specific candidate helpers."""

from minecraft_ore_detector.detection.ores.coal.primary import CoalPrimaryDetector
from minecraft_ore_detector.detection.ores.diamond_expander import DiamondCandidateExpander

__all__ = [
    "CoalPrimaryDetector",
    "DiamondCandidateExpander",
]
