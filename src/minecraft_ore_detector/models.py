"""Shared data shapes used across detection, debug and evaluation code."""

from typing import List, Tuple, TypedDict

Box = Tuple[int, int, int, int]
class Detection(TypedDict, total=False):
    label: str
    score: float
    box: Box
    variant: str
    source: str
    template_score: float
    anchor_box: Box


DetectionList = List[Detection]
