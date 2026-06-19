# -*- coding: utf-8 -*-
"""Non-maximum suppression for overlapping detections."""

from typing import Dict, List

from minecraft_ore_detector.detection.geometry import (
    box_iou, center_distance, containment_ratio,
)

def non_max_suppression(
    detections: List[Dict],
    iou_threshold: float = 0.25
) -> List[Dict]:
    """
    Entfernt doppelte oder stark überlappende Detektionen.

    GEÄNDERT:
    Neben IoU werden auch Containment und Mittelpunkt-Abstand geprüft.
    Dadurch werden Mehrfachbewertungen desselben Erzblocks reduziert.

    Parameters
    ----------
    detections : List[Dict]
        Liste der Detektionen.
    iou_threshold : float
        IoU-Schwelle.

    Returns
    -------
    List[Dict]
        Gefilterte Detektionen.
    """

    if not detections:
        return []

    ordered = sorted(
        detections,
        key=lambda d: d["score"],
        reverse=True
    )

    kept = []

    for det in ordered:
        keep = True

        for k in kept:
            iou = box_iou(det["box"], k["box"])
            containment = containment_ratio(det["box"], k["box"])
            dist = center_distance(det["box"], k["box"])

            x, y, w, h = det["box"]
            kx, ky, kw, kh = k["box"]

            avg_size = (min(w, h) + min(kw, kh)) / 2.0

            centers_close = (
                avg_size > 0
                and dist < 0.55 * avg_size
            )

            if (
                iou >= iou_threshold
                or containment >= 0.60
                or centers_close
            ):
                keep = False
                break

        if keep:
            kept.append(det)

    return kept
