# -*- coding: utf-8 -*-
"""Ore color-support and compatibility measurements."""

from typing import List, Tuple

import cv2
import numpy as np

from minecraft_ore_detector.detection.rules import get_ore_rule

def _color_support_mask(ore: str, roi_bgr: np.ndarray) -> np.ndarray:
    """
    NEU HINZUGEFÜGT:
    Erstellt eine Binärmaske für alle Pixel in einer ROI, die farblich
    zum angegebenen Erz passen.

    Wichtig:
    Ein Minecraft-Erzblock besteht größtenteils aus Stone/Deepslate und nur
    aus einigen farbigen Erzpixeln. Deshalb ist die Durchschnittsfarbe der
    ganzen ROI oft nicht aussagekräftig.

    Parameters
    ----------
    ore : str
        Erz-ID.
    roi_bgr : np.ndarray
        Region of Interest im BGR-Farbraum.

    Returns
    -------
    np.ndarray
        Binärmaske mit 255 für passende Pixel, sonst 0.
    """

    if roi_bgr.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    ore = ore.lower()
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    out = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in get_ore_rule(ore).plausibility_ranges:
        lo = np.array(lower, dtype=np.uint8)
        hi = np.array(upper, dtype=np.uint8)

        part = cv2.inRange(hsv, lo, hi)
        out = cv2.bitwise_or(out, part)

    return out


def _color_support_ratio(ore: str, roi_bgr: np.ndarray) -> float:
    """
    NEU HINZUGEFÜGT:
    Berechnet den Anteil farblich passender Pixel in einer ROI.

    Beispiel:
    1000 Pixel ROI, davon 40 passende Erzpixel -> support = 0.04

    Parameters
    ----------
    ore : str
        Erz-ID.
    roi_bgr : np.ndarray
        Region of Interest im BGR-Farbraum.

    Returns
    -------
    float
        Anteil passender Pixel zwischen 0.0 und 1.0.
    """

    if roi_bgr.size == 0:
        return 0.0

    mask = _color_support_mask(ore, roi_bgr)

    if mask.size == 0:
        return 0.0

    return float((mask > 0).sum()) / float(mask.size)


def _range_support_ratio(
    roi_bgr: np.ndarray,
    ranges: List[Tuple[List[int], List[int]]]
) -> float:
    """
    Berechnet den Anteil der Pixel, die in frei definierte HSV-Bereiche fallen.
    """

    if roi_bgr.size == 0:
        return 0.0

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    out = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in ranges:
        lo = np.array(lower, dtype=np.uint8)
        hi = np.array(upper, dtype=np.uint8)
        out = cv2.bitwise_or(out, cv2.inRange(hsv, lo, hi))

    return float((out > 0).sum()) / float(out.size)


def _copper_orange_support(roi_bgr: np.ndarray) -> float:
    """
    Misst den orange-braunen Copper-Anteil in einer ROI.
    """

    return _range_support_ratio(
        roi_bgr,
        [
            ([5, 45, 30], [30, 255, 255])
        ]
    )


def _copper_green_support(roi_bgr: np.ndarray) -> float:
    """
    Misst den grün-türkisen Copper-Anteil in einer ROI.
    """

    return _range_support_ratio(
        roi_bgr,
        [
            ([70, 25, 25], [98, 255, 255])
        ]
    )


def _min_color_support(ore: str) -> float:
    """
    Gibt den minimal notwendigen Anteil farblich passender Pixel zurück.

    GEÄNDERT:
    Copper und Iron sind bewusst strenger, weil sie in Höhlen sehr leicht
    mit warmem Stein, Holz oder Erde verwechselt werden.

    Parameters
    ----------
    ore : str
        Erz-ID.

    Returns
    -------
    float
        Mindest-Farbsupport.
    """

    return get_ore_rule(ore).min_color_support


def _good_color_support(ore: str) -> float:
    """
    NEU HINZUGEFÜGT:
    Gibt an, ab welchem Farbsupport ein Kandidat farblich stark wirkt.

    Dieser Wert wird für Sonderfälle verwendet, in denen der Template-Score
    etwas niedriger sein darf, wenn die Farbe sehr eindeutig ist.
    """

    return get_ore_rule(ore).good_color_support


def _min_compatibility(ore: str) -> float:
    """
    Gibt die minimale Farbkompatibilität zurück.

    GEÄNDERT:
    Copper und Iron sind strenger als Diamond/Lapis/Redstone, weil sie
    in echten Höhlenbildern deutlich mehr False Positives erzeugen.

    Parameters
    ----------
    ore : str
        Erz-ID.

    Returns
    -------
    float
        Minimale Kompatibilität.
    """

    return get_ore_rule(ore).min_compatibility


def _color_compatibility(ore: str, roi_bgr: np.ndarray) -> float:
    """
    Berechnet einen Farb-Plausibilitätswert zwischen 0.0 und 1.0.

    Bewertet werden:
    - Wie viele passende Erzfarbpixel existieren?
    - Wie gesättigt sind diese Pixel?
    - Wie hell sind diese Pixel?

    GEÄNDERT:
    Es wird nicht nur die Durchschnittsfarbe betrachtet, sondern gezielt
    die farblich passenden Pixel innerhalb der ROI.

    Parameters
    ----------
    ore : str
        Erz-ID.
    roi_bgr : np.ndarray
        Region of Interest im BGR-Format.

    Returns
    -------
    float
        Farbkompatibilität zwischen 0.0 und 1.0.
    """

    if roi_bgr.size == 0:
        return 0.0

    ore = ore.lower()

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = _color_support_mask(ore, roi_bgr)
    support = _color_support_ratio(ore, roi_bgr)

    # Spezialfall Kohle:
    # Kohle ist sehr dunkel und deshalb kaum von Schatten unterscheidbar.
    if ore == "coal":
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        dark_ratio = float((gray < 85).sum()) / float(gray.size)

        support_score = min(
            1.0,
            support / max(_min_color_support(ore), 0.0001)
        )

        dark_score = min(
            1.0,
            dark_ratio / 0.35
        )

        return min(
            1.0,
            0.55 * support_score + 0.45 * dark_score
        )

    if mask.size == 0 or (mask > 0).sum() == 0:
        return 0.0

    selected = hsv[mask > 0]

    s_mean = float(np.mean(selected[:, 1]))
    v_mean = float(np.mean(selected[:, 2]))

    min_support = _min_color_support(ore)

    density_score = min(
        1.0,
        support / max(min_support * 3.0, 0.0001)
    )

    sat_score = max(
        0.0,
        min(1.0, (s_mean - 35.0) / 100.0)
    )

    val_score = max(
        0.0,
        min(1.0, (v_mean - 30.0) / 140.0)
    )

    return min(
        1.0,
        0.55 * density_score + 0.30 * sat_score + 0.15 * val_score
    )
