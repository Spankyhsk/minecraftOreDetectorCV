# -*- coding: utf-8 -*-
"""
Modul für die Farbraumbereich-Isolierung und Kantenfilterung (Segmentierung).
Dieses Modul definiert die HSV-Grenzwerte für die verschiedenen Erztypen
und kombiniert Farbmasken mit Kantenmasken zur präzisen Segmentierung von Erzblöcken.
"""

import cv2
import numpy as np
from typing import List


# HSV-Konfigurationsgrenzen für jedes Erz (OpenCV nutzt Hue: 0-179, Saturation: 0-255, Value: 0-255).
# Diese Bereiche sind bewusst etwas weiter gefasst, um verschiedene Lichtverhältnisse (Fackellicht, Schatten)
# abzudecken. Später filtert und validiert das Template-Matching Fehlalarme heraus.
# Hinweis zu Redstone: Da rot im HSV-Farbraum sowohl bei 0 als auch bei 179 liegt, hat Redstone zwei Bereiche.
ORE_CONFIG = {
    "coal": [([0, 0, 0], [179, 70, 135])],
    "copper": [([5, 55, 40], [23, 255, 255])],
    "diamond": [([75, 28, 30], [112, 255, 255])],
    "emerald": [([45, 35, 30], [85, 255, 255])],
    "gold": [([15, 40, 40], [42, 255, 255])],
    "iron": [([8, 12, 45], [28, 140, 255])],
    "lapis": [([95, 45, 30], [135, 255, 255])],
    "redstone": [([0, 55, 35], [10, 255, 255]), ([165, 55, 35], [179, 255, 255])],
}


def supported_ores() -> List[str]:
    """
    Gibt die Liste aller unterstützten Erztypen zurück.

    Returns
    -------
    List[str]
        Die Liste der Erz-IDs (z.B. ["coal", "copper", "diamond", ...]).
    """
    return list(ORE_CONFIG.keys())


def use_edges_for_ore(ore: str) -> bool:
    """
    Bestimmt, ob für ein bestimmtes Erz zusätzlich Kanteninformationen (Canny Edge)
    genutzt werden sollen.

    Kohle (coal) ist sehr dunkel und verhält sich ähnlich wie Schatten an Steinblöcken.
    Die Kombination mit Kanten würde hier zu extrem großen, unbrauchbaren Masken führen.
    Daher wird Kohle ausschließlich farbbasiert segmentiert.

    Parameters
    ----------
    ore : str
        Die Erz-ID.

    Returns
    -------
    bool
        True, wenn eine hybride Maske (Farbe + Kanten) verwendet werden soll, sonst False.
    """
    return ore != "coal"


def refine_mask_for_ore(ore: str, mask: np.ndarray) -> np.ndarray:
    """
    Erlaubt erzwurzel-spezifische, morphologische Feinjustierungen der Segmentierungsmaske.

    Für Kohle (coal) wird beispielsweise eine stärkere Dilatation durchgeführt und kleine
    Löcher geschlossen, da Kohle-Texturen oft sehr unzusammenhängende dunkle Flecken erzeugen.

    Parameters
    ----------
    ore : str
        Die Erz-ID.
    mask : np.ndarray
        Die binäre Segmentierungsmaske (2D numpy.ndarray mit Werten 0 oder 255).

    Returns
    -------
    np.ndarray
        Die verfeinerte binäre Maske.
    """
    if ore == "coal":
        # Stärkere Dilatation und morphologische Operationen, um vereinzelte schwarze Punkte
        # der Kohle zu einer zusammenhängenden Region zu verbinden.
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def color_mask(hsv: np.ndarray, ore: str = "diamond") -> np.ndarray:
    """
    Erzeugt eine Binärmaske basierend auf den HSV-Farbgrenzwerten eines Erzes.

    Die Funktion prüft jeden Pixel des HSV-Bildes, ob er innerhalb eines der für
    das jeweilige Erz definierten HSV-Bereiche liegt.

    Parameters
    ----------
    hsv : np.ndarray
        Das Eingabebild im HSV-Farbraum.
    ore : str, optional
        Das zu isolierende Erz (Standard ist "diamond").

    Returns
    -------
    np.ndarray
        Die binäre Farbmaske (Wert 255 für passende Pixel, sonst 0).
    """
    ranges = ORE_CONFIG[ore]

    # Leere Ausgabemaske in der gleichen Breite und Höhe wie das Eingabebild anlegen
    out = np.zeros(hsv.shape[:2], dtype=np.uint8)

    # Für jeden definierten Bereich (wichtig für Ores mit Hue-Wraparound wie Redstone)
    for lower, upper in ranges:
        lo = np.array(lower, dtype=np.uint8)
        hi = np.array(upper, dtype=np.uint8)
        # cv2.inRange erzeugt 255 für Pixel im Wertebereich, sonst 0.
        # bitwise_or kombiniert die Bereiche, falls mehrere konfiguriert sind.
        out = cv2.bitwise_or(out, cv2.inRange(hsv, lo, hi))

    return out


def edge_mask(img: np.ndarray) -> np.ndarray:
    """
    Erstellt eine Kantenmaske mithilfe des Canny-Edge-Algorithmus.

    Die Kantenerkennung hilft dabei, Strukturveränderungen an Blockgrenzen und
    innerhalb der Erztexturen zu detektieren, um unvollständige Farbmasken zu stützen.

    Parameters
    ----------
    img : np.ndarray
        Das Eingabebild im BGR-Format.

    Returns
    -------
    np.ndarray
        Die binäre Kantenmaske (Wert 255 an Kanten, sonst 0).
    """
    # Canny arbeitet am besten auf Graustufenbildern
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Die Thresholds 50 (unterer Schwellenwert) und 150 (oberer Schwellenwert)
    # filtern schwächere Gradienten und erfassen starke Kontrastübergänge.
    return cv2.Canny(gray, 50, 150)


def hybrid_mask(color: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Kombiniert eine Farbmaske und eine Kantenmaske durch eine logische ODER-Verknüpfung.

    Dadurch wird sichergestellt, dass Kanten, die innerhalb eines Erzkandidaten oder an
    dessen Rändern liegen, die Segmentierung robuster machen (selbst wenn die Farbe
    durch Schatten oder Lichtreflexe leicht verfälscht ist).

    Parameters
    ----------
    color : np.ndarray
        Die binäre Farbmaske.
    edges : np.ndarray
        Die binäre Kantenmaske.

    Returns
    -------
    np.ndarray
        Die kombinierte binäre Maske.
    """
    return cv2.bitwise_or(color, edges)
