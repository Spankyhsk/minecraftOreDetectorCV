# -*- coding: utf-8 -*-
"""
Modul für die Farbraumbereich-Isolierung und Kantenfilterung (Segmentierung).
Dieses Modul definiert die HSV-Grenzwerte für die verschiedenen Erztypen
und kombiniert Farbmasken mit Kantenmasken zur präzisen Segmentierung von Erzblöcken.
"""

import cv2
import numpy as np
from typing import List

from ore_rules import get_ore_rule, supported_ores as _supported_ores


def supported_ores() -> List[str]:
    """
    Gibt die Liste aller unterstützten Erztypen zurück.

    Returns
    -------
    List[str]
        Die Liste der Erz-IDs (z.B. ["coal", "copper", "diamond", ...]).
    """
    return _supported_ores()


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
    return get_ore_rule(ore).use_edges


def refine_mask_for_ore(ore: str, mask: np.ndarray) -> np.ndarray:
    """
    Erlaubt erz-spezifische, morphologische Feinjustierungen der Segmentierungsmaske.

    Für Kohle (coal) wird beispielsweise eine stärkere Dilatation durchgeführt und kleine
    Löcher geschlossen, da Kohle-Texturen oft sehr unzusammenhängende dunkle Flecken erzeugen.
    """
    if ore == "coal":
        # Stärkere Dilatation und morphologische Operationen, um vereinzelte schwarze Punkte
        # der Kohle zu einer zusammenhängenden Region zu verbinden.
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # NEU HINZUGEFÜGT:
    # Sehr leichte Verbindung kleiner Diamond-/Emerald-Farbinseln.
    # Absichtlich nur 3x3 und nur einmal, damit keine riesigen Wandbereiche entstehen.
    # GEÄNDERT:
    # Nur Diamond wird leicht verbunden.
    # Emerald bleibt unverändert, weil Emerald aktuell gut funktioniert.
    elif get_ore_rule(ore).connect_fragments:
        kernel = np.ones((3, 3), np.uint8)

        # Kleine Diamond-Farbinseln im dunklen Block verbinden
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

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
    ranges = get_ore_rule(ore).segmentation_ranges

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

    # GEÄNDERT:
    # Leichte Glättung reduziert Texturrauschen, ohne die Blockkanten komplett zu verlieren.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Bewusst nicht zu streng, weil sonst dunkle Erzstrukturen verloren gehen.
    return cv2.Canny(gray, 50, 150)


def hybrid_mask(color: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Kombiniert Farbmaske und Kantenmaske, aber nur in der Nähe farbiger Erzpixel.
    Dadurch werden Kanten aus Holz, HUD, Steintexturen usw. nicht überall als Kandidaten benutzt.
    """

    kernel = np.ones((5, 5), np.uint8)

    # Bereich um die Farbmaske leicht vergrößern
    color_zone = cv2.dilate(color, kernel, iterations=2)

    # Nur Kanten behalten, die in der Nähe der Farbmaske liegen
    edges_near_color = cv2.bitwise_and(edges, color_zone)

    # Farbmaske + relevante Kanten kombinieren
    return cv2.bitwise_or(color, edges_near_color)
