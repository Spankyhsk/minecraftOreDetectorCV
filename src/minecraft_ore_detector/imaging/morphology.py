# -*- coding: utf-8 -*-
"""
Modul für morphologische Bildoperationen (Morphology).
Dieses Modul bietet Hilfsfunktionen zur Bereinigung von Binärmasken.
"""

import cv2
import numpy as np


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Bereinigt eine binäre Segmentierungsmaske von Bildrauschen.

    Verwendet morphologische Standard-Operationen mit einem 3x3-Strukturelement (Kernel):
    1. MORPH_OPEN (Opening = Erosion gefolgt von Dilatation):
       Entfernt kleine, isolierte weiße Punkte (Rauschen/Noise), die nicht zu Blöcken gehören.
    2. MORPH_CLOSE (Closing = Dilatation gefolgt von Erosion):
       Schließt kleine schwarze Löcher und Lücken innerhalb von erkannten weißen Clustern.

    Parameters
    ----------
    mask : np.ndarray
        Die zu bereinigende binäre Maske (2D numpy.ndarray mit Werten 0 oder 255).

    Returns
    -------
    np.ndarray
        Die bereinigte binäre Maske.
    """
    # 3x3-Kernel bestehend aus Einsen (Rechteck-Kernel)
    kernel = np.ones((3, 3), np.uint8)

    # Schritt 1: Opening, um kleine "Störpixel" zu entfernen
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Schritt 2: Closing, um Risse/Löcher in Objekten zu schließen
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask
