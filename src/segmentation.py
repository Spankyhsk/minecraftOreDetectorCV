#Farben isolieren
import cv2
import numpy as np


def create_mask(hsv, lower, upper):
    """
    Erstellt eine Binärmaske für einen HSV-Farbbereich.

    Pixel innerhalb des Bereichs werden weiß,
    alle anderen schwarz.

    Parameters
    ----------
    hsv : numpy.ndarray
        HSV-Bild.

    lower : list
        Untere HSV-Grenze.

    upper : list
        Obere HSV-Grenze.

    Returns
    -------
    numpy.ndarray
        Binärmaske.
    """

    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)

    return cv2.inRange(hsv, lower, upper)


def diamond_mask(hsv):
    """
    Erstellt eine Maske für Diamanterz.

    Die HSV-Werte müssen später
    anhand echter Screenshots angepasst werden.
    """
    return create_mask(
        hsv,
        [80, 50, 50],
        [105, 255, 255]
    )


def gold_mask(hsv):
    """
    Erstellt eine Maske für Golderz.
    """
    return create_mask(
        hsv,
        [15, 100, 100],
        [35, 255, 255]
    )