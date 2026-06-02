#Rauschen entfernen
import cv2
import numpy as np


def clean_mask(mask):
    """
    Entfernt Rauschen aus einer Binärmaske.

    Opening:
        entfernt kleine weiße Pixelinseln

    Closing:
        schließt kleine Löcher innerhalb von Objekten

    Parameters
    ----------
    mask : numpy.ndarray
        Binärmaske.

    Returns
    -------
    numpy.ndarray
        Bereinigte Maske.
    """

    kernel = np.ones((3, 3), np.uint8)

    # Kleine Artefakte entfernen
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Löcher schließen
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask