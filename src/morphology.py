#Rauschen entfernen
import cv2
import numpy as np


def clean_mask(mask):
    # Morphologische Operationen entfernen Rauschen in binären Bildern

    kernel = np.ones((3, 3), np.uint8)

    # MORPH_OPEN:
    # entfernt kleine weiße Punkte (Noise)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # MORPH_CLOSE:
    # schließt kleine Lücken in Objekten
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask