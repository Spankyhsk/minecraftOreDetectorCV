#Farben isolieren
import cv2
import numpy as np


# feste Farbbereiche für Diamant-Erz in HSV
# cv2.inRange nutzt diese Werte zur Pixel-Selektion
ORE_CONFIG = {
    "diamond": {
        "lower": [80, 50, 50],
        "upper": [105, 255, 255]
    }
}


def color_mask(hsv, ore="diamond"):
    # cv2.inRange erzeugt eine Binärmaske:
    # Pixel im Bereich → 255 (weiß)
    # andere → 0 (schwarz)

    cfg = ORE_CONFIG[ore]

    lower = np.array(cfg["lower"])
    upper = np.array(cfg["upper"])

    return cv2.inRange(hsv, lower, upper)


def edge_mask(img):
    # cv2.Canny = Kantenerkennung
    # erkennt starke Helligkeitsänderungen im Bild

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 50 und 150 = Thresholds für Kantenstärke
    return cv2.Canny(gray, 50, 150)


def hybrid_mask(color, edges):
    # bitwise_or kombiniert zwei Masken
    # Pixel werden markiert, wenn sie in einer der beiden Masken aktiv sind

    return cv2.bitwise_or(color, edges)