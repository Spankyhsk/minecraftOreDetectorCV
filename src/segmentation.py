#Farben isolieren
import cv2
import numpy as np


# HSV-Bereiche je Erz (Hue 0-179 in OpenCV).
# Die Bereiche sind bewusst etwas breit gewählt und werden
# anschließend durch Template-Matching validiert.
ORE_CONFIG = {
    "coal": [([0, 0, 0], [179, 70, 135])],
    "copper": [([5, 55, 40], [23, 255, 255])],
    "diamond": [([75, 28, 30], [112, 255, 255])],
    "emerald": [([45, 35, 30], [85, 255, 255])],
    "gold": [([15, 40, 40], [42, 255, 255])],
    "iron": [([8, 12, 45], [28, 140, 255])],
    "lapis": [([95, 45, 30], [135, 255, 255])],
    # Rot hat einen Hue-Wraparound (nahe 0 und 179)
    "redstone": [([0, 55, 35], [10, 255, 255]), ([165, 55, 35], [179, 255, 255])],
}


def supported_ores():
    return list(ORE_CONFIG.keys())


def use_edges_for_ore(ore):
    # Coal ist sehr dunkel und wird durch OR mit Kanten zu breit;
    # hier hilft die reine Farbsegmentierung besser.
    return ore != "coal"


def refine_mask_for_ore(ore, mask):
    if ore == "coal":
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def color_mask(hsv, ore="diamond"):
    # cv2.inRange erzeugt eine Binärmaske:
    # Pixel im Bereich → 255 (weiß)
    # andere → 0 (schwarz)

    ranges = ORE_CONFIG[ore]

    out = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        lo = np.array(lower, dtype=np.uint8)
        hi = np.array(upper, dtype=np.uint8)
        out = cv2.bitwise_or(out, cv2.inRange(hsv, lo, hi))

    return out


def edge_mask(img):
    # cv2.Canny = Kantenerkennung
    # erkennt starke Helligkeitsänderungen im Bild

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 50 und 150 = Thresholds für Kantenstärke
    return cv2.Canny(gray, 50, 150)


def hybrid_mask(color, edges):
    # Standard: kombiniere Farb-Maske und Kanten-Maske.
    # Wir verwenden bitwise_or, die später durch Kandidaten-Filterung
    # (z.B. Anteil an Farb-Pixeln) weiter bereinigt wird.
    return cv2.bitwise_or(color, edges)
