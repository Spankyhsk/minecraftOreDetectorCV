#Bild verbessern
import cv2


def load_image(path):
    # cv2.imread lädt ein Bild von der Festplatte
    # Standardmäßig im BGR-Format (nicht RGB!)
    img = cv2.imread(path)

    # Sicherheitscheck: wenn Datei nicht existiert → Fehler
    if img is None:
        raise FileNotFoundError(path)

    return img


def apply_clahe(img):
    # CLAHE = Contrast Limited Adaptive Histogram Equalization
    # verbessert lokalen Kontrast (wichtig für dunkle Minecraft-Höhlen)

    # Umwandlung in LAB-Farbraum:
    # L = Helligkeit, A/B = Farbkanäle
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Kanäle trennen
    l, a, b = cv2.split(lab)

    # CLAHE nur auf Helligkeit anwenden (verhindert Farbverfälschung)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    l = clahe.apply(l)

    # Kanäle wieder zusammenführen
    lab = cv2.merge((l, a, b))

    # zurück in BGR (OpenCV Standardformat)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def blur(img):
    # GaussianBlur glättet das Bild
    # reduziert Rauschen und kleine Details
    # → wichtig damit später weniger False Positives entstehen

    return cv2.GaussianBlur(img, (3, 3), 0)


def to_hsv(img):
    # Konvertiert Bild in HSV-Farbraum
    # HSV trennt Farbe (Hue) von Helligkeit → besser für Segmentierung

    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)