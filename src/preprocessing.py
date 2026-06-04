# -*- coding: utf-8 -*-
"""
Modul für die Vorverarbeitung (Preprocessing) von Minecraft-Screenshots.
Dieses Modul bietet Funktionen zum Laden von Bildern, zur Kontrastverbesserung (CLAHE),
zur Rauschreduzierung (Gauß-Filter) und zur Farbraumkonvertierung (BGR nach HSV).
"""

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """
    Lädt ein Bild von der Festplatte unter Verwendung von OpenCV.

    Hinweis: OpenCV lädt Bilder standardmäßig im BGR-Farbraum (Blue, Green, Red)
    und nicht im im Web oder bei anderen Bibliotheken üblichen RGB-Farbraum.

    Parameters
    ----------
    path : str
        Der absolute oder relative Dateipfad zum Bild.

    Returns
    -------
    np.ndarray
        Das geladene Bild als 3D-Numpy-Array (Höhe, Breite, Kanäle).

    Raises
    ------
    FileNotFoundError
        Wenn die Datei nicht existiert oder nicht von OpenCV geladen werden kann.
    """
    img = cv2.imread(path)

    # Sicherheitscheck: Falls Datei fehlerhaft oder Pfad falsch ist
    if img is None:
        raise FileNotFoundError(f"Bild unter '{path}' konnte nicht geladen werden.")

    return img


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """
    Wendet CLAHE (Contrast Limited Adaptive Histogram Equalization) an,
    um den lokalen Kontrast des Bildes zu verbessern.

    Dies ist besonders hilfreich für Minecraft-Screenshots aus dunklen Umgebungen (z.B. Höhlen),
    da es schwach beleuchtete Erze hervorhebt, ohne helle Bereiche überzubelichten.
    Die Operation wird im LAB-Farbraum auf dem Helligkeitskanal (L) ausgeführt,
    um Farbverfälschungen zu verhindern.

    Parameters
    ----------
    img : np.ndarray
        Das Eingabebild im BGR-Format.

    Returns
    -------
    np.ndarray
        Das kontrastoptimierte Bild im BGR-Format.
    """
    # Umwandlung von BGR in den LAB-Farbraum:
    # L = Helligkeit (Luminance), A = Rot-Grün-Kanal, B = Blau-Gelb-Kanal
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Trennung der Kanäle, um CLAHE isoliert auf die Helligkeit anzuwenden
    l, a, b = cv2.split(lab)

    # CLAHE-Objekt erstellen:
    # - clipLimit=2.0 limitiert die Kontrastverstärkung, um Rauschen zu vermeiden.
    # - tileGridSize=(8, 8) teilt das Bild in ein 8x8-Gitter auf, auf dem die
    #   Histogramme lokal ausgeglichen werden (adaptive Methode).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Zusammenführen des modifizierten L-Kanals mit den originalen A- und B-Kanälen
    lab = cv2.merge((l, a, b))

    # Zurückkonvertieren in den standardmäßigen BGR-Farbraum
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def blur(img: np.ndarray) -> np.ndarray:
    """
    Wendet einen Gaußschen Weichzeichner (Gaussian Blur) an, um Bildrauschen
    und störende Detailstrukturen zu minimieren.

    Diese Glättung verhindert, dass kleine Texturvariationen in Minecraft-Blöcken
    fälschlicherweise als Kanten oder Kantenübergänge interpretiert werden.

    Parameters
    ----------
    img : np.ndarray
        Das Eingabebild im BGR-Format.

    Returns
    -------
    np.ndarray
        Das geglättete Bild im BGR-Format.
    """
    # Kernel-Größe (3, 3) reicht aus, um Rauschen zu filtern, ohne wichtige
    # Erzstrukturen zu stark zu verwaschen. Standardabweichung (sigmaX) ist auf 0 gesetzt
    # (wird automatisch aus Kernel-Größe berechnet).
    return cv2.GaussianBlur(img, (3, 3), 0)


def to_hsv(img: np.ndarray) -> np.ndarray:
    """
    Konvertiert ein BGR-Bild in den HSV-Farbraum (Hue, Saturation, Value).

    Der HSV-Farbraum eignet sich hervorragend für die Segmentierung, da er den
    Farbton (Hue), die Farbsättigung (Saturation) und den Helligkeitswert (Value) trennt.
    Dadurch ist er deutlich robuster gegenüber Helligkeits- und Schattenänderungen
    als der BGR-Farbraum.

    Parameters
    ----------
    img : np.ndarray
        Das Eingabebild im BGR-Format.

    Returns
    -------
    np.ndarray
        Das konvertierte Bild im HSV-Format.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)