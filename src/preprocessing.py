#Bild verbessern
import cv2
import numpy as np


def load_image(path):
    """
    Lädt ein Bild von der Festplatte.

    Parameters
    ----------
    path : str
        Pfad zum Bild.

    Returns
    -------
    numpy.ndarray
        Geladenes Bild im BGR-Farbraum.
    """
    return cv2.imread(path)


def to_hsv(img):
    """
    Konvertiert ein Bild von BGR nach HSV.

    HSV eignet sich besser für Farberkennung als RGB/BGR,
    da Farbton (Hue) von Helligkeit getrennt wird.

    Parameters
    ----------
    img : numpy.ndarray
        Eingabebild.

    Returns
    -------
    numpy.ndarray
        Bild im HSV-Farbraum.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def apply_clahe(img):
    """
    Verbessert den lokalen Kontrast mithilfe von CLAHE.

    Besonders hilfreich für dunkle Minecraft-Höhlen,
    da Details sichtbar werden ohne das Bild zu überbelichten.

    Parameters
    ----------
    img : numpy.ndarray
        Eingabebild.

    Returns
    -------
    numpy.ndarray
        Kontrastverbessertes Bild.
    """

    # Umwandlung in LAB-Farbraum
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Helligkeitskanal von Farbkanälen trennen
    l, a, b = cv2.split(lab)

    # CLAHE-Objekt erstellen
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    # Kontrastverbesserung auf Helligkeitskanal anwenden
    l = clahe.apply(l)

    # Kanäle wieder zusammensetzen
    lab = cv2.merge((l, a, b))

    # Zurück nach BGR konvertieren
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def blur(img, ksize=3):
    """
    Führt einen Gaussian Blur durch.

    Reduziert Bildrauschen und kleine Artefakte,
    die später bei der Segmentierung stören könnten.

    Parameters
    ----------
    img : numpy.ndarray
        Eingabebild.

    ksize : int
        Größe des Faltungsfensters.

    Returns
    -------
    numpy.ndarray
        Geglättetes Bild.
    """
    return cv2.GaussianBlur(img, (ksize, ksize), 0)