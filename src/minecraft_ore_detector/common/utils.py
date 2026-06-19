# -*- coding: utf-8 -*-
"""
Hilfsmodul (Utilities) mit Funktionen zum Skalieren, Speichern
von Bildern sowie zur einheitlichen Konsolenausgabe (Logging).
"""

import cv2
import numpy as np


def resize_image(img: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Skaliert ein Bild proportional mit einem Skalierungsfaktor.

    Parameters
    ----------
    img : np.ndarray
        Das Eingabebild als Numpy-Array.
    scale : float, optional
        Der Skalierungsfaktor (z. B. 0.5 für halbe Größe, 2.0 für doppelte Größe).
        Standardwert ist 1.0.

    Returns
    -------
    np.ndarray
        Das proportional skalierte Bild.
    """
    return cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_LINEAR
    )


def save_image(path: str, img: np.ndarray) -> None:
    """
    Speichert ein Bild im Dateisystem.

    Parameters
    ----------
    path : str
        Der Dateipfad, unter dem das Bild gespeichert werden soll.
    img : np.ndarray
        Das zu speichernde Bild (im BGR- oder Graustufenformat).
    """
    cv2.imwrite(path, img)


def log(msg: str) -> None:
    """
    Gibt eine Standard-Informationsmeldung auf der Konsole aus.

    Parameters
    ----------
    msg : str
        Die auszugebende Nachricht.
    """
    print(f"[INFO] {msg}")


def log_debug(msg: str) -> None:
    """
    Gibt eine Debug-Meldung auf der Konsole aus.
    Kann zur Fehlersuche oder Analyse interner Werte verwendet werden.

    Parameters
    ----------
    msg : str
        Die auszugebende Nachricht.
    """
    print(f"[DEBUG] {msg}")


def log_warning(msg: str) -> None:
    """
    Gibt eine Warnmeldung auf der Konsole aus.

    Parameters
    ----------
    msg : str
        Die auszugebende Warnung.
    """
    print(f"[WARN] {msg}")
