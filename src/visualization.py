# -*- coding: utf-8 -*-
"""
Modul für die Visualisierung der Ergebnisse.
Dieses Modul bietet Zeichenfunktionen, um erzeugte Kandidaten und gefundene Erze
in Screenshots einzuzeichnen, sowie ein Tkinter-gestütztes, responsives OpenCV-Ausgabefenster.
"""

import cv2
import numpy as np
import tkinter as tk
from typing import List, Dict, Tuple


def draw_debug(img: np.ndarray, candidates: List[Tuple[int, int, int, int]], detections: List[Dict]) -> np.ndarray:
    """
    Zeichnet sowohl die Rohkandidaten (Blau) als auch die finalen Detektionen (Grün)
    in das Bild ein. Dies hilft bei der Analyse von Fehlalarmen (False Positives)
    oder verpassten Erzen (False Negatives).

    Parameters
    ----------
    img : np.ndarray
        Das Originalbild.
    candidates : List[Tuple[int, int, int, int]]
        Eine Liste von Bounding Boxes der Kandidaten (x, y, w, h).
    detections : List[Dict]
        Eine Liste von Detektions-Dictionaries nach der Filterung und NMS.

    Returns
    -------
    np.ndarray
        Das annotierte Bild.
    """
    out = img.copy()

    # ==========================================
    # 1. Kandidaten aus Segmentierung (BLAU)
    # ==========================================
    for (x, y, w, h) in candidates:
        # Dünnes blaues Rechteck
        cv2.rectangle(
            out,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),  # BGR: Blau
            1
        )
        # Kurzes Label "C" (für Candidate)
        cv2.putText(
            out,
            "C",
            (x, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1
        )

    # ==========================================
    # 2. Finale Treffer nach Matching & NMS (GRÜN)
    # ==========================================
    for d in detections:
        x, y, w, h = d["box"]

        # Dickeres grünes Rechteck
        cv2.rectangle(
            out,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),  # BGR: Grün
            2
        )

        # Text: "Erztyp" und der Übereinstimmungsgrad (Score)
        text = f"{d['label']} {d['score']:.2f}"

        # Text in Rot, um ihn vom grünen Kasten abzuheben
        cv2.putText(
            out,
            text,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),  # BGR: Rot
            1
        )

    return out


def draw_candidates(img: np.ndarray, candidates: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """
    Zeichnet ausschließlich die erzeugten Segmentierungskandidaten (Blau) in das Bild ein.

    Parameters
    ----------
    img : np.ndarray
        Das Originalbild.
    candidates : List[Tuple[int, int, int, int]]
        Die Bounding-Boxen der Kandidaten.

    Returns
    -------
    np.ndarray
        Das annotierte Bild.
    """
    out = img.copy()

    for (x, y, w, h) in candidates:
        cv2.rectangle(
            out,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),  # BGR: Blau
            2
        )
        cv2.putText(
            out,
            "cand",
            (x, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1
        )

    return out


def draw(img: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """
    Zeichnet nur die finalen erkannten Erze (Grün) in das Bild ein.
    Dies entspricht der Standard-Ausgabe für den Endnutzer.

    Parameters
    ----------
    img : np.ndarray
        Das Originalbild.
    detections : List[Dict]
        Die Liste der finalen Detektionen.

    Returns
    -------
    np.ndarray
        Das fertig annotierte Bild.
    """
    out = img.copy()

    for d in detections:
        x, y, w, h = d["box"]

        # Grünes Rechteck um das erkannte Erz
        cv2.rectangle(
            out,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),  # BGR: Grün
            2
        )

        # Label und Score
        text = f'{d["label"]}: {d["score"]:.2f}'
        cv2.putText(
            out,
            text,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    return out


def show(img: np.ndarray, window_name: str = "VoxelVision") -> None:
    """
    Zeigt das Bild in einem OpenCV-Fenster an, das sich automatisch an die
    aktuelle Bildschirmauflösung anpasst, um ein Abschneiden des Fensters zu verhindern.

    Verwendet die Bibliothek 'tkinter', um die Bildschirmmaße des Betriebssystems auszulesen.

    Parameters
    ----------
    img : np.ndarray
        Das anzuzeigende Bild.
    window_name : str, optional
        Der Name des Fensters (Standard ist "VoxelVision").
    """
    # ==========================================
    # 1. Bildschirmgröße per Tkinter ermitteln
    # ==========================================
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    img_h, img_w = img.shape[:2]

    # ==========================================
    # 2. Skalierungsfaktor berechnen
    # ==========================================
    # Wir stellen sicher, dass das Fenster maximal 90% der Breite/Höhe ausfüllt.
    # Zudem verhindern wir ein Hochskalieren (scale > 1.0), um Pixel-Matsch zu vermeiden.
    scale = min(
        (screen_width * 0.9) / img_w,
        (screen_height * 0.9) / img_h,
        1.0
    )

    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # ==========================================
    # 3. Fenster anpassen und anzeigen
    # ==========================================
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, new_w, new_h)
    cv2.imshow(window_name, resized)

    # Wartet auf einen Tastendruck, bevor das Fenster geschlossen wird
    cv2.waitKey(0)
    cv2.destroyAllWindows()