# -*- coding: utf-8 -*-
"""
Hauptmodul (Main Entry Point) des Projekts 'Minecraft Ore Detector CV'.
Dieses Skript steuert den gesamten Ablauf der Pipeline:
1. Laden des Minecraft-Screenshots
2. Vorverarbeitung (Helligkeitsanpassung und Rauschminderung)
3. Erzeugung von Farb- und Kantenmasken
4. Erz-spezifische Filterung und Konturanalyse zur Kandidatengewinnung
5. Template-Matching zur detaillierten Block-Validierung
6. Non-Maximum Suppression zur Zusammenfassung überlappender Treffer
7. Visualisierung der Ergebnisse (mit optionalem Debug-Overlay)
"""

import os
from typing import Dict
import numpy as np

from preprocessing import load_image, apply_clahe, blur, to_hsv
from segmentation import color_mask, edge_mask, hybrid_mask, supported_ores, use_edges_for_ore, refine_mask_for_ore
from morphology import clean_mask
from detection import (
    find_candidates,
    load_template,
    detect_with_template_bank,
    non_max_suppression,
)
from visualization import draw, draw_debug, show
from utils import log, log_debug


# Debug-Modus steuert die Anzeige:
# - True: Zeichnet zusätzlich alle gefundenen Roh-Kandidaten (blaue Boxen mit 'C')
#         sowie die finalen Treffer (grüne Boxen).
# - False: Zeichnet ausschließlich die finalen Treffer (grüne Boxen).
DEBUG = True


# ==========================================
# Pfadkonfiguration (Relativ zum Skript)
# ==========================================
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Der Pfad zum Standard-Testbild
IMAGE_PATH = os.path.join(
    DATA_DIR,
    "screenshots",
    "test17.png"
)

# Ordner mit den Erz-Templates
TEMPLATES_DIR = os.path.join(DATA_DIR, "templates")

# Minimale Ähnlichkeitsgrenzen (Thresholds) für das Template Matching pro Erztyp.
# Diese Werte balancieren Precision (Genauigkeit) und Recall (Abdeckung).
ORE_MATCH_THRESHOLD = {
    "coal": 0.50,
    "copper": 0.60,
    "diamond": 0.71,
    "emerald": 0.67,
    "gold": 0.82,
    "iron": 0.76,
    "lapis": 0.58,
    "redstone": 0.72,
}


def _ore_label(ore_key: str) -> str:
    """Konvertiert die Erz-ID in ein formatiertes Label für die Anzeige (z. B. 'coal' -> 'Coal')."""
    return ore_key.capitalize()


def _load_template_bank_for_ore(ore_key: str) -> Dict[str, np.ndarray]:
    """
    Lädt alle Template-Varianten für einen bestimmten Erztyp aus dem Template-Verzeichnis.
    Extrahiert automatisch den relevanten Erzblock aus den Template-Screenshots.

    Parameters
    ----------
    ore_key : str
        Die Erz-ID (z. B. "diamond").

    Returns
    -------
    Dict[str, np.ndarray]
        Ein Dictionary, das den Variantennamen auf das extrahierte Graustufen-Template abbildet.
    """
    bank = {}
    for name in os.listdir(TEMPLATES_DIR):
        # Nur PNG-Dateien betrachten
        if not name.endswith(".png"):
            continue
        # Nur Dateien betrachten, die mit der Erz-ID beginnen (z. B. 'diamond_')
        if not name.startswith(f"{ore_key}_"):
            continue

        path = os.path.join(TEMPLATES_DIR, name)
        variant = name[:-4]  # .png Endung abschneiden
        bank[variant] = load_template(path)
    return bank


def main() -> None:
    """Hauptmethode zum Ausführen der kompletten Computer-Vision-Pipeline."""
    log("Minecraft Ore Detector CV gestartet")

    # ==========================================
    # 1. Bild laden
    # ==========================================
    log(f"Lade Bild aus '{IMAGE_PATH}'...")
    img = load_image(IMAGE_PATH)
    log("Bild erfolgreich geladen.")

    # ==========================================
    # 2. Vorverarbeitung (Preprocessing)
    # ==========================================
    log("Starte Vorverarbeitung (CLAHE-Kontrastausgleich + Gauß-Glättung)...")
    img_preprocessed = apply_clahe(img)
    img_preprocessed = blur(img_preprocessed)
    log("Vorverarbeitung abgeschlossen.")

    # ==========================================
    # 3. Maskenerzeugung (HSV + Kanten)
    # ==========================================
    log("Erzeuge HSV-Farbbilder und Kantenmasken...")
    hsv = to_hsv(img_preprocessed)
    edges = edge_mask(img_preprocessed)

    all_raw_detections = []
    all_candidates = []

    # ==========================================
    # 4. Iteration über alle unterstützten Erztypen
    # ==========================================
    for ore in supported_ores():
        log(f"--- Verarbeite Erztyp: {ore} ---")

        # 4.1 Segmentierung & Kombination
        # Farbmaske im HSV-Farbraum erzeugen
        color = color_mask(hsv, ore)
        # Hybride Maske mit Canny-Kanten erstellen (außer bei Kohle)
        mask = hybrid_mask(color, edges) if use_edges_for_ore(ore) else color
        # Erzspezifische Anpassungen vornehmen
        mask = refine_mask_for_ore(ore, mask)
        # Morphologische Rauschunterdrückung (Opening/Closing)
        mask = clean_mask(mask)

        # 4.2 Kandidatengewinnung (Konturanalyse)
        candidates = find_candidates(mask, color)
        all_candidates.extend(candidates)
        log(f"Kandidaten gefunden für '{ore}': {len(candidates)}")

        if len(candidates) == 0:
            continue

        # 4.3 Templates laden
        template_bank = _load_template_bank_for_ore(ore)
        if len(template_bank) == 0:
            log(f"WARNUNG: Keine Templates für '{ore}' in '{TEMPLATES_DIR}' gefunden.")
            continue

        for name, tpl in template_bank.items():
            log_debug(f"Template '{name}' geladen (Shape: {tpl.shape})")

        # 4.4 Template-Matching auf den Kandidaten-ROIs ausführen
        raw = detect_with_template_bank(
            img,
            candidates,
            template_bank,
            label=_ore_label(ore),
            threshold=ORE_MATCH_THRESHOLD.get(ore, 0.8),
            brightness_split=95.0,  # Steuert die Aufteilung in Stone vs. Deepslate
        )
        log(f"Rohe Treffer für '{ore}': {len(raw)}")
        all_raw_detections.extend(raw)

    # ==========================================
    # 5. Non-Maximum Suppression (NMS)
    # ==========================================
    # Bereinigt überlappende Bounding Boxes, um Doppel-Detektionen zu vermeiden
    detections = non_max_suppression(all_raw_detections, iou_threshold=0.25)
    log(f"Pipeline beendet. Rohtreffer gesamt: {len(all_raw_detections)} | Nach NMS gefiltert: {len(detections)}")

    if len(detections) == 0:
        log("WARNUNG: Keine Erze im Bild gefunden (Eventuell Thresholds anpassen).")

    # ==========================================
    # 6. Visualisierung & GUI-Ausgabe
    # ==========================================
    if DEBUG:
        log("Debug-Modus aktiv: Zeichne Bounding Boxes der Roh-Kandidaten (Blau) + Treffer (Grün)")
        debug_img = draw_debug(
            img,
            all_candidates,
            detections
        )
        show(debug_img)
    else:
        log("Erzeuge Standard-Ausgabebild (nur final erkannte Erze)...")
        out = draw(img, detections)
        show(out)

    log("Programm beendet.")


if __name__ == "__main__":
    main()