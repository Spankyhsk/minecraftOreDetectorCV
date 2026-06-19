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

from config import OreDetectorConfig
from pipeline import OreDetector
from preprocessing import load_image
from utils import log
from visualization import show_image


def main():
    config = OreDetectorConfig()

    log("Minecraft Ore Detector CV gestartet")
    log(f"Lade Bild aus '{config.image_path}'...")

    img = load_image(config.image_path)

    log("Bild erfolgreich geladen.")

    output = OreDetector(config).detect_and_render(img)

    show_image(output)

    log("Programm beendet.")


if __name__ == "__main__":
    main()
