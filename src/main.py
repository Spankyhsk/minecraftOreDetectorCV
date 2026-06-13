# -*- coding: utf-8 -*-
"""
Einstiegspunkt fuer die Minecraft-Ore-Detection.
"""

from config import OreDetectorConfig
from pipeline import OreDetector
from preprocessing import load_image
from utils import log
from visualization import show


def run_pipeline(img):
    """
    Rueckwaertskompatibler Wrapper fuer Skripte wie live_runner.py.
    """

    return OreDetector().run(img)


def main():
    config = OreDetectorConfig()

    log("Minecraft Ore Detector CV gestartet")
    log(f"Lade Bild aus '{config.image_path}'...")

    img = load_image(config.image_path)

    log("Bild erfolgreich geladen.")

    output = OreDetector(config).run(img)

    show(output)

    log("Programm beendet.")


if __name__ == "__main__":
    main()

