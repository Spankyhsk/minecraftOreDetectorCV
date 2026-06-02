import os
from preprocessing import *
from segmentation import *
from morphology import *
from detection import *
from visualization import *
from utils import log


# =========================
# ROOT VERZEICHNIS FIX
# =========================
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATA_DIR = os.path.join(ROOT_DIR, "data")

IMAGE_PATH = os.path.join(
    DATA_DIR,
    "screenshots",
    "test1.png"
)

TEMPLATE_PATH = os.path.join(
    DATA_DIR,
    "templates",
    "diamond_ore.png"
)


def main():

    log("VoxelVision gestartet")

    # 1. Bild laden
    log("Lade Bild...")
    img = load_image(IMAGE_PATH)
    log("Bild erfolgreich geladen")

    # 2. Preprocessing
    log("Starte Preprocessing (CLAHE + Blur)")
    img = apply_clahe(img)
    img = blur(img)
    log("Preprocessing abgeschlossen")

    # 3. HSV + Kanten
    log("Erzeuge HSV + Edge Masken")
    hsv = to_hsv(img)

    color = color_mask(hsv, "diamond")
    edges = edge_mask(img)

    mask = hybrid_mask(color, edges)
    log("Hybrid-Maske erstellt")

    # 4. Maske bereinigen
    log("Bereinige Maske (Morphologie)")
    mask = clean_mask(mask)
    log("Maske bereinigt")

    # 5. Kandidaten finden
    log("Suche Kandidaten...")
    candidates = find_candidates(mask)
    log(f"Kandidaten gefunden: {len(candidates)}")

    # Debug: wenn nichts gefunden wurde
    if len(candidates) == 0:
        log("WARNUNG: Keine Kandidaten gefunden!")

    # 6. Template laden
    log("Lade Template...")
    template = load_template(TEMPLATE_PATH)
    log("Template geladen")

    # 7. Detection
    log("Starte Template Matching...")
    detections = detect_diamond(
        img,
        candidates,
        template
    )

    log(f"Diamanten erkannt: {len(detections)}")

    # Debug: wenn nichts erkannt wurde
    if len(detections) == 0:
        log("WARNUNG: Keine Matches gefunden (Threshold evtl. zu hoch)")

    # 8. Visualisierung
    log("Erzeuge Output Bild")
    out = draw(img, detections)

    log("Zeige Ergebnis")
    show(out)

    log("Fertig")


if __name__ == "__main__":
    main()