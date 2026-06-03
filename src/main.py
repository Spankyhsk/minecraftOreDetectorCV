import os
from preprocessing import *
from segmentation import *
from morphology import *
from detection import *
from visualization import *
from utils import log
from utils import log_debug


DEBUG = True  # <<< NEU


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

TEMPLATES_DIR = os.path.join(DATA_DIR, "templates")

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


def _ore_label(ore_key):
    return ore_key.capitalize()


def _load_template_bank_for_ore(ore_key):
    bank = {}
    for name in os.listdir(TEMPLATES_DIR):
        if not name.endswith(".png"):
            continue
        if not name.startswith(f"{ore_key}_"):
            continue

        path = os.path.join(TEMPLATES_DIR, name)
        variant = name[:-4]
        bank[variant] = load_template(path)
    return bank


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
    edges = edge_mask(img)

    # 4-7. Pro Erztyp segmentieren + Kandidaten + Template Matching
    all_raw_detections = []
    all_candidates = []

    for ore in supported_ores():
        log(f"--- Erztyp: {ore} ---")

        color = color_mask(hsv, ore)
        mask = hybrid_mask(color, edges) if use_edges_for_ore(ore) else color
        mask = refine_mask_for_ore(ore, mask)
        mask = clean_mask(mask)

        candidates = find_candidates(mask, color)
        all_candidates.extend(candidates)
        log(f"Kandidaten ({ore}): {len(candidates)}")

        if len(candidates) == 0:
            continue

        template_bank = _load_template_bank_for_ore(ore)
        if len(template_bank) == 0:
            log(f"WARNUNG: Keine Templates für {ore} gefunden")
            continue

        for name, tpl in template_bank.items():
            log_debug(f"Template {name} shape: {tpl.shape}")

        raw = detect_with_template_bank(
            img,
            candidates,
            template_bank,
            label=_ore_label(ore),
            threshold=ORE_MATCH_THRESHOLD.get(ore, 0.8),
            brightness_split=95.0,
        )
        log(f"Rohtreffer ({ore}): {len(raw)}")
        all_raw_detections.extend(raw)

    detections = non_max_suppression(all_raw_detections, iou_threshold=0.25)
    log(f"Rohtreffer gesamt: {len(all_raw_detections)} | nach NMS: {len(detections)}")

    if len(detections) == 0:
        log("WARNUNG: Keine Matches gefunden (Threshold evtl. zu hoch)")

    # =========================
    # DEBUG OVERLAY (NEU)
    # =========================
    if DEBUG:

        log("DEBUG MODE: Zeichne Kandidaten + Treffer")

        debug_img = draw_debug(
            img,
            all_candidates,
            detections
        )

        show(debug_img)

    else:

        # normaler Output
        log("Erzeuge Output Bild")
        out = draw(img, detections)

        log("Zeige Ergebnis")
        show(out)

    log("Fertig")


if __name__ == "__main__":
    main()