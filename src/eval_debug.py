# -*- coding: utf-8 -*-
"""
Modul zur Headless-Evaluierung der Detektionspipeline (Evaluation & Debugging).
Dieses Skript führt die vollständige Bildverarbeitungspipeline für die Test-Screenshots
'test1.png' und 'test2.png' aus und gibt detaillierte Statistiken über die Anzahl
der Kandidaten, die Roh-Detektionen und die finalen Treffer nach NMS auf der Konsole aus.
Es benötigt keine grafische Oberfläche (GUI) und ist ideal für schnelle Tests.
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


# Pfadkonfiguration
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
TEMPLATES_DIR = os.path.join(DATA_DIR, "templates")

# Schwellenwerte (Thresholds) für das Template Matching je Erztyp.
# Erze wie Gold oder Eisen benötigen höhere Schwellenwerte, da ihre Texturen
# sonst leicht mit normalen Stein- oder Deepslate-Strukturen verwechselt werden.
# Lapis und Kohle haben niedrigere Schwellenwerte aufgrund ihrer unregelmäßigen
# oder dunklen Formverteilung im Spiel.
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
    """Konvertiert die Erz-ID (Kleinbuchstaben) in ein schönes Label (Capitalized)."""
    return ore_key.capitalize()


def _load_template_bank_for_ore(ore_key: str) -> Dict[str, np.ndarray]:
    """
    Lädt alle Template-Varianten eines bestimmten Erzes aus dem Template-Verzeichnis.
    Zum Beispiel werden für 'diamond' die Templates 'diamond_ore.png' und
    'diamond_deepslate_ore.png' geladen und vorverarbeitet.

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
        if not name.endswith(".png"):
            continue
        if not name.startswith(f"{ore_key}_"):
            continue

        variant = name[:-4]  # Dateiendung ".png" abschneiden
        # load_template extrahiert automatisch den reinen Erzblock aus dem Template-Screenshot
        bank[variant] = load_template(os.path.join(TEMPLATES_DIR, name))
    return bank


def run_eval() -> None:
    """
    Führt die Evaluierungsschleife über alle Test-Screenshots aus
    und gibt die Metriken in der Konsole aus.
    """
    # Über die beiden vorhandenen Test-Screenshots iterieren
    for shot in ["test1.png", "test2.png"]:
        img_path = os.path.join(DATA_DIR, "screenshots", shot)
        img = load_image(img_path)
        
        # 1. Vorverarbeitung: Kontrast anpassen und glätten
        img = blur(apply_clahe(img))
        hsv = to_hsv(img)
        edges = edge_mask(img)

        all_candidates = []
        all_raw = []

        # 2. Pipeline für jeden unterstützten Erztyp durchlaufen
        for ore in supported_ores():
            template_bank = _load_template_bank_for_ore(ore)
            if len(template_bank) == 0:
                continue

            # Segmentierung
            color = color_mask(hsv, ore)
            # Hybride Maske (Farbe + Kanten) verwenden, falls konfiguriert (alles außer Kohle)
            mask = hybrid_mask(color, edges) if use_edges_for_ore(ore) else color
            # Erz-spezifische Anpassungen & morphologische Bereinigung
            mask = clean_mask(refine_mask_for_ore(ore, mask))
            
            # Konturerkennung & Boxen-Zusammenführung
            candidates = find_candidates(mask, color)
            all_candidates.extend(candidates)

            # Template-Matching für die Kandidaten
            raw = detect_with_template_bank(
                img,
                candidates,
                template_bank,
                label=_ore_label(ore),
                threshold=ORE_MATCH_THRESHOLD.get(ore, 0.8),
                brightness_split=95.0,
            )
            all_raw.extend(raw)

        # 3. Non-Maximum Suppression (NMS) anwenden, um mehrfach erkannte Blöcke zusammenzufassen
        final = non_max_suppression(all_raw, iou_threshold=0.25)

        # 4. Textbasierte Ausgabe der Ergebnisse
        print(f"\n=== Evaluierung für {shot} ===")
        print("Anzahl gefundener Kandidaten-Boxen:", len(all_candidates))
        
        print(f"Roh-Detektionen (insgesamt {len(all_raw)}):")
        for d in all_raw:
            print(f"  - {d['label']} (Score: {d['score']:.3f}, Template: '{d.get('variant')}', Box: {d['box']})")
            
        print(f"Finale Detektionen nach NMS (insgesamt {len(final)}):")
        for d in final:
            print(f"  * {d['label']} (Score: {d['score']:.3f}, Template: '{d.get('variant')}', Box: {d['box']})")


if __name__ == "__main__":
    run_eval()
