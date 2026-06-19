# -*- coding: utf-8 -*-
"""
Temporäres Hilfsskript zum Vergleich des Originalbildes mit der CLAHE-vorverarbeiteten Version.
Dieses Skript lädt einen Minecraft-Screenshot, wendet CLAHE an, fügt zur Kennzeichnung Textlabels hinzu,
speichert den Vergleich als Datei unter 'data/clahe_comparison.png' und zeigt ihn in einem Fenster an.
"""

import os
import argparse
import numpy as np
import cv2

# Importieren der Pipeline-Funktionen
from minecraft_ore_detector.imaging.preprocessing import load_image, normalize_scene_brightness, convert_bgr_to_hsv
from minecraft_ore_detector.imaging.segmentation import color_mask, supported_ores, hybrid_mask, edge_mask, use_edges_for_ore, refine_mask_for_ore
from minecraft_ore_detector.imaging.morphology import clean_mask

from minecraft_ore_detector.presentation.visualization import show_image
from minecraft_ore_detector.paths import PROJECT_ROOT
from minecraft_ore_detector.common.utils import log, log_warning


def draw_label(img: np.ndarray, text: str, position=(20, 50)) -> None:
    """Zeichnet ein lesbares Text-Label mit schwarzem Hintergrund auf das Bild."""
    # Größe des Texts ermitteln
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    # Halbdurchsichtiger schwarzer Hintergrund für das Label
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 10, y - h - 15), (x + w + 10, y + 10), (0, 0, 0), -1)
    
    # Blenden des Overlays mit dem Originalbild (Alpha = 0.6)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    # Weisser Text
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def main():
    # Pfadkonfiguration
    root_dir = str(PROJECT_ROOT)
    data_dir = os.path.join(root_dir, "data")
    output_dir = os.path.join(root_dir, "output")
    screenshots_dir = os.path.join(data_dir, "screenshots")
    
    # ArgumentParser für einfache Auswahl des Bildes
    parser = argparse.ArgumentParser(description="Vergleiche ein Minecraft-Bild mit seiner CLAHE-angepassten Version.")
    parser.format_class = argparse.ArgumentDefaultsHelpFormatter
    parser.add_argument(
        "--image", 
        type=str, 
        default="test1.png", 
        help="Name des Testbildes im Ordner 'data/screenshots/' (z.B. test1.png oder test2.png) oder ein absoluter Pfad."
    )
    args = parser.parse_args()
    
    # Pfad bestimmen
    if os.path.isabs(args.image) or os.path.exists(args.image):
        img_path = args.image
    else:
        img_path = os.path.join(screenshots_dir, args.image)
    
    log(f"Lade Bild aus: {img_path}")
    try:
        orig = load_image(img_path)
    except FileNotFoundError as e:
        log_warning(str(e))
        # Falls das angegebene Bild nicht existiert, versuchen wir test1.png als Fallback
        fallback_path = os.path.join(screenshots_dir, "test1.png")
        log(f"Versuche Fallback-Bild: {fallback_path}")
        try:
            orig = load_image(fallback_path)
            img_path = fallback_path
        except FileNotFoundError:
            log_warning("Kein Testbild gefunden. Bitte stelle sicher, dass 'data/screenshots/test1.png' existiert.")
            return

    normalized_img = normalize_scene_brightness(orig)
    preprocessed_img = normalized_img
    hsv = convert_bgr_to_hsv(preprocessed_img)
    edges = edge_mask(hsv)

    for ore in supported_ores():
        color = color_mask(hsv, ore)
        mask = hybrid_mask(color, edges) if use_edges_for_ore(ore) else color
        mask = refine_mask_for_ore(ore, mask) # bringt aktuell rein gar nichts da bei Coal keine nützlichen Masken hat. (wird auch nur für Coal genutzt)
        mask = clean_mask(mask)

        out_path = os.path.join(output_dir, f"{ore}_clean_mask.png")
        cv2.imwrite(out_path, mask)
        log(f"Vergleichsbild erfolgreich gespeichert unter: {out_path}")


    # Kopien für die Visualisierung erstellen, um die Originalbilder nicht zu verändern
    # orig_vis = orig.copy()
    # preprocessed_vis = preprocessed_img.copy()
    # hsv_vis = hsv.copy()
    #
    # # Labels hinzufügen
    # draw_label(orig_vis, "Original")
    # draw_label(preprocessed_vis, "Helligkeitsnormalisierung")
    # draw_label(hsv_vis, "HSV-Farbbild (Masken)")
    #
    # # Bilder horizontal nebeneinander zusammenfügen
    # comparison = np.hstack((preprocessed_vis, hsv_vis))
    #
    # # Speicherpfad für den Vergleich
    # out_path = os.path.join(output_dir, "hsv.png")
    # cv2.imwrite(out_path, hsv_vis)
    # log(f"Vergleichsbild erfolgreich gespeichert unter: {out_path}")
    #
    # # Bild anzeigen
    # log("Zeige Vergleichsbild an. Drücke eine beliebige Taste im Bildfenster, um es zu schließen.")
    # show_image(hsv_vis, window_name="Vergleich: Vorverarbeitung")


if __name__ == "__main__":
    main()
