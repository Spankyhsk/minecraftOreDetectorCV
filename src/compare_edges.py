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
from preprocessing import load_image, apply_clahe, blur, to_hsv
from segmentation import edge_mask
from visualization import show
from utils import log, log_warning


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
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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

    log("Wende CLAHE-Kontrastausgleich an...")
    clahe_img = apply_clahe(orig)
    #blur_img = blur(clahe_img)
    hsv = to_hsv(clahe_img)
    edges_img = edge_mask(hsv)
    # Kopien für die Visualisierung erstellen, um die Originalbilder nicht zu verändern
    orig_vis = orig.copy()
    hsv_vis = hsv.copy()
    edges_vis = edges_img.copy()
    
    # Labels hinzufügen
    draw_label(orig_vis, "Original")
    draw_label(hsv_vis, "HSV-Farbbild (Masken)")
    draw_label(edges_vis, "Kantenmaske (Canny)")

    # Bilder horizontal nebeneinander zusammenfügen
    comparison = edges_vis
    
    # Speicherpfad für den Vergleich
    out_path = os.path.join(output_dir, "edgesWithOutBlur.png")
    cv2.imwrite(out_path, comparison)
    log(f"Vergleichsbild erfolgreich gespeichert unter: {out_path}")
    
    # Bild anzeigen
    log("Zeige Vergleichsbild an. Drücke eine beliebige Taste im Bildfenster, um es zu schließen.")
    show(comparison, window_name="Vergleich: Original vs. CLAHE (Vorverarbeitung)")


if __name__ == "__main__":
    main()
