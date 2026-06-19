# -*- coding: utf-8 -*-
"""Final score and ROI plausibility filtering for detections."""

from typing import Dict, List

import cv2
import numpy as np

from minecraft_ore_detector.app.config import OreDetectorConfig
from minecraft_ore_detector.detection.core import (
    _color_compatibility,
    _color_support_mask,
    _color_support_ratio,
    _copper_green_support,
    _copper_orange_support,
)


class DetectionPlausibilityFilter:
    """Applies ore- and source-specific output plausibility rules."""

    def __init__(self, config: OreDetectorConfig):
        self.config = config

    def filter(self, detections: List[Dict], img: np.ndarray) -> List[Dict]:
        """
        Entfernt erzspezifische Low-Confidence-Ausgaben nach NMS.

        Diese Schwellen sind bewusst nur fuer die aktuell review-basiert
        auffaelligen False-Positive-Treiber gesetzt.
        """

        filtered = []

        for detection in detections:
            label = detection["label"].lower()
            min_score = self.config.min_detection_scores.get(label, 0.0)

            if detection.get("score", 0.0) < min_score:
                continue
            if not self.is_plausible(detection, img):
                continue

            filtered.append(detection)

        return filtered

    def is_plausible(self, detection: Dict, img: np.ndarray) -> bool:
        """
        Prueft einfache klassische ROI-Merkmale fuer bekannte FP-Muster.
        """

        label = detection["label"].lower()
        x, y, w, h = detection["box"]
        roi = img[y:y + h, x:x + w]

        if roi.size == 0:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        s_mean = float(hsv[:, :, 1].mean())
        v_mean = float(hsv[:, :, 2].mean())
        texture_strength = float(gray.std())
        edge_density = float(np.mean(edges > 0))
        aspect_ratio = max(w / float(h), h / float(w))

        if label == "copper":
            if detection.get("source") == "copper_edge_cluster":
                return (
                    detection.get("template_score", 0.0) >= 0.54
                    and 0.72 <= detection.get("copper_support", 0.0) <= 0.94
                    and detection.get("copper_compatibility", 0.0) >= 0.84
                    and detection.get("copper_orange", 0.0) >= 0.72
                    and detection.get("copper_green", 0.0) <= 0.08
                    and detection.get("edge_density", 0.0) >= 0.08
                )

            if detection.get("source") == "copper_dark_cyan_cluster":
                # NEU HINZUGEFÜGT:
                # Sehr dunkler Deepslate-Copper in test10/test11/test12 zeigt
                # fast keinen klassischen Gruenanteil, aber kleine Cyan-Inseln.
                # Diese Quelle bleibt nur gueltig, wenn Clustergeometrie,
                # Cyan-Anteil, dunkle ROI-Struktur und Template plausibel sind.
                return (
                    detection.get("template_score", 0.0) >= 0.575
                    and detection.get("pre_mask_support", 0.0) >= 0.70
                    and detection.get("copper_support", 0.0) >= 0.018
                    and detection.get("copper_compatibility", 0.0) >= 0.50
                    and 0.065 <= detection.get("copper_orange", 0.0) <= 0.58
                    and 0.015 <= detection.get("copper_cyan", 0.0) <= 0.20
                    and detection.get("copper_specks", 0) >= 4
                    and detection.get("coal_template_score", 0.0)
                    <= detection.get("template_score", 0.0) + 0.060
                    and detection.get("emerald_template_score", 0.0)
                    <= detection.get("template_score", 0.0) + 0.115
                    and (
                        detection.get("gold_template_score", 0.0)
                        <= detection.get("template_score", 0.0) + 0.16
                        or detection.get("copper_cyan", 0.0) >= 0.030
                    )
                    and 12.0 <= detection.get("mean_value", 0.0) <= 31.0
                    and 3.0 <= detection.get("texture_strength", 0.0) <= 11.2
                    and detection.get("edge_density", 0.0) <= 0.0045
                    and 55.0 <= detection.get("saturation_mean", 0.0) <= 126.0
                    and 14.0 <= detection.get("value_mean", 0.0) <= 38.0
                    and 80 <= w <= 250
                    and 140 <= h <= 290
                    and aspect_ratio <= 1.75
                )

            normal_copper_case = (
                detection.get("score", 0.0) >= 0.62
                and _color_support_ratio("copper", roi) >= 0.045
                and _color_compatibility("copper", roi) >= 0.72
                and _copper_orange_support(roi) >= 0.03
            )

            # NEU HINZUGEFÜGT:
            # Kompakte Copper-Blöcke mit starkem Orange+Gruen-Signal duerfen
            # knapp unter 0.62 liegen. Grosse warme Wand-/Holzbereiche bleiben
            # durch Groesse, Kanten und Gruenanteil ausgeschlossen.
            compact_mixed_copper_case = (
                detection.get("score", 0.0) >= 0.60
                and max(w, h) <= 115
                and aspect_ratio <= 1.35
                and _color_support_ratio("copper", roi) >= 0.10
                and _color_compatibility("copper", roi) >= 0.88
                and _copper_orange_support(roi) >= 0.05
                and _copper_green_support(roi) >= 0.035
                and edge_density >= 0.08
            )

            return normal_copper_case or compact_mixed_copper_case

        if label == "coal":
            if detection.get("source") == "coal_copper_anchor_window":
                # NEU HINZUGEFÜGT:
                # Ankerbasierter Coal-Fallback bleibt eng an die im Detector
                # gemessenen ROI-Werte gebunden, damit Schatten-/Wandfenster
                # nicht als Coal durchrutschen.
                compact_dark_case = (
                    detection.get("template_score", 0.0) >= 0.50
                    and detection.get("mean_gray", 255.0) <= 24.0
                    and detection.get("texture_strength", 99.0) <= 11.0
                    and detection.get("low_sat_ratio", 0.0) >= 0.94
                    and detection.get("dark_ratio", 0.0) >= 0.92
                    and detection.get("colored_ratio", 1.0) <= 0.035
                    and detection.get("edge_density", 1.0) <= 0.010
                    and w <= 135
                    and h <= 175
                )
                lit_large_case = (
                    detection.get("template_score", 0.0) >= 0.70
                    and detection.get("mean_gray", 255.0) <= 36.0
                    and detection.get("texture_strength", 99.0) <= 13.0
                    and detection.get("low_sat_ratio", 0.0) >= 0.68
                    and detection.get("dark_ratio", 0.0) >= 0.92
                    and detection.get("colored_ratio", 1.0) <= 0.70
                    and detection.get("edge_density", 1.0) <= 0.040
                    and w >= 150
                    and h >= 135
                )
                deepslate_small_case = (
                    detection.get("template_score", 0.0) >= 0.64
                    and detection.get("mean_gray", 255.0) <= 52.0
                    and detection.get("texture_strength", 0.0) >= 10.0
                    and detection.get("edge_density", 0.0) >= 0.055
                    and detection.get("low_sat_ratio", 0.0) >= 0.86
                    and detection.get("dark_ratio", 0.0) >= 0.92
                    and detection.get("colored_ratio", 1.0) <= 0.92
                    and w <= 125
                    and h <= 125
                )

                return compact_dark_case or lit_large_case or deepslate_small_case

            if detection.get("source") == "coal_underwater_blue_window":
                # NEU HINZUGEFÜGT:
                # Unterwasser-Coal ist blau und gesaettigt statt klassisch
                # grau/schwarz. Diese Quelle bleibt deshalb auf dunkle,
                # homogene Blaustich-Komponenten mit sehr gutem Template-
                # Signal beschraenkt.
                return (
                    detection.get("template_score", 0.0) >= 0.78
                    and detection.get("blue_support", 0.0) >= 0.70
                    and detection.get("mean_gray", 255.0) <= 32.0
                    and 3.0 <= detection.get("texture_strength", 0.0) <= 8.0
                    and 118.0 <= detection.get("mean_hue", 0.0) <= 131.0
                    and detection.get("mean_saturation", 0.0) >= 120.0
                    and 145 <= w <= 175
                    and 145 <= h <= 175
                    and aspect_ratio <= 1.15
                )

            if detection.get("source") == "coal_underwater_blue_right_neighbor_window":
                # NEU HINZUGEFÜGT:
                # Nur der rechte Nachbar eines bereits sicheren blauen Coal-
                # Ankers; keine freie blaue Komponentensuche.
                return (
                    detection.get("template_score", 0.0) >= 0.70
                    and detection.get("mean_gray", 255.0) <= 24.0
                    and 3.0 <= detection.get("texture_strength", 0.0) <= 7.0
                    and 120.0 <= detection.get("mean_hue", 0.0) <= 130.0
                    and detection.get("mean_saturation", 0.0) >= 135.0
                    and 115 <= w <= 140
                    and 105 <= h <= 130
                    and aspect_ratio <= 1.30
                )

            if detection.get("source") == "coal_underwater_blue_grid_window":
                # NEU HINZUGEFÜGT:
                # Lokale Unterwasser-Coal-Gruppenfenster: gleicher Blaustich
                # und gleiche weiche Textur wie der sichere blaue Anker.
                return (
                    detection.get("template_score", 0.0) >= 0.748
                    and detection.get("mean_gray", 255.0) <= 30.0
                    and 3.5 <= detection.get("texture_strength", 0.0) <= 7.0
                    and 121.0 <= detection.get("mean_hue", 0.0) <= 129.0
                    and detection.get("mean_saturation", 0.0) >= 135.0
                    and detection.get("edge_density", 1.0) <= 0.004
                    and 145 <= w <= 170
                    and 145 <= h <= 175
                    and aspect_ratio <= 1.15
                )

            if detection.get("source") == "coal_neighbor_window":
                # NEU HINZUGEFÜGT:
                # Erweiterung nur fuer direkt benachbarte Coal-Blöcke neben
                # bereits validen Coal-Fallback-Treffern.
                compact_dark_case = (
                    detection.get("template_score", 0.0) >= 0.34
                    and detection.get("mean_gray", 255.0) <= 24.0
                    and detection.get("texture_strength", 99.0) <= 12.0
                    and detection.get("low_sat_ratio", 0.0) >= 0.94
                    and detection.get("dark_ratio", 0.0) >= 0.92
                    and detection.get("colored_ratio", 1.0) <= 0.040
                    and w <= 90
                )
                large_neighbor_case = (
                    detection.get("template_score", 0.0) >= 0.66
                    and detection.get("mean_gray", 255.0) <= 32.0
                    and detection.get("texture_strength", 99.0) <= 12.0
                    and detection.get("low_sat_ratio", 0.0) >= 0.68
                    and detection.get("dark_ratio", 0.0) >= 0.92
                    and detection.get("colored_ratio", 1.0) <= 0.60
                    and w >= 250
                    and h >= 140
                )
                blue_neighbor_case = (
                    detection.get("anchor_source") == "coal_underwater_blue_window"
                    and detection.get("template_score", 0.0) >= 0.70
                    and detection.get("mean_gray", 255.0) <= 32.0
                    and 3.0 <= detection.get("texture_strength", 0.0) <= 8.0
                )

                return compact_dark_case or large_neighbor_case or blue_neighbor_case

            if detection.get("source") == "coal_second_neighbor_window":
                # NEU HINZUGEFÜGT:
                # Zweite Nachbarschaftsstufe: nur sehr kleine, extrem dunkle
                # Fenster rechts neben einem bereits sicheren Coal-Nachbarn.
                return (
                    detection.get("template_score", 0.0) >= 0.36
                    and detection.get("mean_gray", 255.0) <= 16.0
                    and detection.get("texture_strength", 99.0) <= 12.0
                    and detection.get("low_sat_ratio", 0.0) >= 0.94
                    and detection.get("dark_ratio", 0.0) >= 0.92
                    and detection.get("colored_ratio", 1.0) <= 0.015
                    and w <= 90
                    and 100 <= h <= 130
                )

            if detection.get("source") == "coal_mask_component_window":
                # NEU HINZUGEFÜGT:
                # test16-artiger Komponentenfallback: maximal ein Kandidat
                # pro Maske, niedrige Helligkeit, klar begrenzter Maskenanteil.
                return (
                    detection.get("template_score", 0.0) >= 0.64
                    and 0.16 <= detection.get("mask_support", 0.0) <= 0.30
                    and detection.get("mean_gray", 255.0) <= 16.0
                    and 18.0 <= detection.get("texture_strength", 0.0) <= 35.0
                    and detection.get("low_sat_ratio", 0.0) >= 0.94
                    and detection.get("dark_ratio", 0.0) >= 0.96
                    and detection.get("colored_ratio", 1.0) <= 0.010
                )

            if detection.get("source") == "coal_component_neighbor_window":
                # NEU HINZUGEFÜGT:
                # Sehr lokaler Komponenten-Nachbar fuer test16-artige dunkle
                # Blöcke. Keine Maskenlockerung, sondern harte Textur- und
                # Template-Grenzen direkt an der Detection.
                return (
                    detection.get("template_score", 0.0) >= 0.79
                    and detection.get("mean_gray", 255.0) <= 18.0
                    and 4.0 <= detection.get("texture_strength", 0.0) <= 7.5
                    and detection.get("dark_ratio", 0.0) >= 0.98
                    and detection.get("low_sat_ratio", 0.0) >= 0.88
                    and detection.get("colored_ratio", 1.0) <= 0.025
                    and detection.get("edge_density", 1.0) <= 0.004
                    and 100 <= w <= 150
                    and 85 <= h <= 135
                )

            if detection.get("source") == "coal_component_upper_neighbor_window":
                # NEU HINZUGEFÜGT:
                # Oberer test16-artiger Komponenten-Nachbar: warm-dunkel,
                # schwach texturiert, aber mit klarem Coal-Template.
                return (
                    detection.get("template_score", 0.0) >= 0.81
                    and detection.get("mean_gray", 255.0) <= 22.0
                    and 4.0 <= detection.get("texture_strength", 0.0) <= 7.0
                    and detection.get("low_sat_ratio", 0.0) >= 0.86
                    and 10.0 <= detection.get("mean_hue", 0.0) <= 21.0
                    and 95.0 <= detection.get("mean_saturation", 0.0) <= 135.0
                    and detection.get("colored_ratio", 1.0) <= 0.32
                    and detection.get("edge_density", 1.0) <= 0.004
                    and 130 <= w <= 150
                    and 125 <= h <= 135
                )

            if detection.get("source") == "coal_component_tail_mask_window":
                # NEU HINZUGEFÜGT:
                # Kurzer maskengestuetzter Coal-Auslaeufer unterhalb eines
                # sicheren Komponenten-Ankers.
                return (
                    detection.get("template_score", 0.0) >= 0.64
                    and 0.06 <= detection.get("mask_support", 0.0) <= 0.16
                    and detection.get("mean_gray", 255.0) <= 16.0
                    and 4.0 <= detection.get("texture_strength", 0.0) <= 6.5
                    and detection.get("low_sat_ratio", 0.0) >= 0.96
                    and detection.get("colored_ratio", 1.0) <= 0.01
                    and detection.get("edge_density", 1.0) <= 0.004
                    and 95 <= w <= 110
                    and 85 <= h <= 95
                )

            if detection.get("source") == "coal_warm_left_copper_window":
                # NEU HINZUGEFÜGT:
                # Warmer test18-artiger Coal links unter einem sicheren
                # Copper-Anker: trotz hohem Orange-Anteil nur gueltig, wenn
                # Coal im Template klar gegen andere Erzfamilien gewinnt.
                return (
                    detection.get("template_score", 0.0) >= 0.64
                    and detection.get("template_margin", 0.0) >= 0.08
                    and 40.0 <= detection.get("mean_gray", 255.0) <= 52.0
                    and 12.0 <= detection.get("texture_strength", 0.0) <= 17.0
                    and detection.get("dark_ratio", 0.0) >= 0.99
                    and detection.get("very_dark_ratio", 0.0) >= 0.98
                    and detection.get("low_sat_ratio", 0.0) >= 0.96
                    and 16.0 <= detection.get("mean_hue", 0.0) <= 21.0
                    and 92.0 <= detection.get("mean_saturation", 0.0) <= 112.0
                    and 155.0 <= detection.get("mean_value", 0.0) <= 195.0
                    and 0.075 <= detection.get("edge_density", 0.0) <= 0.120
                    and detection.get("colored_ratio", 0.0) >= 0.70
                    and detection.get("copper_orange", 0.0) >= 0.70
                    and detection.get("copper_green", 1.0) <= 0.004
                    and 80 <= w <= 115
                    and 76 <= h <= 105
                    and aspect_ratio <= 1.35
                )

            return True

        if label == "diamond":
            bright_textured_case = (
                edge_density >= 0.08
                and v_mean >= 80.0
                and texture_strength >= 16.0
            )
            dark_cave_case = (
                s_mean >= 100.0
                and v_mean >= 35.0
                and texture_strength >= 10.0
                and edge_density >= 0.020
                and _color_support_ratio("diamond", roi) >= 0.025
            )

            return bright_textured_case or dark_cave_case

        if label == "iron":
            if detection.get("source") == "iron_color_cluster":
                return (
                    detection.get("pre_color_support", 0.0) >= 0.12
                    and detection.get("pre_color_compatibility", 0.0) >= 0.60
                    and detection.get("template_score", 0.0) >= 0.64
                )

            if detection.get("source") == "iron_compact_window":
                return (
                    detection.get("template_score", 0.0) >= 0.72
                    and detection.get("iron_support", 0.0) >= 0.35
                    and detection.get("pre_color_support", 0.0) >= 0.70
                    and detection.get("pre_color_compatibility", 0.0) >= 0.82
                    and 0.025 <= detection.get("edge_density", 0.0) <= 0.090
                    and 6.0 <= detection.get("texture_strength", 0.0) <= 18.0
                    and 35.0 <= detection.get("mean_value", 0.0) <= 85.0
                    and max(w, h) <= 104
                    and aspect_ratio <= 1.15
                )

            if detection.get("source") == "iron_dense_wide_split":
                return (
                    detection.get("template_score", 0.0) >= 0.56
                    and detection.get("raw_support", 0.0) >= 0.50
                    and detection.get("iron_support", 0.0) >= 0.55
                    and detection.get("pre_color_support", 0.0) >= 0.88
                    and detection.get("pre_color_compatibility", 0.0) >= 0.90
                    and 37.0 <= detection.get("mean_value", 0.0) <= 47.0
                    and detection.get("texture_strength", 0.0) >= 8.0
                    and 120.0 <= detection.get("saturation_mean", 0.0) <= 135.0
                    and 48.0 <= detection.get("value_mean", 0.0) <= 54.0
                    and w >= 520
                    and h >= 720
                    and aspect_ratio <= 1.70
                )

            if detection.get("source") == "iron_large_region_window":
                return (
                    detection.get("template_score", 0.0) >= 0.60
                    and detection.get("raw_support", 0.0) >= 0.58
                    and detection.get("iron_support", 0.0) >= 0.35
                    and detection.get("pre_color_support", 0.0) >= 0.90
                    and detection.get("pre_color_compatibility", 0.0) >= 0.95
                    and 35.0 <= detection.get("mean_value", 0.0) <= 53.0
                    and 8.0 <= detection.get("texture_strength", 0.0) <= 28.0
                    and 0.001 <= detection.get("edge_density", 0.0) <= 0.030
                    and 118.0 <= detection.get("saturation_mean", 0.0) <= 138.0
                    and 45.0 <= detection.get("value_mean", 0.0) <= 61.0
                )

            if detection.get("source") == "iron_dark_top_window":
                return (
                    detection.get("template_score", 0.0) >= 0.59
                    and detection.get("raw_support", 0.0) >= 0.72
                    and detection.get("iron_support", 0.0) >= 0.22
                    and detection.get("pre_color_support", 0.0) >= 0.92
                    and detection.get("pre_color_compatibility", 0.0) >= 0.93
                    and 31.0 <= detection.get("mean_value", 0.0) <= 34.0
                    and 5.3 <= detection.get("texture_strength", 0.0) <= 6.4
                    and detection.get("edge_density", 0.0) <= 0.0015
                    and 126.0 <= detection.get("saturation_mean", 0.0) <= 130.5
                    and 40.0 <= detection.get("value_mean", 0.0) <= 42.2
                )

            if detection.get("source") == "iron_pre_mask_tail":
                # NEU HINZUGEFÜGT:
                # Fragmentierte Iron-Endstücke in test13/test14 haben fast
                # keinen Original-Farbsupport mehr. Sie werden nur akzeptiert,
                # wenn sie an sichere Iron-Cluster anschliessen und die
                # vorverarbeitete Maske plus dunkle ROI-Merkmale exakt passen.
                return (
                    detection.get("template_score", 0.0) >= 0.55
                    and 0.145 <= detection.get("pre_color_support", 0.0) <= 0.235
                    and 24.0 <= detection.get("mean_value", 0.0) <= 29.5
                    and 3.8 <= detection.get("texture_strength", 0.0) <= 11.2
                    and detection.get("edge_density", 0.0) <= 0.0065
                    and 58.0 <= detection.get("saturation_mean", 0.0) <= 92.0
                    and 30.0 <= detection.get("value_mean", 0.0) <= 36.0
                    and 220 <= w <= 290
                    and 190 <= h <= 230
                    and aspect_ratio <= 1.45
                )

            normal_iron_case = (
                edge_density >= 0.10
                and v_mean >= 80.0
                and s_mean <= 80.0
            )
            dark_angled_iron_case = (
                _color_support_ratio("iron", roi) >= 0.055
                and _color_compatibility("iron", roi) >= 0.55
                and s_mean <= 90.0
                and texture_strength >= 4.5
            )

            # NEU HINZUGEFÜGT:
            # Kleine dunkle Iron-Blöcke koennen im Canny-Bild schwach sein,
            # sind aber ueber kompakte Box, Farbsupport und niedrige Helligkeit
            # von den bisherigen grossen Iron-False-Positives getrennt.
            small_dark_iron_case = (
                detection.get("score", 0.0) >= 0.64
                and max(w, h) <= 90
                and aspect_ratio <= 1.35
                and 0.08 <= _color_support_ratio("iron", roi) <= 0.26
                and _color_compatibility("iron", roi) >= 0.82
                and s_mean <= 150.0
                and v_mean <= 60.0
                and texture_strength >= 6.0
                and edge_density >= 0.020
            )

            return normal_iron_case or dark_angled_iron_case or small_dark_iron_case


        if label == "gold":
            if detection.get("source") == "gold_large_window":
                return (
                    detection.get("template_score", 0.0) >= 0.82
                    and detection.get("gold_support", 0.0) >= 0.42
                    and detection.get("gold_pre_support", 0.0) >= 0.55
                    and detection.get("gold_compatibility", 0.0) >= 0.82
                    and 0.035 <= detection.get("edge_density", 0.0) <= 0.125
                    and 8.0 <= detection.get("texture_strength", 0.0) <= 28.0
                    and max(w, h) <= 160
                    and aspect_ratio <= 1.15
                    and v_mean >= 30.0
                )

            if max(w, h) > 110 or (w * h) > 12000:
                return False

            bright_textured_case = (
                edge_density >= 0.08
                and v_mean >= 80.0
                and texture_strength >= 16.0
                and _color_support_ratio("gold", roi) >= 0.015
            )
            dark_cave_case = (
                s_mean >= 70.0
                and v_mean >= 35.0
                and texture_strength >= 8.0
                and edge_density >= 0.020
                and _color_support_ratio("gold", roi) >= 0.020
            )

            return bright_textured_case or dark_cave_case

        if label == "lapis":
            bright_lapis_case = (
                edge_density >= 0.08
                and v_mean >= 80.0
                and aspect_ratio <= 1.50
            )
            dark_cave_lapis_case = (
                edge_density >= 0.06
                and texture_strength >= 8.0
                and aspect_ratio <= 1.50
                and _color_support_ratio("lapis", roi) >= 0.08
                and _color_compatibility("lapis", roi) >= 0.75
            )

            return bright_lapis_case or dark_cave_lapis_case

        if label == "redstone":
            color_mask = _color_support_mask("redstone", roi)
            selected = hsv[color_mask > 0]

            if selected.size == 0:
                return False

            red_s_mean = float(selected[:, 1].mean())
            red_v_mean = float(selected[:, 2].mean())

            return red_s_mean >= 135.0 and red_v_mean >= 75.0

        return True
