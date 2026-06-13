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

import cv2
import numpy as np

from preprocessing import load_image, match_scene_brightness, to_hsv
from segmentation import (
    color_mask,
    edge_mask,
    hybrid_mask,
    supported_ores,
    use_edges_for_ore,
    refine_mask_for_ore,
)

from detection import (
    find_candidates,
    load_template,
    detect_with_template_bank,
    non_max_suppression,
)
from visualization import draw, draw_debug, show
from utils import log


# Debug-Modus steuert die Anzeige:
# - True: Zeichnet zusätzlich alle gefundenen Roh-Kandidaten (blaue Boxen mit 'C')
#         sowie die finalen Treffer (grüne Boxen).
# - False: Zeichnet ausschließlich die finalen Treffer (grüne Boxen).
DEBUG = True

# Optionaler Debug-Schalter.
# Wenn True, werden Zwischenmasken im Ordner data/debug_masks gespeichert.
SAVE_DEBUG_MASKS = True


# ==========================================
# Pfadkonfiguration (Relativ zum Skript)
# ==========================================

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATA_DIR = os.path.join(ROOT_DIR, "data")

# GEÄNDERT:
# Hier stellst du dein Testbild ein.
# Wichtig: In dieser Datei darf IMAGE_PATH nur EINMAL vorkommen.
IMAGE_PATH = os.path.join(
    DATA_DIR,
    "screenshots",
    "test6.png"
)

# Ordner mit den Erz-Templates
TEMPLATES_DIR = os.path.join(DATA_DIR, "templates")

# Ordner für optionale Debug-Masken.
DEBUG_MASK_DIR = os.path.join(DATA_DIR, "debug_masks")

Vorverarbeitung_DIR = os.path.join(DATA_DIR, "vorverarbeitung")


# Minimale Ähnlichkeitsgrenzen (Thresholds) für das Template Matching pro Erztyp.
# Diese Werte balancieren Precision (Genauigkeit) und Recall (Abdeckung).
ORE_MATCH_THRESHOLD = {
    "coal": 0.30,
    "copper": 0.56,
    "diamond": 0.55,
    "emerald": 0.55,
    "gold": 0.58,
    "iron": 0.61,
    "lapis": 0.50,
    "redstone": 0.53,
}


def expand_diamond_candidates(candidates, img_shape):
    """
    NEU HINZUGEFÜGT / GEÄNDERT:
    Diamond-Spezialfall für dunkle Diamond-Blöcke.

    Problem:
    Dunkle Diamond-Blöcke erzeugen mehrere kleine Farbinseln.
    Wenn jede kleine Farbinsel einzeln erweitert wird, entstehen mehrere Boxen
    auf demselben Block.

    Lösung:
    Nahe Diamond-Kandidaten werden zuerst zu einer Gruppe zusammengeführt.
    Danach wird pro Gruppe nur eine einzige Block-Box erzeugt.
    """

    img_h, img_w = img_shape[:2]

    # Geschätzte Blockgröße für dein Testbild.
    block_size = int(img_w * 0.055)
    block_size = max(70, min(block_size, 115))

    if len(candidates) == 0:
        return candidates

    small_candidates = []
    normal_candidates = []

    for candidate in candidates:
        x, y, w, h = candidate

        # Kleine Diamond-Farbinseln sammeln
        if max(w, h) < block_size * 0.75:
            small_candidates.append(candidate)
        else:
            normal_candidates.append(candidate)

    clusters = []

    for candidate in small_candidates:
        x, y, w, h = candidate
        cx = x + w / 2
        cy = y + h / 2

        added = False

        for cluster in clusters:
            # Mittelpunkt des vorhandenen Clusters berechnen
            xs = []
            ys = []

            for bx, by, bw, bh in cluster:
                xs.append(bx + bw / 2)
                ys.append(by + bh / 2)

            cluster_cx = sum(xs) / len(xs)
            cluster_cy = sum(ys) / len(ys)

            # Wenn die Farbinseln nahe beieinander sind, gehören sie zum selben Block
            if abs(cx - cluster_cx) < block_size * 0.75 and abs(cy - cluster_cy) < block_size * 0.75:
                cluster.append(candidate)
                added = True
                break

        if not added:
            clusters.append([candidate])

    merged_candidates = []

    for cluster in clusters:
        x1 = min(x for x, y, w, h in cluster)
        y1 = min(y for x, y, w, h in cluster)
        x2 = max(x + w for x, y, w, h in cluster)
        y2 = max(y + h for x, y, w, h in cluster)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        new_w = block_size
        new_h = block_size

        new_x = int(cx - new_w / 2)
        new_y = int(cy - new_h / 2)

        # Box innerhalb des Bildes halten
        new_x = max(0, new_x)
        new_y = max(0, new_y)

        if new_x + new_w > img_w:
            new_w = img_w - new_x

        if new_y + new_h > img_h:
            new_h = img_h - new_y

        merged_candidates.append((new_x, new_y, new_w, new_h))

    return normal_candidates + merged_candidates


def remove_hud_regions(mask: np.ndarray) -> np.ndarray:
    """
    Entfernt typische Minecraft-HUD-Bereiche aus einer Maske.

    Dadurch werden Hotbar, Hand, gehaltene Items, Screenshot-Text und Crosshair
    nicht als Erze erkannt.
    """

    h, w = mask.shape[:2]
    out = mask.copy()

    # Unterer Bereich: Screenshot-Text, Hotbar, Inventarleiste
    out[int(0.82 * h):, :] = 0

    # Rechte untere Ecke: Hand / gehaltenes Item
    out[int(0.55 * h):, int(0.72 * w):] = 0

    # Bildmitte: Crosshair entfernen
    cx = w // 2
    cy = h // 2

    cross_w = max(12, int(0.015 * w))
    cross_h = max(12, int(0.020 * h))

    x0 = max(0, cx - cross_w)
    x1 = min(w, cx + cross_w)
    y0 = max(0, cy - cross_h)
    y1 = min(h, cy + cross_h)

    out[y0:y1, x0:x1] = 0

    return out


def _colored_ore_ratio_for_coal_reject(roi_bgr: np.ndarray) -> float:
    """
    NEU HINZUGEFÜGT:
    Prüft, ob ein Coal-Kandidat eigentlich farbige Erzpixel enthält.

    Grund:
    Dunkler Diamond besteht aus dunklem Stein + cyanfarbenen Erzpixeln.
    Ohne diesen Filter kann Coal den dunklen Diamond als Kohle-Kandidat nehmen
    und später durch Non-Maximum Suppression den echten Diamond verdrängen.
    """

    if roi_bgr.size == 0:
        return 0.0

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    colored_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    colored_ranges = [
        # Diamond / Cyan
        ([75, 25, 18], [112, 255, 255]),

        # Emerald / Grün
        ([45, 25, 18], [85, 255, 255]),

        # Copper / Orange
        ([5, 45, 30], [30, 255, 255]),

        # Gold / Gelb
        ([15, 40, 40], [42, 255, 255]),

        # Lapis / Blau
        ([95, 45, 30], [135, 255, 255]),

        # Redstone / Rot
        ([0, 55, 35], [10, 255, 255]),
        ([165, 55, 35], [179, 255, 255]),
    ]

    for lower, upper in colored_ranges:
        lo = np.array(lower, dtype=np.uint8)
        hi = np.array(upper, dtype=np.uint8)
        colored_mask = cv2.bitwise_or(
            colored_mask,
            cv2.inRange(hsv, lo, hi)
        )

    return cv2.countNonZero(colored_mask) / float(colored_mask.shape[0] * colored_mask.shape[1])


def find_coal_candidates_from_mask(img: np.ndarray, coal_mask: np.ndarray):
    """
    GEÄNDERT:
    Spezielle Kandidatenerzeugung nur für Coal.

    Ziel:
    - Dunkle Coal sauber als ganzen Block erkennen.
    - Helle Coal zusätzlich über lokale dunkle Flecken erkennen.
    - Box nicht mehr zu weit nach oben setzen.
    - Diamond, Emerald, Copper, Iron usw. NICHT beeinflussen.
    """

    img_h, img_w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    block_size = int(img_w * 0.055)
    block_size = max(70, min(block_size, 105))

    candidates = []

    # ---------------------------------------------------------
    # 1. Normale Coal-Maske vorbereiten
    # ---------------------------------------------------------

    strict_seed = remove_hud_regions(coal_mask.copy())

    low_saturation = cv2.inRange(hsv[:, :, 1], 0, 110)
    dark_gray = cv2.inRange(gray, 0, 170)

    strict_seed = cv2.bitwise_and(strict_seed, low_saturation)
    strict_seed = cv2.bitwise_and(strict_seed, dark_gray)

    # ---------------------------------------------------------
    # 2. Helle Coal über lokale dunkle Flecken erkennen
    # ---------------------------------------------------------

    local_mean = cv2.GaussianBlur(gray, (41, 41), 0)

    local_dark = np.zeros_like(gray, dtype=np.uint8)

    local_dark[
        ((local_mean.astype(np.int16) - gray.astype(np.int16)) > 13)
        & (gray < 175)
        & (hsv[:, :, 1] < 110)
    ] = 255

    local_dark = remove_hud_regions(local_dark)

    # Falsche Treffer oben an Decke/Wand verhindern
    strict_seed[:int(0.12 * img_h), :] = 0
    local_dark[:int(0.12 * img_h), :] = 0

    # ---------------------------------------------------------
    # 3. Beide Coal-Hinweise kombinieren
    # ---------------------------------------------------------

    coal_seed = cv2.bitwise_or(strict_seed, local_dark)

    coal_seed = cv2.morphologyEx(
        coal_seed,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8)
    )

    grouped = cv2.dilate(
        coal_seed,
        np.ones((9, 9), np.uint8),
        iterations=2
    )

    grouped = cv2.morphologyEx(
        grouped,
        cv2.MORPH_CLOSE,
        np.ones((11, 11), np.uint8)
    )

    contours, _ = cv2.findContours(
        grouped,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w < 8 or h < 8:
            continue

        if w > block_size * 1.65 or h > block_size * 1.65:
            continue

        if y < int(0.12 * img_h):
            continue

        cx = x + w / 2.0

        # Nicht zu stark aufblasen, damit die Box enger am echten Coal-Block bleibt.
        side = int(max(w, h) * 1.05)
        side = max(int(block_size * 0.82), side)
        side = min(int(block_size * 0.98), side)

        new_x = int(cx - side * 0.50)

        # Box nicht aus der Mitte stark nach oben ziehen.
        # Dadurch sitzt sie bei dunkler Coal nicht mehr zu weit oben.
        new_y = int(y + h * 0.08)

        new_w = side
        new_h = side

        new_x = max(0, new_x)
        new_y = max(0, new_y)

        if new_x + new_w > img_w:
            new_w = img_w - new_x

        if new_y + new_h > img_h:
            new_h = img_h - new_y

        if new_w <= 0 or new_h <= 0:
            continue

        roi_gray = gray[new_y:new_y + new_h, new_x:new_x + new_w]
        roi_hsv = hsv[new_y:new_y + new_h, new_x:new_x + new_w]
        roi_bgr = img[new_y:new_y + new_h, new_x:new_x + new_w]
        roi_seed = coal_seed[new_y:new_y + new_h, new_x:new_x + new_w]
        roi_local_dark = local_dark[new_y:new_y + new_h, new_x:new_x + new_w]

        if roi_gray.size == 0 or roi_hsv.size == 0 or roi_seed.size == 0:
            continue

        coal_support = cv2.countNonZero(roi_seed) / float(new_w * new_h)
        local_dark_support = cv2.countNonZero(roi_local_dark) / float(new_w * new_h)

        very_dark_ratio = float(np.mean(roi_gray < 85))
        dark_ratio = float(np.mean(roi_gray < 145))
        low_sat_ratio = float(np.mean(roi_hsv[:, :, 1] < 120))
        texture_strength = float(np.std(roi_gray))

        colored_ratio = _colored_ore_ratio_for_coal_reject(roi_bgr)

        # Helle Coal wird erlaubt, farbige Erze bleiben blockiert.
        if colored_ratio > 0.080:
            continue

        if coal_support < 0.010 and local_dark_support < 0.018:
            continue

        if low_sat_ratio < 0.70:
            continue

        if very_dark_ratio < 0.006 and dark_ratio < 0.070:
            continue

        if texture_strength < 5.5:
            continue

        candidates.append((new_x, new_y, new_w, new_h))

    return _merge_candidate_boxes(
        candidates,
        iou_threshold=0.35
    )


def detect_coal_direct(img: np.ndarray, candidates):
    """
    GEÄNDERT:
    Direkte Coal-Erkennung ohne Template-Matching.

    Wichtig:
    - Coal bleibt unabhängig von Diamond/Iron/Copper.
    - Die Box aus find_coal_candidates_from_mask() wird direkt benutzt.
    - Kein Template-Matching, damit die Box nicht wieder nach oben verschoben wird.
    """

    detections = []

    img_h, img_w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for x, y, w, h in candidates:

        if y < int(0.12 * img_h):
            continue

        roi_gray = gray[y:y + h, x:x + w]
        roi_bgr = img[y:y + h, x:x + w]
        roi_hsv = hsv[y:y + h, x:x + w]

        if roi_gray.size == 0 or roi_bgr.size == 0 or roi_hsv.size == 0:
            continue

        colored_ratio = _colored_ore_ratio_for_coal_reject(roi_bgr)

        # Wichtig:
        # Nicht wieder auf 0.012 setzen, sonst verschwindet helle Coal wieder.
        if colored_ratio > 0.080:
            continue

        low_sat_ratio = float(np.mean(roi_hsv[:, :, 1] < 120))
        very_dark_ratio = float(np.mean(roi_gray < 85))
        dark_ratio = float(np.mean(roi_gray < 145))
        texture_strength = float(np.std(roi_gray))

        if low_sat_ratio < 0.70:
            continue

        if very_dark_ratio < 0.006 and dark_ratio < 0.070:
            continue

        if texture_strength < 5.5:
            continue

        score = (
            0.42
            + min(0.25, dark_ratio * 1.05)
            + min(0.22, texture_strength / 80.0)
            + min(0.10, very_dark_ratio * 1.8)
        )

        score = max(0.0, min(0.99, score))

        detections.append({
            "label": "Coal",
            "variant": "coal_direct",
            "score": float(score),
            "box": (x, y, w, h),
        })

    return detections


def _merge_candidate_boxes(candidates, iou_threshold=0.35):
    """
    NEU HINZUGEFÜGT:
    Verhindert mehrere Coal-Boxen auf demselben Block.
    """

    if len(candidates) == 0:
        return []

    result = []

    for box in candidates:
        keep = True

        for existing in result:
            if _candidate_iou(box, existing) > iou_threshold:
                keep = False
                break

        if keep:
            result.append(box)

    return result


def _candidate_iou(box_a, box_b):
    """
    NEU HINZUGEFÜGT:
    Berechnet Überlappung zweier Kandidatenboxen.
    """

    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h
    area_a = aw * ah
    area_b = bw * bh

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / float(union)


def remove_water_regions(mask: np.ndarray, hsv: np.ndarray) -> np.ndarray:
    """
    Entfernt große Wasserflächen aus einer Maske.

    Grund:
    Wasser kann farblich ähnlich wie Diamond oder Lapis wirken.
    Kleine blaue Erzpixel sollen bleiben, aber große zusammenhängende blaue
    Flächen wie Wasser sollen entfernt werden.
    """

    h, w = mask.shape[:2]
    out = mask.copy()

    # Grober HSV-Bereich für Minecraft-Wasser / bläuliche transparente Flächen
    lower_water = np.array([85, 25, 20], dtype=np.uint8)
    upper_water = np.array([125, 255, 255], dtype=np.uint8)

    water_mask = cv2.inRange(hsv, lower_water, upper_water)

    # Wasserbereiche leicht vergrößern, damit auch Wasser-Kanten entfernt werden.
    kernel = np.ones((7, 7), np.uint8)
    water_mask = cv2.dilate(water_mask, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        water_mask,
        connectivity=8
    )

    image_area = h * w

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])

        # Nur große Wasserflächen entfernen.
        # Kleine blaue Erzbereiche bleiben erhalten.
        if area > int(image_area * 0.0035):
            out[labels == i] = 0

    return out


def remove_large_mask_regions(mask: np.ndarray) -> np.ndarray:
    """
    NEU HINZUGEFÜGT:
    Entfernt extrem große zusammenhängende Maskenbereiche.

    Grund:
    Große zusammenhängende Regionen entstehen häufig durch Wasser, Schatten,
    Holzflächen oder große Wandbereiche. Solche Regionen sind meistens keine
    einzelnen Erzblöcke und führen später zu falschen Bounding Boxes.
    """

    h, w = mask.shape[:2]
    out = mask.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        out,
        connectivity=8
    )

    image_area = h * w

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])

        # Sehr große Regionen entfernen
        if area > int(image_area * 0.020):
            out[labels == i] = 0
            continue

        # Sehr breite oder sehr hohe Regionen entfernen
        if bw > int(0.25 * w) or bh > int(0.25 * h):
            out[labels == i] = 0

    return out


def save_debug_mask(name: str, mask: np.ndarray) -> None:
    """
    NEU HINZUGEFÜGT:
    Speichert Debug-Masken, falls SAVE_DEBUG_MASKS aktiviert ist.
    """

    if not SAVE_DEBUG_MASKS:
        return

    os.makedirs(DEBUG_MASK_DIR, exist_ok=True)

    path = os.path.join(DEBUG_MASK_DIR, f"{name}.png")
    cv2.imwrite(path, mask)

def save_vorverarbeitung(name: str, mask: np.ndarray) -> None:
    """
    NEU HINZUGEFÜGT:
    Speichert Debug-Masken, falls SAVE_DEBUG_MASKS aktiviert ist.
    """

    if not SAVE_DEBUG_MASKS:
        return

    os.makedirs(Vorverarbeitung_DIR, exist_ok=True)

    path = os.path.join(Vorverarbeitung_DIR, f"{name}.png")
    cv2.imwrite(path, mask)


def _ore_label(ore_key: str) -> str:
    """
    Konvertiert die Erz-ID in ein formatiertes Label für die Anzeige.
    """

    return ore_key.capitalize()


def _load_template_bank_for_ore(ore_key: str) -> Dict[str, np.ndarray]:
    """
    Lädt alle Template-Varianten für einen bestimmten Erztyp aus dem Template-Verzeichnis.
    Extrahiert automatisch den relevanten Erzblock aus den Template-Screenshots.
    """

    bank = {}

    if not os.path.isdir(TEMPLATES_DIR):
        log(f"Template-Ordner nicht gefunden: {TEMPLATES_DIR}")
        return bank

    for name in os.listdir(TEMPLATES_DIR):

        # Nur PNG-Dateien betrachten
        if not name.lower().endswith(".png"):
            continue

        # Nur Dateien betrachten, die mit der Erz-ID beginnen.
        if not name.startswith(f"{ore_key}_"):
            continue

        path = os.path.join(TEMPLATES_DIR, name)
        variant = name[:-4]  # ".png" entfernen

        try:
            bank[variant] = load_template(path)
        except Exception as exc:
            log(f"Template konnte nicht geladen werden: {path} ({exc})")

    return bank


def run_pipeline(img: np.ndarray) -> np.ndarray:
    """
    Führt die komplette Ore-Detection auf einem bereits geladenen Bild aus.
    """

    # ---------------------------------------------------------
    # 1. Vorverarbeitung
    # ---------------------------------------------------------

    img_preprocessed = match_scene_brightness(img)

    # ---------------------------------------------------------
    # 2. Farbraumkonvertierung und Kantenerkennung
    # ---------------------------------------------------------

    hsv = to_hsv(img_preprocessed)
    save_vorverarbeitung("vorverarbeitung", hsv)

    edges = edge_mask(img_preprocessed)

    edges = remove_hud_regions(edges)
    edges = remove_water_regions(edges, hsv)
    edges = remove_large_mask_regions(edges)

    save_debug_mask("00_edges_cleaned", edges)

    # ---------------------------------------------------------
    # 3. Listen für Zwischenergebnisse
    # ---------------------------------------------------------

    all_raw_detections = []
    all_candidates = []

    # ---------------------------------------------------------
    # 4. Alle unterstützten Erztypen durchlaufen
    # ---------------------------------------------------------

    for ore in supported_ores():

        # -----------------------------------------------------
        # 4.1 Farbmaske für den aktuellen Erztyp erzeugen
        # -----------------------------------------------------

        color = color_mask(hsv, ore)

        color = remove_hud_regions(color)
        color = remove_water_regions(color, hsv)
        color = remove_large_mask_regions(color)

        save_debug_mask(f"01_color_{ore}", color)

        # -----------------------------------------------------
        # 4.2 Hybridmaske erzeugen
        # -----------------------------------------------------

        if use_edges_for_ore(ore):
            mask = hybrid_mask(color, edges)
        else:
            mask = color

        # -----------------------------------------------------
        # 4.3 Erz-spezifische Maskenverfeinerung
        # -----------------------------------------------------

        mask = refine_mask_for_ore(ore, mask)

        # -----------------------------------------------------
        # 4.4 Morphologische Bereinigung
        # -----------------------------------------------------

        # mask = clean_mask(mask)

        mask = remove_hud_regions(mask)
        mask = remove_water_regions(mask, hsv)
        mask = remove_large_mask_regions(mask)

        save_debug_mask(f"02_mask_{ore}", mask)

        # -----------------------------------------------------
        # 4.5 Kandidaten aus der bereinigten Maske finden
        # -----------------------------------------------------

        # GEÄNDERT:
        # Coal wird komplett separat behandelt.
        # Wichtig:
        # Coal läuft NICHT durch detect_with_template_bank,
        # damit die Box nicht wieder verschoben wird.
        if ore == "coal":
            candidates = find_coal_candidates_from_mask(img, color)

            all_candidates.extend(candidates)

            if len(candidates) == 0:
                continue

            raw = detect_coal_direct(img, candidates)
            all_raw_detections.extend(raw)

            continue

        candidates = find_candidates(mask, color)

        # Nur Diamond bekommt die Spezial-Erweiterung.
        if ore == "diamond":
            candidates = expand_diamond_candidates(candidates, img.shape)

        all_candidates.extend(candidates)

        if len(candidates) == 0:
            continue

        # -----------------------------------------------------
        # 4.6 Template-Bank für aktuellen Erztyp laden
        # -----------------------------------------------------

        template_bank = _load_template_bank_for_ore(ore)

        if len(template_bank) == 0:
            continue

        # -----------------------------------------------------
        # 4.7 Template Matching auf den Kandidaten ausführen
        # -----------------------------------------------------

        # GEÄNDERT:
        # Stone- und Deepslate-Templates werden nicht anhand der mittleren Helligkeit
        # vorgefiltert, weil die weiße Wand im Testraum mean_gray verfälscht.
        raw = detect_with_template_bank(
            img,
            candidates,
            template_bank,
            label=_ore_label(ore),
            threshold=ORE_MATCH_THRESHOLD.get(ore, 0.8),
            brightness_split=None
        )

        all_raw_detections.extend(raw)

    # ---------------------------------------------------------
    # 5. Non-Maximum Suppression
    # ---------------------------------------------------------

    detections = non_max_suppression(
        all_raw_detections,
        iou_threshold=0.25
    )

    # ---------------------------------------------------------
    # 6. Visualisierung
    # ---------------------------------------------------------

    if DEBUG:
        return draw_debug(
            img,
            all_candidates,
            detections
        )

    return draw(
        img,
        detections
    )


def main():
    """
    Startpunkt des Programms.

    Lädt das Testbild, führt die komplette Pipeline aus
    und zeigt das Ergebnis in einem OpenCV-Fenster an.
    """

    log("Minecraft Ore Detector CV gestartet")
    log(f"Lade Bild aus '{IMAGE_PATH}'...")

    img = load_image(IMAGE_PATH)

    log("Bild erfolgreich geladen.")

    output = run_pipeline(img)

    show(output)

    log("Programm beendet.")


# WICHTIG:
# Diese Zeilen dürfen genau EINMAL vorkommen und müssen ganz unten in der Datei stehen.
if __name__ == "__main__":
    main()