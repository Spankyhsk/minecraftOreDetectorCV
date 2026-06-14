# -*- coding: utf-8 -*-
"""
Modul für die Objekterkennung (Detection) von Minecraft-Erzen.
Dieses Modul enthält die Kernlogik der klassischen Bildverarbeitungspipeline:
- Extraktion und Zuschnitt von Erz-Templates aus Vorlage-Screenshots
- Erkennung von Erzkandidaten durch Konturanalyse (mit Flächen- und Dichtefilterung)
- Zusammenführung benachbarter Detektionen (Bounding-Box-Merging)
- Multi-Scale-Template-Matching auf Graustufen- und Kantenbildern
- Plausibilitätsprüfung durch Farbanalysen im HSV-Farbraum
- Non-Maximum Suppression (NMS) zur Reduzierung von Doppel-Detektionen
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

from ore_rules import get_ore_rule
from utils import log_debug


# NEU HINZUGEFÜGT:
# Diese Erztypen sind in echten Höhlenbildern besonders fehleranfällig.
# - Coal ähnelt Schatten.
# - Copper und Iron ähneln warm beleuchtetem Stein, Holz und Erde.
WEAK_ORES = {"coal", "copper", "iron"}


def _template_family(template_name: Optional[str]) -> Optional[str]:
    """
    Extrahiert den Erz-Basisnamen aus einem Template-Namen.

    Beispiel:
    'diamond_deepslate_ore' -> 'diamond'

    Parameters
    ----------
    template_name : Optional[str]
        Name des Templates.

    Returns
    -------
    Optional[str]
        Erz-ID oder None.
    """

    if template_name is None:
        return None

    return template_name.split("_", 1)[0]


def _color_support_mask(ore: str, roi_bgr: np.ndarray) -> np.ndarray:
    """
    NEU HINZUGEFÜGT:
    Erstellt eine Binärmaske für alle Pixel in einer ROI, die farblich
    zum angegebenen Erz passen.

    Wichtig:
    Ein Minecraft-Erzblock besteht größtenteils aus Stone/Deepslate und nur
    aus einigen farbigen Erzpixeln. Deshalb ist die Durchschnittsfarbe der
    ganzen ROI oft nicht aussagekräftig.

    Parameters
    ----------
    ore : str
        Erz-ID.
    roi_bgr : np.ndarray
        Region of Interest im BGR-Farbraum.

    Returns
    -------
    np.ndarray
        Binärmaske mit 255 für passende Pixel, sonst 0.
    """

    if roi_bgr.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    ore = ore.lower()
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    out = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in get_ore_rule(ore).plausibility_ranges:
        lo = np.array(lower, dtype=np.uint8)
        hi = np.array(upper, dtype=np.uint8)

        part = cv2.inRange(hsv, lo, hi)
        out = cv2.bitwise_or(out, part)

    return out


def _color_support_ratio(ore: str, roi_bgr: np.ndarray) -> float:
    """
    NEU HINZUGEFÜGT:
    Berechnet den Anteil farblich passender Pixel in einer ROI.

    Beispiel:
    1000 Pixel ROI, davon 40 passende Erzpixel -> support = 0.04

    Parameters
    ----------
    ore : str
        Erz-ID.
    roi_bgr : np.ndarray
        Region of Interest im BGR-Farbraum.

    Returns
    -------
    float
        Anteil passender Pixel zwischen 0.0 und 1.0.
    """

    if roi_bgr.size == 0:
        return 0.0

    mask = _color_support_mask(ore, roi_bgr)

    if mask.size == 0:
        return 0.0

    return float((mask > 0).sum()) / float(mask.size)



# NEU HINZUGEFÜGT:
# Minimaler Schutz gegen die frühere Verwechslung Copper -> Emerald/Diamond.
# Wichtig: Das ist nur eine Plausibilitätsprüfung in detection.py.
# Die eigentliche Segmentierung bleibt weiterhin in segmentation.py.
def _range_support_ratio(
    roi_bgr: np.ndarray,
    ranges: List[Tuple[List[int], List[int]]]
) -> float:
    """
    Berechnet den Anteil der Pixel, die in frei definierte HSV-Bereiche fallen.
    """

    if roi_bgr.size == 0:
        return 0.0

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    out = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in ranges:
        lo = np.array(lower, dtype=np.uint8)
        hi = np.array(upper, dtype=np.uint8)
        out = cv2.bitwise_or(out, cv2.inRange(hsv, lo, hi))

    return float((out > 0).sum()) / float(out.size)


# NEU HINZUGEFÜGT:
# Copper hat einen orange-braunen Anteil. Wenn dieser deutlich vorhanden ist,
# darf derselbe Kandidat nicht leichtfertig als Diamond oder Emerald akzeptiert werden.
def _copper_orange_support(roi_bgr: np.ndarray) -> float:
    """
    Misst den orange-braunen Copper-Anteil in einer ROI.
    """

    return _range_support_ratio(
        roi_bgr,
        [
            ([5, 45, 30], [30, 255, 255])
        ]
    )



# NEU HINZUGEFÜGT:
# Misst den grün-türkisen Anteil von Copper.
# Wichtig: Dieser Wert wird nur zusammen mit Orange benutzt.
# Dadurch werden Diamond/Emerald nicht einfach zu Copper, weil dort der orange Anteil fehlt.
def _copper_green_support(roi_bgr: np.ndarray) -> float:
    """
    Misst den grün-türkisen Copper-Anteil in einer ROI.
    """

    return _range_support_ratio(
        roi_bgr,
        [
            ([70, 25, 25], [98, 255, 255])
        ]
    )


def _min_color_support(ore: str) -> float:
    """
    Gibt den minimal notwendigen Anteil farblich passender Pixel zurück.

    GEÄNDERT:
    Copper und Iron sind bewusst strenger, weil sie in Höhlen sehr leicht
    mit warmem Stein, Holz oder Erde verwechselt werden.

    Parameters
    ----------
    ore : str
        Erz-ID.

    Returns
    -------
    float
        Mindest-Farbsupport.
    """

    return get_ore_rule(ore).min_color_support


def _good_color_support(ore: str) -> float:
    """
    NEU HINZUGEFÜGT:
    Gibt an, ab welchem Farbsupport ein Kandidat farblich stark wirkt.

    Dieser Wert wird für Sonderfälle verwendet, in denen der Template-Score
    etwas niedriger sein darf, wenn die Farbe sehr eindeutig ist.
    """

    return get_ore_rule(ore).good_color_support


def _min_compatibility(ore: str) -> float:
    """
    Gibt die minimale Farbkompatibilität zurück.

    GEÄNDERT:
    Copper und Iron sind strenger als Diamond/Lapis/Redstone, weil sie
    in echten Höhlenbildern deutlich mehr False Positives erzeugen.

    Parameters
    ----------
    ore : str
        Erz-ID.

    Returns
    -------
    float
        Minimale Kompatibilität.
    """

    return get_ore_rule(ore).min_compatibility


def _color_compatibility(ore: str, roi_bgr: np.ndarray) -> float:
    """
    Berechnet einen Farb-Plausibilitätswert zwischen 0.0 und 1.0.

    Bewertet werden:
    - Wie viele passende Erzfarbpixel existieren?
    - Wie gesättigt sind diese Pixel?
    - Wie hell sind diese Pixel?

    GEÄNDERT:
    Es wird nicht nur die Durchschnittsfarbe betrachtet, sondern gezielt
    die farblich passenden Pixel innerhalb der ROI.

    Parameters
    ----------
    ore : str
        Erz-ID.
    roi_bgr : np.ndarray
        Region of Interest im BGR-Format.

    Returns
    -------
    float
        Farbkompatibilität zwischen 0.0 und 1.0.
    """

    if roi_bgr.size == 0:
        return 0.0

    ore = ore.lower()

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = _color_support_mask(ore, roi_bgr)
    support = _color_support_ratio(ore, roi_bgr)

    # Spezialfall Kohle:
    # Kohle ist sehr dunkel und deshalb kaum von Schatten unterscheidbar.
    if ore == "coal":
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        dark_ratio = float((gray < 85).sum()) / float(gray.size)

        support_score = min(
            1.0,
            support / max(_min_color_support(ore), 0.0001)
        )

        dark_score = min(
            1.0,
            dark_ratio / 0.35
        )

        return min(
            1.0,
            0.55 * support_score + 0.45 * dark_score
        )

    if mask.size == 0 or (mask > 0).sum() == 0:
        return 0.0

    selected = hsv[mask > 0]

    s_mean = float(np.mean(selected[:, 1]))
    v_mean = float(np.mean(selected[:, 2]))

    min_support = _min_color_support(ore)

    density_score = min(
        1.0,
        support / max(min_support * 3.0, 0.0001)
    )

    sat_score = max(
        0.0,
        min(1.0, (s_mean - 35.0) / 100.0)
    )

    val_score = max(
        0.0,
        min(1.0, (v_mean - 30.0) / 140.0)
    )

    return min(
        1.0,
        0.55 * density_score + 0.30 * sat_score + 0.15 * val_score
    )


def _best_center_component_bbox(
    mask: np.ndarray,
    min_area: int = 120
) -> Optional[Tuple[int, int, int, int]]:
    """
    Sucht die am besten zentrierte zusammenhängende Komponente in einer Maske.

    Diese Funktion wird beim Laden der Templates benutzt, um aus einem Template-
    Screenshot möglichst den eigentlichen Erzblock herauszuschneiden.

    Parameters
    ----------
    mask : np.ndarray
        Binärmaske.
    min_area : int
        Mindestfläche.

    Returns
    -------
    Optional[Tuple[int, int, int, int]]
        Bounding Box (x, y, w, h) oder None.
    """

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    h, w = mask.shape[:2]
    img_area = h * w

    cx_img = w / 2.0
    cy_img = h / 2.0

    best_idx = -1
    best_score = float("inf")

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area < min_area:
            continue

        if area > int(0.25 * img_area):
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])

        if bw < 8 or bh < 8:
            continue

        cx, cy = centroids[i]

        dist = ((cx - cx_img) / w) ** 2 + ((cy - cy_img) / h) ** 2
        ratio = max(bw / float(bh), bh / float(bw))

        score = dist + 0.08 * (ratio - 1.0)

        if score < best_score:
            best_score = score
            best_idx = i

    if best_idx < 0:
        return None

    x = int(stats[best_idx, cv2.CC_STAT_LEFT])
    y = int(stats[best_idx, cv2.CC_STAT_TOP])
    bw = int(stats[best_idx, cv2.CC_STAT_WIDTH])
    bh = int(stats[best_idx, cv2.CC_STAT_HEIGHT])

    return x, y, bw, bh


def _merge_nearby_boxes(
    boxes: List[Tuple[int, int, int, int]],
    gap: int = 12
) -> List[Tuple[int, int, int, int]]:
    """
    Führt nahe oder überlappende Bounding Boxes zusammen.

    Das ist wichtig, weil ein einzelner Erzblock oft aus mehreren kleinen farbigen
    Bereichen besteht, die sonst als einzelne Kandidaten behandelt würden.

    Parameters
    ----------
    boxes : List[Tuple[int, int, int, int]]
        Liste von Boxen im Format (x, y, w, h).
    gap : int
        Maximale Lücke zwischen Boxen, damit sie zusammengeführt werden.

    Returns
    -------
    List[Tuple[int, int, int, int]]
        Zusammengeführte Boxen.
    """

    if not boxes:
        return []

    work = [
        [x, y, x + w, y + h]
        for (x, y, w, h) in boxes
    ]

    changed = True

    while changed:
        changed = False
        new_boxes = []
        used = [False] * len(work)

        for i in range(len(work)):
            if used[i]:
                continue

            x1, y1, x2, y2 = work[i]
            used[i] = True

            merged_this_round = True

            while merged_this_round:
                merged_this_round = False

                for j in range(len(work)):
                    if used[j]:
                        continue

                    a1, b1, a2, b2 = work[j]

                    overlap_or_close = not (
                        x2 + gap < a1
                        or a2 + gap < x1
                        or y2 + gap < b1
                        or b2 + gap < y1
                    )

                    if overlap_or_close:
                        x1 = min(x1, a1)
                        y1 = min(y1, b1)
                        x2 = max(x2, a2)
                        y2 = max(y2, b2)

                        used[j] = True
                        merged_this_round = True
                        changed = True

            new_boxes.append([x1, y1, x2, y2])

        work = new_boxes

    return [
        (x1, y1, x2 - x1, y2 - y1)
        for (x1, y1, x2, y2) in work
    ]


def _expand_box(
    box: Tuple[int, int, int, int],
    img_shape: Tuple[int, ...],
    pad_factor: float = 0.40,
    min_pad: int = 8
) -> Tuple[int, int, int, int]:
    """
    Erweitert eine Bounding Box für das Template Matching.

    Wichtig:
    Diese erweiterte Box wird nur intern für das Matching benutzt.
    Für die Ausgabe wird weiterhin die ursprüngliche Kandidatenbox verwendet.
    Dadurch entstehen keine riesigen grünen Boxen.

    Parameters
    ----------
    box : Tuple[int, int, int, int]
        Ursprüngliche Box.
    img_shape : Tuple[int, ...]
        Bildform.
    pad_factor : float
        Erweiterungsfaktor.
    min_pad : int
        Mindest-Padding.

    Returns
    -------
    Tuple[int, int, int, int]
        Erweiterte Box.
    """

    x, y, w, h = box

    pad = max(
        min_pad,
        int(max(w, h) * pad_factor)
    )

    img_h, img_w = img_shape[:2]

    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img_w, x + w + pad)
    y1 = min(img_h, y + h + pad)

    return x0, y0, x1 - x0, y1 - y0


def load_template(path: str) -> np.ndarray:
    """
    Lädt ein Template-Bild und extrahiert daraus möglichst den zentralen Erzblock.

    Parameters
    ----------
    path : str
        Pfad zum Template-Bild.

    Returns
    -------
    np.ndarray
        Zugeschnittenes Graustufen-Template.

    Raises
    ------
    FileNotFoundError
        Falls das Template nicht geladen werden kann.
    """

    tpl_bgr = cv2.imread(path, cv2.IMREAD_COLOR)

    if tpl_bgr is None:
        raise FileNotFoundError(
            f"Template unter '{path}' konnte nicht geladen werden."
        )

    tpl_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2LAB)

    h, w = tpl_gray.shape[:2]

    strip = max(
        8,
        min(h, w) // 40
    )

    border = np.concatenate([
        lab[:strip, :, :].reshape(-1, 3),
        lab[-strip:, :, :].reshape(-1, 3),
        lab[:, :strip, :].reshape(-1, 3),
        lab[:, -strip:, :].reshape(-1, 3),
    ], axis=0)

    bg = np.median(border, axis=0)

    dist = np.linalg.norm(
        lab.astype(np.float32) - bg.astype(np.float32),
        axis=2
    )

    sat = hsv[:, :, 1]

    fg = ((dist > 14.0) | (sat > 36)).astype(np.uint8) * 255

    # Unteren HUD-Bereich im Template ignorieren
    fg[int(0.86 * h):, :] = 0

    kernel = np.ones((5, 5), np.uint8)

    fg = cv2.morphologyEx(
        fg,
        cv2.MORPH_OPEN,
        kernel
    )

    fg = cv2.morphologyEx(
        fg,
        cv2.MORPH_CLOSE,
        kernel
    )

    bbox = _best_center_component_bbox(
        fg,
        min_area=max(120, int(h * w * 0.00008))
    )

    if bbox is not None:
        x, y, bw, bh = bbox

        if bw < int(0.55 * w) and bh < int(0.55 * h):
            pad = 10

            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(w, x + bw + pad)
            y1 = min(h, y + bh + pad)

            cropped = tpl_gray[y0:y1, x0:x1]

            if (
                cropped.size > 0
                and cropped.shape[0] >= 8
                and cropped.shape[1] >= 8
            ):
                return cropped

    # Fallback:
    # Grober Zuschnitt über nicht-weiße Pixel.
    mask = (tpl_gray < 245)
    mask[int(0.86 * h):, :] = False

    if mask.any():
        ys, xs = np.nonzero(mask)

        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())

        cropped = tpl_gray[y0:y1 + 1, x0:x1 + 1]

        if (
            cropped.size > 0
            and cropped.shape[0] >= 8
            and cropped.shape[1] >= 8
            and cropped.shape[0] < int(0.65 * h)
            and cropped.shape[1] < int(0.65 * w)
        ):
            return cropped

    # Letzter Fallback:
    # Zentraler Bildausschnitt.
    side = int(min(h, w) * 0.18)
    side = max(96, min(side, 320))

    cx, cy = w // 2, h // 2

    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(w, x0 + side)
    y1 = min(h, y0 + side)

    center_crop = tpl_gray[y0:y1, x0:x1]

    if (
        center_crop.size > 0
        and center_crop.shape[0] >= 8
        and center_crop.shape[1] >= 8
    ):
        return center_crop

    return tpl_gray


def find_candidates(
    mask: np.ndarray,
    color_mask: Optional[np.ndarray] = None
) -> List[Tuple[int, int, int, int]]:
    """
    Sucht Kandidatenregionen in einer binären Maske.

    GEÄNDERT:
    Die Filterung ist stabil, aber nicht zu aggressiv.
    Sehr große Wandflächen, extrem lange dünne Linien und Mini-Rauschen
    werden entfernt.

    Parameters
    ----------
    mask : np.ndarray
        Binärmaske.
    color_mask : Optional[np.ndarray]
        Reine Farbmaske des jeweiligen Erzes.

    Returns
    -------
    List[Tuple[int, int, int, int]]
        Kandidatenboxen im Format (x, y, w, h).
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    img_h, img_w = mask.shape[:2]
    img_area = img_h * img_w

    min_area = max(
        30,
        int(img_area * 0.000035)
    )

    max_area = int(img_area * 0.012)

    max_w = int(img_w * 0.18)
    max_h = int(img_h * 0.18)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        area = w * h

        if area < min_area:
            continue

        if area > max_area:
            continue

        if w > max_w or h > max_h:
            continue

        ratio = max(w / float(h), h / float(w))

        if ratio > 4.8:
            continue

        if color_mask is not None:
            crop = color_mask[y:y + h, x:x + w]

            if crop.size == 0:
                continue

            white_pixels = int((crop > 0).sum())
            density = white_pixels / float(crop.size)

            if density < 0.015 and white_pixels < 18:
                continue

        candidates.append((x, y, w, h))

    merged = _merge_nearby_boxes(
        candidates,
        gap=12
    )

    final = []

    for (x, y, w, h) in merged:
        area = w * h

        if area < max(55, int(img_area * 0.000045)):
            continue

        if area > max_area:
            continue

        if w > max_w or h > max_h:
            continue

        ratio = max(w / float(h), h / float(w))

        if ratio > 5.0:
            continue

        if color_mask is not None:
            crop = color_mask[y:y + h, x:x + w]

            if crop.size == 0:
                continue

            white_pixels = int((crop > 0).sum())
            density = white_pixels / float(crop.size)

            if density < 0.010 and white_pixels < 14:
                continue

        final.append((x, y, w, h))

    return final


def match_template(
    roi: np.ndarray,
    template: np.ndarray
) -> float:
    """
    Führt einfaches Template Matching auf einer ROI durch.

    Parameters
    ----------
    roi : np.ndarray
        Bildausschnitt im BGR-Format.
    template : np.ndarray
        Template im Graustufenformat.

    Returns
    -------
    float
        Matching-Score.
    """

    if roi.size == 0:
        return 0.0

    roi_gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    th, tw = template.shape

    if roi_gray.shape[0] < th or roi_gray.shape[1] < tw:
        return 0.0

    result = cv2.matchTemplate(
        roi_gray,
        template,
        cv2.TM_CCOEFF_NORMED
    )

    return float(np.max(result))


def match_template_multiscale(
    roi: np.ndarray,
    template: np.ndarray,
    scales: Optional[List[float]] = None
) -> float:
    """
    Führt Multi-Scale-Template-Matching auf einer ROI durch.

    WICHTIG:
    Diese stabile Version benutzt wieder nur proportionale Skalierung.
    Die vorherigen nicht-proportionalen Varianten wurden entfernt, weil sie
    zu viele falsche Treffer auf Wasser, Holz, Hand und Schatten erzeugt haben.

    Parameters
    ----------
    roi : np.ndarray
        Bildausschnitt im BGR-Format.
    template : np.ndarray
        Template im Graustufenformat.
    scales : Optional[List[float]]
        Skalierungsfaktoren.

    Returns
    -------
    float
        Bester Matching-Score.
    """

    if roi.size == 0:
        return 0.0

    if scales is None:
        scales = [0.18, 0.25, 0.35, 0.50, 0.70, 0.90, 1.10, 1.30]

    roi_gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    if roi_gray.size == 0:
        return 0.0

    # Sehr gleichförmige Bereiche sind meist Wand/Schatten/Wasser.
    if float(np.std(roi_gray)) < 3.0:
        return 0.0

    roi_edges = cv2.Canny(
        roi_gray,
        50,
        150
    )

    best_score = 0.0

    for scale in scales:
        resized_template = cv2.resize(
            template,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

        th, tw = resized_template.shape

        if th < 10 or tw < 10:
            continue

        if roi_gray.shape[0] < th or roi_gray.shape[1] < tw:
            continue

        result_gray = cv2.matchTemplate(
            roi_gray,
            resized_template,
            cv2.TM_CCOEFF_NORMED
        )

        score_gray = float(np.max(result_gray))

        tpl_edges = cv2.Canny(
            resized_template,
            50,
            150
        )

        score_edges = 0.0

        if (
            tpl_edges.sum() > 0
            and roi_edges.shape[0] >= tpl_edges.shape[0]
            and roi_edges.shape[1] >= tpl_edges.shape[1]
        ):
            try:
                result_edges = cv2.matchTemplate(
                    roi_edges,
                    tpl_edges,
                    cv2.TM_CCOEFF_NORMED
                )

                score_edges = float(np.max(result_edges))

            except cv2.error:
                score_edges = 0.0

        # GEÄNDERT:
        # Graustufen-Matching dominiert. Kanten unterstützen nur leicht.
        score = 0.90 * score_gray + 0.10 * score_edges

        if score > best_score:
            best_score = score

    return best_score


def _decision_score(
    ore: str,
    best_score: float,
    compat: float,
    support: float
) -> float:
    """
    NEU HINZUGEFÜGT:
    Kombiniert Template-Score, Farbkompatibilität und Farbsupport.

    Diese Bewertung wird für Anzeige und Sortierung verwendet.
    Die eigentliche Annahmeentscheidung passiert zusätzlich über harte Regeln.

    Parameters
    ----------
    ore : str
        Erz-ID.
    best_score : float
        Template-Score.
    compat : float
        Farbkompatibilität.
    support : float
        Farbsupport.

    Returns
    -------
    float
        Kombinierter Score zwischen 0.0 und 1.0.
    """

    ore = ore.lower()

    support_score = min(
        1.0,
        support / max(_good_color_support(ore), 0.0001)
    )

    score = (
        0.70 * best_score
        + 0.20 * compat
        + 0.10 * support_score
    )

    # GEÄNDERT:
    # Problematische Klassen leicht abwerten.
    if ore == "copper":
        # GEÄNDERT:
        # Copper wird nicht mehr stark abgewertet, damit Copper gegen
        # fälschliche Diamond/Emerald-Treffer gewinnen kann.
        score -= 0.02

    if ore == "iron":
        # GEÄNDERT:
        # Iron nur noch leicht abwerten; die Box wird später sauber vergrößert.
        score -= 0.02

    if ore == "coal":
        score -= 0.10

    return float(max(0.0, min(1.0, score)))


def _final_output_box(
    ore: str,
    box: Tuple[int, int, int, int],
    img_shape: Tuple[int, ...]
) -> Tuple[int, int, int, int]:
    """
    NEU HINZUGEFÜGT:
    Korrigiert nur die sichtbare Ausgabe-Box.

    Hintergrund:
    Bei Iron besteht die Farbmaske oft nur aus einem hellen, horizontalen
    Erzstreifen. Die Erkennung ist richtig, aber die gezeichnete Box wäre
    dadurch zu klein und sitzt nur oben auf dem Block. Für Iron wird die Box
    deshalb zu einer blockähnlichen, quadratischen Box erweitert.

    Wichtig:
    Diese Funktion verändert nicht das Template Matching selbst, sondern nur
    die finale gezeichnete Bounding Box.
    """

    x, y, w, h = box
    img_h, img_w = img_shape[:2]
    ore = ore.lower()

    if ore == "iron":
        ratio = max(w / float(max(h, 1)), h / float(max(w, 1)))

        # Iron wird häufig nur als flacher Streifen erkannt.
        if ratio >= 1.55:
            side = int(max(w * 1.15, h * 2.65, 42))
            side = min(side, int(min(img_w, img_h) * 0.13))

            cx = x + w / 2.0

            # Wenn die Kandidatenbox ein oberer Streifen ist, muss die
            # finale Box nach unten erweitert werden.
            if w >= h:
                y0 = int(y - 0.16 * side)
            else:
                y0 = int(y + h / 2.0 - side / 2.0)

            x0 = int(cx - side / 2.0)

            x0 = max(0, min(x0, img_w - side))
            y0 = max(0, min(y0, img_h - side))

            return x0, y0, side, side

    return x, y, w, h


def detect_with_template_bank(
    img: np.ndarray,
    candidates: List[Tuple[int, int, int, int]],
    template_bank: Dict[str, np.ndarray],
    label: str = "Diamond",
    threshold: float = 0.72,
    brightness_split: Optional[float] = 95.0,
) -> List[Dict]:
    """
    Vergleicht Kandidaten mit einer Template-Bank.

    GEÄNDERT:
    Template Matching entscheidet nicht mehr alleine.
    Ein Treffer wird nur akzeptiert, wenn Template-Score, Farbkompatibilität
    und Farbsupport gemeinsam plausibel sind.

    WICHTIG:
    Für das Matching wird intern eine erweiterte ROI benutzt.
    Für die Ausgabe wird aber die ursprüngliche Kandidatenbox verwendet.
    Dadurch entstehen keine riesigen falschen Boxen.

    Parameters
    ----------
    img : np.ndarray
        Eingabebild.
    candidates : List[Tuple[int, int, int, int]]
        Kandidatenboxen.
    template_bank : Dict[str, np.ndarray]
        Template-Bank für ein Erz.
    label : str
        Anzeigename des Erzes.
    threshold : float
        Template-Schwellenwert.
    brightness_split : Optional[float]
        Helligkeitsgrenze für Stone-/Deepslate-Templates.

    Returns
    -------
    List[Dict]
        Validierte Detektionen.
    """

    detections = []
    label_key = label.lower()

    for idx, (x, y, w, h) in enumerate(candidates):

        # -----------------------------------------------------
        # ROI für Template Matching erweitern
        # -----------------------------------------------------

        if label_key == "coal":
            roi_box = _expand_box(
                (x, y, w, h),
                img.shape,
                pad_factor=0.60,
                min_pad=10
            )
        else:
            roi_box = _expand_box(
                (x, y, w, h),
                img.shape,
                pad_factor=0.35,
                min_pad=8
            )

        rx, ry, rw, rh = roi_box
        roi = img[ry:ry + rh, rx:rx + rw]

        if roi.size == 0:
            continue

        roi_gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        mean_gray = float(roi_gray.mean()) if roi_gray.size > 0 else 0.0

        template_items = list(template_bank.items())

        # -----------------------------------------------------
        # Stone-/Deepslate-Auswahl über Helligkeit
        # -----------------------------------------------------

        if brightness_split is not None and len(template_items) >= 2:
            if mean_gray >= brightness_split:
                filtered = [
                    (n, t)
                    for (n, t) in template_items
                    if "deepslate" not in n
                ]

                if filtered:
                    template_items = filtered
            else:
                filtered = [
                    (n, t)
                    for (n, t) in template_items
                    if "deepslate" in n
                ]

                if filtered:
                    template_items = filtered

        # -----------------------------------------------------
        # Bestes Template suchen
        # -----------------------------------------------------

        best_score = 0.0
        best_name = None

        for name, tpl in template_items:
            score = match_template_multiscale(
                roi,
                tpl
            )

            if score > best_score:
                best_score = score
                best_name = name

        family = _template_family(best_name) if best_name else label_key

        compat_template_family = _color_compatibility(
            family,
            roi
        )

        compat_label = _color_compatibility(
            label_key,
            roi
        )

        compat_main = max(
            compat_template_family,
            compat_label
        )

        color_support = _color_support_ratio(
            label_key,
            roi
        )

        min_support = _min_color_support(label_key)

        final_score = _decision_score(
            label_key,
            best_score,
            compat_main,
            color_support
        )

        log_debug(
            f"Candidate {idx} box={(x, y, w, h)} roi_box={(rx, ry, rw, rh)} "
            f"mean_gray={mean_gray:.1f} best_template={best_name} "
            f"score={best_score:.3f} compat={compat_main:.2f} "
            f"support={color_support:.3f} final={final_score:.3f}"
        )

        if best_name is None:
            continue

        # Für farbige Erze muss wenigstens etwas passende Farbe vorhanden sein.
        if label_key != "coal" and color_support <= 0.0:
            continue

        # NEU HINZUGEFÜGT:
        # Sehr kleiner, gezielter Schutz gegen den alten Fehler:
        # Copper-Blöcke enthalten oft türkis/grüne Pixel und wurden dadurch manchmal
        # als Diamond oder Emerald akzeptiert. Wenn aber gleichzeitig deutlich
        # orange-braune Copper-Pixel vorhanden sind, wird Diamond/Emerald verworfen.
        if label_key in {"diamond", "emerald"}:
            copper_orange = _copper_orange_support(roi)
            copper_compat = _color_compatibility("copper", roi)

            # GEÄNDERT:
            # Stärkerer Schutz gegen Copper -> Diamond/Emerald.
            # Copper enthält oft türkis/grüne Pixel, aber zusätzlich orange-braune Pixel.
            # Sobald dieser Copper-Anteil vorhanden ist, darf Diamond/Emerald nur noch
            # akzeptiert werden, wenn der eigene Farbsupport deutlich stärker ist.
            copper_like_region = (
                copper_orange >= 0.006
                and (
                    copper_orange >= color_support * 0.30
                    or copper_compat >= 0.30
                )
            )

            own_color_dominates = (
                (
                    label_key == "diamond"
                    and color_support >= 0.020
                    and compat_label >= copper_compat + 0.025
                )
                or (
                    label_key == "emerald"
                    and best_score >= 0.70
                    and 0.060 <= color_support <= 0.140
                    and compat_label >= copper_compat + 0.040
                )
            )

            if copper_like_region and not own_color_dominates:
                continue

        has_basic_template = best_score >= threshold
        has_good_color = compat_main >= _min_compatibility(label_key)
        has_enough_support = color_support >= min_support

        # -----------------------------------------------------
        # Annahmebedingungen
        # -----------------------------------------------------

        # GEÄNDERT:
        # Farblich sehr starke Kandidaten dürfen minimal schwächer im Template sein.
        strong_color_case = (
            compat_main >= 0.78
            and color_support >= _good_color_support(label_key)
            and best_score >= max(0.35, threshold - 0.04)
        )

        # GEÄNDERT:
        # Sehr guter Template-Treffer darf etwas weniger Farbdichte haben,
        # aber niemals komplett ohne Farbe.
        strong_template_case = (
            best_score >= threshold + 0.12
            and compat_main >= max(0.10, _min_compatibility(label_key) * 0.75)
            and color_support >= min_support
        )

        if label_key == "copper":
            # GEÄNDERT:
            # Copper wird wieder gezielt akzeptiert, wenn das typische
            # Copper-Signal vorhanden ist: Orange/Braun + grün-türkische Pixel.
            # Dadurch werden genau die zwei Copper-Blöcke in Zeile 3 erkannt,
            # ohne Diamond/Emerald wieder zu beschädigen.
            copper_orange = _copper_orange_support(roi)
            copper_green = _copper_green_support(roi)

            normal_copper_case = (
                has_basic_template
                and has_good_color
                and has_enough_support
            )

            mixed_copper_case = (
                best_score >= max(0.46, threshold - 0.10)
                and compat_main >= 0.25
                and color_support >= min_support
                and copper_orange >= 0.004
                and copper_green >= 0.004
            )

            accepted = normal_copper_case or mixed_copper_case

            if not accepted:
                continue

        elif label_key == "iron":
            # Copper und Iron sehr vorsichtig behandeln.
            accepted = (
                has_basic_template
                and has_good_color
                and has_enough_support
            )

            if not accepted:
                continue

        elif label_key == "coal":
            # Kohle ist extrem anfällig für Schatten-Fehltreffer.
            coal_ok = (
                best_score >= threshold
                and compat_main >= _min_compatibility(label_key)
                and color_support >= min_support
            )

            if not coal_ok:
                continue

        else:
            accepted = (
                (has_basic_template and has_good_color and has_enough_support)
                or strong_color_case
                or strong_template_case
            )

            if not accepted:
                continue

        # GEÄNDERT:
        # Für die Ausgabe wird grundsätzlich die Kandidatenbox verwendet.
        # Nur bei Iron wird sie leicht zu einer blockähnlichen Box korrigiert.
        output_box = _final_output_box(
            label_key,
            (x, y, w, h),
            img.shape
        )

        detections.append({
            "label": label,
            "variant": best_name,
            "score": float(final_score),
            "box": output_box,
        })

    return detections


def _iou(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int]
) -> float:
    """
    Berechnet Intersection over Union zweier Bounding Boxes.

    Parameters
    ----------
    box_a : Tuple[int, int, int, int]
        Erste Box.
    box_b : Tuple[int, int, int, int]
        Zweite Box.

    Returns
    -------
    float
        IoU-Wert.
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
    union_area = aw * ah + bw * bh - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / float(union_area)


def _containment(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int]
) -> float:
    """
    NEU HINZUGEFÜGT:
    Berechnet, wie stark die kleinere Box in der größeren Box enthalten ist.

    Das hilft bei Fällen, in denen normale IoU niedrig ist, aber eine kleine Box
    eigentlich innerhalb derselben Objektregion liegt.

    Parameters
    ----------
    box_a : Tuple[int, int, int, int]
        Erste Box.
    box_b : Tuple[int, int, int, int]
        Zweite Box.

    Returns
    -------
    float
        Enthaltenheitswert bezogen auf die kleinere Box.
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

    smaller_area = min(area_a, area_b)

    if smaller_area <= 0:
        return 0.0

    return inter_area / float(smaller_area)


def _center_distance(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int]
) -> float:
    """
    NEU HINZUGEFÜGT:
    Berechnet den Abstand der Box-Zentren.

    Parameters
    ----------
    box_a : Tuple[int, int, int, int]
        Erste Box.
    box_b : Tuple[int, int, int, int]
        Zweite Box.

    Returns
    -------
    float
        Abstand der Mittelpunkte.
    """

    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    acx = ax + aw / 2.0
    acy = ay + ah / 2.0

    bcx = bx + bw / 2.0
    bcy = by + bh / 2.0

    return float(((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5)


def non_max_suppression(
    detections: List[Dict],
    iou_threshold: float = 0.25
) -> List[Dict]:
    """
    Entfernt doppelte oder stark überlappende Detektionen.

    GEÄNDERT:
    Neben IoU werden auch Containment und Mittelpunkt-Abstand geprüft.
    Dadurch werden Mehrfachbewertungen desselben Erzblocks reduziert.

    Parameters
    ----------
    detections : List[Dict]
        Liste der Detektionen.
    iou_threshold : float
        IoU-Schwelle.

    Returns
    -------
    List[Dict]
        Gefilterte Detektionen.
    """

    if not detections:
        return []

    ordered = sorted(
        detections,
        key=lambda d: d["score"],
        reverse=True
    )

    kept = []

    for det in ordered:
        keep = True

        for k in kept:
            iou = _iou(det["box"], k["box"])
            containment = _containment(det["box"], k["box"])
            dist = _center_distance(det["box"], k["box"])

            x, y, w, h = det["box"]
            kx, ky, kw, kh = k["box"]

            avg_size = (min(w, h) + min(kw, kh)) / 2.0

            centers_close = (
                avg_size > 0
                and dist < 0.55 * avg_size
            )

            if (
                iou >= iou_threshold
                or containment >= 0.60
                or centers_close
            ):
                keep = False
                break

        if keep:
            kept.append(det)

    return kept


def detect_diamond(
    img: np.ndarray,
    candidates: List[Tuple[int, int, int, int]],
    template: np.ndarray,
    threshold: float = 0.65
) -> List[Dict]:
    """
    Legacy-Funktion zur Erkennung von Diamanterz mit einem einzelnen Template.

    Diese Funktion wird aktuell in der Hauptpipeline nicht mehr benötigt,
    bleibt aber aus Kompatibilitätsgründen erhalten.

    Parameters
    ----------
    img : np.ndarray
        Eingabebild.
    candidates : List[Tuple[int, int, int, int]]
        Kandidatenboxen.
    template : np.ndarray
        Einzelnes Template.
    threshold : float
        Schwellenwert.

    Returns
    -------
    List[Dict]
        Erkannte Diamanten.
    """

    detections = []
    idx = 0

    for (x, y, w, h) in candidates:
        roi = img[y:y + h, x:x + w]

        score = match_template_multiscale(
            roi,
            template
        )

        log_debug(
            f"Candidate {idx} box={(x, y, w, h)} score={score:.3f}"
        )

        idx += 1

        if score >= threshold:
            detections.append({
                "label": "Diamond",
                "score": float(score),
                "box": (x, y, w, h),
            })

    return detections
