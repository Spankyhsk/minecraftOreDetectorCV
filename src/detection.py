# -*- coding: utf-8 -*-
"""
Modul für die Objekterkennung (Detection) von Minecraft-Erzen.
Dieses Modul enthält die Kernlogik der klassischen Bildverarbeitungspipeline:
- Extraktion und Zuschnitt von Erz-Templates aus Vorlage-Screenshots
- Erkennung von Erzkandidaten durch Konturanalyse (mit Flächen- und Dichtefilterung)
- Zusammenführung benachbarter Detektionen (Bounding-Box-Merging)
- Multi-Scale-Template-Matching sowohl auf Graustufen- als auch auf Kantenbildern
- Plausibilitätsprüfung durch Farbanalysen im HSV-Farbraum (Farb-Kompatibilität)
- Non-Maximum Suppression (NMS) zur Reduzierung von Doppel-Detektionen
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from utils import log_debug


def _hue_distance(a: float, b: float) -> float:
    """
    Berechnet den kreisförmigen Abstand zweieinhalb Farbtönen (Hues) im HSV-Raum.
    Da der Farbton kreisförmig angeordnet ist (0 bis 179 in OpenCV), beträgt die
    maximale Distanz 90 Grad.

    Parameters
    ----------
    a : float
        Erster Farbtonwert (0.0 bis 179.0).
    b : float
        Zweiter Farbtonwert (0.0 bis 179.0).

    Returns
    -------
    float
        Die minimale kreisförmige Distanz (0.0 bis 90.0).
    """
    d = abs(float(a) - float(b))
    return min(d, 180.0 - d)


def _template_family(template_name: Optional[str]) -> Optional[str]:
    """
    Extrahiert den Basisnamen des Erzes aus dem Template-Dateinamen.
    Beispiel: 'diamond_deepslate_ore' -> 'diamond'.

    Parameters
    ----------
    template_name : Optional[str]
        Der Name der Template-Variante.

    Returns
    -------
    Optional[str]
        Die Erz-ID oder None.
    """
    return template_name.split("_", 1)[0] if template_name else None


def _color_compatibility(ore: str, roi_bgr: np.ndarray) -> float:
    """
    Berechnet einen Plausibilitäts-Score [0..1] für die Übereinstimmung der
    durchschnittlichen Pixelfarbe in einem Ausschnitt (ROI) mit dem gesuchten Erztyp.

    Dies ist eine heuristische Plausibilitätsprüfung, um Fehltreffer des Template-Matchings
    (z. B. geometrische Steinstrukturen, die wie Diamanten geformt sind, aber grau statt blau sind)
    auszusortieren oder deren Score abzuwerten.

    Parameters
    ----------
    ore : str
        Die Erz-ID (z. B. "diamond").
    roi_bgr : np.ndarray
        Die Region of Interest (ROI) im BGR-Format.

    Returns
    -------
    float
        Der Kompatibilitäts-Score von 0.0 (unplausibel) bis 1.0 (sehr passend).
    """
    if roi_bgr.size == 0:
        return 0.0

    # Durchschnittliche HSV-Werte der Region berechnen
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h = float(np.mean(hsv[:, :, 0]))
    s = float(np.mean(hsv[:, :, 1]))
    v = float(np.mean(hsv[:, :, 2]))

    # Kohle ist extrem dunkel und farblos (geringe Sättigung)
    if ore == "coal":
        dark = max(0.0, 1.0 - (v / 150.0))
        desat = max(0.0, 1.0 - (s / 120.0))
        return min(1.0, 0.55 * dark + 0.45 * desat)

    # Farbprofile: (Erwarteter Hue-Mittelpunkt, Toleranz)
    profiles = {
        "copper": [(15.0, 18.0)],
        "diamond": [(98.0, 20.0)],
        "emerald": [(70.0, 18.0)],
        "gold": [(28.0, 16.0)],
        "iron": [(22.0, 20.0)],
        "lapis": [(112.0, 22.0)],
        "redstone": [(0.0, 12.0), (179.0, 12.0)],  # Redstone hat Hue-Wraparound
    }

    if ore not in profiles:
        return 1.0

    # Abstand zum erwarteten Hue-Wert berechnen
    hue_score = 0.0
    for center, tol in profiles[ore]:
        hue_score = max(hue_score, max(0.0, 1.0 - (_hue_distance(h, center) / tol)))

    # Plausibilitätsprüfungen für Sättigung (S) und Helligkeit (V)
    if ore == "redstone":
        sat_score = max(0.0, min(1.0, (s - 45.0) / 90.0))
        val_score = max(0.0, min(1.0, (v - 25.0) / 120.0))
        return min(1.0, 0.7 * hue_score + 0.15 * sat_score + 0.15 * val_score)

    if ore == "diamond":
        sat_score = max(0.0, min(1.0, (s - 25.0) / 90.0))
        val_score = max(0.0, min(1.0, (v - 25.0) / 120.0))
        return min(1.0, 0.7 * hue_score + 0.15 * sat_score + 0.15 * val_score)

    if ore == "coal":
        return hue_score

    # Standard-Erz-Profil (z.B. Gold, Kupfer, Eisen, Lapis)
    sat_score = max(0.0, min(1.0, (s - 20.0) / 120.0))
    val_score = max(0.0, min(1.0, (v - 20.0) / 140.0))
    return min(1.0, 0.75 * hue_score + 0.15 * sat_score + 0.10 * val_score)


def _min_compatibility(ore: str) -> float:
    """
    Gibt die minimale Farbkompatibilität zurück, die ein Erz-Kandidat aufweisen muss,
    um überhaupt akzeptiert zu werden. Verhindert Fehlalarme bei strukturell ähnlichen,
    aber farblich völlig falschen Steinen.

    Parameters
    ----------
    ore : str
        Die Erz-ID.

    Returns
    -------
    float
        Die minimale Kompatibilitätsgrenze.
    """
    return {
        "coal": 0.35,
        "copper": 0.04,
        "diamond": 0.05,
        "emerald": 0.08,
        "gold": 0.05,
        "iron": 0.05,
        "lapis": 0.05,
        "redstone": 0.12,
    }.get(ore, 0.0)


def _best_center_component_bbox(mask: np.ndarray, min_area: int = 120) -> Optional[Tuple[int, int, int, int]]:
    """
    Sucht das am besten zentrierte und ausreichend große zusammenhängende Objekt
    (Connected Component) in einer Binärmaske.

    Hintergrund: Die Template-Bilder zeigen ein Erz im Bildzentrum vor einer
    neutralen weißen Betonwand. Mit dieser Funktion wird der Erzblock zuverlässig
    vom weißen Hintergrund und etwaigen HUD-Elementen isoliert.

    Parameters
    ----------
    mask : np.ndarray
        Die binäre Maske des Template-Bildes.
    min_area : int, optional
        Mindestfläche des Objekts in Pixeln (Standard ist 120).

    Returns
    -------
    Optional[Tuple[int, int, int, int]]
        Die Bounding Box (x, y, w, h) des zentralen Objekts, oder None.
    """
    # Connected Components mit Statistiken ermitteln (8er Nachbarschaft)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape[:2]
    img_area = h * w
    cx_img = w / 2.0
    cy_img = h / 2.0

    best_idx = -1
    best_score = float("inf")

    # Index 0 ist der Hintergrund, daher bei 1 starten
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        # Wenn das Objekt mehr als 25% des Bildes einnimmt, ist es wohl kein Erzblock
        if area > int(0.25 * img_area):
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if bw < 8 or bh < 8:
            continue

        cx, cy = centroids[i]
        # Distanz zum Bildmittelpunkt (normalisiert auf Bildgröße)
        dist = ((cx - cx_img) / w) ** 2 + ((cy - cy_img) / h) ** 2
        # Geringe Strafe für sehr asymmetrische Bounding Boxes
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


def _merge_nearby_boxes(boxes: List[Tuple[int, int, int, int]], gap: int = 16) -> List[Tuple[int, int, int, int]]:
    """
    Führt Bounding-Boxes, die sich überschneiden oder sehr nah beieinander liegen,
    zu einer einzigen größeren Bounding-Box zusammen.

    Dadurch werden zerstückelte Farbbereiche eines einzelnen Erzblocks wieder
    als ein zusammenhängendes Kandidatenareal erfasst.

    Parameters
    ----------
    boxes : List[Tuple[int, int, int, int]]
        Die Liste der ursprünglichen Bounding-Boxes (x, y, w, h).
    gap : int, optional
        Die maximale Distanz in Pixeln, bis zu der Boxen gemerged werden (Standard ist 16).

    Returns
    -------
    List[Tuple[int, int, int, int]]
        Die Liste der zusammengeführten Bounding-Boxes.
    """
    if not boxes:
        return []

    # Umwandlung von (x, y, w, h) zu (x1, y1, x2, y2) für einfachere Berechnung
    work = [[x, y, x + w, y + h] for (x, y, w, h) in boxes]

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

                    # Prüfung, ob die Boxen unter Berücksichtigung des 'gap' überlappen
                    if not (x2 + gap < a1 or a2 + gap < x1 or y2 + gap < b1 or b2 + gap < y1):
                        # Bounding Box erweitern, um beide Bereiche einzuschließen
                        x1 = min(x1, a1)
                        y1 = min(y1, b1)
                        x2 = max(x2, a2)
                        y2 = max(y2, b2)
                        used[j] = True
                        merged_this_round = True
                        changed = True

            new_boxes.append([x1, y1, x2, y2])

        work = new_boxes

    # Zurückkonvertieren zu (x, y, w, h)
    return [(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in work]


def _expand_box(box: Tuple[int, int, int, int], img_shape: Tuple[int, ...], pad_factor: float = 0.75, min_pad: int = 10) -> Tuple[int, int, int, int]:
    """
    Erweitert eine Bounding-Box proportional um einen Sicherheitsabstand (Padding).
    Wird z. B. bei Kohle eingesetzt, um auch die umgebenden Steinstrukturen für das
    Template matching einzubeziehen, da die reine Maske oft zu klein ist.

    Parameters
    ----------
    box : Tuple[int, int, int, int]
        Die ursprüngliche Box (x, y, w, h).
    img_shape : Tuple[int, ...]
        Die Bildgröße des Originalbilds (Höhe, Breite, ...), um ein Überschreiten der Ränder zu verhindern.
    pad_factor : float, optional
        Prozentualer Erweiterungsfaktor bezogen auf die maximale Boxdimension (Standard ist 0.75).
    min_pad : int, optional
        Mindest-Padding in Pixeln (Standard ist 10).

    Returns
    -------
    Tuple[int, int, int, int]
        Die vergrößerte Box (x, y, w, h).
    """
    x, y, w, h = box
    pad = max(min_pad, int(max(w, h) * pad_factor))
    img_h, img_w = img_shape[:2]
    
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img_w, x + w + pad)
    y1 = min(img_h, y + h + pad)
    return x0, y0, x1 - x0, y1 - y0


def _full_image_template_search(img: np.ndarray, template_bank: Dict[str, np.ndarray], label: str, threshold: float = 0.36) -> List[Dict]:
    """
    Führt eine globale Template-Matching-Suche über das gesamte Bild durch.

    Dient als Fallback für Erze (insb. Kohle), falls diese durch die normale
    Farbsegmentierung (z.B. weil sie extrem dunkel oder klein sind) keine Kandidaten erzeugen konnten.
    Arbeitet auf Graustufenbildern bei verschiedenen Skalierungsfaktoren.

    Parameters
    ----------
    img : np.ndarray
        Das Eingabebild.
    template_bank : Dict[str, np.ndarray]
        Die Template-Bank des Erzes.
    label : str
        Das Label des Erzes (z. B. "Coal").
    threshold : float, optional
        Der minimale Schwellenwert (Standard ist 0.36).

    Returns
    -------
    List[Dict]
        Eine Liste gefundener Detektionen.
    """
    detections = []
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Skalierungsfaktoren für das Template (Kohle-Erze können unterschiedlich weit entfernt sein)
    scales = [0.05, 0.07, 0.09, 0.12, 0.15, 0.18, 0.22, 0.28, 0.34]

    best = []
    for name, tpl in template_bank.items():
        for scale in scales:
            # Template skalieren
            resized = cv2.resize(tpl, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            th, tw = resized.shape
            # Schutz: Template darf nicht größer als das Suchbild sein
            if img_gray.shape[0] < th or img_gray.shape[1] < tw:
                continue

            # Standard Template Matching (normierte Kreuzkorrelation)
            result = cv2.matchTemplate(img_gray, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val < threshold:
                continue

            x, y = max_loc
            det = {
                "label": label,
                "variant": name,
                "score": float(max_val),
                "box": (x, y, tw, th),
            }
            best.append(det)

    if not best:
        return []

    # Nach Score sortieren und die stärksten Treffer mittels NMS filtern
    best = sorted(best, key=lambda d: d["score"], reverse=True)
    return non_max_suppression(best, iou_threshold=0.25)[:2]


def load_template(path: str) -> np.ndarray:
    """
    Lädt ein Erz-Template aus dem Screenshot und extrahiert daraus automatisch den
    relevanten Erz-Block (Crop).

    Der Algorithmus schätzt die Hintergrundfarbe (hellgrauer/weißer Beton) am Bildrand,
    erstellt eine Maske für alle Pixel, die farblich oder sättigungstechnisch davon
    abweichen, filtert das HUD am unteren Rand aus und schneidet das zentrale Objekt aus.
    Dies ermöglicht das Vergleichen kleiner Bildausschnitte mit den echten Erzen im Spiel.

    Parameters
    ----------
    path : str
        Dateipfad zum Template-Screenshot.

    Returns
    -------
    np.ndarray
        Das freigestellte, zugeschnittene Template als Graustufenbild.

    Raises
    ------
    FileNotFoundError
        Wenn der Screenshot nicht geladen werden kann.
    """
    # Template als Farbbild laden
    tpl_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if tpl_bgr is None:
        raise FileNotFoundError(f"Template unter '{path}' konnte nicht geladen werden.")

    tpl_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2LAB)

    h, w = tpl_gray.shape[:2]

    # 1. Hintergrundfarbe abschätzen (Auswertung des Bildrandes)
    strip = max(8, min(h, w) // 40)
    border = np.concatenate([
        lab[:strip, :, :].reshape(-1, 3),   # Oberer Rand
        lab[-strip:, :, :].reshape(-1, 3),  # Unterer Rand
        lab[:, :strip, :].reshape(-1, 3),   # Linker Rand
        lab[:, -strip:, :].reshape(-1, 3),  # Rechter Rand
    ], axis=0)
    bg = np.median(border, axis=0)

    # 2. Distanz zum geschätzten Hintergrund berechnen + Sättigung als Feature nutzen
    dist = np.linalg.norm(lab.astype(np.float32) - bg.astype(np.float32), axis=2)
    sat = hsv[:, :, 1]

    # Pixel markieren, die sich deutlich vom Hintergrund unterscheiden (dist > 14.0) oder farbig sind (sat > 36)
    fg = ((dist > 14.0) | (sat > 36)).astype(np.uint8) * 255

    # 3. HUD am unteren Rand ausblenden (verhindert fälschliche Erkennung von Hotbar-Elementen)
    fg[int(0.86 * h):, :] = 0

    # 4. Rauschbereinigung per Opening/Closing
    kernel = np.ones((5, 5), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

    # 5. Connected Component Suche nach dem zentrierten Erz-Block
    bbox = _best_center_component_bbox(fg, min_area=max(120, int(h * w * 0.00008)))
    if bbox is not None:
        x, y, bw, bh = bbox
        # Plausibilität: Das extrahierte Erz sollte nicht mehr als 55% der Bildmaße einnehmen
        if bw < int(0.55 * w) and bh < int(0.55 * h):
            pad = 10
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(w, x + bw + pad)
            y1 = min(h, y + bh + pad)
            cropped = tpl_gray[y0:y1, x0:x1]
            if cropped.size > 0 and cropped.shape[0] >= 8 and cropped.shape[1] >= 8:
                return cropped

    # 6. Erster Fallback: Alle Pixel ausschneiden, die nicht weiß/hellgrau sind (< 245)
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

    # 7. Zweiter Fallback: Zentraler quadratischer Ausschnitt (Heuristik für Test-Screenshots)
    side = int(min(h, w) * 0.18)
    side = max(96, min(side, 320))
    cx, cy = w // 2, h // 2
    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(w, x0 + side)
    y1 = min(h, y0 + side)
    
    center_crop = tpl_gray[y0:y1, x0:x1]
    if center_crop.size > 0 and center_crop.shape[0] >= 8 and center_crop.shape[1] >= 8:
        return center_crop

    # Letzter Ausweg: Komplettes Graustufenbild zurückgeben
    return tpl_gray


def find_candidates(mask: np.ndarray, color_mask: Optional[np.ndarray] = None) -> List[Tuple[int, int, int, int]]:
    """
    Sucht nach erzwürdigen Regionen (Kandidaten) in der binären Segmentierungsmaske.

    Nutzt die OpenCV-Konturanalyse. Die gefundenen Konturen werden nach ihrer
    Fläche gefiltert (zu kleine Strukturen sind Bildrauschen, zu große sind Wände/Böden).
    Falls eine Farbmaske übergeben wurde, wird zusätzlich geprüft, wie hoch der Anteil
    von Pixeln der gesuchten Farbe in der Region ist, um geometrisch bedingte Kanten-Fehltreffer zu reduzieren.
    Abschließend werden nahe beieinander liegende Boxen verschmolzen.

    Parameters
    ----------
    mask : np.ndarray
        Die binäre hybride Maske (Farbe + Kanten).
    color_mask : Optional[np.ndarray], optional
        Die reine binäre Farbmaske des Erzes zur Validierung der Farbdichte (Standard ist None).

    Returns
    -------
    List[Tuple[int, int, int, int]]
        Eine Liste von Bounding Boxes (x, y, w, h) der Kandidaten.
    """
    # Konturen in der Binärmaske suchen
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,      # Nur äußere Konturen erfassen
        cv2.CHAIN_APPROX_SIMPLE  # Konturen komprimieren (spart Speicher)
    )

    candidates = []

    img_h, img_w = mask.shape[:2]
    # Dynamische Flächengrenzen basierend auf der Bildauflösung
    min_area = max(40, int(img_h * img_w * 0.00008))
    max_area = int(img_h * img_w * 0.02)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # 1. Filter: Zu kleine Artefakte ausschließen
        if w * h < min_area:
            continue
        # 2. Filter: Riesige Areale (z. B. ganze Decken oder Schluchten) ausschließen
        if w * h > max_area:
            continue

        # 3. Filter: Farbdichte prüfen (falls color_mask angegeben)
        if color_mask is not None:
            crop = color_mask[y:y+h, x:x+w]
            if crop.size == 0:
                continue

            # Anteil der farbsegmentierten Pixel an der Box-Fläche berechnen
            white_pixels = (crop > 0).sum()
            density = white_pixels / float(crop.size)

            # Plausibilität: Entweder weisen mind. 3% der Box die gesuchte Farbe auf
            # ODER es gibt eine absolute Mindestanzahl an Erzfarb-Pixeln (wichtig bei kleinen Blöcken)
            if density < 0.03 and white_pixels < 28:
                continue

        candidates.append((x, y, w, h))

    # 4. Nahe beieinander liegende Kandidatenboxen verschmelzen (z. B. getrennte Adern)
    merged = _merge_nearby_boxes(candidates, gap=18)

    # 5. Nach dem Mergen erneut auf Plausibilität der Größe prüfen
    final = []
    for (x, y, w, h) in merged:
        if w * h < max(120, int(img_h * img_w * 0.00012)):
            continue
        if w * h > max_area:
            continue
        final.append((x, y, w, h))

    return final


def match_template(roi: np.ndarray, template: np.ndarray) -> float:
    """
    Führt ein einfaches Template-Matching auf einer Region of Interest (ROI) durch.

    Parameters
    ----------
    roi : np.ndarray
        Der Bildausschnitt (BGR-Format).
    template : np.ndarray
        Das zugeschnittene Template (Graustufen).

    Returns
    -------
    float
        Der normierte Übereinstimmungskoeffizient [0..1].
    """
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    th, tw = template.shape

    # Falls die ROI kleiner als das Template ist, kann nicht gematcht werden
    if roi_gray.shape[0] < th or roi_gray.shape[1] < tw:
        return 0.0

    # Normiertes Template Matching (TM_CCOEFF_NORMED)
    result = cv2.matchTemplate(
        roi_gray,
        template,
        cv2.TM_CCOEFF_NORMED
    )

    # Maximalen Korrelationskoeffizienten zurückgeben
    return float(np.max(result))


def match_template_multiscale(roi: np.ndarray, template: np.ndarray, scales: Optional[List[float]] = None) -> float:
    """
    Führt ein Multi-Scale-Template-Matching auf einer ROI durch.
    Das Template wird in verschiedenen Größen skaliert, um Erze in unterschiedlichen
    Entfernungen (Kamera-Distanzen) zu erkennen.

    Zusätzlich zum Graustufenbild wird auch ein Kanten-Matching (Canny Edge) durchgeführt,
    da Kantenstrukturen im Minecraft-Erzblock (die Pixel-Art-Struktur) unter veränderten
    Helligkeiten stabiler sind als Graustufenwerte. Der beste Score aus beiden Matchings
    wird zurückgegeben.

    Parameters
    ----------
    roi : np.ndarray
        Die zu überprüfende Region of Interest (BGR-Farbbild).
    template : np.ndarray
        Das Template (Graustufen).
    scales : Optional[List[float]], optional
        Liste der Skalierungsfaktoren (Standard ist [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25]).

    Returns
    -------
    float
        Der beste normierte Übereinstimmungsscore [0..1] über alle Skalen und Modi.
    """
    if scales is None:
        # Breite Palette an Skalierungen, um nahe sowie sehr weit entfernte Blöcke abzudecken
        scales = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25]

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    best_score = 0.0

    for scale in scales:
        # Template proportional skalieren (INTER_AREA eignet sich hervorragend zum Verkleinern)
        resized_template = cv2.resize(
            template,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

        th, tw = resized_template.shape

        # Passt das skaliert Template überhaupt in die ROI?
        if roi_gray.shape[0] < th or roi_gray.shape[1] < tw:
            continue

        # 1. Matching auf Graustufenbild
        result_gray = cv2.matchTemplate(
            roi_gray,
            resized_template,
            cv2.TM_CCOEFF_NORMED
        )
        score_gray = float(np.max(result_gray))

        # 2. Matching auf Canny-Kantenbildern (Beleuchtungsunabhängig)
        roi_edges = cv2.Canny(roi_gray, 50, 150)
        tpl_edges = cv2.Canny(resized_template, 50, 150)

        score_edges = 0.0
        # Prüfen, ob das Kanten-Template Kanten enthält und in die ROI passt
        if tpl_edges.sum() > 0 and roi_edges.shape[0] >= tpl_edges.shape[0] and roi_edges.shape[1] >= tpl_edges.shape[1]:
            try:
                result_edges = cv2.matchTemplate(roi_edges, tpl_edges, cv2.TM_CCOEFF_NORMED)
                score_edges = float(np.max(result_edges))
            except cv2.error:
                score_edges = 0.0

        # Höchsten Score dieser Skalierung wählen
        score = max(score_gray, score_edges)

        if score > best_score:
            best_score = score

    return best_score


def detect_with_template_bank(
    img: np.ndarray,
    candidates: List[Tuple[int, int, int, int]],
    template_bank: Dict[str, np.ndarray],
    label: str = "Diamond",
    threshold: float = 0.72,
    brightness_split: Optional[float] = 95.0,
) -> List[Dict]:
    """
    Abgleich aller Kandidaten-ROIs mit einer Template-Bank (z. B. Stein- und Deepslate-Varianten).

    Features:
    1. Padding-Erweiterung für spezielle Ores (Kohle).
    2. Helligkeitssplit: Unterscheidet anhand der mittleren Helligkeit der ROI
       zwischen Stone (heller) und Deepslate (dunkler). Nur die passenden Templates
       werden verglichen, was Rechenzeit spart und False Positives reduziert.
    3. Gewichtung durch Farbkompatibilität: Der Matching-Score wird mit der Farbkompatibilität
       multipliziert (gewichtete Anpassung), um farblich falsche Strukturen abzuwerten.

    Parameters
    ----------
    img : np.ndarray
        Das Eingabebild.
    candidates : List[Tuple[int, int, int, int]]
        Die Liste der Kandidatenboxen.
    template_bank : Dict[str, np.ndarray]
        Die Dictionary-Struktur aller Templates dieses Erzes.
    label : str, optional
        Der Anzeigename des Erzes (z. B. "Diamond").
    threshold : float, optional
        Der Akzeptanz-Schwellenwert (Standard ist 0.72).
    brightness_split : Optional[float], optional
        Grauwert-Grenze zur Trennung von Stone und Deepslate (Standard ist 95.0).

    Returns
    -------
    List[Dict]
        Die Liste der validierten Detektionen.
    """
    detections = []

    for idx, (x, y, w, h) in enumerate(candidates):
        roi_box = (x, y, w, h)
        # Kohle-Kandidaten vergrößern, da die Maske oft sehr eng an den Erz-Pixeln liegt
        if label.lower() == "coal":
            roi_box = _expand_box(roi_box, img.shape, pad_factor=1.5, min_pad=20)

        rx, ry, rw, rh = roi_box
        roi = img[ry:ry + rh, rx:rx + rw]
        
        # Mittlere Helligkeit im Graubild schätzen
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_gray = float(roi_gray.mean()) if roi_gray.size > 0 else 0.0

        # Templates filtern basierend auf der Helligkeit (Stone vs Deepslate)
        template_items = list(template_bank.items())
        if brightness_split is not None and len(template_items) >= 2:
            if mean_gray >= brightness_split:
                # Stone-Variante bevorzugen (enthält kein 'deepslate' im Dateinamen)
                filtered = [(n, t) for (n, t) in template_items if "deepslate" not in n]
                if filtered:
                    template_items = filtered
            else:
                # Deepslate-Variante bevorzugen (enthält 'deepslate' im Dateinamen)
                filtered = [(n, t) for (n, t) in template_items if "deepslate" in n]
                if filtered:
                    template_items = filtered

        best_score = 0.0
        best_name = None

        # Bestes Template für diesen Kandidaten ermitteln
        for name, tpl in template_items:
            score = match_template_multiscale(roi, tpl)
            if score > best_score:
                best_score = score
                best_name = name

        # Farbanalyse zur Validierung
        family = _template_family(best_name) if best_name else label.lower()
        compat = _color_compatibility(family, roi)
        label_compat = _color_compatibility(label.lower(), roi)
        
        # Der finale Score setzt sich zu 85% aus der Formähnlichkeit (Template Matching)
        # und zu 15% aus der Farbplausibilität zusammen.
        adjusted_score = best_score * (0.85 + 0.15 * compat)

        log_debug(
            f"Candidate {idx} box={(x, y, w, h)} roi_box={(rx, ry, rw, rh)} "
            f"mean_gray={mean_gray:.1f} best_template={best_name} "
            f"score={best_score:.3f} compat={compat:.2f} adjusted={adjusted_score:.3f}"
        )

        # Akzeptieren, wenn der Score über dem Threshold liegt und die Farbe eine Mindestkompatibilität aufweist
        if best_score >= threshold and max(compat, label_compat) >= _min_compatibility(label.lower()):
            detections.append({
                "label": label,
                "variant": best_name,
                "score": float(adjusted_score),
                "box": (x, y, w, h),
            })

    # Kohle-Spezialfall: Wenn die normale Pipeline gar keine Kohle findet,
    # probieren wir die globale Suche über das gesamte Bild.
    if label.lower() == "coal" and len(detections) == 0:
        detections.extend(_full_image_template_search(img, template_bank, label, threshold=0.52))

    return detections


def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """
    Berechnet die Intersection over Union (IoU) zweier Bounding Boxes.
    IoU = Schnittfläche / Vereinigungsfläche.
    Ein IoU von 1.0 bedeutet identische Boxen, ein Wert von 0.0 bedeutet keine Überschneidung.

    Parameters
    ----------
    box_a : Tuple[int, int, int, int]
        Erste Bounding Box (x, y, w, h).
    box_b : Tuple[int, int, int, int]
        Zweite Bounding Box (x, y, w, h).

    Returns
    -------
    float
        Der IoU-Wert [0.0..1.0].
    """
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    # Koordinaten der Schnittmenge (Intersection) berechnen
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Vereinigungsmenge (Union) = Fläche A + Fläche B - Schnittmenge
    union_area = aw * ah + bw * bh - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / float(union_area)


def non_max_suppression(detections: List[Dict], iou_threshold: float = 0.25) -> List[Dict]:
    """
    Entfernt mehrfache, stark überlappende Detektionen desselben Erzblocks.

    Es wird immer der Treffer mit dem höchsten Score behalten. Alle anderen Treffer,
    die mit diesem Treffer einen IoU-Wert aufweisen, der über dem 'iou_threshold' liegt,
    werden unterdrückt (verworfen).

    Parameters
    ----------
    detections : List[Dict]
        Die Liste aller rohen Detektionen.
    iou_threshold : float, optional
        Grenzwert für die Überlappung (Standard ist 0.25).

    Returns
    -------
    List[Dict]
        Die bereinigte Liste der Detektionen.
    """
    if not detections:
        return []

    # Sortieren nach Score absteigend (stärkster Treffer zuerst)
    ordered = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []

    for det in ordered:
        keep = True
        for k in kept:
            # Wenn der aktuelle Treffer stark mit einer bereits behaltenen Box überlappt
            if _iou(det["box"], k["box"]) >= iou_threshold:
                keep = False
                break
        if keep:
            kept.append(det)

    return kept


def detect_diamond(img: np.ndarray, candidates: List[Tuple[int, int, int, int]], template: np.ndarray, threshold: float = 0.65) -> List[Dict]:
    """
    Legacy-Funktion zur Erkennung von Diamanten mit einem einzigen Template.
    (Wurde in früheren Versionen verwendet und für Abwärtskompatibilität behalten).

    Parameters
    ----------
    img : np.ndarray
        Das Eingabebild.
    candidates : List[Tuple[int, int, int, int]]
        Die Liste der Kandidatenboxen.
    template : np.ndarray
        Das Diamanterz-Template.
    threshold : float, optional
        Schwellenwert (Standard ist 0.65).

    Returns
    -------
    List[Dict]
        Die erkannten Diamanten.
    """
    detections = []
    idx = 0

    for (x, y, w, h) in candidates:
        roi = img[y:y+h, x:x+w]
        score = match_template_multiscale(roi, template)

        log_debug(f"Candidate {idx} box={(x,y,w,h)} score={score:.3f}")
        idx += 1

        if score >= threshold:
            detections.append({
                "label": "Diamond",
                "score": score,
                "box": (x, y, w, h)
            })

    return detections