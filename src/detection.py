#Erze finden
import cv2
import numpy as np
from utils import log_debug


def _hue_distance(a, b):
    d = abs(float(a) - float(b))
    return min(d, 180.0 - d)


def _template_family(template_name):
    return template_name.split("_", 1)[0] if template_name else None


def _color_compatibility(ore, roi_bgr):
    """Return a loose [0..1] plausibility score for the ore color."""
    if roi_bgr.size == 0:
        return 0.0

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h = float(np.mean(hsv[:, :, 0]))
    s = float(np.mean(hsv[:, :, 1]))
    v = float(np.mean(hsv[:, :, 2]))

    if ore == "coal":
        dark = max(0.0, 1.0 - (v / 150.0))
        desat = max(0.0, 1.0 - (s / 120.0))
        return min(1.0, 0.55 * dark + 0.45 * desat)

    profiles = {
        "copper": [(15.0, 18.0)],
        "diamond": [(98.0, 20.0)],
        "emerald": [(70.0, 18.0)],
        "gold": [(28.0, 16.0)],
        "iron": [(22.0, 20.0)],
        "lapis": [(112.0, 22.0)],
        "redstone": [(0.0, 12.0), (179.0, 12.0)],
    }

    if ore not in profiles:
        return 1.0

    hue_score = 0.0
    for center, tol in profiles[ore]:
        hue_score = max(hue_score, max(0.0, 1.0 - (_hue_distance(h, center) / tol)))

    # small sanity terms to prefer the expected saturation/value range
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

    sat_score = max(0.0, min(1.0, (s - 20.0) / 120.0))
    val_score = max(0.0, min(1.0, (v - 20.0) / 140.0))
    return min(1.0, 0.75 * hue_score + 0.15 * sat_score + 0.10 * val_score)


def _min_compatibility(ore):
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


def _best_center_component_bbox(mask, min_area=120):
    """Find a plausible ore component near image center.

    The template screenshots contain one ore on a mostly white wall.
    We choose the connected component that is sufficiently large and
    closest to the image center to avoid selecting HUD elements.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
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
        # normalized distance-to-center (primary criterion)
        dist = ((cx - cx_img) / w) ** 2 + ((cy - cy_img) / h) ** 2
        # mild penalty for very non-square boxes
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


def _merge_nearby_boxes(boxes, gap=16):
    """Merge overlapping / nearby candidate boxes into larger block regions."""
    if not boxes:
        return []

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

                    # overlap or near-overlap test with small gap
                    if not (x2 + gap < a1 or a2 + gap < x1 or y2 + gap < b1 or b2 + gap < y1):
                        x1 = min(x1, a1)
                        y1 = min(y1, b1)
                        x2 = max(x2, a2)
                        y2 = max(y2, b2)
                        used[j] = True
                        merged_this_round = True
                        changed = True

            new_boxes.append([x1, y1, x2, y2])

        work = new_boxes

    return [(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in work]


def _expand_box(box, img_shape, pad_factor=0.75, min_pad=10):
    x, y, w, h = box
    pad = max(min_pad, int(max(w, h) * pad_factor))
    img_h, img_w = img_shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img_w, x + w + pad)
    y1 = min(img_h, y + h + pad)
    return x0, y0, x1 - x0, y1 - y0


def _full_image_template_search(img, template_bank, label, threshold=0.36):
    """Search the whole image for a template variant.

    Used as a fallback for ores like coal that are often too dark / too small
    for the normal candidate mask to isolate reliably.
    """
    detections = []
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scales = [0.05, 0.07, 0.09, 0.12, 0.15, 0.18, 0.22, 0.28, 0.34]

    best = []
    for name, tpl in template_bank.items():
        for scale in scales:
            resized = cv2.resize(tpl, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            th, tw = resized.shape
            if img_gray.shape[0] < th or img_gray.shape[1] < tw:
                continue

            result = cv2.matchTemplate(img_gray, resized, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
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

    # keep the strongest few, then suppress overlap
    best = sorted(best, key=lambda d: d["score"], reverse=True)
    return non_max_suppression(best, iou_threshold=0.25)[:2]


def load_template(path):
    # Template als Farbbild laden; wir extrahieren daraus automatisch
    # den relevanten Erzbereich und geben anschließend ein Graubild zurück.
    tpl_bgr = cv2.imread(path, cv2.IMREAD_COLOR)

    if tpl_bgr is None:
        raise FileNotFoundError(path)

    tpl_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2LAB)

    h, w = tpl_gray.shape[:2]

    # Hintergrundfarbe aus Bildrand schätzen (weiße Concrete-Wand)
    strip = max(8, min(h, w) // 40)
    border = np.concatenate([
        lab[:strip, :, :].reshape(-1, 3),
        lab[-strip:, :, :].reshape(-1, 3),
        lab[:, :strip, :].reshape(-1, 3),
        lab[:, -strip:, :].reshape(-1, 3),
    ], axis=0)
    bg = np.median(border, axis=0)

    # Distanz zur geschätzten Hintergrundfarbe + Sättigung als Foreground-Hinweis
    dist = np.linalg.norm(lab.astype(np.float32) - bg.astype(np.float32), axis=2)
    sat = hsv[:, :, 1]

    fg = ((dist > 14.0) | (sat > 36)).astype(np.uint8) * 255

    # HUD / UI am unteren Rand ausblenden (nur für Templates)
    fg[int(0.86 * h):, :] = 0

    kernel = np.ones((5, 5), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

    bbox = _best_center_component_bbox(fg, min_area=max(120, int(h * w * 0.00008)))
    if bbox is not None:
        x, y, bw, bh = bbox
        # Wenn die Box zu groß ist, war die Extraktion vermutlich falsch.
        if bw < int(0.55 * w) and bh < int(0.55 * h):
            pad = 10
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(w, x + bw + pad)
            y1 = min(h, y + bh + pad)
            cropped = tpl_gray[y0:y1, x0:x1]
            if cropped.size > 0 and cropped.shape[0] >= 8 and cropped.shape[1] >= 8:
                return cropped

    # Fallback: klassisches Nicht-Weiß-Cropping (mit HUD-Ausblendung)
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

    # Robuster Fallback für eure Template-Screenshots:
    # zentralen quadratischen Ausschnitt nutzen (ein Ore-Block im Zentrum).
    # Diese Heuristik passt gut zu eurem "weiße Box + 1 Erz" Setup.
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

    # letzte Rückfalloption
    return tpl_gray


def find_candidates(mask, color_mask=None):
    # cv2.findContours findet zusammenhängende weiße Bereiche in einer Maske

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    # dynamische Min-Fläche: abhängig von Bildgröße
    img_h, img_w = mask.shape[:2]
    min_area = max(40, int(img_h * img_w * 0.00008))
    max_area = int(img_h * img_w * 0.02)

    for c in contours:
        # cv2.boundingRect berechnet umschließendes Rechteck
        x, y, w, h = cv2.boundingRect(c)

        # kleine Artefakte filtern
        if w * h < min_area:
            continue
        # sehr große Flächen sind kein einzelner Erzblock
        if w * h > max_area:
            continue

        # Falls eine Farb-Maske übergeben wurde, prüfe wieviel der Region
        # tatsächlich farblich passt. Das reduziert False-Positives die
        # nur aufgrund von Kanten entstanden sind.
        if color_mask is not None:
            # Ausschnitt aus Farbmaske
            crop = color_mask[y:y+h, x:x+w]
            if crop.size == 0:
                continue

            # Anteil weißer Pixel in der Farbmaske
            white = (crop > 0).sum()
            frac = white / float(crop.size)

            # Regel: entweder ausreichend prozentualer Anteil ODER genug absolute
            # Farb-Pixel (wichtig wenn Erze im Bild klein sind).
            if frac < 0.03 and white < 28:
                continue

        candidates.append((x, y, w, h))

    # Kleine Teilsegmente zu Block-Kandidaten zusammenfassen
    merged = _merge_nearby_boxes(candidates, gap=18)

    # Nach dem Mergen erneut kleine Boxen verwerfen
    final = []
    for (x, y, w, h) in merged:
        if w * h < max(120, int(img_h * img_w * 0.00012)):
            continue
        if w * h > max_area:
            continue
        final.append((x, y, w, h))

    return final


def match_template(roi, template):
    # ROI = Region of Interest (Kandidatenbereich aus Bild)

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    th, tw = template.shape

    # Schutz: ROI muss größer sein als Template
    if roi_gray.shape[0] < th or roi_gray.shape[1] < tw:
        return 0.0

    # cv2.matchTemplate vergleicht Bildausschnitt mit Template
    # TM_CCOEFF_NORMED gibt Ähnlichkeit zwischen 0 und 1 zurück
    result = cv2.matchTemplate(
        roi_gray,
        template,
        cv2.TM_CCOEFF_NORMED
    )

    # höchster Ähnlichkeitswert
    return float(np.max(result))


def match_template_multiscale(roi, template, scales=None):

    # Standard-Skalen (kannst du später erweitern)
    if scales is None:
        # Erweiterte Skalen inkl. kleiner Werte, damit auch große Templates
        # auf kleine ROIs skaliert werden können.
        scales = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25]

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    best_score = 0.0

    for scale in scales:

        # Template skalieren
        resized_template = cv2.resize(
            template,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

        th, tw = resized_template.shape

        # ROI muss größer sein als Template
        if roi_gray.shape[0] < th or roi_gray.shape[1] < tw:
            continue

        # Normales Matching auf Graustufen
        result_gray = cv2.matchTemplate(
            roi_gray,
            resized_template,
            cv2.TM_CCOEFF_NORMED
        )

        score_gray = float(np.max(result_gray))

        # Zusätzlich: Matching auf Kantemaps (robuster gegen Beleuchtung)
        roi_edges = cv2.Canny(roi_gray, 50, 150)
        tpl_edges = cv2.Canny(resized_template, 50, 150)

        score_edges = 0.0
        # template edges könnte leer sein (z.B. sehr kleines Template)
        if tpl_edges.sum() > 0 and roi_edges.shape[0] >= tpl_edges.shape[0] and roi_edges.shape[1] >= tpl_edges.shape[1]:
            try:
                result_edges = cv2.matchTemplate(roi_edges, tpl_edges, cv2.TM_CCOEFF_NORMED)
                score_edges = float(np.max(result_edges))
            except cv2.error:
                score_edges = 0.0

        # Kombiniere Scores: wir nehmen das Maximum (falls einer der beiden gut passt)
        score = max(score_gray, score_edges)

        if score > best_score:
            best_score = score

    return best_score


def detect_with_template_bank(
    img,
    candidates,
    template_bank,
    label="Diamond",
    threshold=0.72,
    brightness_split=None,
):
    """Match each candidate against multiple templates and keep best score."""
    detections = []

    for idx, (x, y, w, h) in enumerate(candidates):
        roi_box = (x, y, w, h)
        if label.lower() == "coal":
            roi_box = _expand_box(roi_box, img.shape, pad_factor=1.5, min_pad=20)

        rx, ry, rw, rh = roi_box
        roi = img[ry:ry + rh, rx:rx + rw]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_gray = float(roi_gray.mean()) if roi_gray.size > 0 else 0.0

        template_items = list(template_bank.items())
        # Optional: helle ROIs eher gegen Stone-Template matchen,
        # dunkle ROIs eher gegen Deepslate-Template.
        if brightness_split is not None and len(template_items) >= 2:
            if mean_gray >= brightness_split:
                filtered = [(n, t) for (n, t) in template_items if "deepslate" not in n]
                if filtered:
                    template_items = filtered
            else:
                filtered = [(n, t) for (n, t) in template_items if "deepslate" in n]
                if filtered:
                    template_items = filtered

        best_score = 0.0
        best_name = None

        for name, tpl in template_items:
            score = match_template_multiscale(roi, tpl)
            if score > best_score:
                best_score = score
                best_name = name

        family = _template_family(best_name) if best_name else label.lower()
        compat = _color_compatibility(family, roi)
        label_compat = _color_compatibility(label.lower(), roi)
        adjusted_score = best_score * (0.85 + 0.15 * compat)

        log_debug(
            f"Candidate {idx} box={(x, y, w, h)} roi_box={(rx, ry, rw, rh)} mean_gray={mean_gray:.1f} best_template={best_name} score={best_score:.3f} compat={compat:.2f} adjusted={adjusted_score:.3f}"
        )

        if best_score >= threshold and max(compat, label_compat) >= _min_compatibility(label.lower()):
            detections.append({
                "label": label,
                "variant": best_name,
                "score": float(adjusted_score),
                "box": (x, y, w, h),
            })

    # Fallback für Coal: wenn die Masken-Detektion zu schwach ist,
    # suche direkt im ganzen Bild nach dem besten dunklen Block.
    if label.lower() == "coal" and len(detections) == 0:
        detections.extend(_full_image_template_search(img, template_bank, label, threshold=0.52))

    return detections


def _iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / float(union)


def non_max_suppression(detections, iou_threshold=0.25):
    """Suppress duplicate detections that refer to the same ore block."""
    if not detections:
        return []

    ordered = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []

    for det in ordered:
        keep = True
        for k in kept:
            if _iou(det["box"], k["box"]) >= iou_threshold:
                keep = False
                break
        if keep:
            kept.append(det)

    return kept


def detect_diamond(img, candidates, template, threshold=0.65):
    detections = []
    idx = 0

    for (x, y, w, h) in candidates:

        # ROI aus Originalbild ausschneiden
        roi = img[y:y+h, x:x+w]

        score = match_template_multiscale(roi, template)

        log_debug(f"Candidate {idx} box={(x,y,w,h)} score={score:.3f}")
        idx += 1

        # nur akzeptieren, wenn Ähnlichkeit hoch genug ist
        if score >= threshold:
            detections.append({
                "label": "Diamond",
                "score": score,
                "box": (x, y, w, h)
            })

    return detections