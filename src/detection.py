#Erze finden
import cv2
import numpy as np


def load_template(path):
    # cv2.imread im Graustufenmodus
    # Templates werden einfacher verglichen als Graubild
    tpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if tpl is None:
        raise FileNotFoundError(path)

    return tpl


def find_candidates(mask):
    # cv2.findContours findet zusammenhängende weiße Bereiche in einer Maske

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for c in contours:
        # cv2.boundingRect berechnet umschließendes Rechteck
        x, y, w, h = cv2.boundingRect(c)

        # kleine Artefakte filtern
        if w * h < 30:
            continue

        candidates.append((x, y, w, h))

    return candidates


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

import cv2
import numpy as np


def match_template_multiscale(roi, template, scales=None):

    # Standard-Skalen (kannst du später erweitern)
    if scales is None:
        scales = [0.6, 0.75, 1.0, 1.25, 1.5]

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

        result = cv2.matchTemplate(
            roi_gray,
            resized_template,
            cv2.TM_CCOEFF_NORMED
        )

        score = np.max(result)

        if score > best_score:
            best_score = score

    return best_score


def detect_diamond(img, candidates, template, threshold=0.65):
    detections = []

    for (x, y, w, h) in candidates:

        # ROI aus Originalbild ausschneiden
        roi = img[y:y+h, x:x+w]

        score = match_template_multiscale(roi, template)

        # nur akzeptieren, wenn Ähnlichkeit hoch genug ist
        if score >= threshold:
            detections.append({
                "label": "Diamond",
                "score": score,
                "box": (x, y, w, h)
            })

    return detections