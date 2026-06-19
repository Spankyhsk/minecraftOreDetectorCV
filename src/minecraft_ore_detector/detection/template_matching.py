# -*- coding: utf-8 -*-
"""Template loading and image matching."""

from typing import List, Optional, Tuple

import cv2
import numpy as np

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
