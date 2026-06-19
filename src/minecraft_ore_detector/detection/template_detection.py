# -*- coding: utf-8 -*-
"""Builds scored detections from candidates and template banks."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from minecraft_ore_detector.common.utils import log_debug
from minecraft_ore_detector.detection.color_validation import (
    _color_compatibility, _color_support_ratio, _copper_green_support,
    _copper_orange_support, _good_color_support, _min_color_support,
    _min_compatibility,
)
from minecraft_ore_detector.detection.template_matching import match_template_multiscale

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
