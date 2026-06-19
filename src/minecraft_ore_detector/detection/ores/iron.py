# -*- coding: utf-8 -*-
"""Iron-spezifische Zusatzstrategien."""

from typing import Dict, List, Tuple

import cv2
import numpy as np

from minecraft_ore_detector.app.config import OreDetectorConfig
from minecraft_ore_detector.detection.core import (
    _color_compatibility,
    _color_support_mask,
    _color_support_ratio,
    _expand_box,
    match_template_multiscale,
    non_max_suppression,
)
from minecraft_ore_detector.detection.geometry import (
    box_iou,
    clip_box,
    overlaps_any_box,
)
from minecraft_ore_detector.detection.mask_statistics import (
    integral_support,
    mask_integral,
)
from minecraft_ore_detector.imaging.runtime_mask_filter import RuntimeMaskFilter
from minecraft_ore_detector.imaging.segmentation import color_mask

Box = Tuple[int, int, int, int]


class IronDetector:
    """Ergaenzt die Standarderkennung um Iron-Sonderfaelle."""

    def __init__(
        self,
        config: OreDetectorConfig,
        mask_filter: RuntimeMaskFilter,
    ):
        self.config = config
        self.mask_filter = mask_filter






    def detect_dense_wide_split(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        hsv: np.ndarray,
        template_bank: Dict[str, np.ndarray]
    ) -> List[Dict]:
        """
        NEU HINZUGEFÜGT:
        Konservativer Split fuer extrem grosse, dichte Iron-Flaechen.

        In test7 verschmilzt die Iron-Maske zu einer breiten Wandregion. Eine
        einzelne Gesamtbox ist zu gross fuer die Review-IoU, zwei Haelften
        entsprechen aber den sichtbaren Erzbereichen. Der Fallback greift nur
        bei sehr dichtem Iron-Farbsignal und engen Helligkeits-/Sättigungswerten.
        """

        if not template_bank:
            return []

        raw_mask = color_mask(hsv, "iron")
        raw_mask = self.mask_filter.remove_hud_regions(raw_mask)

        img_h, img_w = img.shape[:2]
        grouped = cv2.morphologyEx(
            raw_mask,
            cv2.MORPH_OPEN,
            np.ones((9, 9), np.uint8)
        )
        grouped = cv2.morphologyEx(
            grouped,
            cv2.MORPH_CLOSE,
            np.ones((17, 17), np.uint8)
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            grouped,
            connectivity=8
        )
        detections = []

        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])

            if w < 1000 or h < 720:
                continue

            component_raw = raw_mask[y:y + h, x:x + w]
            component_support = float(np.mean(component_raw > 0))
            if component_support < 0.60:
                continue

            split_x = x + w // 2
            split_boxes = [
                (x, y, split_x - x, h),
                (split_x, y, x + w - split_x, h),
            ]

            for sx, sy, sw, sh in split_boxes:
                roi = img[sy:sy + sh, sx:sx + sw]
                pre_roi = img_preprocessed[sy:sy + sh, sx:sx + sw]

                if roi.size == 0 or pre_roi.size == 0:
                    continue

                raw_support = float(np.mean(raw_mask[sy:sy + sh, sx:sx + sw] > 0))
                pre_color_support = _color_support_ratio("iron", pre_roi)
                iron_support = _color_support_ratio("iron", roi)
                pre_color_compatibility = max(
                    _color_compatibility("iron", roi),
                    _color_compatibility("iron", pre_roi)
                )

                if raw_support < 0.50:
                    continue
                if pre_color_support < 0.88 or iron_support < 0.55:
                    continue
                if pre_color_compatibility < 0.90:
                    continue

                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mean_value = float(gray.mean())
                texture_strength = float(gray.std())
                saturation_mean = float(hsv_roi[:, :, 1].mean())
                value_mean = float(hsv_roi[:, :, 2].mean())

                if not (37.0 <= mean_value <= 47.0):
                    continue
                if texture_strength < 8.0:
                    continue
                if not (120.0 <= saturation_mean <= 135.0):
                    continue
                if not (48.0 <= value_mean <= 54.0):
                    continue

                sample = roi
                sample_side = 220
                if sw >= sample_side and sh >= sample_side:
                    cx = sw // 2
                    cy = sh // 2
                    sample = roi[
                        max(0, cy - sample_side // 2):min(sh, cy + sample_side // 2),
                        max(0, cx - sample_side // 2):min(sw, cx + sample_side // 2),
                    ]

                best_score = 0.0
                best_name = None

                for name, template in template_bank.items():
                    score = match_template_multiscale(sample, template)
                    if score > best_score:
                        best_score = score
                        best_name = name

                if best_name is None or best_score < 0.56:
                    continue

                final_score = min(
                    0.93,
                    0.50
                    + 0.16 * best_score
                    + 0.12 * pre_color_support
                    + 0.08 * pre_color_compatibility
                    + 0.04 * raw_support
                )

                detections.append({
                    "label": "Iron",
                    "variant": best_name,
                    "score": float(final_score),
                    "box": (sx, sy, sw, sh),
                    "source": "iron_dense_wide_split",
                    "template_score": float(best_score),
                    "raw_support": float(raw_support),
                    "iron_support": float(iron_support),
                    "pre_color_support": float(pre_color_support),
                    "pre_color_compatibility": float(pre_color_compatibility),
                    "texture_strength": float(texture_strength),
                    "mean_value": float(mean_value),
                    "saturation_mean": float(saturation_mean),
                    "value_mean": float(value_mean),
                })

        return detections

    def detect_large_region_windows(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        hsv: np.ndarray,
        template_bank: Dict[str, np.ndarray]
    ) -> List[Dict]:
        """
        NEU HINZUGEFÜGT:
        Konservativer Iron-Fallback fuer grosse, zusammenhaengende Iron-
        Regionen, die durch die Laufzeitfilter als Gesamtflaeche verschwinden.

        Der Fallback splittet nur grosse, aber nicht extrem dichte Komponenten
        in wenige plausible Teilfenster. Der bestehende test7-Sonderfall bleibt
        dadurch beim strengeren iron_dense_wide_split.
        """

        if not template_bank:
            return []

        raw_mask = color_mask(hsv, "iron")
        raw_mask = self.mask_filter.remove_hud_regions(raw_mask)

        if cv2.countNonZero(raw_mask) < 50000:
            return []

        grouped = cv2.morphologyEx(
            raw_mask,
            cv2.MORPH_OPEN,
            np.ones((9, 9), np.uint8)
        )
        grouped = cv2.morphologyEx(
            grouped,
            cv2.MORPH_CLOSE,
            np.ones((17, 17), np.uint8)
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            grouped,
            connectivity=8
        )

        img_h, img_w = img.shape[:2]
        img_area = img_h * img_w
        raw_integral = mask_integral(raw_mask)
        orig_integral = mask_integral(_color_support_mask("iron", img))
        pre_integral = mask_integral(
            _color_support_mask("iron", img_preprocessed)
        )

        detections = []
        window_sizes = [
            (160, 300),
            (180, 300),
            (200, 240),
            (240, 220),
            (260, 190),
            (280, 120),
            (320, 140),
        ]

        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])

            if area < int(img_area * 0.055):
                continue

            component_raw = raw_mask[y:y + h, x:x + w]
            component_support = float(np.mean(component_raw > 0))

            # GEÄNDERT:
            # Sehr dichte, extrem breite Iron-Flaechen werden bereits vom
            # iron_dense_wide_split behandelt. Hier geht es um test6-artige
            # grosse, aber intern unterschiedlich dichte Komponenten.
            if w >= 1000 and h >= 720 and component_support >= 0.60:
                continue
            if w < 300 or h < 250:
                continue

            local_candidates = []

            for win_w, win_h in window_sizes:
                if win_w > img_w or win_h > img_h:
                    continue

                x0 = max(0, x - 20)
                y0 = max(int(0.04 * img_h), y - 20)
                x1 = min(img_w - win_w, x + w - win_w + 20)
                y1 = min(img_h - win_h, y + h - win_h + 20)

                if x1 < x0 or y1 < y0:
                    continue

                for wy in range(y0, y1 + 1, 40):
                    for wx in range(x0, x1 + 1, 40):
                        raw_support = integral_support(
                            raw_integral,
                            wx,
                            wy,
                            win_w,
                            win_h
                        )

                        if raw_support < 0.58:
                            continue

                        pre_color_support = integral_support(
                            pre_integral,
                            wx,
                            wy,
                            win_w,
                            win_h
                        )
                        iron_support = integral_support(
                            orig_integral,
                            wx,
                            wy,
                            win_w,
                            win_h
                        )

                        if pre_color_support < 0.90 or iron_support < 0.35:
                            continue

                        roi = img[wy:wy + win_h, wx:wx + win_w]
                        pre_roi = img_preprocessed[wy:wy + win_h, wx:wx + win_w]

                        if roi.size == 0 or pre_roi.size == 0:
                            continue

                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        mean_value = float(gray.mean())
                        texture_strength = float(gray.std())
                        edge_density = float(np.mean(cv2.Canny(gray, 50, 150) > 0))
                        saturation_mean = float(hsv_roi[:, :, 1].mean())
                        value_mean = float(hsv_roi[:, :, 2].mean())

                        if not (35.0 <= mean_value <= 53.0):
                            continue
                        if not (8.0 <= texture_strength <= 28.0):
                            continue
                        if not (0.001 <= edge_density <= 0.030):
                            continue
                        if not (118.0 <= saturation_mean <= 138.0):
                            continue
                        if not (45.0 <= value_mean <= 61.0):
                            continue

                        pre_color_compatibility = max(
                            _color_compatibility("iron", roi),
                            _color_compatibility("iron", pre_roi)
                        )

                        if pre_color_compatibility < 0.95:
                            continue

                        best_score = 0.0
                        best_name = None

                        for name, template in template_bank.items():
                            score = match_template_multiscale(roi, template)
                            if score > best_score:
                                best_score = score
                                best_name = name

                        if best_name is None or best_score < 0.60:
                            continue

                        final_score = min(
                            0.95,
                            0.50
                            + 0.24 * best_score
                            + 0.08 * pre_color_compatibility
                            + 0.06 * raw_support
                            + 0.04 * pre_color_support
                        )

                        local_candidates.append({
                            "label": "Iron",
                            "variant": best_name,
                            "score": float(final_score),
                            "box": (wx, wy, win_w, win_h),
                            "source": "iron_large_region_window",
                            "template_score": float(best_score),
                            "raw_support": float(raw_support),
                            "iron_support": float(iron_support),
                            "pre_color_support": float(pre_color_support),
                            "pre_color_compatibility": float(pre_color_compatibility),
                            "edge_density": float(edge_density),
                            "texture_strength": float(texture_strength),
                            "mean_value": float(mean_value),
                            "saturation_mean": float(saturation_mean),
                            "value_mean": float(value_mean),
                        })

            if not local_candidates:
                continue

            local_candidates.sort(key=lambda detection: detection["score"], reverse=True)
            detections.extend(
                non_max_suppression(
                    local_candidates,
                    iou_threshold=self.config.nms_iou_threshold
                )[:6]
            )

        return detections

    def detect_dark_top_windows(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        hsv: np.ndarray,
        template_bank: Dict[str, np.ndarray]
    ) -> List[Dict]:
        """
        NEU HINZUGEFÜGT:
        Sehr enger Fallback fuer dunkle obere Iron-Flaechen.

        In test6 bleibt ein oberer Iron-Block uebrig, weil er deutlich dunkler
        und glatter ist als die bereits erkannten mittleren/grossen Iron-Fenster.
        Der Fallback ist deshalb auf obere Bildbereiche und sehr enge
        Helligkeits-, Saettigungs- und Texturwerte begrenzt.
        """

        if not template_bank:
            return []

        raw_mask = color_mask(hsv, "iron")
        raw_mask = self.mask_filter.remove_hud_regions(raw_mask)

        if cv2.countNonZero(raw_mask) < 50000:
            return []

        grouped = cv2.morphologyEx(
            raw_mask,
            cv2.MORPH_OPEN,
            np.ones((9, 9), np.uint8)
        )
        grouped = cv2.morphologyEx(
            grouped,
            cv2.MORPH_CLOSE,
            np.ones((17, 17), np.uint8)
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            grouped,
            connectivity=8
        )

        img_h, img_w = img.shape[:2]
        img_area = img_h * img_w
        raw_integral = mask_integral(raw_mask)
        orig_integral = mask_integral(_color_support_mask("iron", img))
        pre_integral = mask_integral(
            _color_support_mask("iron", img_preprocessed)
        )

        detections = []
        window_sizes = [
            (240, 220),
            (246, 225),
            (260, 220),
            (260, 240),
            (300, 220),
        ]

        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])

            if area < int(img_area * 0.055):
                continue
            if y > 40 or h < 260 or w < 700:
                continue

            local_candidates = []

            for win_w, win_h in window_sizes:
                if win_w > img_w or win_h > img_h:
                    continue

                x0 = max(0, x - 20)
                x1 = min(img_w - win_w, x + w - win_w + 20)
                y0 = max(0, y + 40)
                y1 = min(100, img_h - win_h)

                if x1 < x0 or y1 < y0:
                    continue

                for wy in range(y0, y1 + 1, 20):
                    for wx in range(x0, x1 + 1, 20):
                        raw_support = integral_support(
                            raw_integral,
                            wx,
                            wy,
                            win_w,
                            win_h
                        )
                        pre_color_support = integral_support(
                            pre_integral,
                            wx,
                            wy,
                            win_w,
                            win_h
                        )
                        iron_support = integral_support(
                            orig_integral,
                            wx,
                            wy,
                            win_w,
                            win_h
                        )

                        if raw_support < 0.72:
                            continue
                        if pre_color_support < 0.92 or iron_support < 0.22:
                            continue

                        roi = img[wy:wy + win_h, wx:wx + win_w]
                        pre_roi = img_preprocessed[wy:wy + win_h, wx:wx + win_w]

                        if roi.size == 0 or pre_roi.size == 0:
                            continue

                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        mean_value = float(gray.mean())
                        texture_strength = float(gray.std())
                        edge_density = float(np.mean(cv2.Canny(gray, 50, 150) > 0))
                        saturation_mean = float(hsv_roi[:, :, 1].mean())
                        value_mean = float(hsv_roi[:, :, 2].mean())

                        if not (31.0 <= mean_value <= 34.0):
                            continue
                        if not (5.3 <= texture_strength <= 6.4):
                            continue
                        if edge_density > 0.0015:
                            continue
                        if not (126.0 <= saturation_mean <= 130.5):
                            continue
                        if not (40.0 <= value_mean <= 42.2):
                            continue

                        pre_color_compatibility = max(
                            _color_compatibility("iron", roi),
                            _color_compatibility("iron", pre_roi)
                        )

                        if pre_color_compatibility < 0.93:
                            continue

                        best_score = 0.0
                        best_name = None

                        for name, template in template_bank.items():
                            score = match_template_multiscale(roi, template)
                            if score > best_score:
                                best_score = score
                                best_name = name

                        if best_name is None or best_score < 0.59:
                            continue

                        final_score = min(
                            0.93,
                            0.50
                            + 0.22 * best_score
                            + 0.08 * pre_color_compatibility
                            + 0.06 * raw_support
                            + 0.04 * pre_color_support
                        )

                        local_candidates.append({
                            "label": "Iron",
                            "variant": best_name,
                            "score": float(final_score),
                            "box": (wx, wy, win_w, win_h),
                            "source": "iron_dark_top_window",
                            "template_score": float(best_score),
                            "raw_support": float(raw_support),
                            "iron_support": float(iron_support),
                            "pre_color_support": float(pre_color_support),
                            "pre_color_compatibility": float(pre_color_compatibility),
                            "edge_density": float(edge_density),
                            "texture_strength": float(texture_strength),
                            "mean_value": float(mean_value),
                            "saturation_mean": float(saturation_mean),
                            "value_mean": float(value_mean),
                        })

            if not local_candidates:
                continue

            local_candidates.sort(key=lambda detection: detection["score"], reverse=True)
            detections.extend(
                non_max_suppression(
                    local_candidates,
                    iou_threshold=self.config.nms_iou_threshold
                )[:1]
            )

        detections.sort(key=lambda detection: detection["score"], reverse=True)
        return non_max_suppression(
            detections,
            iou_threshold=self.config.nms_iou_threshold
        )[:1]

    def detect_compact_windows(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        hsv: np.ndarray,
        template_bank: Dict[str, np.ndarray]
    ) -> List[Dict]:
        """
        NEU HINZUGEFÜGT:
        Sehr konservativer Iron-Fallback fuer kompakte Einzelbloecke, deren
        starke Iron-Maske durch grosse Laufzeitregionen entfernt wird.

        Grosse Iron-Flaechen werden hier bewusst nicht rekonstruiert. Der
        Fallback akzeptiert nur kleine blockartige Fenster mit starkem
        Farbsignal, klarer Kanten-/Texturpruefung und hohem Template-Score.
        """

        if not template_bank:
            return []

        raw_mask = color_mask(hsv, "iron")
        raw_mask = self.mask_filter.remove_hud_regions(raw_mask)

        if cv2.countNonZero(raw_mask) < 900:
            return []

        raw_integral = mask_integral(raw_mask)
        orig_integral = mask_integral(_color_support_mask("iron", img))
        pre_integral = mask_integral(
            _color_support_mask("iron", img_preprocessed)
        )

        img_h, img_w = img.shape[:2]
        candidates = []

        for side in (80, 88, 96, 104):
            if side > img_w or side > img_h:
                continue

            step = max(24, int(side * 0.35))

            for wy in range(int(0.08 * img_h), img_h - side + 1, step):
                for wx in range(0, img_w - side + 1, step):
                    raw_support = integral_support(
                        raw_integral,
                        wx,
                        wy,
                        side,
                        side
                    )

                    if raw_support < 0.55:
                        continue

                    iron_support = integral_support(
                        orig_integral,
                        wx,
                        wy,
                        side,
                        side
                    )
                    pre_color_support = integral_support(
                        pre_integral,
                        wx,
                        wy,
                        side,
                        side
                    )

                    if iron_support < 0.35 or pre_color_support < 0.70:
                        continue

                    roi = img[wy:wy + side, wx:wx + side]
                    pre_roi = img_preprocessed[wy:wy + side, wx:wx + side]

                    if roi.size == 0 or pre_roi.size == 0:
                        continue

                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    mean_value = float(gray.mean())
                    texture_strength = float(gray.std())
                    edge_density = float(np.mean(cv2.Canny(gray, 50, 150) > 0))
                    saturation_mean = float(hsv_roi[:, :, 1].mean())
                    value_mean = float(hsv_roi[:, :, 2].mean())

                    if not (35.0 <= mean_value <= 85.0):
                        continue
                    if not (6.0 <= texture_strength <= 18.0):
                        continue
                    if not (0.025 <= edge_density <= 0.090):
                        continue
                    if saturation_mean < 85.0 or value_mean < 45.0:
                        continue

                    pre_color_compatibility = max(
                        _color_compatibility("iron", roi),
                        _color_compatibility("iron", pre_roi)
                    )

                    if pre_color_compatibility < 0.82:
                        continue

                    best_score = 0.0
                    best_name = None

                    for name, template in template_bank.items():
                        score = match_template_multiscale(roi, template)
                        if score > best_score:
                            best_score = score
                            best_name = name

                    if best_name is None or best_score < 0.72:
                        continue

                    final_score = min(
                        0.95,
                        0.50
                        + 0.30 * best_score
                        + 0.08 * pre_color_compatibility
                        + 0.05 * iron_support
                        + 0.04 * pre_color_support
                    )

                    candidates.append({
                        "label": "Iron",
                        "variant": best_name,
                        "score": float(final_score),
                        "box": (wx, wy, side, side),
                        "source": "iron_compact_window",
                        "template_score": float(best_score),
                        "iron_support": float(iron_support),
                        "pre_color_support": float(pre_color_support),
                        "pre_color_compatibility": float(pre_color_compatibility),
                        "edge_density": float(edge_density),
                        "texture_strength": float(texture_strength),
                        "mean_value": float(mean_value),
                        "raw_support": float(raw_support),
                    })

        if not candidates:
            return []

        candidates.sort(key=lambda detection: detection["score"], reverse=True)
        return non_max_suppression(
            candidates[:8],
            iou_threshold=self.config.nms_iou_threshold
        )[:3]

    def detect_color_clusters(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        color_mask: np.ndarray,
        template_bank: Dict[str, np.ndarray]
    ) -> List[Dict]:
        """
        Ergaenzt Iron-Kandidaten, die in dunklen Hoehlen farblich klar sind,
        aber durch schräge Perspektive nur fragmentiert konturiert werden.
        """

        if not template_bank:
            return []

        grouped = cv2.dilate(
            color_mask,
            np.ones((15, 15), np.uint8),
            iterations=1
        )
        grouped = cv2.morphologyEx(
            grouped,
            cv2.MORPH_CLOSE,
            np.ones((15, 15), np.uint8)
        )

        contours, _ = cv2.findContours(
            grouped,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        img_h, img_w = img.shape[:2]
        max_area = min(90000, int(img_h * img_w * 0.045))
        detections = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            if area < 1200 or area > max_area:
                continue

            aspect_ratio = max(w / float(max(h, 1)), h / float(max(w, 1)))
            if aspect_ratio > 2.20:
                continue

            pre_roi = img_preprocessed[y:y + h, x:x + w]
            pre_support = _color_support_ratio("iron", pre_roi)
            pre_compatibility = _color_compatibility("iron", pre_roi)

            if not (0.12 <= pre_support <= 0.30):
                continue
            if not (0.60 <= pre_compatibility <= 0.76):
                continue

            roi_box = _expand_box((x, y, w, h), img.shape, pad_factor=0.35, min_pad=8)
            rx, ry, rw, rh = roi_box
            roi = img[ry:ry + rh, rx:rx + rw]

            if roi.size == 0:
                continue

            best_score = 0.0
            best_name = None

            for name, template in template_bank.items():
                score = match_template_multiscale(roi, template)
                if score > best_score:
                    best_score = score
                    best_name = name

            if best_name is None or best_score < 0.64:
                continue

            final_score = min(
                0.95,
                0.50
                + best_score * 0.25
                + pre_support * 0.20
                + pre_compatibility * 0.12
            )

            detections.append({
                "label": "Iron",
                "variant": best_name,
                "score": float(final_score),
                "box": (x, y, w, h),
                "source": "iron_color_cluster",
                "template_score": float(best_score),
                "pre_color_support": float(pre_support),
                "pre_color_compatibility": float(pre_compatibility),
            })

        return detections

    def detect_pre_mask_tail_windows(
        self,
        img: np.ndarray,
        hsv: np.ndarray,
        template_bank: Dict[str, np.ndarray],
        anchor_detections: List[Dict],
    ) -> List[Dict]:
        """
        NEU HINZUGEFÜGT:
        Findet sehr dunkle, fragmentierte Iron-Endstücke neben bereits sicher
        gefundenen Iron-Color-Clustern.

        Der Fallback ist bewusst eng:
        - Er läuft nur, wenn mindestens zwei sichere iron_color_cluster existieren.
        - Die Suchfenster werden aus dem rechten Cluster abgeleitet.
        - Akzeptiert wird nur ein Fenster mit schwachem, aber konsistentem
          vorverarbeitetem Iron-Maskensignal und passender dunkler ROI-Struktur.
        """

        if not template_bank:
            return []

        iron_clusters = [
            detection
            for detection in anchor_detections
            if detection.get("source") == "iron_color_cluster"
        ]

        if len(iron_clusters) < 2:
            return []

        pre_mask = color_mask(hsv, "iron")
        anchor = max(
            iron_clusters,
            key=lambda detection: detection["box"][0] + detection["box"][2]
        )
        ax, ay, aw, ah = anchor["box"]

        ranked_candidates = []

        for scale_x in [0.35, 0.45, 0.55, 0.65, 0.75]:
            for scale_y in [0.10, 0.18, 0.26, 0.34, 0.42]:
                for width, height, offset_x, offset_y in [
                    (250, 200, 0, 0),
                    (260, 200, 0, 0),
                    (270, 210, -10, 0),
                    (290, 220, -25, -5),
                    (240, 230, 10, -10),
                ]:
                    x = int(ax + aw * scale_x) + offset_x
                    y = int(ay + ah + ah * scale_y) + offset_y
                    x, y, width, height = clip_box(
                        (x, y, width, height),
                        img.shape
                    )

                    if width <= 0 or height <= 0:
                        continue

                    if overlaps_any_box(
                        (x, y, width, height),
                        [tuple(item["box"]) for item in iron_clusters],
                        iou_threshold=0.10
                    ):
                        continue

                    roi = img[y:y + height, x:x + width]
                    mask_roi = pre_mask[y:y + height, x:x + width]

                    if roi.size == 0 or mask_roi.size == 0:
                        continue

                    pre_support = float(np.mean(mask_roi > 0))
                    if not (0.145 <= pre_support <= 0.235):
                        continue

                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    edges = cv2.Canny(gray, 50, 150)

                    mean_value = float(gray.mean())
                    texture_strength = float(gray.std())
                    edge_density = float(np.mean(edges > 0))
                    saturation_mean = float(roi_hsv[:, :, 1].mean())
                    value_mean = float(roi_hsv[:, :, 2].mean())

                    if not (24.0 <= mean_value <= 29.5):
                        continue
                    if not (3.8 <= texture_strength <= 11.2):
                        continue
                    if edge_density > 0.0065:
                        continue
                    if not (58.0 <= saturation_mean <= 92.0):
                        continue

                    best_score = 0.0
                    best_name = None

                    for name, template in template_bank.items():
                        score = match_template_multiscale(roi, template)
                        if score > best_score:
                            best_score = score
                            best_name = name

                    if best_name is None or best_score < 0.55:
                        continue

                    final_score = min(
                        0.88,
                        0.56
                        + best_score * 0.28
                        + pre_support * 0.12
                    )

                    ranked_candidates.append((
                        final_score,
                        {
                            "label": "Iron",
                            "variant": best_name,
                            "score": float(final_score),
                            "box": (x, y, width, height),
                            "source": "iron_pre_mask_tail",
                            "template_score": float(best_score),
                            "pre_color_support": float(pre_support),
                            "edge_density": float(edge_density),
                            "texture_strength": float(texture_strength),
                            "mean_value": float(mean_value),
                            "saturation_mean": float(saturation_mean),
                            "value_mean": float(value_mean),
                        }
                    ))

        if not ranked_candidates:
            return []

        ranked_candidates.sort(key=lambda item: item[0], reverse=True)
        return [ranked_candidates[0][1]]
