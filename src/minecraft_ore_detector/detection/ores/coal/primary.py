# -*- coding: utf-8 -*-
"""Primäre Coal-Kandidaten- und Template-Erkennung."""

from typing import Dict, List, Optional

import cv2
import numpy as np

from minecraft_ore_detector.detection import match_template_multiscale
from minecraft_ore_detector.imaging.runtime_mask_filter import RuntimeMaskFilter
from minecraft_ore_detector.models import Box


class CoalPrimaryDetector:
    """
    Fuehrt die primaere Coal-Kandidatensuche und -Erkennung aus.
    """

    def __init__(self, mask_filter: RuntimeMaskFilter):
        self.mask_filter = mask_filter

    def find_candidates(self, img: np.ndarray, coal_mask: np.ndarray) -> List[Box]:
        img_h, img_w = img.shape[:2]
        gray, hsv, coal_seed, local_dark, grouped, block_size = self._build_coal_seed(
            img,
            coal_mask
        )
        candidates = []

        contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w < 8 or h < 8:
                continue
            if w > block_size * 1.65 or h > block_size * 1.65:
                continue
            if y < int(0.12 * img_h):
                continue

            cx = x + w / 2.0
            side = int(max(w, h) * 1.05)
            side = max(int(block_size * 0.82), side)
            side = min(int(block_size * 0.98), side)

            new_x = int(cx - side * 0.50)
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
            edge_density = self._edge_density(roi_gray)
            colored_ratio = self._colored_ore_ratio_for_coal_reject(roi_bgr)

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
            if edge_density < 0.045:
                continue

            candidates.append((new_x, new_y, new_w, new_h))

        return self._merge_candidate_boxes(candidates, iou_threshold=0.35)

    def detect_template_fallback(
        self,
        img: np.ndarray,
        coal_mask: np.ndarray,
        template_bank: Dict[str, np.ndarray],
    ) -> List[dict]:
        """
        NEU HINZUGEFÜGT:
        Sehr konservativer Coal-Fallback fuer Faelle, in denen die direkte
        Kohle-Erkennung keine Kandidaten liefert.

        Der Fallback tastet nur bereits dunkle Seed-Komponenten blockweise ab
        und akzeptiert pro Komponente nur das beste Fenster. Dadurch werden
        grosse Schattenflaechen nicht in viele Coal-False-Positives zerlegt.
        """

        if not template_bank:
            return []

        img_h, img_w = img.shape[:2]
        gray, hsv, coal_seed, local_dark, grouped, block_size = self._build_coal_seed(
            img,
            coal_mask
        )

        side = int(block_size * 0.98)
        stride = max(24, int(side * 0.45))
        detections = []

        contours, _ = cv2.findContours(
            grouped,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w < 8 or h < 8:
                continue
            if y < int(0.12 * img_h):
                continue

            too_large = w > block_size * 1.65 or h > block_size * 1.65
            windows = []

            if too_large:
                y_start = max(y, int(0.12 * img_h))
                y_stop = min(y + h, img_h - side)
                x_stop = min(x + w, img_w - side)

                for wy in range(y_start, y_stop + 1, stride):
                    for wx in range(x, x_stop + 1, stride):
                        component_crop = grouped[wy:wy + side, wx:wx + side]

                        if component_crop.size == 0:
                            continue

                        component_support = (
                            cv2.countNonZero(component_crop)
                            / float(component_crop.size)
                        )

                        if component_support >= 0.16:
                            windows.append((wx, wy, side, side))
            else:
                cx = x + w / 2.0
                local_side = int(max(w, h) * 1.05)
                local_side = max(int(block_size * 0.82), local_side)
                local_side = min(int(block_size * 0.98), local_side)
                wx = int(cx - local_side * 0.50)
                wy = int(y + h * 0.08)
                wx = max(0, min(wx, img_w - local_side))
                wy = max(0, min(wy, img_h - local_side))
                windows.append((wx, wy, local_side, local_side))

            best_detection = None

            for window in windows:
                detection = self._evaluate_template_fallback_candidate(
                    img,
                    gray,
                    hsv,
                    coal_seed,
                    local_dark,
                    template_bank,
                    window,
                )

                if detection is None:
                    continue

                if (
                    best_detection is None
                    or detection["score"] > best_detection["score"]
                ):
                    best_detection = detection

            if best_detection is not None:
                detections.append(best_detection)

        return detections

    def detect_from_candidates(self, img: np.ndarray, candidates: List[Box]) -> List[dict]:
        detections = []
        img_h, _ = img.shape[:2]
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

            colored_ratio = self._colored_ore_ratio_for_coal_reject(roi_bgr)
            if colored_ratio > 0.080:
                continue

            low_sat_ratio = float(np.mean(roi_hsv[:, :, 1] < 120))
            very_dark_ratio = float(np.mean(roi_gray < 85))
            dark_ratio = float(np.mean(roi_gray < 145))
            texture_strength = float(np.std(roi_gray))
            edge_density = self._edge_density(roi_gray)

            if low_sat_ratio < 0.70:
                continue
            if very_dark_ratio < 0.006 and dark_ratio < 0.070:
                continue
            if texture_strength < 5.5:
                continue
            if edge_density < 0.045:
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

    def _build_coal_seed(self, img: np.ndarray, coal_mask: np.ndarray):
        """
        GEÄNDERT:
        Gemeinsamer Seed-Aufbau fuer direkte Coal-Erkennung und Fallback.
        """

        img_h, img_w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        block_size = int(img_w * 0.055)
        block_size = max(70, min(block_size, 105))

        strict_seed = self.mask_filter.remove_hud_regions(coal_mask.copy())
        low_saturation = cv2.inRange(hsv[:, :, 1], 0, 110)
        dark_gray = cv2.inRange(gray, 0, 170)
        strict_seed = cv2.bitwise_and(strict_seed, low_saturation)
        strict_seed = cv2.bitwise_and(strict_seed, dark_gray)

        local_mean = cv2.GaussianBlur(gray, (41, 41), 0)
        local_dark = np.zeros_like(gray, dtype=np.uint8)
        local_dark[
            ((local_mean.astype(np.int16) - gray.astype(np.int16)) > 13)
            & (gray < 175)
            & (hsv[:, :, 1] < 110)
        ] = 255
        local_dark = self.mask_filter.remove_hud_regions(local_dark)

        strict_seed[:int(0.12 * img_h), :] = 0
        local_dark[:int(0.12 * img_h), :] = 0

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

        return gray, hsv, coal_seed, local_dark, grouped, block_size

    def _evaluate_template_fallback_candidate(
        self,
        img: np.ndarray,
        gray: np.ndarray,
        hsv: np.ndarray,
        coal_seed: np.ndarray,
        local_dark: np.ndarray,
        template_bank: Dict[str, np.ndarray],
        window: Box,
    ) -> Optional[dict]:
        x, y, w, h = window

        if w <= 0 or h <= 0:
            return None

        roi_gray = gray[y:y + h, x:x + w]
        roi_hsv = hsv[y:y + h, x:x + w]
        roi_bgr = img[y:y + h, x:x + w]
        roi_seed = coal_seed[y:y + h, x:x + w]
        roi_local_dark = local_dark[y:y + h, x:x + w]

        if (
            roi_gray.size == 0
            or roi_hsv.size == 0
            or roi_bgr.size == 0
            or roi_seed.size == 0
        ):
            return None

        area = float(w * h)
        coal_support = cv2.countNonZero(roi_seed) / area
        local_dark_support = cv2.countNonZero(roi_local_dark) / area
        low_sat_ratio = float(np.mean(roi_hsv[:, :, 1] < 120))
        texture_strength = float(np.std(roi_gray))
        edge_density = self._edge_density(roi_gray)
        colored_ratio = self._colored_ore_ratio_for_coal_reject(roi_bgr)

        if colored_ratio > 0.080:
            return None
        if coal_support < 0.010 and local_dark_support < 0.018:
            return None
        if low_sat_ratio < 0.70:
            return None
        if texture_strength < 12.0:
            return None
        if edge_density < 0.050:
            return None

        best_score = 0.0
        best_name = None

        for name, template in template_bank.items():
            score = match_template_multiscale(roi_bgr, template)

            if score > best_score:
                best_score = score
                best_name = name

        # GEÄNDERT:
        # Der Fallback darf nur sehr sichere Coal-Template-Treffer ergaenzen.
        # Niedrigere Werte erzeugten in den Review-Bildern Schatten-FPs.
        if best_name is None or best_score < 0.80:
            return None

        final_score = min(
            0.99,
            0.55
            + best_score * 0.30
            + min(0.10, texture_strength / 140.0)
            + min(0.05, edge_density)
        )

        return {
            "label": "Coal",
            "variant": best_name,
            "score": float(final_score),
            "box": (x, y, w, h),
            "source": "coal_template_fallback",
            "template_score": float(best_score),
            "coal_support": float(coal_support),
            "local_dark_support": float(local_dark_support),
            "edge_density": float(edge_density),
        }

    def _edge_density(self, roi_gray: np.ndarray) -> float:
        if roi_gray.size == 0:
            return 0.0

        edges = cv2.Canny(roi_gray, 50, 150)
        return float(np.mean(edges > 0))

    def _colored_ore_ratio_for_coal_reject(self, roi_bgr: np.ndarray) -> float:
        if roi_bgr.size == 0:
            return 0.0

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        colored_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        colored_ranges = [
            ([75, 25, 18], [112, 255, 255]),
            ([45, 25, 18], [85, 255, 255]),
            ([5, 45, 30], [30, 255, 255]),
            ([15, 40, 40], [42, 255, 255]),
            ([95, 45, 30], [135, 255, 255]),
            ([0, 55, 35], [10, 255, 255]),
            ([165, 55, 35], [179, 255, 255]),
        ]

        for lower, upper in colored_ranges:
            lo = np.array(lower, dtype=np.uint8)
            hi = np.array(upper, dtype=np.uint8)
            colored_mask = cv2.bitwise_or(colored_mask, cv2.inRange(hsv, lo, hi))

        return cv2.countNonZero(colored_mask) / float(colored_mask.shape[0] * colored_mask.shape[1])

    def _merge_candidate_boxes(self, candidates: List[Box], iou_threshold: float = 0.35) -> List[Box]:
        if len(candidates) == 0:
            return []

        result = []

        for box in candidates:
            keep = True
            for existing in result:
                if self._candidate_iou(box, existing) > iou_threshold:
                    keep = False
                    break
            if keep:
                result.append(box)

        return result

    def _candidate_iou(self, box_a: Box, box_b: Box) -> float:
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
        union = aw * ah + bw * bh - inter_area

        if union <= 0:
            return 0.0

        return inter_area / float(union)
