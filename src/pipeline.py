# -*- coding: utf-8 -*-
"""
Zentrale Ore-Detection-Pipeline.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from candidate_filters import (
    CoalDetector,
    DiamondCandidateExpander,
)
from config import OreDetectorConfig
from detection import (
    _color_compatibility,
    _copper_green_support,
    _copper_orange_support,
    _expand_box,
    _color_support_mask,
    _color_support_ratio,
    detect_with_template_bank,
    find_candidates,
    match_template_multiscale,
    non_max_suppression,
)
from mask_filters import MaskRegionFilter
from morphology import clean_mask
from preprocessing import match_scene_brightness, to_hsv
from segmentation import (
    color_mask,
    edge_mask,
    hybrid_mask,
    refine_mask_for_ore,
    supported_ores,
    use_edges_for_ore,
)
from template_repository import TemplateRepository
from visualization import draw, draw_debug

Box = Tuple[int, int, int, int]


@dataclass
class OreDetectionResult:
    """
    Ergebnisdaten einer Pipeline-Ausfuehrung.
    """

    image: np.ndarray
    detections: List[Dict]
    candidates: List[Box]


class DebugMaskWriter:
    """
    Speichert Zwischenmasken, wenn Debug-Ausgabe aktiviert ist.
    """

    def __init__(self, output_dir: str, enabled: bool):
        self.output_dir = output_dir
        self.enabled = enabled

    def save(self, name: str, mask: np.ndarray) -> None:
        if not self.enabled:
            return

        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"{name}.png")
        cv2.imwrite(path, mask)


class OreDetector:
    """
    Orchestriert Preprocessing, Segmentierung, Kandidatenbildung und Matching.
    """

    def __init__(self, config: Optional[OreDetectorConfig] = None):
        self.config = config or OreDetectorConfig()
        self.mask_filter = MaskRegionFilter()
        self.template_repository = TemplateRepository(self.config.templates_dir)
        self.debug_masks = DebugMaskWriter(
            self.config.debug_mask_dir,
            self.config.save_debug_masks
        )
        self.coal_detector = CoalDetector(self.mask_filter)
        self.diamond_expander = DiamondCandidateExpander()

    def run(self, img: np.ndarray) -> np.ndarray:
        """
        Fuehrt die Pipeline aus und gibt das annotierte Bild zurueck.
        """

        result = self.detect(img)

        if self.config.debug:
            return draw_debug(img, result.candidates, result.detections)

        return draw(img, result.detections)

    def detect(self, img: np.ndarray) -> OreDetectionResult:
        """
        Fuehrt die Pipeline aus und gibt strukturierte Zwischenergebnisse zurueck.
        """

        img_preprocessed = match_scene_brightness(img)
        hsv = to_hsv(img_preprocessed)

        edges = edge_mask(img_preprocessed)
        edges = self.mask_filter.clean_runtime_mask(edges, hsv)
        self.debug_masks.save("00_edges_cleaned", edges)

        all_raw_detections = []
        all_candidates: List[Box] = []

        for ore in supported_ores():
            color = color_mask(hsv, ore)
            color = self.mask_filter.clean_runtime_mask(color, hsv, ore=ore)
            self.debug_masks.save(f"01_color_{ore}", color)

            mask = hybrid_mask(color, edges) if use_edges_for_ore(ore) else color
            mask = refine_mask_for_ore(ore, mask)
            mask = clean_mask(mask)
            mask = self.mask_filter.clean_runtime_mask(mask, hsv, ore=ore)
            self.debug_masks.save(f"02_mask_{ore}", mask)

            if ore == "coal":
                candidates = self.coal_detector.find_candidates(img, color)
                all_candidates.extend(candidates)

                if candidates:
                    all_raw_detections.extend(
                        self.coal_detector.detect_direct(img, candidates)
                    )
                continue

            candidates = find_candidates(mask, color, ore=ore)

            if ore == "diamond":
                candidates = self.diamond_expander.expand(candidates, img.shape)

            template_bank = self.template_repository.get_for_ore(ore)
            if not template_bank:
                all_candidates.extend(candidates)
                continue

            all_candidates.extend(candidates)

            if not candidates:
                continue

            raw = detect_with_template_bank(
                img,
                candidates,
                template_bank,
                label=self._ore_label(ore),
                threshold=self.config.ore_match_thresholds.get(ore, 0.8),
                brightness_split=None
            )
            all_raw_detections.extend(raw)

            if ore == "copper":
                all_raw_detections.extend(
                    self._detect_copper_edge_clusters(
                        img,
                        img_preprocessed,
                        edges,
                        template_bank
                    )
                )

            if ore == "iron":
                all_raw_detections.extend(
                    self._detect_iron_color_clusters(
                        img,
                        img_preprocessed,
                        color,
                        template_bank
                    )
                )

        detections = non_max_suppression(
            all_raw_detections,
            iou_threshold=self.config.nms_iou_threshold
        )
        detections = self._filter_low_confidence_outputs(detections, img)
        detections = self._merge_close_diamond_detections(detections)
        detections = self._expand_small_diamond_cluster_boxes(detections, img)

        return OreDetectionResult(
            image=img,
            detections=detections,
            candidates=all_candidates
        )

    def _ore_label(self, ore_key: str) -> str:
        return ore_key.capitalize()

    def _filter_low_confidence_outputs(self, detections: List[Dict], img: np.ndarray) -> List[Dict]:
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
            if not self._passes_roi_plausibility(detection, img):
                continue

            filtered.append(detection)

        return filtered

    def _passes_roi_plausibility(self, detection: Dict, img: np.ndarray) -> bool:
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

            return (
                detection.get("score", 0.0) >= 0.62
                and _color_support_ratio("copper", roi) >= 0.045
                and _color_compatibility("copper", roi) >= 0.72
                and _copper_orange_support(roi) >= 0.03
            )

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

            return normal_iron_case or dark_angled_iron_case

        if label == "gold":
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

    def _detect_copper_edge_clusters(
        self,
        img: np.ndarray,
        img_preprocessed: np.ndarray,
        edges: np.ndarray,
        template_bank: Dict[str, np.ndarray]
    ) -> List[Dict]:
        """
        Ergaenzt Copper-Kandidaten in warmen Hoehlen, wo die Farbflaeche zu
        gross wird und deshalb nur die lokale Kantenkomponente brauchbar ist.
        """

        if not template_bank:
            return []

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            if area < 4500 or area > 18000:
                continue

            aspect_ratio = max(w / float(max(h, 1)), h / float(max(w, 1)))
            if aspect_ratio > 1.75:
                continue

            edge_roi = edges[y:y + h, x:x + w]
            edge_density = cv2.countNonZero(edge_roi) / float(area)
            if edge_density < 0.10:
                continue

            roi = img[y:y + h, x:x + w]
            pre_roi = img_preprocessed[y:y + h, x:x + w]

            copper_support = max(
                _color_support_ratio("copper", roi),
                _color_support_ratio("copper", pre_roi)
            )
            copper_compatibility = max(
                _color_compatibility("copper", roi),
                _color_compatibility("copper", pre_roi)
            )
            copper_orange = max(
                _copper_orange_support(roi),
                _copper_orange_support(pre_roi)
            )
            copper_green = max(
                _copper_green_support(roi),
                _copper_green_support(pre_roi)
            )

            if not (0.72 <= copper_support <= 0.94):
                continue
            if copper_compatibility < 0.84:
                continue
            if copper_orange < 0.72 or copper_green > 0.08:
                continue

            pad = int(max(w, h) * 0.60)
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(img.shape[1], x + w + pad)
            y1 = min(img.shape[0], y + h + pad)
            match_roi = img[y0:y1, x0:x1]

            if match_roi.size == 0:
                continue

            best_score = 0.0
            best_name = None

            for name, template in template_bank.items():
                score = match_template_multiscale(match_roi, template)
                if score > best_score:
                    best_score = score
                    best_name = name

            if best_name is None or best_score < 0.54:
                continue

            detections.append({
                "label": "Copper",
                "variant": best_name,
                "score": float(
                    0.50
                    + best_score * 0.20
                    + copper_support * 0.08
                    + copper_compatibility * 0.06
                ),
                "box": (x, y, w, h),
                "source": "copper_edge_cluster",
                "template_score": float(best_score),
                "copper_support": float(copper_support),
                "copper_compatibility": float(copper_compatibility),
                "copper_orange": float(copper_orange),
                "copper_green": float(copper_green),
                "edge_density": float(edge_density),
            })

        return detections

    def _detect_iron_color_clusters(
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

    def _merge_close_diamond_detections(self, detections: List[Dict]) -> List[Dict]:
        diamonds = [d for d in detections if d["label"].lower() == "diamond"]
        others = [d for d in detections if d["label"].lower() != "diamond"]

        if len(diamonds) <= 1:
            return detections

        work = [dict(d) for d in diamonds]
        changed = True

        while changed:
            changed = False
            merged = []
            used = [False] * len(work)

            for i, det in enumerate(work):
                if used[i]:
                    continue

                current = dict(det)
                used[i] = True

                for j in range(i + 1, len(work)):
                    if used[j]:
                        continue

                    if self._boxes_close(current["box"], work[j]["box"], gap=45):
                        current = self._merge_detection_pair(current, work[j])
                        used[j] = True
                        changed = True

                merged.append(current)

            work = merged

        return others + work

    def _boxes_close(self, box_a: Box, box_b: Box, gap: int) -> bool:
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b

        return not (
            ax + aw + gap < bx
            or bx + bw + gap < ax
            or ay + ah + gap < by
            or by + bh + gap < ay
        )

    def _merge_detection_pair(self, det_a: Dict, det_b: Dict) -> Dict:
        ax, ay, aw, ah = det_a["box"]
        bx, by, bw, bh = det_b["box"]

        x1 = min(ax, bx)
        y1 = min(ay, by)
        x2 = max(ax + aw, bx + bw)
        y2 = max(ay + ah, by + bh)

        keep = det_a if det_a.get("score", 0.0) >= det_b.get("score", 0.0) else det_b
        merged = dict(keep)
        merged["box"] = (x1, y1, x2 - x1, y2 - y1)
        merged["score"] = max(det_a.get("score", 0.0), det_b.get("score", 0.0))
        return merged

    def _expand_small_diamond_cluster_boxes(self, detections: List[Dict], img: np.ndarray) -> List[Dict]:
        expanded = []

        for detection in detections:
            if detection["label"].lower() != "diamond":
                expanded.append(detection)
                continue

            x, y, w, h = detection["box"]
            if max(w, h) > 140:
                expanded.append(detection)
                continue

            new_box = self._diamond_color_cluster_box((x, y, w, h), img)
            if new_box is None:
                expanded.append(detection)
                continue

            updated = dict(detection)
            updated["box"] = new_box
            expanded.append(updated)

        return expanded

    def _diamond_color_cluster_box(self, box: Box, img: np.ndarray) -> Optional[Box]:
        x, y, w, h = box
        img_h, img_w = img.shape[:2]
        pad = 140

        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(img_w, x + w + pad)
        y1 = min(img_h, y + h + pad)

        local_img = img[y0:y1, x0:x1]
        if local_img.size == 0:
            return None

        local_color = _color_support_mask("diamond", local_img)
        if cv2.countNonZero(local_color) < 80:
            return None

        grouped = cv2.dilate(local_color, np.ones((25, 25), np.uint8), iterations=2)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)

        center_x = x + w / 2.0 - x0
        center_y = y + h / 2.0 - y0

        for i in range(1, num_labels):
            lx = int(stats[i, cv2.CC_STAT_LEFT])
            ly = int(stats[i, cv2.CC_STAT_TOP])
            lw = int(stats[i, cv2.CC_STAT_WIDTH])
            lh = int(stats[i, cv2.CC_STAT_HEIGHT])

            if not (lx <= center_x <= lx + lw and ly <= center_y <= ly + lh):
                continue

            if lw < w or lh < h:
                return None
            if lw > 320 or lh > 260:
                return None
            if max(lw / float(max(lh, 1)), lh / float(max(lw, 1))) > 1.80:
                return None

            return x0 + lx, y0 + ly, lw, lh

        return None
