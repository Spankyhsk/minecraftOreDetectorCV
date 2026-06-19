# -*- coding: utf-8 -*-
"""Diamond-spezifische Nachbearbeitung finaler Detektionen."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from detection import _color_support_mask

Box = Tuple[int, int, int, int]


class DiamondPostprocessor:
    """Fuehrt nahe Diamond-Treffer zusammen und passt kleine Clusterboxen an."""

    def postprocess(
        self,
        detections: List[Dict],
        img: np.ndarray,
    ) -> List[Dict]:
        detections = self._merge_close_detections(detections)
        return self._expand_small_cluster_boxes(detections, img)

    def _merge_close_detections(
        self,
        detections: List[Dict],
    ) -> List[Dict]:
        diamonds = [
            detection
            for detection in detections
            if detection["label"].lower() == "diamond"
        ]
        others = [
            detection
            for detection in detections
            if detection["label"].lower() != "diamond"
        ]

        if len(diamonds) <= 1:
            return detections

        work = [dict(detection) for detection in diamonds]
        changed = True

        while changed:
            changed = False
            merged = []
            used = [False] * len(work)

            for index, detection in enumerate(work):
                if used[index]:
                    continue

                current = dict(detection)
                used[index] = True

                for other_index in range(index + 1, len(work)):
                    if used[other_index]:
                        continue

                    if self._boxes_are_close(
                        current["box"],
                        work[other_index]["box"],
                        gap=45,
                    ):
                        current = self._merge_pair(
                            current,
                            work[other_index],
                        )
                        used[other_index] = True
                        changed = True

                merged.append(current)

            work = merged

        return others + work

    @staticmethod
    def _boxes_are_close(box_a: Box, box_b: Box, gap: int) -> bool:
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b

        return not (
            ax + aw + gap < bx
            or bx + bw + gap < ax
            or ay + ah + gap < by
            or by + bh + gap < ay
        )

    @staticmethod
    def _merge_pair(det_a: Dict, det_b: Dict) -> Dict:
        ax, ay, aw, ah = det_a["box"]
        bx, by, bw, bh = det_b["box"]

        x1 = min(ax, bx)
        y1 = min(ay, by)
        x2 = max(ax + aw, bx + bw)
        y2 = max(ay + ah, by + bh)

        keep = (
            det_a
            if det_a.get("score", 0.0) >= det_b.get("score", 0.0)
            else det_b
        )
        merged = dict(keep)
        merged["box"] = (x1, y1, x2 - x1, y2 - y1)
        merged["score"] = max(
            det_a.get("score", 0.0),
            det_b.get("score", 0.0),
        )
        return merged

    def _expand_small_cluster_boxes(
        self,
        detections: List[Dict],
        img: np.ndarray,
    ) -> List[Dict]:
        expanded = []

        for detection in detections:
            if detection["label"].lower() != "diamond":
                expanded.append(detection)
                continue

            box = tuple(detection["box"])
            if max(box[2], box[3]) > 140:
                expanded.append(detection)
                continue

            new_box = self._find_color_cluster_box(box, img)
            if new_box is None:
                expanded.append(detection)
                continue

            updated = dict(detection)
            updated["box"] = new_box
            expanded.append(updated)

        return expanded

    @staticmethod
    def _find_color_cluster_box(
        box: Box,
        img: np.ndarray,
    ) -> Optional[Box]:
        x, y, width, height = box
        img_h, img_w = img.shape[:2]
        padding = 140

        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(img_w, x + width + padding)
        y1 = min(img_h, y + height + padding)

        local_img = img[y0:y1, x0:x1]
        if local_img.size == 0:
            return None

        local_color = _color_support_mask("diamond", local_img)
        if cv2.countNonZero(local_color) < 80:
            return None

        grouped = cv2.dilate(
            local_color,
            np.ones((25, 25), np.uint8),
            iterations=2,
        )
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            grouped,
            connectivity=8,
        )

        center_x = x + width / 2.0 - x0
        center_y = y + height / 2.0 - y0

        for label_index in range(1, num_labels):
            local_x = int(stats[label_index, cv2.CC_STAT_LEFT])
            local_y = int(stats[label_index, cv2.CC_STAT_TOP])
            local_width = int(stats[label_index, cv2.CC_STAT_WIDTH])
            local_height = int(stats[label_index, cv2.CC_STAT_HEIGHT])

            if not (
                local_x <= center_x <= local_x + local_width
                and local_y <= center_y <= local_y + local_height
            ):
                continue

            if local_width < width or local_height < height:
                return None
            if local_width > 320 or local_height > 260:
                return None
            if max(
                local_width / float(max(local_height, 1)),
                local_height / float(max(local_width, 1)),
            ) > 1.80:
                return None

            return (
                x0 + local_x,
                y0 + local_y,
                local_width,
                local_height,
            )

        return None
