# -*- coding: utf-8 -*-
"""Diamond-spezifische Kandidaten-Erweiterung."""

from typing import List, Tuple

from minecraft_ore_detector.models import Box


class DiamondCandidateExpander:
    """
    Fasst kleine Diamond-Farbinseln zu blockgrossen Kandidaten zusammen.
    """

    def expand_candidates(self, candidates: List[Box], img_shape: Tuple[int, ...]) -> List[Box]:
        img_h, img_w = img_shape[:2]
        block_size = int(img_w * 0.055)
        block_size = max(70, min(block_size, 115))

        if len(candidates) == 0:
            return candidates

        small_candidates = []
        normal_candidates = []

        for candidate in candidates:
            x, y, w, h = candidate
            if max(w, h) < block_size * 0.75:
                small_candidates.append(candidate)
            else:
                normal_candidates.append(candidate)

        clusters = []

        for candidate in small_candidates:
            x, y, w, h = candidate
            cx = x + w / 2
            cy = y + h / 2
            added = False

            for cluster in clusters:
                xs = [bx + bw / 2 for bx, by, bw, bh in cluster]
                ys = [by + bh / 2 for bx, by, bw, bh in cluster]
                cluster_cx = sum(xs) / len(xs)
                cluster_cy = sum(ys) / len(ys)

                if abs(cx - cluster_cx) < block_size * 0.75 and abs(cy - cluster_cy) < block_size * 0.75:
                    cluster.append(candidate)
                    added = True
                    break

            if not added:
                clusters.append([candidate])

        merged_candidates = []

        for cluster in clusters:
            x1 = min(x for x, y, w, h in cluster)
            y1 = min(y for x, y, w, h in cluster)
            x2 = max(x + w for x, y, w, h in cluster)
            y2 = max(y + h for x, y, w, h in cluster)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            new_w = block_size
            new_h = block_size
            new_x = int(cx - new_w / 2)
            new_y = int(cy - new_h / 2)

            new_x = max(0, new_x)
            new_y = max(0, new_y)

            if new_x + new_w > img_w:
                new_w = img_w - new_x
            if new_y + new_h > img_h:
                new_h = img_h - new_y

            merged_candidates.append((new_x, new_y, new_w, new_h))

        return normal_candidates + merged_candidates
