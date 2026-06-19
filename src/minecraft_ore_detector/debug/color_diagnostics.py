# -*- coding: utf-8 -*-
"""
Diagnose der Erz-Farbregeln gegen annotierte Ground-Truth-Boxen.

Das Skript erkennt keine Erze, sondern misst nur, welche Farbregeln innerhalb
der manuell annotierten Boxen anschlagen. Das hilft beim Tuning der HSV-Regeln.
"""

import argparse
import json
import os
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from minecraft_ore_detector.app.config import DATA_DIR
from minecraft_ore_detector.evaluation.evaluate import load_annotations, normalize_label
from minecraft_ore_detector.detection.rules import ORE_RULES, supported_ores
from minecraft_ore_detector.imaging.preprocessing import load_image, normalize_scene_brightness
from minecraft_ore_detector.models import Box



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose HSV color rules inside GT boxes.")
    parser.add_argument(
        "--annotations",
        default=os.path.join(DATA_DIR, "annotations", "ground_truth.json"),
        help="Path to ground-truth JSON.",
    )
    parser.add_argument(
        "--screenshots",
        default=os.path.join(DATA_DIR, "screenshots"),
        help="Directory containing screenshots.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Optional single image name, e.g. test13.png.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional GT label filter, e.g. iron.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=4,
        help="Number of strongest competing color rules to print.",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="Optional path for machine-readable JSON output.",
    )
    return parser.parse_args()


def mask_ratio(hsv: np.ndarray, ranges: Iterable[Tuple[List[int], List[int]]]) -> float:
    if hsv.size == 0:
        return 0.0

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in ranges:
        lo = np.array(lower, dtype=np.uint8)
        hi = np.array(upper, dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

    return cv2.countNonZero(mask) / float(mask.shape[0] * mask.shape[1])


def hsv_percentiles(hsv: np.ndarray) -> Dict[str, List[float]]:
    if hsv.size == 0:
        return {"h": [], "s": [], "v": []}

    return {
        "h": [float(v) for v in np.percentile(hsv[:, :, 0], [5, 25, 50, 75, 95])],
        "s": [float(v) for v in np.percentile(hsv[:, :, 1], [5, 25, 50, 75, 95])],
        "v": [float(v) for v in np.percentile(hsv[:, :, 2], [5, 25, 50, 75, 95])],
    }


def score_box(img: np.ndarray, preprocessed: np.ndarray, box: Box) -> Dict:
    x, y, w, h = box
    roi_orig = img[y:y + h, x:x + w]
    roi_pre = preprocessed[y:y + h, x:x + w]
    hsv_orig = cv2.cvtColor(roi_orig, cv2.COLOR_BGR2HSV)
    hsv_pre = cv2.cvtColor(roi_pre, cv2.COLOR_BGR2HSV)

    scores = {}
    for ore in supported_ores():
        rule = ORE_RULES[ore]
        scores[ore] = {
            "orig_seg": mask_ratio(hsv_orig, rule.segmentation_ranges),
            "orig_plaus": mask_ratio(hsv_orig, rule.plausibility_ranges),
            "pre_seg": mask_ratio(hsv_pre, rule.segmentation_ranges),
            "pre_plaus": mask_ratio(hsv_pre, rule.plausibility_ranges),
        }

    return {
        "hsv_orig": hsv_percentiles(hsv_orig),
        "hsv_pre": hsv_percentiles(hsv_pre),
        "scores": scores,
    }


def strongest(scores: Dict[str, Dict[str, float]], key: str, limit: int) -> List[Tuple[str, float]]:
    ranked = sorted(
        ((ore, values[key]) for ore, values in scores.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[:limit]


def print_case(case: Dict, top: int) -> None:
    box = case["box"]
    difficulty = case["difficulty"]
    print(f"\n{case['image']} | GT={case['label']} | {difficulty} | box={box}")

    for source, title in [("orig", "original"), ("pre", "preprocessed")]:
        hsv = case[f"hsv_{source}"]
        print(
            f"  {title:12s} HSV p5/25/50/75/95 "
            f"H={hsv['h']} S={hsv['s']} V={hsv['v']}"
        )

    scores = case["scores"]
    gt = normalize_label(case["label"])
    if gt in scores:
        gt_scores = scores[gt]
        print(
            "  GT color      "
            f"orig_seg={gt_scores['orig_seg']:.3f} "
            f"orig_plaus={gt_scores['orig_plaus']:.3f} "
            f"pre_seg={gt_scores['pre_seg']:.3f} "
            f"pre_plaus={gt_scores['pre_plaus']:.3f}"
        )

    for key in ["orig_seg", "pre_seg", "orig_plaus", "pre_plaus"]:
        parts = [f"{ore}:{value:.3f}" for ore, value in strongest(scores, key, top)]
        print(f"  top {key:10s} " + ", ".join(parts))


def collect_cases(args: argparse.Namespace) -> List[Dict]:
    annotations = load_annotations(args.annotations)
    label_filter = normalize_label(args.label) if args.label else None
    cases = []

    for image_name in sorted(annotations.keys()):
        if args.image and image_name != args.image:
            continue

        image_path = os.path.join(args.screenshots, image_name)
        if not os.path.exists(image_path):
            print(f"[WARN] Missing image: {image_name}")
            continue

        img = load_image(image_path)
        preprocessed = normalize_scene_brightness(img)

        for item in annotations[image_name]:
            label = normalize_label(item["label"])
            if label_filter and label != label_filter:
                continue

            box = tuple(item["box"])
            case = {
                "image": image_name,
                "label": label,
                "difficulty": item.get("difficulty", "normal"),
                "box": list(box),
            }
            case.update(score_box(img, preprocessed, box))
            cases.append(case)

    return cases


def main() -> None:
    args = parse_args()
    cases = collect_cases(args)

    for case in cases:
        print_case(case, args.top)

    if args.json_path:
        os.makedirs(os.path.dirname(args.json_path), exist_ok=True)
        with open(args.json_path, "w", encoding="utf-8") as file:
            json.dump(cases, file, indent=2)
            file.write("\n")
        print(f"\nWrote {args.json_path}")


if __name__ == "__main__":
    main()
