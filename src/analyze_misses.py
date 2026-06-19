# -*- coding: utf-8 -*-
"""
Diagnose verpasster Ground-Truth-Boxen.

Das Skript erklaert pro annotierter Box, an welcher Stufe die Pipeline gerade
scheitert: Farbe, Maske/Kandidaten, Template-Matching oder finaler ROI-Filter.
Es ist bewusst ein Debug-Werkzeug und keine Erkennung.
"""

import argparse
import contextlib
import io
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import DATA_DIR, OreDetectorConfig
from detection import (
    _color_compatibility,
    _color_support_ratio,
    _expand_box,
    detect_with_template_bank,
    find_candidates,
    match_template_multiscale,
    non_max_suppression,
)
from evaluate import (
    box_iou,
    box_partial_overlap,
    load_annotations,
    load_review,
    match_image,
    normalize_label,
)
from runtime_mask_filter import RuntimeMaskFilter
from morphology import clean_mask
from pipeline import OreDetector
from preprocessing import load_image, normalize_scene_brightness, convert_bgr_to_hsv
from segmentation import (
    color_mask,
    edge_mask,
    hybrid_mask,
    refine_mask_for_ore,
    use_edges_for_ore,
)
from template_repository import TemplateRepository

Box = Tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze why GT ore boxes are missed.")
    parser.add_argument(
        "--annotations",
        default=os.path.join(DATA_DIR, "annotations", "ground_truth.json"),
    )
    parser.add_argument(
        "--review",
        default=os.path.join(DATA_DIR, "annotations", "manual_review.json"),
    )
    parser.add_argument(
        "--screenshots",
        default=os.path.join(DATA_DIR, "screenshots"),
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Optional screenshot filename, e.g. test18.png.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional label filter, e.g. copper.",
    )
    parser.add_argument(
        "--include-hard",
        action="store_true",
        help="Also analyze GT boxes marked as hard.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of nearest candidates/detections to print.",
    )
    return parser.parse_args()


def roi_stats(img: np.ndarray, box: Box, ore: str) -> Dict[str, float]:
    x, y, w, h = box
    roi = img[y:y + h, x:x + w]

    if roi.size == 0:
        return {
            "support": 0.0,
            "compat": 0.0,
            "s_mean": 0.0,
            "v_mean": 0.0,
            "gray_std": 0.0,
            "edge_density": 0.0,
        }

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    return {
        "support": _color_support_ratio(ore, roi),
        "compat": _color_compatibility(ore, roi),
        "s_mean": float(hsv[:, :, 1].mean()),
        "v_mean": float(hsv[:, :, 2].mean()),
        "gray_std": float(gray.std()),
        "edge_density": float(np.mean(edges > 0)),
    }


def template_score_for_box(
    img: np.ndarray,
    box: Box,
    template_bank: Dict[str, np.ndarray],
) -> Tuple[float, Optional[str]]:
    x, y, w, h = box
    roi = img[y:y + h, x:x + w]

    best_score = 0.0
    best_name = None

    for name, template in template_bank.items():
        with contextlib.redirect_stdout(io.StringIO()):
            score = match_template_multiscale(roi, template)
        if score > best_score:
            best_score = score
            best_name = name

    return best_score, best_name


def build_stage_data(
    img: np.ndarray,
    ore: str,
    detector: OreDetector,
    template_repo: TemplateRepository,
) -> Dict:
    preprocessed = normalize_scene_brightness(img)
    hsv = convert_bgr_to_hsv(preprocessed)
    mask_filter = RuntimeMaskFilter()

    edges = edge_mask(preprocessed)
    edges = mask_filter.filter_mask(edges, hsv)

    color_raw = color_mask(hsv, ore)
    color_no_hud = mask_filter.remove_hud_regions(color_raw)
    color_no_water = mask_filter.remove_water_regions(color_no_hud, hsv)
    color_clean = mask_filter.remove_large_mask_regions(color_no_water, hsv, ore=ore)

    mask = hybrid_mask(color_clean, edges) if use_edges_for_ore(ore) else color_clean
    mask = refine_mask_for_ore(ore, mask)
    mask = clean_mask(mask)
    mask = mask_filter.filter_mask(mask, hsv, ore=ore)

    candidates = find_candidates(mask, color_clean, ore=ore)

    template_bank = template_repo.get_templates_for_ore(ore)
    raw = []

    if template_bank and candidates:
        with contextlib.redirect_stdout(io.StringIO()):
            raw = detect_with_template_bank(
                img,
                candidates,
                template_bank,
                label=ore.capitalize(),
                threshold=detector.config.ore_match_thresholds.get(ore, 0.8),
                brightness_split=None,
            )

    nms = non_max_suppression(raw, detector.config.nms_iou_threshold)
    filtered = detector.processor.filter_plausible_detections(nms, img)

    return {
        "preprocessed": preprocessed,
        "edges": edges,
        "color_raw": color_raw,
        "color_no_hud": color_no_hud,
        "color_no_water": color_no_water,
        "color": color_clean,
        "mask": mask,
        "candidates": candidates,
        "raw": raw,
        "nms": nms,
        "filtered": filtered,
        "template_bank": template_bank,
    }


def rank_boxes(boxes: List[Box], gt_box: Box, limit: int) -> List[Tuple[float, float, Box]]:
    ranked = []
    for box in boxes:
        ranked.append((box_iou(box, gt_box), box_partial_overlap(box, gt_box), box))
    return sorted(ranked, key=lambda item: (item[0], item[1]), reverse=True)[:limit]


def rank_detections(detections: List[Dict], gt_box: Box, limit: int) -> List[Tuple[float, float, Dict]]:
    ranked = []
    for detection in detections:
        box = tuple(detection["box"])
        ranked.append((box_iou(box, gt_box), box_partial_overlap(box, gt_box), detection))
    return sorted(ranked, key=lambda item: (item[0], item[1]), reverse=True)[:limit]


def mask_density(mask: np.ndarray, box: Box) -> float:
    x, y, w, h = box
    crop = mask[y:y + h, x:x + w]
    if crop.size == 0:
        return 0.0
    return cv2.countNonZero(crop) / float(crop.shape[0] * crop.shape[1])


def print_stats(title: str, stats: Dict[str, float]) -> None:
    print(
        f"  {title:12s} support={stats['support']:.3f} "
        f"compat={stats['compat']:.3f} "
        f"S={stats['s_mean']:.1f} V={stats['v_mean']:.1f} "
        f"std={stats['gray_std']:.1f} edge={stats['edge_density']:.3f}"
    )


def explain_case(
    image_name: str,
    gt_item: Dict,
    img: np.ndarray,
    detector: OreDetector,
    template_repo: TemplateRepository,
    top: int,
) -> None:
    ore = normalize_label(gt_item["label"])
    gt_box = tuple(gt_item["box"])
    data = build_stage_data(img, ore, detector, template_repo)
    preprocessed = data["preprocessed"]

    print(f"\n{image_name} | {ore} | {gt_item.get('difficulty', 'normal')} | GT={gt_box}")

    print_stats("original", roi_stats(img, gt_box, ore))
    print_stats("processed", roi_stats(preprocessed, gt_box, ore))
    print(
        f"  mask density raw={mask_density(data['color_raw'], gt_box):.3f} "
        f"no_hud={mask_density(data['color_no_hud'], gt_box):.3f} "
        f"no_water={mask_density(data['color_no_water'], gt_box):.3f} "
        f"clean={mask_density(data['color'], gt_box):.3f} "
        f"final={mask_density(data['mask'], gt_box):.3f} "
        f"edges={mask_density(data['edges'], gt_box):.3f}"
    )

    gt_score, gt_template = template_score_for_box(img, gt_box, data["template_bank"])
    expanded = _expand_box(gt_box, img.shape, pad_factor=0.35, min_pad=8)
    expanded_score, expanded_template = template_score_for_box(
        img,
        expanded,
        data["template_bank"],
    )
    print(
        f"  template GT score={gt_score:.3f} template={gt_template} | "
        f"expanded={expanded} score={expanded_score:.3f} template={expanded_template}"
    )

    print(
        f"  stage counts candidates={len(data['candidates'])} "
        f"raw={len(data['raw'])} nms={len(data['nms'])} filtered={len(data['filtered'])}"
    )

    print("  nearest candidates:")
    for iou, partial, box in rank_boxes(data["candidates"], gt_box, top):
        score, template = template_score_for_box(img, box, data["template_bank"])
        print(
            f"    iou={iou:.3f} partial={partial:.3f} box={box} "
            f"tpl={score:.3f} {template}"
        )

    print("  nearest raw detections:")
    for iou, partial, detection in rank_detections(data["raw"], gt_box, top):
        passes = detector.processor.is_detection_plausible(detection, img)
        print(
            f"    iou={iou:.3f} partial={partial:.3f} "
            f"score={detection.get('score', 0.0):.3f} "
            f"box={tuple(detection['box'])} variant={detection.get('variant')} "
            f"roi_ok={passes}"
        )

    print("  nearest final detections:")
    for iou, partial, detection in rank_detections(data["filtered"], gt_box, top):
        print(
            f"    iou={iou:.3f} partial={partial:.3f} "
            f"score={detection.get('score', 0.0):.3f} "
            f"box={tuple(detection['box'])} variant={detection.get('variant')}"
        )


def main() -> None:
    args = parse_args()
    annotations = load_annotations(args.annotations)
    review = load_review(args.review)
    label_filter = normalize_label(args.label) if args.label else None
    template_repo = TemplateRepository(os.path.join(DATA_DIR, "templates"))

    for image_name in sorted(annotations.keys()):
        if args.image and image_name != args.image:
            continue

        image_path = os.path.join(args.screenshots, image_name)
        if not os.path.exists(image_path):
            print(f"[WARN] Missing image: {image_name}")
            continue

        img = load_image(image_path)
        detector = OreDetector(OreDetectorConfig(image_path=image_path, save_debug_masks=False))

        with contextlib.redirect_stdout(io.StringIO()):
            result = detector.detect(img)

        tp, fp, fn = match_image(
            annotations[image_name],
            result.detections,
            iou_threshold=0.35,
            hard_iou_threshold=0.20,
            include_hard=args.include_hard,
        )

        reviewed_misses = review.get(image_name, {}).get("misses", {})
        missed_boxes = {tuple(item["box"]) for item in fn}

        for item in annotations[image_name]:
            label = normalize_label(item["label"])
            if label_filter and label != label_filter:
                continue
            if item.get("ignore", False):
                continue
            if item.get("difficulty") == "hard" and not args.include_hard:
                continue

            box = tuple(item["box"])
            if box not in missed_boxes:
                continue

            miss_key = f"{label}:{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}"
            if reviewed_misses.get(miss_key, {}).get("decision") == "acceptable":
                continue

            explain_case(image_name, item, img, detector, template_repo, args.top)


if __name__ == "__main__":
    main()
