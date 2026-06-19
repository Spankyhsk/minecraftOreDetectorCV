# -*- coding: utf-8 -*-
"""
Visuelles Debug-Board fuer einzelne Screenshots.
"""

import argparse
import contextlib
import io
import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

from minecraft_ore_detector.app.config import DATA_DIR, OreDetectorConfig
from minecraft_ore_detector.detection.core import detect_with_template_bank, find_candidates, non_max_suppression
from minecraft_ore_detector.evaluation.evaluate import detection_key, normalize_label
from minecraft_ore_detector.imaging.runtime_mask_filter import RuntimeMaskFilter
from minecraft_ore_detector.imaging.morphology import clean_mask
from minecraft_ore_detector.app.pipeline import OreDetector
from minecraft_ore_detector.imaging.preprocessing import load_image, normalize_scene_brightness, convert_bgr_to_hsv
from minecraft_ore_detector.imaging.segmentation import (
    color_mask,
    edge_mask,
    hybrid_mask,
    refine_mask_for_ore,
    supported_ores,
    use_edges_for_ore,
)
from minecraft_ore_detector.repositories.template_repository import TemplateRepository
from minecraft_ore_detector.presentation.visualization import draw_candidate_boxes

Box = Tuple[int, int, int, int]

LABEL_COLORS = {
    "coal": (70, 70, 70),
    "copper": (45, 120, 210),
    "diamond": (255, 220, 40),
    "emerald": (70, 220, 70),
    "gold": (0, 215, 255),
    "iron": (180, 180, 210),
    "lapis": (230, 80, 30),
    "redstone": (40, 40, 230),
}

REVIEW_COLORS = {
    "valid": (70, 220, 70),
    "invalid": (40, 40, 230),
    "ignore": (160, 160, 160),
    "unknown": (0, 220, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create visual pipeline debug boards.")
    parser.add_argument(
        "--image",
        default=None,
        help="Screenshot filename from data/screenshots. Omit to process all annotated images.",
    )
    parser.add_argument(
        "--annotations",
        default=os.path.join(DATA_DIR, "annotations", "ground_truth.json"),
    )
    parser.add_argument(
        "--review",
        default=os.path.join(DATA_DIR, "annotations", "manual_review.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(DATA_DIR, "debug_visual"),
    )
    parser.add_argument("--panel-width", type=int, default=420)
    parser.add_argument("--ore", default=None, help="Optional single ore mask board.")
    parser.add_argument(
        "--candidates-only",
        action="store_true",
        help="Write an original-image overlay with candidate boxes only.",
    )
    return parser.parse_args()


def load_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_images(args: argparse.Namespace, annotations: Dict) -> List[str]:
    if args.image:
        return [args.image]

    if annotations:
        return sorted(annotations.keys())

    screenshots_dir = os.path.join(DATA_DIR, "screenshots")
    return sorted(name for name in os.listdir(screenshots_dir) if name.endswith(".png"))


def make_panel(img: np.ndarray, title: str, width: int) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    scale = width / float(w)
    resized = cv2.resize(img, (width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    header_h = 32
    header = np.full((header_h, width, 3), 24, dtype=np.uint8)
    cv2.putText(header, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([header, resized])


def stack_grid(panels: List[np.ndarray], columns: int) -> np.ndarray:
    if not panels:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    max_w = max(panel.shape[1] for panel in panels)
    max_h = max(panel.shape[0] for panel in panels)
    padded = [pad_to(panel, max_w, max_h) for panel in panels]
    rows = []

    for i in range(0, len(padded), columns):
        row_panels = padded[i:i + columns]
        while len(row_panels) < columns:
            row_panels.append(np.full((max_h, max_w, 3), 18, dtype=np.uint8))
        rows.append(np.hstack(row_panels))

    return np.vstack(rows)


def pad_to(img: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.full((height, width, 3), 18, dtype=np.uint8)
    out[:img.shape[0], :img.shape[1]] = img
    return out


def draw_box(img: np.ndarray, box: Box, color: Tuple[int, int, int], label: str, thickness: int = 2) -> None:
    x, y, w, h = [int(v) for v in box]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    cv2.putText(img, label, (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def draw_ground_truth(img: np.ndarray, items: List[dict]) -> None:
    for item in items:
        label = normalize_label(item["label"])
        color = LABEL_COLORS.get(label, (220, 220, 220))
        title = f"GT {item['label']}"
        if item.get("difficulty") == "hard":
            title += " hard"
        if item.get("ignore", False):
            title += " ignore"
        draw_box(img, tuple(item["box"]), color, title, 1)


def draw_detections(img: np.ndarray, detections: List[dict], image_review: Dict) -> None:
    reviewed = image_review.get("detections", {})

    for det in detections:
        key = detection_key(det)
        decision = reviewed.get(key, {}).get("decision", "unknown")
        color = REVIEW_COLORS.get(decision, REVIEW_COLORS["unknown"])
        label = f"{det['label']} {det.get('score', 0.0):.2f} {decision}"
        draw_box(img, tuple(det["box"]), color, label, 2)


def collect_pipeline_debug(img: np.ndarray, config: OreDetectorConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, dict], List[dict]]:
    mask_filter = RuntimeMaskFilter()
    templates = TemplateRepository(config.templates_dir)

    preprocessed = normalize_scene_brightness(img)
    hsv = convert_bgr_to_hsv(preprocessed)
    edges = edge_mask(preprocessed)
    edges_clean = mask_filter.filter_mask(edges, hsv)

    per_ore = {}
    raw_detections = []

    for ore in supported_ores():
        color = color_mask(hsv, ore)
        color_clean = mask_filter.filter_mask(color, hsv, ore=ore)

        mask = hybrid_mask(color_clean, edges_clean) if use_edges_for_ore(ore) else color_clean
        mask = refine_mask_for_ore(ore, mask)
        mask = clean_mask(mask)
        mask = mask_filter.filter_mask(mask, hsv, ore=ore)

        if ore == "coal":
            from minecraft_ore_detector.detection.candidate_detection import CoalPrimaryDetector
            coal_detector = CoalPrimaryDetector(mask_filter)
            candidates = coal_detector.find_candidates(img, color_clean)
            raw = coal_detector.detect_from_candidates(img, candidates)
        else:
            candidates = find_candidates(mask, color_clean, ore=ore)
            if ore == "diamond":
                from minecraft_ore_detector.detection.candidate_detection import DiamondCandidateExpander
                candidates = DiamondCandidateExpander().expand_candidates(candidates, img.shape)

            bank = templates.get_templates_for_ore(ore)
            raw = []
            if bank:
                raw = detect_with_template_bank(
                    img,
                    candidates,
                    bank,
                    label=ore.capitalize(),
                    threshold=config.ore_match_thresholds.get(ore, 0.8),
                    brightness_split=None,
                )

        raw_detections.extend(raw)
        per_ore[ore] = {
            "color": color_clean,
            "mask": mask,
            "candidates": candidates,
            "raw": raw,
        }

    detections = non_max_suppression(raw_detections, iou_threshold=config.nms_iou_threshold)
    return preprocessed, edges_clean, per_ore, detections


def draw_candidates_on_mask(mask: np.ndarray, candidates: List[Box], raw: List[dict]) -> np.ndarray:
    out = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for box in candidates:
        draw_box(out, box, (255, 0, 0), "cand", 1)

    for det in raw:
        draw_box(out, tuple(det["box"]), (0, 255, 0), f"{det['label']} {det.get('score', 0.0):.2f}", 2)

    return out


def create_debug_board(image_name: str, args: argparse.Namespace, annotations: Dict, review: Dict) -> str:
    image_path = os.path.join(DATA_DIR, "screenshots", image_name)
    img = load_image(image_path)
    config = OreDetectorConfig(image_path=image_path, save_debug_masks=False)

    with contextlib.redirect_stdout(io.StringIO()):
        preprocessed, edges, per_ore, detections = collect_pipeline_debug(img, config)
        final = OreDetector(config).detect_and_render(img)

    gt_items = annotations.get(image_name, [])
    image_review = review.get(image_name, {})

    gt_overlay = img.copy()
    draw_ground_truth(gt_overlay, gt_items)

    review_overlay = img.copy()
    draw_ground_truth(review_overlay, gt_items)
    draw_detections(review_overlay, detections, image_review)

    panels = [
        make_panel(img, f"{image_name} original", args.panel_width),
        make_panel(preprocessed, "preprocessed", args.panel_width),
        make_panel(edges, "edges cleaned", args.panel_width),
        make_panel(final, "final detections", args.panel_width),
        make_panel(gt_overlay, "ground truth", args.panel_width),
        make_panel(review_overlay, "review overlay", args.panel_width),
    ]

    ores = [args.ore] if args.ore else supported_ores()

    for ore in ores:
        if ore not in per_ore:
            continue
        data = per_ore[ore]
        panels.append(make_panel(data["color"], f"{ore} color mask", args.panel_width))
        panels.append(make_panel(
            draw_candidates_on_mask(data["mask"], data["candidates"], data["raw"]),
            f"{ore} clean mask + candidates",
            args.panel_width,
        ))

    board = stack_grid(panels, columns=2)
    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f"_{args.ore}" if args.ore else ""
    out_path = os.path.join(args.output_dir, f"{os.path.splitext(image_name)[0]}{suffix}_debug.png")
    cv2.imwrite(out_path, board)
    return out_path


def create_candidate_overlay(image_name: str, args: argparse.Namespace) -> str:
    image_path = os.path.join(DATA_DIR, "screenshots", image_name)
    img = load_image(image_path)
    config = OreDetectorConfig(image_path=image_path, save_debug_masks=False)

    with contextlib.redirect_stdout(io.StringIO()):
        if args.ore:
            _, _, per_ore, _ = collect_pipeline_debug(img, config)
            candidates = per_ore.get(args.ore, {}).get("candidates", [])
        else:
            candidates = OreDetector(config).detect(img).candidates

    out = draw_candidate_boxes(img, candidates)
    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f"_{args.ore}" if args.ore else ""
    out_path = os.path.join(args.output_dir, f"{os.path.splitext(image_name)[0]}{suffix}_candidates.png")
    cv2.imwrite(out_path, out)
    return out_path


def main() -> None:
    args = parse_args()
    annotations = load_json(args.annotations)
    review = load_json(args.review)

    for image_name in resolve_images(args, annotations):
        if args.candidates_only:
            path = create_candidate_overlay(image_name, args)
        else:
            path = create_debug_board(image_name, args, annotations, review)
        print(path)


if __name__ == "__main__":
    main()
