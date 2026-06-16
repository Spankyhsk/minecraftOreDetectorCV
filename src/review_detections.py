# -*- coding: utf-8 -*-
"""
Interaktives Review-Tool fuer Detektionsergebnisse.

Das Tool speichert manuelle Entscheidungen separat von der Ground Truth.
"""

import argparse
import contextlib
import io
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2

from config import DATA_DIR, OreDetectorConfig
from evaluate import (
    detection_key,
    load_annotations,
    match_image,
    miss_key,
    normalize_label,
)
from pipeline import OreDetector
from preprocessing import load_image

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


class ReviewCase:
    def __init__(
        self,
        image_name: str,
        kind: str,
        label: str,
        prediction: Optional[dict],
        ground_truth: Optional[dict],
        iou: Optional[float],
    ):
        self.image_name = image_name
        self.kind = kind
        self.label = normalize_label(label)
        self.prediction = prediction
        self.ground_truth = ground_truth
        self.iou = iou

    def key(self) -> str:
        if self.kind == "fn":
            return miss_key(self.ground_truth)
        return detection_key(self.prediction)

    def current_decision(self, review: Dict[str, dict]) -> Optional[str]:
        image_review = review.get(self.image_name, {})

        if self.kind == "fn":
            return image_review.get("misses", {}).get(self.key(), {}).get("decision")

        return image_review.get("detections", {}).get(self.key(), {}).get("decision")


class ReviewSession:
    def __init__(
        self,
        annotations_path: str,
        screenshots_dir: str,
        review_path: str,
        include_hard: bool,
        iou: float,
        hard_iou: float,
        max_width: int,
        max_height: int,
    ):
        self.annotations_path = annotations_path
        self.screenshots_dir = screenshots_dir
        self.review_path = review_path
        self.include_hard = include_hard
        self.iou = iou
        self.hard_iou = hard_iou
        self.max_width = max_width
        self.max_height = max_height
        self.annotations = load_annotations(annotations_path)
        self.review = self._load_review()
        self.cases = self._build_cases()
        self.index = 0
        self.window_name = "Review detections"

    def run(self) -> None:
        if not self.cases:
            print("No review cases found.")
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        print("Review controls:")
        print("  y  mark detection/case as good")
        print("  n  mark detection/case as bad")
        print("  i  ignore detection or accept missed hard case")
        print("  s  skip/clear manual decision")
        print("  a/d or left/right  previous/next")
        print("  q or ESC           save and quit")

        while True:
            frame = self._render_case(self.cases[self.index])
            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(0) & 0xFF

            if key in {ord("q"), 27}:
                self.save()
                break
            if key in {ord("d"), 83, 3}:
                self.index = min(len(self.cases) - 1, self.index + 1)
            elif key in {ord("a"), 81, 2}:
                self.index = max(0, self.index - 1)
            elif key == ord("y"):
                self._set_decision(self.cases[self.index], "valid")
                self._advance()
            elif key == ord("n"):
                self._set_decision(self.cases[self.index], "invalid")
                self._advance()
            elif key == ord("i"):
                case = self.cases[self.index]
                decision = "acceptable" if case.kind == "fn" else "ignore"
                self._set_decision(case, decision)
                self._advance()
            elif key == ord("s"):
                self._clear_decision(self.cases[self.index])
                self._advance()

        cv2.destroyWindow(self.window_name)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.review_path), exist_ok=True)
        with open(self.review_path, "w", encoding="utf-8") as file:
            json.dump(self.review, file, indent=2, sort_keys=True)
            file.write("\n")
        print(f"Saved review decisions to {self.review_path}")

    def _build_cases(self) -> List[ReviewCase]:
        cases = []

        for image_name in sorted(self.annotations.keys()):
            image_path = os.path.join(self.screenshots_dir, image_name)
            if not os.path.exists(image_path):
                print(f"[WARN] Missing image: {image_name}")
                continue

            img = load_image(image_path)
            config = OreDetectorConfig(image_path=image_path, save_debug_masks=False)
            detector = OreDetector(config)

            with contextlib.redirect_stdout(io.StringIO()):
                result = detector.detect(img)

            tp, fp, fn = match_image(
                self.annotations[image_name],
                result.detections,
                self.iou,
                self.hard_iou,
                self.include_hard,
            )

            for item in tp:
                cases.append(ReviewCase(
                    image_name=image_name,
                    kind="tp",
                    label=item["prediction"]["label"],
                    prediction=item["prediction"],
                    ground_truth=item["ground_truth"],
                    iou=item["iou"],
                ))

            for item in fp:
                cases.append(ReviewCase(
                    image_name=image_name,
                    kind="fp",
                    label=item["label"],
                    prediction=item,
                    ground_truth=None,
                    iou=None,
                ))

            for item in fn:
                cases.append(ReviewCase(
                    image_name=image_name,
                    kind="fn",
                    label=item["label"],
                    prediction=None,
                    ground_truth=item,
                    iou=None,
                ))

        return cases

    def _render_case(self, case: ReviewCase):
        image_path = os.path.join(self.screenshots_dir, case.image_name)
        img = load_image(image_path)
        img_h, img_w = img.shape[:2]
        scale = min(
            self.max_width / float(img_w),
            self.max_height / float(img_h),
            1.0,
        )
        display_w = int(img_w * scale)
        display_h = int(img_h * scale)
        frame = cv2.resize(img, (display_w, display_h), interpolation=cv2.INTER_AREA)
        cv2.resizeWindow(self.window_name, display_w, display_h)

        self._draw_all_ground_truth(frame, case.image_name, scale)

        if case.ground_truth is not None:
            self._draw_box(
                frame,
                tuple(case.ground_truth["box"]),
                scale,
                (255, 255, 255),
                f"GT {case.ground_truth['label']}",
                3,
            )

        if case.prediction is not None:
            color = LABEL_COLORS.get(normalize_label(case.prediction["label"]), (0, 255, 0))
            score = float(case.prediction.get("score", 0.0))
            self._draw_box(
                frame,
                tuple(case.prediction["box"]),
                scale,
                color,
                f"PRED {case.prediction['label']} {score:.2f}",
                3,
            )

        self._draw_status(frame, case)
        return frame

    def _draw_all_ground_truth(self, frame, image_name: str, scale: float) -> None:
        for item in self.annotations.get(image_name, []):
            color = LABEL_COLORS.get(normalize_label(item["label"]), (180, 180, 180))
            thickness = 1
            label = f"GT {item['label']}"

            if item.get("difficulty") == "hard":
                label += " hard"
            if item.get("ignore", False):
                label += " ignore"

            self._draw_box(frame, tuple(item["box"]), scale, color, label, thickness)

    def _draw_box(
        self,
        frame,
        box: Box,
        scale: float,
        color: Tuple[int, int, int],
        label: str,
        thickness: int,
    ) -> None:
        x, y, w, h = box
        sx = int(x * scale)
        sy = int(y * scale)
        sx2 = int((x + w) * scale)
        sy2 = int((y + h) * scale)

        cv2.rectangle(frame, (sx, sy), (sx2, sy2), color, thickness)
        cv2.putText(
            frame,
            label,
            (sx, max(18, sy - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    def _draw_status(self, frame, case: ReviewCase) -> None:
        decision = case.current_decision(self.review) or "none"
        iou_text = "n/a" if case.iou is None else f"{case.iou:.3f}"
        text = (
            f"{self.index + 1}/{len(self.cases)} | {case.image_name} | "
            f"{case.kind.upper()} | {case.label} | IoU {iou_text} | review: {decision}"
        )
        hint = "y=good n=bad i=ignore/acceptable s=skip a/d=prev/next q=save"

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 58), (20, 20, 20), -1)
        cv2.putText(frame, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, hint, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

    def _set_decision(self, case: ReviewCase, decision: str) -> None:
        image_review = self.review.setdefault(case.image_name, {})

        if case.kind == "fn":
            misses = image_review.setdefault("misses", {})
            misses[case.key()] = {
                "decision": "acceptable" if decision in {"valid", "acceptable"} else "missed",
                "label": case.label,
                "box": case.ground_truth["box"],
            }
            return

        detections = image_review.setdefault("detections", {})
        detections[case.key()] = {
            "decision": decision,
            "label": case.label,
            "box": case.prediction["box"],
            "score": float(case.prediction.get("score", 0.0)),
            "original_kind": case.kind,
        }

    def _clear_decision(self, case: ReviewCase) -> None:
        image_review = self.review.setdefault(case.image_name, {})
        bucket = "misses" if case.kind == "fn" else "detections"
        image_review.setdefault(bucket, {}).pop(case.key(), None)

    def _advance(self) -> None:
        self.index = min(len(self.cases) - 1, self.index + 1)

    def _load_review(self) -> Dict[str, dict]:
        if not os.path.exists(self.review_path):
            return {}

        with open(self.review_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("Review JSON must be an object keyed by filename.")

        return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually review detector output.")
    parser.add_argument(
        "--annotations",
        default=os.path.join(DATA_DIR, "annotations", "ground_truth.json"),
    )
    parser.add_argument(
        "--screenshots",
        default=os.path.join(DATA_DIR, "screenshots"),
    )
    parser.add_argument(
        "--review",
        default=os.path.join(DATA_DIR, "annotations", "manual_review.json"),
    )
    parser.add_argument("--iou", type=float, default=0.35)
    parser.add_argument("--hard-iou", type=float, default=0.20)
    parser.add_argument("--include-hard", action="store_true")
    parser.add_argument("--max-width", type=int, default=1600)
    parser.add_argument("--max-height", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = ReviewSession(
        annotations_path=args.annotations,
        screenshots_dir=args.screenshots,
        review_path=args.review,
        include_hard=args.include_hard,
        iou=args.iou,
        hard_iou=args.hard_iou,
        max_width=args.max_width,
        max_height=args.max_height,
    )
    session.run()


if __name__ == "__main__":
    main()

