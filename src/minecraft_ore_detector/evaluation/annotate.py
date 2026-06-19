# -*- coding: utf-8 -*-
"""
Interaktives Tool zum Erstellen von Ground-Truth-Boxen.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2

from minecraft_ore_detector.app.config import DATA_DIR
from minecraft_ore_detector.imaging.preprocessing import load_image

Box = Tuple[int, int, int, int]

LABELS = [
    "Coal",
    "Copper",
    "Diamond",
    "Emerald",
    "Gold",
    "Iron",
    "Lapis",
    "Redstone",
]

KEY_TO_LABEL_INDEX = {
    ord("1"): 0,
    ord("2"): 1,
    ord("3"): 2,
    ord("4"): 3,
    ord("5"): 4,
    ord("6"): 5,
    ord("7"): 6,
    ord("8"): 7,
}

LABEL_COLORS = {
    "Coal": (70, 70, 70),
    "Copper": (45, 120, 210),
    "Diamond": (255, 220, 40),
    "Emerald": (70, 220, 70),
    "Gold": (0, 215, 255),
    "Iron": (180, 180, 210),
    "Lapis": (230, 80, 30),
    "Redstone": (40, 40, 230),
}


class AnnotationSession:
    def __init__(
        self,
        image_path: str,
        image_name: str,
        annotations_path: str,
        max_width: int,
        max_height: int,
    ):
        self.image_path = image_path
        self.image_name = image_name
        self.annotations_path = annotations_path
        self.original = load_image(image_path)
        self.image_h, self.image_w = self.original.shape[:2]
        self.scale = min(
            max_width / float(self.image_w),
            max_height / float(self.image_h),
            1.0,
        )
        self.display_w = int(self.image_w * self.scale)
        self.display_h = int(self.image_h * self.scale)
        self.annotations = self._load_annotations()
        self.boxes: List[dict] = list(self.annotations.get(image_name, []))
        self.current_label = "Diamond"
        self.current_difficulty = "normal"
        self.selected_index: Optional[int] = None
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_current: Optional[Tuple[int, int]] = None
        self.mouse_pos = (0, 0)
        self.window_name = f"Annotate: {image_name}"

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_w, self.display_h)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        print("Annotation controls:")
        print("  1-8  select label: " + ", ".join(f"{idx + 1}={label}" for idx, label in enumerate(LABELS)))
        print("  left mouse drag      draw_detection_boxes box")
        print("  right mouse click    select existing box")
        print("  z                    undo last box")
        print("  backspace/delete     delete selected box")
        print("  h                    toggle hard difficulty on selected/new boxes")
        print("  i                    toggle ignore on selected box")
        print("  s                    save")
        print("  q or ESC             save and quit")
        print("  c                    clear boxes for this image")

        while True:
            cv2.imshow(self.window_name, self._render())
            key = cv2.waitKey(20) & 0xFF

            if key in KEY_TO_LABEL_INDEX:
                self.current_label = LABELS[KEY_TO_LABEL_INDEX[key]]
            elif key == ord("z"):
                if self.boxes:
                    self.boxes.pop()
                    self.selected_index = None
            elif key in {8, 127}:
                self._delete_selected()
            elif key == ord("h"):
                self._toggle_hard()
            elif key == ord("i"):
                self._toggle_ignore()
            elif key == ord("c"):
                self.boxes = []
                self.selected_index = None
            elif key == ord("s"):
                self.save()
            elif key in {ord("q"), 27}:
                self.save()
                break

        cv2.destroyWindow(self.window_name)

    def save(self) -> None:
        self.annotations[self.image_name] = self.boxes
        os.makedirs(os.path.dirname(self.annotations_path), exist_ok=True)

        with open(self.annotations_path, "w", encoding="utf-8") as file:
            json.dump(self.annotations, file, indent=2, sort_keys=True)
            file.write("\n")

        print(f"Saved {len(self.boxes)} boxes for {self.image_name} to {self.annotations_path}")

    def _load_annotations(self) -> Dict[str, List[dict]]:
        if not os.path.exists(self.annotations_path):
            return {}

        with open(self.annotations_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("Annotation JSON must be an object keyed by filename.")

        return data

    def _render(self):
        frame = cv2.resize(
            self.original,
            (self.display_w, self.display_h),
            interpolation=cv2.INTER_AREA,
        )

        for idx, item in enumerate(self.boxes):
            label = item["label"]
            box = tuple(item["box"])
            self._draw_box(
                frame,
                item,
                box,
                committed=True,
                selected=idx == self.selected_index,
            )

        if self.drag_start is not None and self.drag_current is not None:
            x0, y0 = self.drag_start
            x1, y1 = self.drag_current
            box = self._normalize_box(x0, y0, x1, y1)
            preview_item = {
                "label": self.current_label,
                "difficulty": self.current_difficulty,
            }
            self._draw_box(
                frame,
                preview_item,
                box,
                committed=False,
                selected=False,
            )

        self._draw_status(frame)
        return frame

    def _draw_status(self, frame) -> None:
        x, y = self.mouse_pos
        label_text = (
            f"Label: {self.current_label} | New: {self.current_difficulty} "
            f"| Mouse: x={x} y={y} | Boxes: {len(self.boxes)}"
        )

        if self.selected_index is not None:
            selected = self.boxes[self.selected_index]
            selected_state = selected.get("difficulty", "normal")
            if selected.get("ignore", False):
                selected_state = "ignore"
            label_text += f" | Selected: {self.selected_index + 1} {selected['label']} {selected_state}"

        if self.drag_start is not None and self.drag_current is not None:
            box = self._normalize_box(
                self.drag_start[0],
                self.drag_start[1],
                self.drag_current[0],
                self.drag_current[1],
            )
            label_text += f" | Current box: {list(box)}"

        cv2.rectangle(frame, (0, 0), (self.display_w, 34), (20, 20, 20), -1)
        cv2.putText(
            frame,
            label_text,
            (8, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _draw_box(self, frame, item: dict, box: Box, committed: bool, selected: bool) -> None:
        label = item["label"]
        x, y, w, h = box
        sx, sy = self._to_screen(x, y)
        sx2, sy2 = self._to_screen(x + w, y + h)
        color = LABEL_COLORS.get(label, (0, 255, 0))
        thickness = 3 if selected else (2 if committed else 1)
        line_type = cv2.LINE_4 if item.get("difficulty") == "hard" else cv2.LINE_8

        cv2.rectangle(frame, (sx, sy), (sx2, sy2), color, thickness, line_type)

        tags = []
        if item.get("difficulty") == "hard":
            tags.append("hard")
        if item.get("ignore", False):
            tags.append("ignore")

        suffix = f" ({','.join(tags)})" if tags else ""
        text = f"{label}{suffix} [{x},{y},{w},{h}]"
        cv2.putText(
            frame,
            text,
            (sx, max(18, sy - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )

    def _on_mouse(self, event, screen_x: int, screen_y: int, flags, userdata) -> None:
        x, y = self._to_original(screen_x, screen_y)
        self.mouse_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.drag_current = (x, y)
            self.selected_index = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag_start is not None:
                self.drag_current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drag_start is not None:
                box = self._normalize_box(
                    self.drag_start[0],
                    self.drag_start[1],
                    x,
                    y,
                )
                if box[2] >= 4 and box[3] >= 4:
                    item = {
                        "label": self.current_label,
                        "box": list(box),
                    }

                    if self.current_difficulty == "hard":
                        item["difficulty"] = "hard"

                    self.boxes.append(item)
                    self.selected_index = len(self.boxes) - 1

            self.drag_start = None
            self.drag_current = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.selected_index = self._find_box_at(x, y)

    def _find_box_at(self, x: int, y: int) -> Optional[int]:
        for idx in range(len(self.boxes) - 1, -1, -1):
            bx, by, bw, bh = self.boxes[idx]["box"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return idx

        return None

    def _delete_selected(self) -> None:
        if self.selected_index is None:
            return

        self.boxes.pop(self.selected_index)
        self.selected_index = None

    def _toggle_hard(self) -> None:
        if self.selected_index is None:
            self.current_difficulty = (
                "normal"
                if self.current_difficulty == "hard"
                else "hard"
            )
            return

        selected = self.boxes[self.selected_index]
        if selected.get("difficulty") == "hard":
            selected.pop("difficulty", None)
        else:
            selected["difficulty"] = "hard"

    def _toggle_ignore(self) -> None:
        if self.selected_index is None:
            return

        selected = self.boxes[self.selected_index]
        if selected.get("ignore", False):
            selected.pop("ignore", None)
        else:
            selected["ignore"] = True

    def _normalize_box(self, x0: int, y0: int, x1: int, y1: int) -> Box:
        left = max(0, min(x0, x1))
        top = max(0, min(y0, y1))
        right = min(self.image_w - 1, max(x0, x1))
        bottom = min(self.image_h - 1, max(y0, y1))
        return left, top, right - left, bottom - top

    def _to_original(self, screen_x: int, screen_y: int) -> Tuple[int, int]:
        x = int(screen_x / self.scale)
        y = int(screen_y / self.scale)
        return (
            max(0, min(self.image_w - 1, x)),
            max(0, min(self.image_h - 1, y)),
        )

    def _to_screen(self, x: int, y: int) -> Tuple[int, int]:
        return int(x * self.scale), int(y * self.scale)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw ground-truth ore boxes.")
    parser.add_argument(
        "--image",
        required=True,
        help="Screenshot filename from data/screenshots or a full image path.",
    )
    parser.add_argument(
        "--annotations",
        default=os.path.join(DATA_DIR, "annotations", "ground_truth.json"),
        help="Path to annotation JSON.",
    )
    parser.add_argument("--max-width", type=int, default=1600)
    parser.add_argument("--max-height", type=int, default=900)
    return parser.parse_args()


def resolve_image_path(image_arg: str) -> Tuple[str, str]:
    if os.path.exists(image_arg):
        return image_arg, os.path.basename(image_arg)

    image_path = os.path.join(DATA_DIR, "screenshots", image_arg)
    if os.path.exists(image_path):
        return image_path, image_arg

    raise FileNotFoundError(f"Image not found: {image_arg}")


def main() -> None:
    args = parse_args()
    image_path, image_name = resolve_image_path(args.image)
    session = AnnotationSession(
        image_path=image_path,
        image_name=image_name,
        annotations_path=args.annotations,
        max_width=args.max_width,
        max_height=args.max_height,
    )
    session.run()


if __name__ == "__main__":
    main()
