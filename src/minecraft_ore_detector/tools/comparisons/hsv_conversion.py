# -*- coding: utf-8 -*-
"""Visualisiert die HSV-Konvertierung nach der aktuellen Vorverarbeitung."""

import argparse
import os

import cv2
import numpy as np

from minecraft_ore_detector.imaging.preprocessing import (
    convert_bgr_to_hsv,
    load_image,
    normalize_scene_brightness,
)
from minecraft_ore_detector.common.utils import log, log_warning
from minecraft_ore_detector.presentation.visualization import show_image
from minecraft_ore_detector.paths import PROJECT_ROOT


def draw_label(
    img: np.ndarray,
    text: str,
    position=(20, 50),
) -> None:
    """Zeichnet ein lesbares Text-Label auf ein Bild."""

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    x, y = position
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (x - 10, y - text_height - 15),
        (x + text_width + 10, y + 10),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(
        img,
        text,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def main() -> None:
    root_dir = str(PROJECT_ROOT)
    output_dir = os.path.join(root_dir, "output")
    screenshots_dir = os.path.join(root_dir, "data", "screenshots")

    parser = argparse.ArgumentParser(
        description="Visualisiert die HSV-Konvertierung eines Minecraft-Bildes."
    )
    parser.add_argument(
        "--image",
        default="test1.png",
        help="Testbildname oder absoluter Bildpfad.",
    )
    args = parser.parse_args()

    image_path = (
        args.image
        if os.path.isabs(args.image) or os.path.exists(args.image)
        else os.path.join(screenshots_dir, args.image)
    )

    try:
        image = load_image(image_path)
    except FileNotFoundError as exc:
        log_warning(str(exc))
        return

    preprocessed = normalize_scene_brightness(image)
    hsv = convert_bgr_to_hsv(preprocessed)
    hsv_visualization = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    draw_label(hsv_visualization, "HSV nach Helligkeitsnormalisierung")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "hsv_visualization.png")
    cv2.imwrite(output_path, hsv_visualization)
    log(f"HSV-Visualisierung gespeichert unter: {output_path}")
    show_image(hsv_visualization, window_name="HSV-Visualisierung")


if __name__ == "__main__":
    main()
