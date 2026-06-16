# -*- coding: utf-8 -*-
"""
Headless-Auswertung der Detection-Pipeline fuer Test-Screenshots.
"""

import os

from config import DATA_DIR, OreDetectorConfig
from pipeline import OreDetector
from preprocessing import load_image


def run_eval() -> None:
    """
    Fuehrt die Pipeline ohne GUI aus und gibt Kandidaten/Detektionen aus.
    """

    screenshots_dir = os.path.join(DATA_DIR, "screenshots")

    for shot in sorted(os.listdir(screenshots_dir)):
        if not shot.lower().endswith(".png"):
            continue

        img_path = os.path.join(screenshots_dir, shot)
        config = OreDetectorConfig(image_path=img_path, save_debug_masks=False)
        detector = OreDetector(config)
        img = load_image(img_path)
        result = detector.detect(img)

        print(f"\n=== Evaluierung fuer {shot} ===")
        print("Anzahl gefundener Kandidaten-Boxen:", len(result.candidates))

        print(f"Finale Detektionen nach NMS (insgesamt {len(result.detections)}):")
        for d in result.detections:
            print(
                f"  * {d['label']} "
                f"(Score: {d['score']:.3f}, "
                f"Template: '{d.get('variant')}', "
                f"Box: {d['box']})"
            )


if __name__ == "__main__":
    run_eval()

