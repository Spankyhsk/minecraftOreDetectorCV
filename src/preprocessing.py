# -*- coding: utf-8 -*-

from pathlib import Path
import cv2
import numpy as np

# ------------------------------------------------------------
# PROJECT ROOT (von /src aus)
# ------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# ------------------------------------------------------------
# REFERENZBILD IM ROOT
# ------------------------------------------------------------

REF_PATH = PROJECT_ROOT / "data" / "reference" / "ore_scene.png"

_ref_img = cv2.imread(str(REF_PATH))
if _ref_img is None:
    raise FileNotFoundError(f"Referenzbild nicht gefunden: {REF_PATH}")


_ref_hsv = cv2.cvtColor(_ref_img, cv2.COLOR_BGR2HSV)
_, _, _ref_v = cv2.split(_ref_hsv)

REF_BRIGHTNESS = np.median(_ref_v)

# ------------------------------------------------------------
# LOAD IMAGE
# ------------------------------------------------------------
def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)

    if img is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {path}")

    return img


# ------------------------------------------------------------
# HELLIGKEIT AN TEMPLATE ANPASSEN (OHNE PARAMETER)
# ------------------------------------------------------------
def match_scene_brightness(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    img_brightness = np.median(v)

    if img_brightness > 0:
        factor = REF_BRIGHTNESS / img_brightness
    else:
        factor = 1.0

    v = np.clip(v.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """
    Kompatibilitaetswrapper fuer alte Vergleichsskripte.

    Die Pipeline nutzt jetzt match_scene_brightness(), weil diese Methode die
    Szenenhelligkeit auf das Referenzbild normalisiert.
    """

    return match_scene_brightness(img)


def blur(img: np.ndarray) -> np.ndarray:
    """
    Kompatibilitaetswrapper fuer alte Vergleichsskripte.

    Die neue Vorverarbeitung verwendet keinen zusaetzlichen Blur.
    """

    return img


# ------------------------------------------------------------
# HSV KONVERTIERUNG
# ------------------------------------------------------------
def to_hsv(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
