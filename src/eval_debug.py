import os

from preprocessing import load_image, apply_clahe, blur, to_hsv
from segmentation import color_mask, edge_mask, hybrid_mask, supported_ores, use_edges_for_ore, refine_mask_for_ore
from morphology import clean_mask
from detection import (
    find_candidates,
    load_template,
    detect_with_template_bank,
    non_max_suppression,
)


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
TEMPLATES_DIR = os.path.join(DATA_DIR, "templates")

ORE_MATCH_THRESHOLD = {
    "coal": 0.50,
    "copper": 0.60,
    "diamond": 0.71,
    "emerald": 0.67,
    "gold": 0.82,
    "iron": 0.76,
    "lapis": 0.58,
    "redstone": 0.72,
}


def _ore_label(ore_key):
    return ore_key.capitalize()


def _load_template_bank_for_ore(ore_key):
    bank = {}
    for name in os.listdir(TEMPLATES_DIR):
        if not name.endswith(".png"):
            continue
        if not name.startswith(f"{ore_key}_"):
            continue

        variant = name[:-4]
        bank[variant] = load_template(os.path.join(TEMPLATES_DIR, name))
    return bank


def run_eval():
    for shot in ["test1.png", "test2.png"]:
        img = load_image(os.path.join(DATA_DIR, "screenshots", shot))
        img = blur(apply_clahe(img))
        hsv = to_hsv(img)
        edges = edge_mask(img)

        all_candidates = []
        all_raw = []

        for ore in supported_ores():
            template_bank = _load_template_bank_for_ore(ore)
            if len(template_bank) == 0:
                continue

            color = color_mask(hsv, ore)
            mask = hybrid_mask(color, edges) if use_edges_for_ore(ore) else color
            mask = clean_mask(refine_mask_for_ore(ore, mask))
            candidates = find_candidates(mask, color)
            all_candidates.extend(candidates)

            raw = detect_with_template_bank(
                img,
                candidates,
                template_bank,
                label=_ore_label(ore),
                threshold=ORE_MATCH_THRESHOLD.get(ore, 0.8),
                brightness_split=95.0,
            )
            all_raw.extend(raw)

        final = non_max_suppression(all_raw, iou_threshold=0.25)

        print(f"\n=== {shot} ===")
        print("candidates:", len(all_candidates))
        print(
            "raw detections:",
            len(all_raw),
            [(d["label"], round(d["score"], 3), d.get("variant"), d["box"]) for d in all_raw],
        )
        print(
            "final detections:",
            len(final),
            [(d["label"], round(d["score"], 3), d.get("variant"), d["box"]) for d in final],
        )


if __name__ == "__main__":
    run_eval()










