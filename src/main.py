#Pipeline
from preprocessing import *
from segmentation import *
from morphology import *
from detection import *
from visualization import *

img = load_image("data/screenshots/test.png")

img = apply_clahe(img)
img = blur(img)

hsv = to_hsv(img)

mask = diamond_mask(hsv)
mask = clean_mask(mask)

detections = find_ores(mask)

out = draw_detections(img, detections, "diamond")

show(out)