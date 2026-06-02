#Ergebnisse zeichnen
import cv2
import tkinter as tk


def draw_debug(img, candidates, detections):

    out = img.copy()

    # =========================
    # 1. Kandidaten (BLAU)
    # =========================
    for (x, y, w, h) in candidates:

        cv2.rectangle(
            out,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),  # Blau
            1
        )

        cv2.putText(
            out,
            "C",
            (x, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1
        )

    # =========================
    # 2. Treffer (GRÜN)
    # =========================
    for d in detections:

        x, y, w, h = d["box"]

        cv2.rectangle(
            out,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),  # Grün
            2
        )

        text = f"{d['label']} {d['score']:.2f}"

        cv2.putText(
            out,
            text,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    return out

def draw_candidates(img, candidates):

    out = img.copy()

    for (x, y, w, h) in candidates:

        # Rechteck um jeden Kandidaten
        cv2.rectangle(
            out,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),  # blau = Kandidat
            2
        )

        # kleine Debug-Markierung
        cv2.putText(
            out,
            "cand",
            (x, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1
        )

    return out
def draw(img, detections):
    # Kopie erstellen, damit Originalbild unverändert bleibt
    out = img.copy()

    for d in detections:
        x, y, w, h = d["box"]

        # Rechteck um erkanntes Objekt
        cv2.rectangle(
            out,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        # Text mit Label + Confidence Score
        text = f'{d["label"]}: {d["score"]:.2f}'

        cv2.putText(
            out,
            text,
            (x, y-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    return out

def show(img, window_name="VoxelVision"):

    # =========================
    # Bildschirmgröße holen
    # =========================
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    img_h, img_w = img.shape[:2]

    # =========================
    # Skalierung berechnen
    # =========================
    scale = min(
        screen_width / img_w,
        screen_height / img_h,
        1.0  # nie hochskalieren, nur runterskalieren
    )

    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    resized = cv2.resize(img, (new_w, new_h))

    # =========================
    # Fenster anpassen
    # =========================
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, new_w, new_h)

    # =========================
    # anzeigen
    # =========================
    cv2.imshow(window_name, resized)

    cv2.waitKey(0)
    cv2.destroyAllWindows()