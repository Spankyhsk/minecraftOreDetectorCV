#Ergebnisse zeichnen
import cv2


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


def show(img):
    # OpenCV GUI-Funktion
    # zeigt Bild in Fenster

    cv2.imshow("VoxelVision", img)

    # wartet bis Taste gedrückt wird
    cv2.waitKey(0)

    # schließt Fenster
    cv2.destroyAllWindows()