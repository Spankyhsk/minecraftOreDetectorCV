#Ergebnisse zeichnen
import cv2


def draw_detections(img, detections, label="ore"):
    """
    Zeichnet Bounding Boxes um erkannte Erze.

    Parameters
    ----------
    img : numpy.ndarray
        Originalbild.

    detections : list
        Liste erkannter Regionen.

    label : str
        Bezeichnung des Erzes.

    Returns
    -------
    numpy.ndarray
        Annotiertes Bild.
    """

    output = img.copy()

    for (x, y, w, h) in detections:

        # Rechteck zeichnen
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Beschriftung hinzufügen
        cv2.putText(
            output,
            label,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    return output


def show(img):
    """
    Zeigt ein Bild in einem OpenCV-Fenster an.

    Das Fenster bleibt offen,
    bis eine Taste gedrückt wird.
    """

    cv2.imshow("VoxelVision", img)

    cv2.waitKey(0)

    cv2.destroyAllWindows()