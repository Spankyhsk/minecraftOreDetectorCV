#Erze finden
import cv2


def find_ores(mask):
    """
    Findet zusammenhängende Regionen
    innerhalb einer Binärmaske.

    Jede Region wird später als
    möglicher Erzblock betrachtet.

    Parameters
    ----------
    mask : numpy.ndarray
        Binärmaske.

    Returns
    -------
    list
        Liste von Bounding Boxes.
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    for contour in contours:

        # Kleinste umschließende Box berechnen
        x, y, w, h = cv2.boundingRect(contour)

        # Sehr kleine Regionen ignorieren
        if w * h > 50:
            detections.append(
                (x, y, w, h)
            )

    return detections