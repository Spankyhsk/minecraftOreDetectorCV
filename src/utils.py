#Helferfunktionen
import cv2
import time


def resize(img, scale=1.0):
    """
    Skaliert ein Bild.

    Parameters
    ----------
    img : numpy.ndarray
        Eingabebild.

    scale : float
        Skalierungsfaktor.

    Returns
    -------
    numpy.ndarray
        Skaliertes Bild.
    """

    return cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale
    )


def save_image(path, img):
    """
    Speichert ein Bild auf der Festplatte.

    Parameters
    ----------
    path : str
        Speicherpfad.

    img : numpy.ndarray
        Zu speicherndes Bild.
    """

    cv2.imwrite(path, img)

def log(msg):
    print(f"[INFO] {msg}")


def log_debug(msg):
    print(f"[DEBUG] {msg}")


def log_warning(msg):
    print(f"[WARN] {msg}")