import time
import cv2
import mss
import numpy as np

from main import run_pipeline


SHOW_FPS = True


def main():

    print("[INFO] Starte Live-Modus")

    sct = mss.mss()

    print("Verfügbare Monitore:")

    for i, mon in enumerate(sct.monitors[1:], start=1):
        print(f"{i}: {mon}")

    monitor_id = int(input("Minecraft Monitor: "))
    region = sct.monitors[monitor_id]

    print("[INFO] Starte Capture... (ESC zum Beenden)")

    last_time = time.time()

    while True:

        try:
            screenshot = sct.grab(region)

            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            output = run_pipeline(frame)

            if SHOW_FPS:
                now = time.time()
                fps = 1.0 / max(now - last_time, 0.0001)
                last_time = now

                cv2.putText(
                    output,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("VoxelVision Live", output)

        except Exception as e:
            print(f"[ERROR] Pipeline Fehler: {e}")

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()
    print("[INFO] Live-Modus beendet")


if __name__ == "__main__":
    main()