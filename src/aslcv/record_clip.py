"""Record a webcam clip to test the video translator.   python -m aslcv.record_clip

Keys: SPACE start/stop recording (each recording is a new file in clips/) | ESC quit
The window is mirrored for comfort, but the file stores the camera's real (unmirrored) image,
exactly like any other video, so the translator treats it the same way as real-world footage.
"""
import time
from pathlib import Path
import cv2

from .capture import FrameSource

OUT_DIR = Path(__file__).resolve().parents[2] / "clips"
TIPS = ["Spell slowly and clearly, ~1 letter per second",
        "Drop your hand out of view between words",
        "Double letters: small bounce or slide sideways"]


def main():
    OUT_DIR.mkdir(exist_ok=True)
    writer, path, started = None, None, 0.0
    with FrameSource(0) as src:
        fps = src.cap.get(cv2.CAP_PROP_FPS) or 30.0
        while True:
            frame = src.read()
            if frame is None:
                break
            if writer is not None:
                writer.write(frame)                     # real image, not mirrored
            view = cv2.flip(frame, 1)
            if writer is not None:
                cv2.circle(view, (24, 26), 9, (0, 0, 255), -1)
                cv2.putText(view, f"REC {time.time() - started:4.1f}s   SPACE to stop", (42, 33),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(view, "SPACE to start recording   ESC to quit", (12, 33),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                for i, tip in enumerate(TIPS):
                    cv2.putText(view, tip, (12, view.shape[0] - 20 - 24 * (len(TIPS) - 1 - i)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
            cv2.imshow("Record a test clip", view)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord(" "):
                if writer is None:
                    path = OUT_DIR / f"clip-{time.strftime('%Y%m%d-%H%M%S')}.mp4"
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    started = time.time()
                else:
                    writer.release(); writer = None
                    print("saved", path)
    if writer is not None:
        writer.release()
        print("saved", path)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
