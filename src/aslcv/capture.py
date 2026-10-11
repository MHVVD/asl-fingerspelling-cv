"""Stage 1: frame source abstraction (webcam index or video file) with FPS measurement."""
import time
import cv2


class FrameSource:
    def __init__(self, source=0, width=640, height=480):
        self.cap = cv2.VideoCapture(source)
        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source {source!r}")
        self._last = time.perf_counter()
        self.fps = 0.0

    def read(self):
        """Returns a BGR frame (H,W,3 uint8) or None at end of stream."""
        ok, frame = self.cap.read()
        if not ok:
            return None
        now = time.perf_counter()
        inst = 1.0 / max(now - self._last, 1e-6)
        self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst  # smoothed
        self._last = now
        return frame

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
