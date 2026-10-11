"""Stage 2: MediaPipe HandLandmarker wrapper -> (21,3) landmarks + handedness + overlay."""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"

# Standard hand skeleton edges (wrist -> thumb/index/middle/ring/pinky chains + palm).
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


@dataclass
class Hand:
    landmarks: np.ndarray     # (21,3) x,y normalized to image [0,1]; z relative depth (for drawing)
    landmarks_px: np.ndarray  # (21,3) same points in pixel units, aspect-correct (use for features)
    handedness: str           # "Left" / "Right" as seen by the model
    score: float


def to_pixel_space(landmarks, width, height):
    """MediaPipe x is a fraction of width and y of height, so on a 4:3 frame the hand shape is
    squashed relative to a square image. Scaling back to pixels makes shapes comparable across
    image sizes. z is on roughly the same scale as x, so it is scaled by width."""
    return np.asarray(landmarks, dtype=np.float64) * np.array([width, height, width])


class HandDetector:
    def __init__(self, model_path=MODEL_PATH, max_hands=1, min_conf=0.5, mode="video"):
        """mode="video": frames from a stream (tracks between frames, needs increasing timestamps).
        mode="image": independent photos (detects from scratch every time)."""
        self.mode = mode
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self._t_ms = 0

    def detect(self, frame_bgr, t_ms=None):
        """Returns list[Hand] (empty if none). Frame is BGR from OpenCV.
        t_ms: the frame's real timestamp in VIDEO mode (e.g. from a video file); default +33 ms per call."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # MediaPipe expects RGB
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self.mode == "video":
            # VIDEO mode needs strictly increasing timestamps
            self._t_ms = max(int(t_ms), self._t_ms + 1) if t_ms is not None else self._t_ms + 33
            res = self._lm.detect_for_video(image, self._t_ms)
        else:
            res = self._lm.detect(image)
        h, w = frame_bgr.shape[:2]
        hands = []
        for pts, hd in zip(res.hand_landmarks, res.handedness):
            arr = np.array([[p.x, p.y, p.z] for p in pts])
            hands.append(Hand(arr, to_pixel_space(arr, w, h), hd[0].category_name, hd[0].score))
        return hands

    def close(self):
        self._lm.close()


def mirror_hand(hand):
    """The same hand as it appears in a horizontally flipped (mirror-view) frame, for drawing only.
    Detection always runs on the camera's real image; only the on-screen view is mirrored."""
    lm = hand.landmarks.copy()
    lm[:, 0] = 1.0 - lm[:, 0]
    return Hand(lm, hand.landmarks_px, hand.handedness, hand.score)


def draw_hand(frame, hand, color=(0, 255, 0)):
    """Draws skeleton + keypoints onto frame in place."""
    h, w = frame.shape[:2]
    xy = (hand.landmarks[:, :2] * [w, h]).astype(int)
    for a, b in CONNECTIONS:
        cv2.line(frame, tuple(xy[a]), tuple(xy[b]), color, 2)
    for p in xy:
        cv2.circle(frame, tuple(p), 4, (0, 0, 255), -1)
    return frame
