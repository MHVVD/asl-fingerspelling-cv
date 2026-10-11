"""Stage 4 UI: record your own labeled hand samples.   python -m aslcv.record

Window: live webcam (left) + reference hand shape for the selected letter (right).
Keys:  A-Z select letter | SPACE start/stop recording | BACKSPACE undo last burst
       ENTER save now | ESC save and quit
(Control keys are deliberately not letters, so every letter A-Z can be selected.)
"""
import string
from collections import deque
from pathlib import Path
import cv2
import numpy as np
from .capture import FrameSource
from .dataset import Recorder
from .features import normalize_landmarks, resolve_handedness
from .landmarks import HandDetector, draw_hand, mirror_hand
from .reference import render_panel

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "recorded"
KEY_ESC, KEY_ENTER, KEY_BACKSPACE = 27, 13, 8


class HandMismatch:
    """Warns when MediaPipe has mostly seen the OTHER hand recently. Single odd labels are normal
    (e.g. downward P/Q), but a consistent disagreement means the wrong hand, the back of the hand,
    or a flipped camera image, and the samples would not match the rest of the data."""
    def __init__(self, declared, window=15, share=0.8):
        self.declared, self.share = declared, share
        self.labels = deque(maxlen=window)

    def update(self, label):
        self.labels.append(label)

    @property
    def seen(self):
        return max(set(self.labels), key=self.labels.count) if self.labels else None

    @property
    def warning(self):
        if len(self.labels) < self.labels.maxlen:
            return False
        return sum(l != self.declared for l in self.labels) / len(self.labels) >= self.share


def main(target=150, hand="Right"):
    rec = Recorder(OUT_DIR, target=target)
    det = HandDetector()
    check = HandMismatch(hand)
    with FrameSource(0) as src:
        while True:
            frame = src.read()
            if frame is None:
                break
            # Detect on the camera's REAL image (like the public photos and any video file), so MediaPipe's
            # "Right" means the signer's right hand. Only the on-screen view is mirrored (journal 07).
            hands = det.detect(frame)
            frame = cv2.flip(frame, 1)
            stored = False
            if hands:
                h = hands[0]
                draw_hand(frame, mirror_hand(h))
                # use the signer's declared hand, not MediaPipe's per-frame guess (unreliable for P/Q)
                feats = normalize_landmarks(h.landmarks_px, resolve_handedness(h.handedness, hand))
                stored = rec.add(feats, raw=h.landmarks_px, model_label=h.handedness)
                check.update(h.handedness)
            _hud(frame, rec, bool(hands), stored, src.fps)
            _hand_check(frame, hand, check)
            panel = render_panel(rec.letter, width=300, height=frame.shape[0])
            cv2.imshow("ASL recorder", np.hstack([frame, panel]))
            key = cv2.waitKey(1) & 0xFF
            if key == KEY_ESC:
                break
            elif key == ord(" "):
                rec.toggle()
            elif key in (KEY_BACKSPACE, 127):  # 127 = Backspace on some keyboards
                rec.undo()
            elif key in (KEY_ENTER, 10):
                print("saved", rec.save())
            elif chr(key) in string.ascii_letters:
                rec.select(chr(key))
    print("saved", rec.save())
    det.close()
    cv2.destroyAllWindows()


def _hud(frame, rec, hand_found, stored, fps):
    color = (0, 0, 255) if rec.recording else (255, 255, 255)
    status = "REC" if rec.recording else "idle"
    cv2.putText(frame, f"Letter: {rec.letter or '-'}  [{status}]  "
                f"{rec.counts[rec.letter] if rec.letter else 0}/{rec.target}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, "hand: " + ("yes" if hand_found else "NO"), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if hand_found else (0, 0, 255), 2)
    done = sum(1 for v in rec.counts.values() if v >= rec.target)
    cv2.putText(frame, f"letters complete: {done}/26   fps {fps:.0f}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def _hand_check(frame, declared, check):
    if check.seen is None:
        return
    if check.warning:
        h = frame.shape[0]
        cv2.rectangle(frame, (0, h - 70), (frame.shape[1], h), (0, 0, 160), -1)
        cv2.putText(frame, f"CAMERA SEES A {check.seen.upper()} HAND, you declared {declared.lower()}",
                    (10, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Use your declared hand, palm to camera (or restart with --hand)",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        cv2.putText(frame, f"camera sees: {check.seen} hand", (10, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Record labeled hand samples.")
    ap.add_argument("--hand", choices=["right", "left"], default="right", help="the hand you sign with")
    ap.add_argument("--target", type=int, default=150, help="samples per letter")
    args = ap.parse_args()
    main(target=args.target, hand=args.hand.capitalize())
