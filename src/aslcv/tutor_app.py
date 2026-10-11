"""Mode 1: the fingerspelling tutor.   python -m aslcv.tutor_app [--letters ABCLY] [--hand right]

Window: your mirrored webcam (left) + the target letter's reference hand (right).
Hold the target sign until the bar fills. Tips appear if you're stuck; after 12 s it moves on.

Keys: SPACE skip | ENTER teach (save the next 2 s as examples of the target letter, for when you
are signing it right but the model disagrees) | D debug view | ESC quit (saves progress)
"""
import argparse
import time
from pathlib import Path
import cv2
import numpy as np

from .capture import FrameSource
from .dataset import Recorder
from .features import normalize_landmarks
from .hints import hand_measures, hints, load_table
from .infer import LetterStream
from .landmarks import HandDetector, draw_hand, mirror_hand
from .model import LetterModel
from .pipeline import FramePipeline
from .reference import render_panel
from .tutor import Attempt, Planner, PROGRESS_PATH

ROOT = Path(__file__).resolve().parents[2]
GREEN, RED, AMBER, WHITE, GREY = (90, 220, 90), (60, 60, 230), (80, 200, 255), (255, 255, 255), (170, 170, 170)
TEACH_FRAMES = 60


class TutorApp:
    def __init__(self, detector, model, table, planner, hand="Right", hold_s=0.6, timeout_s=12.0,
                 recorder=None, fps=30.0):
        self.pipe = FramePipeline(detector, model, hand=hand)
        self.stream = LetterStream.for_fps(model.classes, fps)
        self.table, self.planner, self.recorder = table, planner, recorder
        self.hold_s, self.timeout_s = hold_s, timeout_s
        self.debug = False
        self.session = {"correct": 0, "attempts": 0, "streak": 0}
        self.tips, self._tips_t = [], -1.0
        self.flash = None            # (text, color, until_t)
        self.teach_left = 0
        self._new_attempt(0.0)

    # ---- state ----
    def _new_attempt(self, t):
        self.target = self.planner.next()
        self.attempt = Attempt(self.target, self.hold_s, self.timeout_s)
        self.tips, self._tips_t = [], t

    def _finish(self, ok, t):
        self.planner.record(self.target, ok)
        self.session["attempts"] += 1
        self.session["correct"] += int(ok)
        self.session["streak"] = self.session["streak"] + 1 if ok else 0
        self.flash = (("Correct!", GREEN) if ok else (f"Missed {self.target}: study the shape", RED)) + (t + (1.0 if ok else 2.5),)
        self._new_attempt(t)

    def skip(self, t):
        self._finish(False, t)

    def teach(self):
        """Save the next TEACH_FRAMES hand frames as examples of the target."""
        if self.recorder is not None and self.teach_left == 0:
            self.recorder.select(self.target)
            self.recorder.toggle()
            self.teach_left = TEACH_FRAMES

    # ---- one frame ----
    def step(self, frame, t):
        r = self.pipe.process(frame)
        self.stream.update(r.probs, t=t, motion=r.motion)
        seen = self.stream.current
        feats = normalize_landmarks(r.hand.landmarks_px, r.side) if r.hand is not None else None

        if self.teach_left and feats is not None:
            self.recorder.add(feats, raw=r.hand.landmarks_px, model_label=r.hand.handedness)
            self.teach_left -= 1
            if self.teach_left == 0 and self.recorder.recording:
                self.recorder.toggle()

        in_flash = self.flash is not None and t < self.flash[2]
        if not in_flash and not self.teach_left:
            if self.attempt.update(seen, t):
                self._finish(self.attempt.result == "success", t)
            elif feats is not None and self.attempt.elapsed > 2.0 and self.attempt.progress == 0 and t - self._tips_t > 0.5:
                top = self.pipe.model.top(feats)
                self.tips = hints(hand_measures(feats), self.table[self.target], target=self.target, seen=top)
                self._tips_t = t

        view = cv2.flip(frame, 1)
        if r.hand is not None:
            draw_hand(view, mirror_hand(r.hand))
        in_flash = self.flash is not None and t < self.flash[2]   # re-check: an attempt may have just finished
        self._draw(view, r, seen, feats, t, in_flash)
        panel = render_panel(self.target, width=300, height=view.shape[0], footer="SPACE skip  ENTER teach  D debug")
        return np.hstack([view, panel])

    # ---- drawing ----
    def _draw(self, img, r, seen, feats, t, in_flash):
        h, w = img.shape[:2]
        top = img[0:70]
        top[:] = (top * 0.4).astype(np.uint8)
        cv2.putText(img, "Sign:", (14, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)
        cv2.putText(img, self.target, (100, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.7, AMBER, 3, cv2.LINE_AA)
        s = self.session
        cv2.putText(img, f"{s['correct']}/{s['attempts']} correct   streak {s['streak']}", (w - 290, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
        remaining = max(0.0, self.timeout_s - self.attempt.elapsed)
        cv2.putText(img, f"{remaining:4.1f}s", (w - 290, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREY, 1, cv2.LINE_AA)
        # hold bar
        cv2.rectangle(img, (170, 30), (330, 46), GREY, 1)
        cv2.rectangle(img, (170, 30), (170 + int(160 * self.attempt.progress), 46), GREEN, -1)

        if self.teach_left:
            self._banner(img, f"Saving examples of {self.target}... keep signing ({self.teach_left})", AMBER)
        elif in_flash:
            self._banner(img, self.flash[0], self.flash[1])
        elif r.hand is None:
            self._banner(img, "Show your hand to the camera", GREY)
        elif self.tips:
            self._banner(img, self.tips, WHITE)

        if self.debug:
            self._debug(img, r, seen, feats)

    @staticmethod
    def _banner(img, lines, color):
        """One message per line at the bottom of the view (tips stack instead of running off-screen)."""
        lines = [lines] if isinstance(lines, str) else lines
        h, step = img.shape[0], 30
        bar = img[h - 20 - step * len(lines):h]
        bar[:] = (bar * 0.35).astype(np.uint8)
        for i, line in enumerate(lines):
            y = h - 18 - step * (len(lines) - 1 - i)
            cv2.putText(img, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)

    def _debug(self, img, r, seen, feats):
        x0, y0 = 10, 82
        box = img[y0:y0 + 190, x0:x0 + 250]
        box[:] = (box * 0.3).astype(np.uint8)
        lines = [f"smoothed: {seen[0] or '-'} {seen[1]:.2f}"]
        if feats is not None:
            for letter, p in self.pipe.model.top(feats, k=3):
                lines.append(f"frame {letter}: {p:.2f}")
            m = hand_measures(feats)
            lines.append("bend I/M/R/P: " + " ".join(f"{v:.0f}" for v in m["fingers"].values()))
            lines.append(f"thumb out {m['thumb_out']:.2f}  spread {m['spread']:.0f}")
            lines.append(f"palm {m['direction']}  hand {r.side}  motion {r.motion:.2f}")
        for i, line in enumerate(lines):
            cv2.putText(img, line, (x0 + 8, y0 + 22 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, AMBER, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--letters", default="ABCDEFGHIJKLMNOPQRSTUVWXYZ", help="practise only these letters")
    ap.add_argument("--hand", choices=["right", "left"], default="right")
    ap.add_argument("--model", default=str(ROOT / "models" / "letter_mlp_final.npz"))
    ap.add_argument("--hold", type=float, default=0.6, help="seconds to hold a sign")
    ap.add_argument("--timeout", type=float, default=12.0, help="seconds before moving on")
    a = ap.parse_args()

    planner = Planner(a.letters, path=PROGRESS_PATH)
    recorder = Recorder(ROOT / "data" / "recorded", session=f"practice-{time.strftime('%Y%m%d-%H%M%S')}", target=10**6)
    det = HandDetector()
    app = TutorApp(det, LetterModel.load(a.model), load_table(), planner, hand=a.hand.capitalize(),
                   hold_s=a.hold, timeout_s=a.timeout, recorder=recorder)
    t0 = time.perf_counter()
    with FrameSource(0) as src:
        while True:
            frame = src.read()
            if frame is None:
                break
            t = time.perf_counter() - t0
            cv2.imshow("ASL tutor", app.step(frame, t))
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord(" "):
                app.skip(t)
            elif key in (13, 10):
                app.teach()
            elif key in (ord("d"), ord("D")):
                app.debug = not app.debug
    planner.save(PROGRESS_PATH)
    saved = recorder.save()
    det.close()
    cv2.destroyAllWindows()
    s = planner.summary()
    print(f"session: {app.session['correct']}/{app.session['attempts']} correct | letters tried {s['tried']}, "
          f"mastered {s['mastered']} | practise next: {' '.join(s['weakest'])}")
    if saved:
        print(f"saved taught examples to {saved} (included in the next training run)")


if __name__ == "__main__":
    main()
