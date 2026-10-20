"""Score every model ONCE on the fresh real-world clips whose answer keys were written beforehand.

    python -m aslcv.score_fresh

Keys: artifacts/fresh_test_keys.json (written by eye, before any model saw the clips; journal 13).
For each key window: detect both hands (MediaPipe, VIDEO mode), keep the signing hand's track for the window,
then decode with each model on the same frames. Results: artifacts/fresh_test_results.json.
"""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

from .extract_fsw import select_signing_track
from .features import majority_handedness
from .infer import LetterStream, transcript
from .landmarks import HandDetector
from .model import LetterModel
from .score_clip import align
from .seqmodel import SeqModel, sequence_features
from .simulate import char_error_rate

ROOT = Path(__file__).resolve().parents[2]
PER_FRAME_TUNED = {"hold_s": 0.10, "enter": 0.4}           # best per-frame settings, chosen on ChicagoFSWild dev


def window_hands(path, t0, t1):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    det, hands, i = HandDetector(max_hands=2, mode="video"), [], 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / fps
            if t0 <= t <= t1:
                hands.append(det.detect(frame, t_ms=t * 1000))
            elif t > t1:
                break
            i += 1
    finally:
        det.close()
    return hands, fps


def per_frame_decode(model, F, fps):
    s = LetterStream.for_fps(model.classes, fps, hold_s=PER_FRAME_TUNED["hold_s"], enter=PER_FRAME_TUNED["enter"])
    ev = []
    for t, row in enumerate(F):
        p = model.predict_proba(row[:63]) if row[65] else None
        ev += s.update(p, t=t / fps, motion=float(np.linalg.norm(row[63:65])))
    return transcript(ev)


def main():
    keys = json.loads((ROOT / "artifacts" / "fresh_test_keys.json").read_text())
    models = {
        "per-frame (dev-tuned)": ("frame", LetterModel.load(ROOT / "models" / "letter_mlp_final.npz")),
        "sequence": ("seq", SeqModel.load(ROOT / "models" / "seq_fsw.npz")),
        "sequence + slow signing": ("seq", SeqModel.load(ROOT / "models" / "seq_fsw_slowaug2.npz")),
    }
    rows = []
    for k in keys["clips"]:
        path = ROOT / k["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == k["sha256"], f"{path} changed since its key was written"
        hands, fps = window_hands(path, *k["window_s"])
        chosen = select_signing_track(hands)
        side = majority_handedness([h.handedness for h in chosen if h is not None])
        F = sequence_features([None if h is None else h.landmarks_px for h in chosen], side)
        row = {"clip": Path(k["file"]).name, "truth": k["truth"], "frames": len(F),
               "hand_found": round(float(F[:, 65].mean()), 3), "handedness": side, "models": {}}
        for name, (kind, m) in models.items():
            out = (per_frame_decode(m, F, fps) if kind == "frame" else m.decode(F)).upper().replace(" ", "")
            al = align(k["truth"], out)
            row["models"][name] = {"output": out, "letters_right": sum(o == "ok" for *_, o in al),
                                   "edits": round(char_error_rate(k["truth"], out) * len(k["truth"]))}
        rows.append(row)

    total = sum(len(r["truth"]) for r in rows)
    summary = {}
    for name in models:
        right = sum(r["models"][name]["letters_right"] for r in rows)
        edits = sum(r["models"][name]["edits"] for r in rows)
        summary[name] = {"letters_right": right, "letters_total": total, "letter_accuracy": round(1 - edits / total, 3)}
    out = {"keys_file": "artifacts/fresh_test_keys.json", "per_frame_settings": PER_FRAME_TUNED, "clips": rows, "summary": summary}
    (ROOT / "artifacts" / "fresh_test_results.json").write_text(json.dumps(out, indent=1))

    for r in rows:
        print(f"\n{r['clip']}  truth {r['truth']}  ({r['frames']} frames, hand {r['hand_found']:.0%}, {r['handedness']})")
        for name, m in r["models"].items():
            print(f"  {name:<26} {m['output']:<18} {m['letters_right']}/{len(r['truth'])} right, {m['edits']} edits")
    print("\nTOTAL over", total, "letters:")
    for name, s in summary.items():
        print(f"  {name:<26} {s['letters_right']}/{total} right   letter accuracy {s['letter_accuracy']:.1%}")


if __name__ == "__main__":
    main()
