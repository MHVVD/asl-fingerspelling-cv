"""Export what the in-browser tutor needs (journal 20): model weights, tip table, reference hands, test vectors.

    PYTHONPATH=src python tools/export_web.py OUT_DIR

Writes OUT_DIR/asl-model.json (per-frame letter model + finger-state table + tip thresholds),
OUT_DIR/ref/A.png .. Z.png (the drawn reference hands) and OUT_DIR/test-vectors.json: real recorded hands as
MediaPipe would report them, with the features, probabilities and tips this Python code computes for them, so the
JavaScript port can be checked for identical results. Only the per-frame model is exported: it is trained on my own
recordings and the public ASL Alphabet set; the sequence model and the detector (ChicagoFSWild) stay local.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from aslcv import hints
from aslcv.features import normalize_landmarks
from aslcv.model import LetterModel
from aslcv.reference import render_panel

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "letter_mlp_final.npz"
CAM_W, CAM_H = 640, 480            # the recording webcam: raw points are in these pixel units


def r(a, nd=6):
    return np.round(np.asarray(a, np.float64), nd).tolist()


def main(out):
    out = Path(out)
    (out / "ref").mkdir(parents=True, exist_ok=True)
    m = LetterModel.load(MODEL)
    model = {"classes": [str(c) for c in m.classes],
             "layers": [{"W": r(W), "b": r(b)} for W, b in zip(m.weights, m.biases)],
             "finger_states": hints.load_table(),
             "thresholds": {"straight_max": hints.STRAIGHT_MAX, "curled_min": hints.CURLED_MIN,
                            "thumb_in_max": hints.THUMB_IN_MAX, "thumb_out_min": hints.THUMB_OUT_MIN,
                            "together_max": hints.TOGETHER_MAX, "apart_min": hints.APART_MIN},
             "source": "github.com/MHVVD/asl-fingerspelling-cv models/letter_mlp_final.npz (MIT)"}
    (out / "asl-model.json").write_text(json.dumps(model, separators=(",", ":")))

    for letter in model["classes"]:
        panel = render_panel(letter, width=240, height=300, footer="")
        cv2.imwrite(str(out / "ref" / f"{letter}.png"), panel, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    d = np.load(ROOT / "data" / "recorded" / "20261011-020828.npz", allow_pickle=True)
    rng = np.random.default_rng(0)
    cases = []
    for i in rng.choice(len(d["y"]), 40, replace=False):
        px = d["raw"][i].astype(np.float64)
        norm = px / [CAM_W, CAM_H, CAM_W]                      # what MediaPipe reports (0..1 of width/height)
        side = "Right" if i % 2 == 0 else "Left"               # exercise both settings of the hand switch
        feats = normalize_landmarks(norm * [CAM_W, CAM_H, CAM_W], side)
        probs = m.predict_proba(feats)
        meas = hints.hand_measures(feats)
        target = str(d["y"][(i + 7) % len(d["y"])])
        cases.append({"landmarks": r(norm, 8), "width": CAM_W, "height": CAM_H, "side": side,
                      "features": r(feats), "probs": r(probs, 7), "top": str(m.classes[int(np.argmax(probs))]),
                      "target": target, "hints": hints.hints(meas, model["finger_states"][target], target,
                                                             (str(m.classes[int(np.argmax(probs))]), float(probs.max())))})
    (out / "test-vectors.json").write_text(json.dumps(cases))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"wrote {out}: model {(out / 'asl-model.json').stat().st_size // 1024} KB, 26 reference images, "
          f"{len(cases)} test vectors, {size // 1024} KB in all")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "web-export")
