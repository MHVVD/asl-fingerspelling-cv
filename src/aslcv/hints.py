"""Tutor feedback: measure the hand's shape and say, in words, what to change.

Measures (all from the normalized 63-number features, so position/size/hand don't matter):
  * finger bend: total angle along each finger (knuckle + 2 joints). ~0° straight, ~180° curled.
  * thumb_out:   thumb-tip to index-knuckle distance, in hand sizes.
  * direction:   which way the palm points (wrist -> middle knuckle): up / down / side.
                 Only "down"/"side" become tips: it follows the palm, not the fingers, so "point up"
                 would mislead e.g. a correct G (index sideways, palm up).
  * spread:      angle between the index and middle fingers (U/R together vs V apart).

What each letter *should* look like is learned from real hands (build_table), not from the drawn
reference poses, which are approximations (their X hook, for example, is far more bent than a real X).
A tip is only given where real samples agree, and only for clear mistakes (straight vs curled).
"""
import json
from pathlib import Path
import numpy as np

FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
STRAIGHT_MAX, CURLED_MIN = 70.0, 130.0      # degrees of total bend
THUMB_IN_MAX, THUMB_OUT_MIN = 0.45, 0.65    # hand sizes
TOGETHER_MAX, APART_MIN = 12.0, 22.0        # degrees between index and middle finger
TABLE_PATH = Path(__file__).resolve().parents[2] / "models" / "finger_states.json"
DIRECTION_WORD = {"up": "up", "down": "down", "side": "sideways"}


def _angle(u, v):
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def hand_measures(features):
    p = np.asarray(features, dtype=float).reshape(21, 3)
    fingers = {}
    for name, (m, q, d, t) in FINGERS.items():
        fingers[name] = (_angle(p[m] - p[0], p[q] - p[m]) + _angle(p[q] - p[m], p[d] - p[q])
                         + _angle(p[d] - p[q], p[t] - p[d]))
    vx, vy = p[9, 0] - p[0, 0], p[9, 1] - p[0, 1]           # image y grows downward
    direction = "up" if -vy > abs(vx) else "down" if vy > abs(vx) else "side"
    spread = _angle((p[8] - p[5])[:2], (p[12] - p[9])[:2])
    return {"fingers": fingers, "thumb_out": float(np.linalg.norm(p[4] - p[5])), "direction": direction,
            "spread": spread}


def finger_state(deg):
    return "straight" if deg < STRAIGHT_MAX else "curled" if deg > CURLED_MIN else "between"


def thumb_state(dist):
    return "out" if dist > THUMB_OUT_MIN else "in" if dist < THUMB_IN_MAX else "between"


def spread_state(deg):
    return "together" if deg < TOGETHER_MAX else "apart" if deg > APART_MIN else "between"


def _agreed(values, agree):
    vals, counts = np.unique(values, return_counts=True)
    i = int(np.argmax(counts))
    return str(vals[i]) if counts[i] / len(values) >= agree and vals[i] != "between" else None


def build_table(X, y, agree=0.8):
    """For each letter: the finger/thumb/direction states that at least `agree` of real samples share."""
    table = {}
    for letter in sorted(set(y)):
        ms = [hand_measures(x) for x in X[y == letter]]
        table[str(letter)] = {
            "fingers": {f: _agreed([finger_state(m["fingers"][f]) for m in ms], agree) for f in FINGERS},
            "thumb": _agreed([thumb_state(m["thumb_out"]) for m in ms], agree),
            "direction": _agreed([m["direction"] for m in ms], agree),
        }
        f = table[str(letter)]["fingers"]
        both_straight = f["index"] == "straight" and f["middle"] == "straight"
        table[str(letter)]["spread"] = _agreed([spread_state(m["spread"]) for m in ms], agree) if both_straight else None
    return table


def hints(measures, expected, target=None, seen=None, max_hints=2, seen_conf=0.6):
    """Short, specific instructions, most important first."""
    out = []
    if expected.get("direction") in ("down", "side") and measures["direction"] != expected["direction"]:
        out.append(f"Point your fingers {DIRECTION_WORD[expected['direction']]}")
    for f in FINGERS:
        want, have = expected["fingers"].get(f), measures["fingers"][f]
        if want == "straight" and have > CURLED_MIN:
            out.append(f"Straighten your {f} finger")
        elif want == "curled" and have < STRAIGHT_MAX:
            out.append(f"Curl your {f} finger down")
    if expected.get("spread") == "apart" and measures["spread"] < TOGETHER_MAX:
        out.append("Spread your index and middle fingers apart")
    elif expected.get("spread") == "together" and measures["spread"] > APART_MIN:
        out.append("Hold your index and middle fingers together")
    if expected.get("thumb") == "out" and measures["thumb_out"] < THUMB_IN_MAX:
        out.append("Stick your thumb out")
    elif expected.get("thumb") == "in" and measures["thumb_out"] > THUMB_OUT_MIN:
        out.append("Tuck your thumb in")
    out = out[:max_hints]
    if seen and target and seen[0] and seen[0] != target and seen[1] >= seen_conf:
        out.append(f"That looks like {seen[0]}")
    return out


def save_table(table, path=TABLE_PATH):
    Path(path).write_text(json.dumps(table, indent=1))


def load_table(path=TABLE_PATH):
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    # python -m aslcv.hints : build the expected-shape table from public + recorded data
    from .train import load_all
    X, y, _, _ = load_all()
    t = build_table(X, y)
    save_table(t)
    print(f"saved {TABLE_PATH}")
    for k, v in t.items():
        f = " ".join(f"{n[0].upper()}:{(s or '?')[0]}" for n, s in v["fingers"].items())
        print(f"  {k}  {f}  thumb:{v['thumb'] or '?'}  dir:{v['direction'] or '?'}  spread:{v['spread'] or '-'}")
