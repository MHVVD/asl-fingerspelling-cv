"""Reference hand poses for the ASL alphabet, built from a simple kinematic hand model.

Each letter is a spec: per-finger flexion (MCP, PIP, DIP angles in degrees), finger spread,
a thumb-tip target, an optional in-plane roll (for sideways/downward letters) and an optional
motion path (J, Z). The model outputs the same (21,3) landmark layout as MediaPipe, so the
tutor can later compare a live hand against these poses.

Coordinates: wrist at origin, x to the right, y DOWN (image convention), z toward the viewer.
Drawn as the signer sees themselves in the mirrored webcam view: right hand, palm facing camera,
thumb on the left of the screen. These are approximations for guidance, not exact photographs.

    python -m aslcv.reference   # saves artifacts/reference_alphabet.png
"""
from pathlib import Path
import cv2
import numpy as np

FINGERS = ["index", "middle", "ring", "pinky"]
FINGER_BASE = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
MCP_POS = {"index": (-0.38, -1.0), "middle": (-0.12, -1.04), "ring": (0.12, -1.0), "pinky": (0.34, -0.9)}
SEG_LEN = {"index": (0.45, 0.27, 0.20), "middle": (0.50, 0.30, 0.22),
           "ring": (0.47, 0.28, 0.20), "pinky": (0.37, 0.22, 0.18)}
DEFAULT_SPREAD = {"index": -6, "middle": 0, "ring": 6, "pinky": 14}
THUMB_CMC = np.array([-0.30, -0.22, 0.05])

# Finger flexion presets: (MCP, PIP, DIP) degrees. 0 = straight; flexion curls toward the viewer.
EXT = (0, 0, 0)
CURL = (80, 100, 60)      # fist
BENT = (25, 100, 75)      # E: fingertips bent down
HOOK = (10, 85, 60)       # X
CURVE_C = (25, 35, 25)
CURVE_O = (40, 55, 40)
FWD = (40, 0, 0)          # K/P middle finger angled forward

TOGETHER = {"index": -2, "middle": 0, "ring": 2, "pinky": 4}
THUMB_ACROSS = (0.08, -0.80, 0.30)


def _spec(text, fingers, thumb, spread=None, roll=0, motion=None):
    return dict(text=text, fingers=fingers, thumb=thumb, spread=spread or {}, roll=roll, motion=motion)


def _all(flex):
    return {f: flex for f in FINGERS}


def _only(up, rest=CURL, flex=EXT):
    return {f: (flex if f in up else rest) for f in FINGERS}


LETTER_SPECS = {
    "A": _spec("Fist, thumb straight up against the side of the index finger.", _all(CURL), (-0.52, -1.00, 0.12)),
    "B": _spec("Flat hand, fingers straight up and together; thumb folded across the palm.",
               _all(EXT), (0.02, -0.72, 0.22), spread=TOGETHER),
    "C": _spec("Curve fingers and thumb into a C shape (from the side it looks like a C).",
               _all(CURVE_C), (-0.55, -0.90, 0.45), spread=TOGETHER),
    "D": _spec("Index straight up; the other fingers curve to touch the thumb tip in a circle.",
               _only({"index"}, rest=CURVE_O), ("touch", "middle")),
    "E": _spec("Fingertips bent down onto the thumb, which is tucked across the palm.",
               _all(BENT), (-0.02, -0.78, 0.30), spread=TOGETHER),
    "F": _spec("Index and thumb touch in a circle; the other three fingers straight up and spread.",
               {"index": CURVE_O, "middle": EXT, "ring": EXT, "pinky": EXT}, ("touch", "index"),
               spread={"middle": 0, "ring": 9, "pinky": 18}),
    "G": _spec("Hand sideways: index points to the side, thumb parallel to it; others curled.",
               _only({"index"}), (-0.62, -1.35, 0.05), roll=-90),
    "H": _spec("Hand sideways: index and middle straight and together, pointing to the side.",
               _only({"index", "middle"}), THUMB_ACROSS, spread=TOGETHER, roll=-90),
    "I": _spec("Fist with the pinky straight up; thumb across the curled fingers.",
               _only({"pinky"}), (-0.05, -0.85, 0.45)),
    "J": _spec("Start in I, then trace a J in the air with the pinky.",
               _only({"pinky"}), (-0.05, -0.85, 0.45), motion="J"),
    "K": _spec("Index up, middle angled forward, thumb tip between them on the middle finger.",
               {"index": EXT, "middle": FWD, "ring": CURL, "pinky": CURL}, (-0.22, -1.35, 0.30),
               spread={"index": -10, "middle": 8}),
    "L": _spec("Index straight up and thumb straight out to the side: an L shape.",
               _only({"index"}), (-0.98, -0.50, 0.0)),
    "M": _spec("Thumb tucked under index, middle and ring; its tip peeks out between ring and pinky.",
               _all(CURL), (0.24, -0.82, 0.12)),
    "N": _spec("Thumb tucked under index and middle; its tip peeks out between middle and ring.",
               _all(CURL), (0.0, -0.82, 0.12)),
    "O": _spec("All fingertips curve down to meet the thumb tip, forming an O.",
               _all(CURVE_O), ("touch", "index"), spread=TOGETHER),
    "P": _spec("Like K, but the hand is turned to point downward.",
               {"index": EXT, "middle": FWD, "ring": CURL, "pinky": CURL}, (-0.22, -1.35, 0.30),
               spread={"index": -10, "middle": 8}, roll=170),
    "Q": _spec("Like G, but the index and thumb point downward.",
               _only({"index"}), (-0.62, -1.35, 0.05), roll=180),
    "R": _spec("Index and middle straight up and crossed, middle over index.",
               _only({"index", "middle"}), THUMB_ACROSS, spread={"index": 4, "middle": -10}),
    "S": _spec("Fist with the thumb wrapped across the front of the fingers.", _all(CURL), (-0.08, -0.92, 0.55)),
    "T": _spec("Fist with the thumb tucked between index and middle, tip poking up.",
               _all(CURL), (-0.30, -1.02, 0.42)),
    "U": _spec("Index and middle straight up and pressed together.",
               _only({"index", "middle"}), THUMB_ACROSS, spread={"index": -1, "middle": 1}),
    "V": _spec("Index and middle straight up and spread into a V.",
               _only({"index", "middle"}), THUMB_ACROSS, spread={"index": -14, "middle": 12}),
    "W": _spec("Index, middle and ring straight up and spread; thumb holds the pinky down.",
               _only({"index", "middle", "ring"}), ("touch", "pinky"),
               spread={"index": -14, "middle": 0, "ring": 14}),
    "X": _spec("Fist with the index finger raised and bent into a hook.",
               {"index": HOOK, "middle": CURL, "ring": CURL, "pinky": CURL}, (-0.05, -0.85, 0.40)),
    "Y": _spec("Thumb and pinky stretched out, other fingers curled ('hang loose').",
               _only({"pinky"}), (-0.98, -0.55, 0.0), spread={"pinky": 28}),
    "Z": _spec("Point with the index finger and trace a Z in the air.",
               _only({"index"}), (0.0, -0.85, 0.40), motion="Z"),
}


def _finger_chain(name, flex, spread_deg):
    """Forward kinematics for one finger: MCP, PIP, DIP, TIP points."""
    s = np.radians(spread_deg)
    along = np.array([np.sin(s), -np.cos(s), 0.0])   # straight-finger direction in the palm plane
    toward_viewer = np.array([0.0, 0.0, 1.0])
    pts = [np.array([*MCP_POS[name], 0.0])]
    total = 0.0
    for length, angle in zip(SEG_LEN[name], flex):
        total += np.radians(angle)                    # angles accumulate along the chain
        direction = along * np.cos(total) + toward_viewer * np.sin(total)
        pts.append(pts[-1] + length * direction)
    return pts


def reference_pose(letter):
    """Returns (21,3) landmarks for a letter in MediaPipe's index layout."""
    spec = LETTER_SPECS[letter.upper()]
    pose = np.zeros((21, 3))
    for f in FINGERS:
        spread = spec["spread"].get(f, DEFAULT_SPREAD[f])
        b = FINGER_BASE[f]
        pose[b:b + 4] = _finger_chain(f, spec["fingers"][f], spread)

    thumb = spec["thumb"]
    if thumb[0] == "touch":                             # thumb tip meets a fingertip
        tip = pose[FINGER_BASE[thumb[1]] + 3] + np.array([0.0, 0.0, 0.02])
    else:
        tip = np.array(thumb, dtype=float)
    bulge = np.array([-0.08, 0.0, 0.06])               # thumb joints bow outward from the palm
    pose[1] = THUMB_CMC
    pose[2] = THUMB_CMC + 0.40 * (tip - THUMB_CMC) + bulge
    pose[3] = THUMB_CMC + 0.72 * (tip - THUMB_CMC) + 0.6 * bulge
    pose[4] = tip

    if spec["roll"]:
        r = np.radians(spec["roll"])
        rot = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
        pose = pose @ rot.T
    return pose


# Bone list shared with the live overlay.
from .landmarks import CONNECTIONS  # noqa: E402

BG = (32, 32, 32)
TEXT = (235, 235, 235)
MUTED = (150, 150, 150)
ACCENT = (80, 200, 255)
HAND_EXTENT = 2.3  # model units spanning the largest pose (wrist to fingertip, thumb out)


def _draw_skeleton(img, pose, box):
    """Draw pose into box=(x0,y0,x1,y1), nearer bones brighter (depth cue)."""
    x0, y0, x1, y1 = box
    xy = pose[:, :2]
    lo, hi = xy.min(0), xy.max(0)
    # One fixed scale for every letter, so a fist really looks smaller than an open hand.
    scale = 0.92 * min(x1 - x0, y1 - y0) / HAND_EXTENT
    center = (lo + hi) / 2
    px = ((xy - center) * scale + [(x0 + x1) / 2, (y0 + y1) / 2]).astype(int)

    z = pose[:, 2]
    zmin, zmax = z.min(), max(z.max(), z.min() + 1e-6)
    bones = sorted(CONNECTIONS, key=lambda e: z[e[0]] + z[e[1]])  # draw far bones first
    for a, b in bones:
        t = ((z[a] + z[b]) / 2 - zmin) / (zmax - zmin)
        color = tuple(int(c) for c in (np.array([70, 120, 70]) * (1 - t) + np.array([120, 255, 140]) * t))
        cv2.line(img, tuple(px[a]), tuple(px[b]), color, 3, cv2.LINE_AA)
    for i, p in enumerate(px):
        color = (60, 140, 255) if i in (4, 8, 12, 16, 20) else (200, 200, 200)
        cv2.circle(img, tuple(p), 4, color, -1, cv2.LINE_AA)
    return px


def _draw_motion(img, kind, anchor, size):
    """Arrow showing the J/Z movement, drawn beside the moving fingertip."""
    ax, ay = anchor
    s = size
    if kind == "Z":
        pts = [(ax, ay), (ax + s, ay), (ax, ay + s), (ax + s, ay + s)]
    else:  # J: down, then hook toward the thumb side
        pts = [(ax + int(s * 0.6), ay), (ax + int(s * 0.6), ay + int(s * 0.7)),
               (ax + int(s * 0.4), ay + s), (ax + int(s * 0.1), ay + int(s * 0.9))]
    for p, q in zip(pts[:-1], pts[1:-1]):
        cv2.line(img, p, q, ACCENT, 2, cv2.LINE_AA)
    cv2.arrowedLine(img, pts[-2], pts[-1], ACCENT, 2, cv2.LINE_AA, tipLength=0.4)


def _wrap(text, max_chars):
    lines, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > max_chars:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + [line] if line else lines


def render_panel(letter, width=300, height=480, footer="SPACE rec  BKSP undo  ESC quit"):
    """BGR panel showing the reference hand for `letter`, or the controls when None."""
    img = np.full((height, width, 3), BG, np.uint8)
    if letter is None:
        cv2.putText(img, "Reference", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT, 2, cv2.LINE_AA)
        help_lines = ["Press a letter key", "(A-Z) to see its", "hand shape here.", "",
                      "SPACE      start/stop rec", "BACKSPACE  undo last burst", "ENTER      save", "ESC        save + quit"]
        for i, line in enumerate(help_lines):
            cv2.putText(img, line, (16, 90 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, MUTED, 1, cv2.LINE_AA)
        return img

    letter = letter.upper()
    spec = LETTER_SPECS[letter]
    cv2.putText(img, letter, (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.8, ACCENT, 3, cv2.LINE_AA)
    cv2.putText(img, "mirror view", (width - 105, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, MUTED, 1, cv2.LINE_AA)
    sk_bottom = int(height * 0.62)
    px = _draw_skeleton(img, reference_pose(letter), (16, 75, width - 16, sk_bottom))
    if spec["motion"]:
        tip = px[20] if spec["motion"] == "J" else px[8]
        size = 40
        # place the motion arrow beside the fingertip, clear of the hand
        x = min(tip[0] + 22, width - size - 8) if spec["motion"] == "J" else min(px[:, 0].max() + 14, width - size - 8)
        _draw_motion(img, spec["motion"], (int(x), int(max(tip[1] - 20, 80))), size)
    for i, line in enumerate(_wrap(spec["text"], max(18, int(width / 9.5)))):
        cv2.putText(img, line, (16, sk_bottom + 26 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT, 1, cv2.LINE_AA)
    if height >= 420:  # key reminder in the live window (skipped on small alphabet-sheet tiles)
        cv2.putText(img, footer, (16, height - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, MUTED, 1, cv2.LINE_AA)
    return img


def render_alphabet_sheet(cols=7, tile_w=220, tile_h=360):
    letters = sorted(LETTER_SPECS)
    rows = -(-len(letters) // cols)
    sheet = np.full((rows * tile_h, cols * tile_w, 3), BG, np.uint8)
    for i, letter in enumerate(letters):
        r, c = divmod(i, cols)
        tile = render_panel(letter, tile_w, tile_h)
        cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), (70, 70, 70), 1)
        sheet[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = tile
    return sheet


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[2] / "artifacts" / "reference_alphabet.png"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), render_alphabet_sheet())
    print("saved", out)
