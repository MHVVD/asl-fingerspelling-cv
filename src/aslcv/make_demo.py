"""Before/after demo video of the first real-world test (journal 11).

    python -m aslcv.make_demo          # -> outputs/demo_realworld_before_after.mp4 (1080x1080, H.264)

Built from the two translator runs on the CC BY-SA Travis Dougherty clip: the run-1 and run-2 captioned
videos (which show which hand was tracked) and their JSON records (which letters were output, and whether
each was right). Output letters are coloured green when they match the alphabet, red when they don't.
"""
import json
import subprocess
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
W = H = 1080
FPS = 29.97
SEG = (16.0, 45.0)
FONT = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
BOLD = "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
BG, WHITE, GREY = (18, 20, 24), (240, 240, 240), (150, 155, 165)
RED, GREEN, AMBER = (235, 87, 87), (80, 200, 120), (255, 196, 60)          # RGB (PIL)
CREDIT = "Video: Travis Dougherty, ASL demo, by Fuzheado, CC BY-SA 3.0, via Wikimedia Commons"
REPO = "github.com/MHVVD/asl-fingerspelling-cv"
TRUTH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def font(path, size):
    return ImageFont.truetype(path, size)


def load_run(path):
    """-> (letters in the scored window [(commit_s, letter, ok)], counts)."""
    rec = json.loads(Path(path).read_text())
    if "letters" in rec:
        ev = [(e["commit_s"], e["letter"]) for e in rec["letters"]]
    else:                                                   # run-1 record format
        ev = [(e[2], e[0]) for e in rec["events"]]
    ev = [(t, l) for t, l in ev if SEG[0] <= t <= SEG[1] + 0.5]
    # alignment entries with an output letter, in order, say whether each output letter was right
    al = rec["alignment"]
    oks = [(a["result"] if isinstance(a, dict) else a[2]) == "ok" for a in al
           if (a["output"] if isinstance(a, dict) else a[1])]
    right = sum(oks)
    return [(t, l, ok) for (t, l), ok in zip(ev, oks)], right


def centered(draw, y, text, f, fill):
    w = draw.textlength(text, font=f)
    draw.text(((W - w) / 2, y), text, font=f, fill=fill)


def card(lines, credit=True):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for y, text, f, fill in lines:
        centered(d, y, text, f, fill)
    if credit:
        centered(d, H - 44, CREDIT, font(FONT, 20), GREY)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def letters_strip(d, x0, y0, width, letters, f):
    """Draw output letters left to right, wrapping, green if right and red if wrong."""
    step = d.textlength("M", font=f) + 4
    per_line = int(width // step)
    for i, (_, l, ok) in enumerate(letters):
        r, c = divmod(i, per_line)
        d.text((x0 + c * step, y0 + r * (f.size + 10)), l, font=f, fill=GREEN if ok else RED)


def main_frame(fa, fb, t, run_a, run_b):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    hdr, sub = font(BOLD, 40), font(FONT, 24)
    for x, title, desc, col in ((0, "BEFORE", "tracking the wrong hand", RED), (540, "AFTER", "tracking the signing hand", GREEN)):
        d.text((x + 24, 18), title, font=hdr, fill=col)
        d.text((x + 24, 70), desc, font=sub, fill=WHITE)
    for x, frame in ((0, fa), (540, fb)):
        crop = cv2.cvtColor(cv2.resize(frame[0:470, 240:720], (540, 529)), cv2.COLOR_BGR2RGB)   # skip burned caption bar
        img.paste(Image.fromarray(crop), (x, 112))
    d.line([(540, 112), (540, 641)], fill=BG, width=6)
    mono = font(MONO, 34)
    d.text((24, 652), "model output:", font=font(FONT, 22), fill=GREY)
    d.text((564, 652), "model output:", font=font(FONT, 22), fill=GREY)
    letters_strip(d, 24, 686, 500, [x for x in run_a if x[0] <= t], mono)
    letters_strip(d, 564, 686, 500, [x for x in run_b if x[0] <= t], mono)
    d.text((24, 862), "correct:", font=font(FONT, 22), fill=GREY)
    d.text((130, 858), TRUTH, font=font(MONO, 30), fill=AMBER)
    centered(d, 930, "Real footage of a signer the model has never seen · laptop CPU · faster than real time",
             font(FONT, 22), WHITE)
    centered(d, H - 44, CREDIT, font(FONT, 20), GREY)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    out = ROOT / "outputs" / "demo_realworld_before_after.mp4"
    run_a, right_a = load_run(ROOT / "artifacts" / "realworld_travis_run1.json")
    run_b, right_b = load_run(ROOT / "artifacts" / "realworld_travis_run2.json")
    ca = cv2.VideoCapture(str(ROOT / "outputs" / "asl_alphabet_travis_subtitled.mp4"))
    cb = cv2.VideoCapture(str(ROOT / "outputs" / "asl_alphabet_travis_travis_run2.mp4"))

    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                           "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                          stdin=subprocess.PIPE)

    def emit(frame, n=1):
        for _ in range(n):
            ff.stdin.write(frame.tobytes())

    def hold_with_fade(frame, seconds):
        n = int(seconds * FPS)
        for i in range(n):
            a = min(1.0, i / 10, (n - i) / 10)                  # fade in/out over 10 frames
            emit((frame * a).astype(np.uint8))

    title = card([(330, "Can my model read a", font(BOLD, 58), WHITE),
                  (405, "stranger's fingerspelling?", font(BOLD, 58), WHITE),
                  (520, "First real-world test: a signer it has never seen,", font(FONT, 32), GREY),
                  (565, "spelling the alphabet A to Z", font(FONT, 32), GREY)])
    hold_with_fade(title, 4.0)

    start, end = int(SEG[0] * FPS), int(SEG[1] * FPS)
    for i in range(end):
        ok_a, fa = ca.read()
        ok_b, fb = cb.read()
        if not (ok_a and ok_b):
            break
        if i >= start:
            emit(main_frame(fa, fb, i / FPS, run_a, run_b))

    results = card([(170, "Result", font(BOLD, 56), WHITE),
                    (280, f"Before:  {right_a} / 26 letters", font(BOLD, 50), RED),
                    (350, f"After:  {right_b} / 26 letters", font(BOLD, 50), GREEN),
                    (480, "The bug: the tracker locked onto the resting hand", font(FONT, 32), WHITE),
                    (522, "at the waist, not the hand doing the spelling.", font(FONT, 32), WHITE),
                    (590, "The fix: detect both hands, read the raised one.", font(FONT, 32), WHITE),
                    (680, "Next: more signers, a 'no letter' state for transitions,", font(FONT, 28), GREY),
                    (718, "and more real-world clips.", font(FONT, 28), GREY),
                    (820, REPO, font(BOLD, 32), AMBER)])
    hold_with_fade(results, 6.0)
    ff.stdin.close()
    ff.wait()
    print("saved", out)


if __name__ == "__main__":
    main()
