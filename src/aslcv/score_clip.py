"""Score the video translator on a clip with a known answer, and keep a record of the run.

    python -m aslcv.score_clip clips/asl_alphabet_travis.ogv --truth ABCDEFGHIJKLMNOPQRSTUVWXYZ \\
        --start 16 --end 45.5 --name travis_run2 --note "two hands, pick raised hand"

Writes artifacts/realworld_<name>.json (truth, output, letter-by-letter alignment, error rate, settings)
plus the captioned video and .srt in outputs/. Only letters committed between --start and --end are
scored (e.g. to skip a spoken/signed introduction that is not fingerspelling).
"""
import argparse
import json
from pathlib import Path
import numpy as np

from .simulate import char_error_rate
from .translate import translate

ROOT = Path(__file__).resolve().parents[2]


def align(ref, hyp):
    """Levenshtein alignment -> [(truth_char|'', output_char|'', 'ok'|'wrong'|'missed'|'extra')]."""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), int)
    D[:, 0], D[0, :] = range(n + 1), range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, out = n, m, []
    while i or j:
        if i and j and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            out.append((ref[i - 1], hyp[j - 1], "ok" if ref[i - 1] == hyp[j - 1] else "wrong"))
            i, j = i - 1, j - 1
        elif i and D[i, j] == D[i - 1, j] + 1:
            out.append((ref[i - 1], "", "missed"))
            i -= 1
        else:
            out.append(("", hyp[j - 1], "extra"))
            j -= 1
    return out[::-1]


def score(clip, truth, start, end, name, note="", **translate_kw):
    stem = Path(clip).stem
    r = translate(clip, ROOT / "outputs" / f"{stem}_{name}.mp4", ROOT / "outputs" / f"{stem}_{name}.srt",
                  verbose=False, **translate_kw)
    letters = [e for e in r["events"] if e.kind == "letter"]
    seg = "".join(e.value for e in letters if start <= e.t_commit <= end)
    al = align(truth, seg)
    counts = {k: sum(o == k for *_, o in al) for k in ("ok", "wrong", "missed", "extra")}
    record = {
        "name": name, "note": note, "clip": str(clip), "truth": truth, "scored_window_s": [start, end],
        "output_scored": seg, "output_full": r["text"],
        "output_before_window": "".join(e.value for e in letters if e.t_commit < start),
        "char_error_rate": round(char_error_rate(truth, seg), 3), "counts": counts,
        "alignment": [{"truth": a, "output": b, "result": o} for a, b, o in al],
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in translate_kw.items()},
        "stats": {k: r[k] for k in ("video_fps", "processed_fps", "hand_detected_share", "handedness_used",
                                    "seconds", "speed_fps")},
        "letters": [{"letter": e.value, "start_s": round(e.t_start, 2), "commit_s": round(e.t_commit, 2),
                     "confidence": round(e.confidence, 3)} for e in letters],
    }
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / f"realworld_{name}.json").write_text(json.dumps(record, indent=1))
    return record


def show(rec):
    print(f"[{rec['name']}] {rec['note']}")
    print(f"  correct : {rec['truth']}")
    print(f"  output  : {rec['output_scored']}")
    c = rec["counts"]
    print(f"  error rate {rec['char_error_rate']:.2f} | right {c['ok']}/{len(rec['truth'])} | wrong {c['wrong']} | "
          f"missed {c['missed']} | extra {c['extra']}")
    print("  aligned : " + " ".join(a["truth"] if a["result"] == "ok" else f"{a['truth'] or '-'}>{a['output'] or '-'}"
                                    for a in rec["alignment"]))
    print(f"  before window: {rec['output_before_window'] or '(nothing)'} | stats {rec['stats']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--truth", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=1e9)
    ap.add_argument("--name", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--hold", type=float, default=0.2)
    a = ap.parse_args()
    show(score(a.clip, a.truth.upper(), a.start, a.end, a.name, a.note, hold_s=a.hold))


if __name__ == "__main__":
    main()
