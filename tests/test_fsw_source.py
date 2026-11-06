import numpy as np

from aslcv.fsw_source import label_frames, load_sources, parse_time, video_id

HEADER = ",filename,url,start_time,number_of_frames,width,height,label_proc,label_raw,label_notes,partition,signer\n"


def row(i, url, start, n, part, signer="0001"):
    return f"{i},src/clip_{i},{url},{start},{n},640,360,abc,abc,,{part},{signer}\n"


def test_parse_time_and_video_id():
    assert parse_time("0:01:05.500000") == 65.5
    assert video_id("https://www.youtube.com/watch?v=4YIYRu2PERg") == "4YIYRu2PERg"


def test_load_sources_keeps_single_split_youtube_videos_sorted(tmp_path):
    a, b, c = "https://www.youtube.com/watch?v=AAA", "https://www.youtube.com/watch?v=BBB", "http://www.deafvideo.tv/1"
    csv = tmp_path / "f.csv"
    csv.write_text(HEADER + row(0, a, "0:00:20.0", 30, "train") + row(1, a, "0:00:05.0", 30, "train")
                   + row(2, b, "0:00:01.0", 30, "train") + row(3, b, "0:00:09.0", 30, "test")   # mixed split
                   + row(4, c, "0:00:01.0", 30, "dev"))                                          # not YouTube
    src = load_sources(csv)
    assert list(src) == [a]
    assert src[a]["partition"] == "train" and [cl["start_s"] for cl in src[a]["clips"]] == [5.0, 20.0]


def test_label_frames_marks_clip_spans():
    times = np.arange(0, 4, 0.5)                      # 0.0 .. 3.5
    clips = [{"start_s": 1.0, "frames": 30}]          # 1 s at 30 fps -> [1.0, 2.0)
    assert label_frames(times, clips, fps=30).tolist() == [0, 0, 1, 1, 0, 0, 0, 0]
