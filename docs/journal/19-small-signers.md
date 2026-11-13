# 19 — Small Signers: Find the Person, Then the Hands

## Goal
The benchmark (journal 18) showed that the translator reads almost nothing when the signer is a small inset in the
corner of the screen, as in the CDC radiation videos. The fix: find the signer first, then track the hands in an
enlarged crop around them.

## How (`seq_translate.find_signer_roi`, `signer_box`, `hand_pass(roi=...)`)
1. **Look for hands at two scales.** On 40 frames spread over the video, the existing hand detector runs on the full
   frame and on 9 overlapping tiles (each half the frame, enlarged 2×). No new model or download is needed.
2. **Decide whether to crop.** If the full frame already finds hands in at least half the samples and they are a
   reasonable size (palm ≥ 5% of the frame height), there's no crop.
3. **Otherwise, choose the box.** It goes around the **densest cluster** of tile detections, padded for the signing
   space (3 palms each side, 6 above for the head). A first version used percentiles of all detections; stray
   detections in the background footage and another person's hand at the frame edge stretched the box over the whole
   frame, so it gave up. The cluster rule fixed that, and has a test.
4. **Track hands in the crop.** The crop is enlarged to 720 px on its longer side. Points used for the models are in
   crop coordinates, so the signer fills the frame as in a normal video; points used for drawing are mapped back to
   the full frame.

On by default; `--no-crop` turns it off. The search adds about 15–30 s per video.

## Developed on a fake inset, not on the test videos
Both real inset videos are in the benchmark's **test** split, so tuning on them would be cheating. Instead I built a
fake inset from the two **tuning** videos: the USDA microwave interpreter shrunk to 40% in the bottom-right corner,
over the lightning video's background footage. Its answer key still applies.

| Microwave interpreter | Hand found | Both hands | Terms read (of 5) |
|---|---|---|---|
| Original, full size | 96% | 78% | 3 |
| Fake inset, no crop | **32%** | 5% | 0 |
| Fake inset, crop | **95%** | 71% | 1 |

- **Hand finding is fully restored.** The decoded letters look like the original's again: `sloelabrutaria`, `watagy`,
  `wtfodesaftisgov`, against `salonelabrtstria`, `watage`, `wtfodesaftytor`.
- **The score recovers only partly.** The strict scorer allows a quarter of a word's letters wrong, and a signer at 40%
  size is noisier to read ("watagy" is 2 edits from "wattage").
- **Full-size videos are left alone:** the finder chose no crop for both normal tuning videos.

## Test videos (settings frozen in commit `1fd870c`, scored once)
- **No crop chosen** for E. coli, FEMA or the West Wing tour, so their results are identical to journal 18.
- **Crop chosen** for both CDC radiation videos.

| CDC inset videos | Hand found | Both hands found | Decoded output |
|---|---|---|---|
| Radiation "inside", no crop | 50% | 15% | `e · lar · or · cc · mop · nurior` |
| Radiation "inside", **crop** | **67%** | **35%** | `lear · radiaton · roton · nuriors · …` |
| Radiation "decon", no crop | 59% | 23% | `radien · sandor · condor · wool` |
| Radiation "decon", **crop** | **72%** | **37%** | `radien · sand · radion · shampa · condor` |

- **Benchmark totals are unchanged**: 26 / 45 terms read, 12 / 18 spoken words correct. The reason is a gap in my
  own answer key: when I checked the key, I couldn't judge the tiny inset signers by eye either. All their terms are
  "unclear", "signed" or "not spelled", so there's nothing there to score.
- **The improvement shows at the input:** both hands are found more than twice as often.
- **The output now contains the videos' main topic word** ("radiaton", "radion" for *radiation*), plus "shampa" where
  the key says "shampoo" was signed; the interpreter may have spelled it as well.
- **The gain is smaller than on the fake inset.** In the real videos the inset overlaps the main footage, the signer's
  edges are cut off, and on the decon video hands in the dramatised scenes stretched the box to the top of the frame,
  so it zooms in less than it should.

## Pitfalls
- **Percentile box:** stretched over the whole frame by stray detections, so no crop at all. Fixed with the densest
  cluster.
- **Scorer bug:** the `video@key` naming (scoring a file against another video's key) first looked up the wrong key.
- **Can't measure what you can't see:** a benchmark can only measure what its answer key can confirm. Inset videos
  need their terms checked on an enlarged view of the signer, without looking at the translator's output.

## Next
- **Re-check the inset videos' key on the enlarged crop,** labelled clearly as a revised key, so the benchmark can
  score them.
- **Keep the box on the signer:** ignore detections in the main footage, for example by preferring a stable cluster
  near the frame edge.
- **Recognising whole signs.**
