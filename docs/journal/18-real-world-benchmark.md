# 18 — A Real-World Benchmark, and Making the Laptop Survive It

## Goal
Every end-to-end translator number so far came from one video (USDA Campylobacter). That isn't enough to trust a
result, or to tune anything without fooling myself. This entry builds a small, honest benchmark: several real videos,
several signers, answer keys written before any model saw the videos, and a fixed tuning/test split.

## The videos (`benchmark/README.md`)
- **Source:** 11 openly licensed ASL videos found on Wikimedia Commons, all US federal government works (public
  domain). Wikimedia refuses downloads without a descriptive User-Agent; with one, they downloaded fine.
- **Kept:** 7 videos with English text. Four CDC COVID videos have no captions, and their transcript links are dead.
- **Seven different signers:**
  - USDA E. coli and USDA Microwave interpreters (the E. coli intro and outro presenter also appears in the
    Campylobacter video)
  - the FEMA interpreter
  - Leah Katz-Hernandez, a Deaf native signer, filmed full-length in the White House West Wing tour
  - two CDC interpreters shown as small insets in the radiation-emergency videos
  - the National Weather Service lightning-safety signer
- **Split, fixed in advance:**
  - tune on USDA Microwave and NWS Lightning (the latter has no fingerspelled terms, so it tests false alarms);
  - test once on the other five.

## The answer key (written and pushed before any model run)
1. **English text with times:** two subtitle tracks from Commons, plus burned-in captions read from frames for the
   other five. The caption area was cropped each time its white text pixels changed, and I read and transcribed
   about 280 caption strips.
2. **Candidate terms:** 77 names, acronyms and technical terms likely to be fingerspelled, chosen from the English
   text alone.
3. **Checked on the video:** 8 frames over [t−0.5, t+5.5] s per term, judged by eye.
   - Result: **50 spelled, 17 unclear, 5 signed** (e.g. "rose", "shampoo"), **5 not spelled**.
   - The inset signers in the radiation videos are too small to judge, so their terms are mostly "unclear".
   - Caveat: "spelled" means fingerspelling is visible near the term. Which exact word was spelled can't always be
     read at that resolution.
4. `key.json` and the transcripts were committed with SHA-256 hashes **and pushed** (commit `2d65266`) before any
   model run, so the public history shows the key came first.

## Scoring (`aslcv.benchmark`)
- **Terms read:** a "spelled" term counts if the translator's letters within [t−2, t+8] s contain it, allowing up to
  a quarter of its letters wrong.
- **Terms spoken:** the exact word is among the words `--speak` would say, in the same window.
- **Spoken words correct:** a spoken word counts if it is in a caption shown 10 s before to 2 s after it.
  Plurals count as the same word; anything else needs an exact match.
- **Letters share:** how much of the video with a hand visible is read as letters at all.

The translator was split into a **hand pass** (slow, cached per video) and a **decode pass**, so settings can be
compared cheaply. A regression check ran the old code (from git) and the new code on the same cached hands. They
produced **identical output**.

## Tuning: one decision
On the tuning videos, the spoken words that were wrong ("ada", "hikes", "tale") were all short dictionary words that
happened to fill a whole segment. That's the same failure as "bus" and "coon" in journal 17. So **short dictionary
words are no longer spoken even when they are a whole segment**; they need 6+ letters like any other dictionary word,
while topic words from `--vocab` still count at any length.

On the tuning videos, spoken words went from 2 of 5 correct to 1 of 1. That's a tiny sample, but it's held-out from
the test videos.

## Test results (five videos, scored once, settings frozen in commit `80e04bf` before the run)

| Test videos | Terms read (of 45 spelled) | Terms spoken | Spoken words correct | Read as letters |
|---|---|---|---|---|
| Without the fingerspelling detector | 21 (47%) | 0 | 3 of 9 | ~100% of the time |
| **With the detector (current translator)** | **26 (58%)** | 4 | **12 of 18 (67%)** | 4–39% |

| Video | Terms read | Spoken correct | Read as letters |
|---|---|---|---|
| USDA E. coli | 14 / 20 | 9 / 14 | 39% |
| FEMA Hurricane Harvey | 5 / 12 | 0 / 0 | 9% |
| White House West Wing (Deaf native signer) | 7 / 13 | 3 / 3 | 22% |
| CDC radiation, inside (small inset) | — (no key terms spelled) | 0 / 0 | 4% |
| CDC radiation, decontamination (small inset) | — | 0 / 1 | 7% |

- **Read correctly:** USDA, E. coli, Escherichia, hemolytic, uremic, pathogens, unpasteurized, spinach, FSIS,
  Karen ×3, FEMA, Denton, DC, Harvey, TV, Leah, Katz, Franklin ×2, Roosevelt ×2 and cabinet.
- **Close decodings:** one segment reads `dehydrationhemolyticuremicayndrome`; the West Wing tour gives
  `afranklindedronsvelt` (Franklin D. Roosevelt).
- **Spoken and correct (12):** bacteria, intestines, bacterium, dehydration, kidney, pathogens, ground, spinach,
  virtual, Franklin, cabinet, Theodore. Many were fingerspelled by the interpreter but aren't in the key, which only
  lists likely candidates.
- **Counted wrong (6):**
  - really wrong: "sisters" (FSIS again), "escher" (a truncated Escherichia), "emails", "condor";
  - plausibly right, but the strict scorer says no: "contaminate" (caption: "contaminated") and "website" (caption:
    "Web site"). I didn't change the scorer after seeing test results, so the strict number stands.
- **Names are rarely spoken** (4 of 45 terms), because names aren't in the dictionary and need `--vocab`. They are
  often *read* correctly in the captions.

**What the benchmark shows**
- The detector helps everywhere: more terms read (21 → 26), far more correct speech (3/9 → 12/18), and junk letters
  cut from all the signing to a fraction of it.
- **Small signers don't work.** In the CDC radiation videos the signer is a corner inset; the hand tracker finds the
  hands only occasionally, and almost nothing is read. Cropping to the signer before tracking would be the fix.
- **FEMA** is read but not spoken: `femac`, `haveyca`, `cadenten`. These are close, but not exact words.

## The laptop kept switching off: diagnosis and fix
- **The symptom:** during this work the PC powered off several times.
- **What the logs show:** every session ended abruptly, with no shutdown or crash messages. The firmware reports an
  invalid critical-temperature threshold (`ACPI: thermal: [Firmware Bug]`), so Linux can't shut down cleanly when
  hot.
- **Two causes found:**
  1. **A memory bug of mine.** The new `load_hands` indexed a compressed `.npz` file once per frame. Every access
     decompressed the whole array again, and each hand kept a reference into its own full copy.
     - Measured: more than 3 GB for a 76-second video.
     - systemd-oomd killed the job under 71% memory pressure.
     - **Fixed:** each array is read once and slices are copied, giving 117 MB and 5 s. A test guards it.
  2. **Heat under sustained full load.** Earlier jobs ran all 8 threads flat out for hours on a thin-laptop CPU.
- **`tools/safe_run.sh`** now runs every heavy job:
  - CPU capped at 2 threads, memory at 6 GB;
  - detached, so jobs survive the session ending;
  - paused above 88 °C and resumed below 75 °C (tested: the job freezes and thaws as expected);
  - a temperature log flushed to disk every 10 s.
- **Result:** the whole benchmark (tuning, regression check and two test configurations) ran this way with a
  **maximum of 73 °C** and no shutdown.

## Next
- **Small signers:** track the person first, then the hands in a crop around them.
- **Near-miss words:** snap close decodings to known words ("femac" → FEMA), tuned on the tuning videos only.
- **Grow the benchmark:** more videos and signers would make the test less noisy. With 45 spelled terms, every
  number here moves by about 2 points per term.
