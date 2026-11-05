# 16 — Making the Translator Speak (ElevenLabs)

## Goal
The video translator writes captions; now it should also **say** the words it reads, as a narration track timed to the
video. It's an optional layer on top: the model and the decoder are unchanged.

```bash
pip install -r requirements-tts.txt                   # optional extra: elevenlabs
cp .env.example .env                                  # ELEVENLABS_API_KEY=...
PYTHONPATH=src python -m aslcv.translate clips/usda_campylobacter.webm --speak --vocab examples/usda_campylobacter_vocab.txt
```
Outputs next to the captioned video and `.srt`: `<name>_spoken.mp3` (narration), `<name>_captioned_spoken.mp4`
(captions + narration) and `<name>_spoken.json` (what was spoken, what was skipped and why, characters used).

## Design
| Module | Job | Knows about ElevenLabs? |
|---|---|---|
| `tts.py` | `Synthesizer.synthesize(text) -> MP3 bytes`, disk cache, readable errors | **only this one** |
| `narrate.py` | place clips on the timeline, mix, write MP3, mux with ffmpeg | no |
| `spotting.py` | choose which decoded text is worth speaking | no |
| `translate.py` | `--speak` flags; checks setup *before* the 4-minute video pass | no |

- **API (checked against the docs and the installed SDK, `elevenlabs` 2.76.0, not from memory):**
  - The call is `client.text_to_speech.convert(voice_id, text=, model_id=, output_format=)`. It returns an iterator of
    byte chunks.
  - Defaults: model `eleven_multilingual_v2` (the API's own default, "most stable"), voice George
    (`JBFqnCBsd6RMkjVDRZzb`, the quickstart voice) and `mp3_44100_128` (available on the free plan).
- **Optional dependency:** `elevenlabs` is imported inside `tts.py` only when speech is needed.
  - Without the package, without a key, or without ffmpeg, the translator still runs and all tests pass.
  - `--speak` alone then fails with one message saying how to set it up.
- **Key:** read from `ELEVENLABS_API_KEY` or a git-ignored `.env`; it is never printed. A small built-in reader handles
  `.env`, so `python-dotenv` isn't needed.
- **Cache:** `cache/tts/<sha256(text, voice, model, format)>.mp3`, written atomically. `--no-cache` turns it off.
- **Errors** become one readable line, with no traceback:
  - bad key, or a *restricted* key missing the Text to Speech permission
  - no credits (the newer `insufficient_credits` code and the older `quota_exceeded`)
  - rate limit (retried twice with back-off)
  - voice not found
  - network failure
- **No corrupt files:** every clip is synthesized and decoded before anything is written. Outputs go to `.part` files
  that are renamed only when complete.
- **Overlaps are queued.** A clip that would start before the previous one ends waits for it. Trimming would cut words
  in half; speeding up distorts the voice and still overlaps when the gap is tiny. Any delay is reported.

## What should be spoken? (the main problem)
The sequence model transcribes **all** signing as letters (journal 15). Speaking a segment like
`strainhumanicampylobacterjejunis` or `reveidisemittar` aloud is useless. Three filters were tried on the
28 segments of the USDA video (3.6 min):

1. **Model confidence**: the mean top probability on frames where it emits a letter.
   - It is 0.69–0.89 for almost every segment, junk included (`reveidisemittar` 0.69, `campylobacter…` 0.83).
   - Only short junk falls below 0.6 (`scull` 0.45, `lim` 0.54, `pm` 0.31, `re` 0.0). **Kept, threshold 0.6.**
2. **How word-like the text is**: the character model's per-letter probability.
   - Every long segment scores low, because real words and junk are run together.
   - **Dropped:** it doesn't separate anything at segment level.
3. **Word spotting** (`spotting.py`): speak only known words found *inside* the decoded letters, each at its estimated
   time. Words come from the system word list (73k words). Inside a longer string a word needs 6+ letters, because
   shorter ones turn up by chance ("oust", "cane", "goof"); a whole segment that is exactly a word counts at any length.
   An optional `--vocab` file adds names and topic terms, which count at any length (3+).
   - This is the same idea as giving a speech recogniser a custom vocabulary.
   - **Default.** `--speak-all` speaks the raw segments instead.

## Results on the USDA video
Each spoken word was checked against the English captions burned into the video (frames at each word's time):

| Words searched | Spoken | Match the caption | Wrong | Unclear |
|---|---|---|---|---|
| System word list only | 10 | 7 (bacteria, bacterial, strain, antibiotic, reactive, website, virtual) | 2 | 1 |
| + 9 topic words (`examples/usda_campylobacter_vocab.txt`) | **22** | **19** (+ campylobacter ×7, jejuni, guillain, FSIS ×2, Karen) | 2 | 1 |

- **Wrong:** "ps", a 2-letter segment that happens to be a word, and "sisters", which is really FSIS (caption
  `www.fsis.usda.gov`).
- **Unclear:** "emails", which isn't in the caption but fits the topic (comments sent by e-mail).
- **Near misses:** two of the "campylobacter"s were "campylobacteriosis" (the decoder dropped the i).
- **Missed:** "meningitis", which the decoder cut short to "meningit".
- **Caveat:** the topic list was written from the video's subject, so it's prior knowledge about this video and should
  be read as "what a user who knows the topic gets". Without it, the system word list alone still gives 7 correct words
  out of 10.

**Cost and speed** (laptop CPU, full 3.6-minute video, `--speak --vocab ...`):

| | Run 1 | Run 2 (same video) |
|---|---|---|
| Characters sent to ElevenLabs | **106** | **0** |
| API requests | 15 | 0 |
| Clips from cache | 7 | **22** |
| Speech step (synthesize, mix, mux) | 10.0 s | 3.6 s |
| Whole command (detection, decoding, rendering, speech) | 293 s | 306 s |

- The speech step is about 3% of the run; the rest is hand detection, decoding and rendering, which ElevenLabs doesn't
  touch. Both runs produced the identical transcript, so the feature doesn't change decoding.
- The first run already reuses its own cache: 22 words were spoken, but repeats ("campylobacter" ×7, FSIS ×2) are
  synthesized only once. So only 15 unique words (106 characters) were billed, instead of 188 characters.
- Speech placement:
  - Total speech is 21.9 s; no clip had to be queued, because spoken words (0.8–1.3 s) are shorter than the gaps
    between them.
  - In the narration track, the level is about 0.05–0.09 RMS inside each word and silence in the half-second before it.

Demo (30 s, captions + narration, with the spoken word shown on screen): `docs/media/usda_spoken_demo.mp4`.

## What went wrong along the way
- **Restricted keys.** ElevenLabs keys can be restricted per feature, and the keys I was given lacked the Text to
  Speech permission.
  - The API answers 401 with `missing_permissions`, and my first error handler reported that as "invalid key", which
    was wrong.
  - Now it says the key needs the Text to Speech permission and where to change it, with a test.
- **A bare key in `.env`.** The `.env` file first held just the key, with no `ELEVENLABS_API_KEY=` before it. A check
  I ran to print variable names with the values hidden printed the key itself, because there was no `=` to hide after.
  - Fix: the `.env` reader now rejects such a line and names only the line number, never the content.
  - Lesson: treat every line of `.env` as secret when inspecting it.
- **`narrate()` defaults.** They were bound at definition time, so tests couldn't swap the decoder; they are now
  resolved at call time.
- **Short words.** The first word filter dropped "hello" and "world" (under 6 letters) on the synthetic clip.
  Whole-segment words are now kept at any length.

## Limits
- The voice speaks each word when its fingerspelling **starts**, which is earlier than the caption that spells it out.
  Speaking at the segment start was the brief; moving it to the end of the word would match the captions better.
- Word spotting can only say words it knows. Names need `--vocab`, and wrong-but-real words ("sisters") get through.
- The real fix is upstream: detecting *when* someone is fingerspelling (journal 15's next step) would remove most junk
  before it reaches the voice.
