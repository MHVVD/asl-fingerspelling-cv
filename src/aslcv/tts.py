"""Text-to-speech for the video translator, via ElevenLabs (journal 16).

The rest of the project only sees `Synthesizer.synthesize(text) -> bytes` (MP3) and `TTSError`; nothing else
knows about ElevenLabs. The `elevenlabs` package is optional and imported only when speech is first needed:

    pip install -r requirements-tts.txt          # or: pip install .[tts]
    echo 'ELEVENLABS_API_KEY=...' > .env           # or export it

Audio is cached on disk by hash(text, voice, model, format), so re-running a video costs no credits.
Errors (no package, no key, bad key, no credits, rate limit, network) become one readable TTSError.
"""
import hashlib
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"
CACHE_DIR = ROOT / "cache" / "tts"
KEY_VAR = "ELEVENLABS_API_KEY"

DEFAULT_VOICE = "JBFqnCBsd6RMkjVDRZzb"     # "George", the voice in the ElevenLabs quickstart
DEFAULT_MODEL = "eleven_multilingual_v2"   # the API's default model; "most stable", 29 languages
OUTPUT_FORMAT = "mp3_44100_128"            # the API default; available on the free tier

SETUP_HELP = f"""\
--speak needs ElevenLabs set up:
  1. pip install -r requirements-tts.txt
  2. put your API key in .env (see .env.example):  {KEY_VAR}=your-key
     (keys: https://elevenlabs.io/app/settings/api-keys)"""


class TTSError(Exception):
    """A failure that should be shown to the user as one message, without a traceback."""


def load_dotenv(path=ENV_FILE):
    """Minimal .env reader: NAME=value lines; existing environment variables win. Values are never printed."""
    path = Path(path)
    if not path.exists():
        return
    for n, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise TTSError(f"{path.name} line {n} has no NAME=value; write it as {KEY_VAR}=your-key")
        name, value = line.split("=", 1)
        name = name.removeprefix("export ").strip()
        os.environ.setdefault(name, value.strip().strip("'\""))


def api_key(env_file=ENV_FILE):
    load_dotenv(env_file)
    key = os.environ.get(KEY_VAR, "").strip()
    if not key:
        raise TTSError(f"{KEY_VAR} is not set.\n{SETUP_HELP}")
    return key


def cache_key(text, voice_id, model_id, output_format=OUTPUT_FORMAT):
    return hashlib.sha256("\x1f".join([text, voice_id, model_id, output_format]).encode()).hexdigest()


def describe_error(e):
    """Exception from the SDK/HTTP layer -> one readable sentence. Duck-typed on ApiError's
    `status_code` and `body`, so it needs no import of elevenlabs."""
    status = getattr(e, "status_code", None)
    body = getattr(e, "body", None)
    detail = body.get("detail", body) if isinstance(body, dict) else body
    code, message = "", ""
    if isinstance(detail, dict):
        code = str(detail.get("code") or detail.get("status") or "")       # "status" is the older field
        message = str(detail.get("message") or "")
    elif detail:
        message = str(detail)

    if code in ("insufficient_credits", "quota_exceeded") or status == 402:
        return ("ElevenLabs: not enough credits for this request. Check your usage at "
                "https://elevenlabs.io/app/usage; clips already made are cached and will not be charged again.")
    if "missing_permissions" in (code, str(detail.get("status", "")) if isinstance(detail, dict) else ""):
        return ("ElevenLabs: this API key is restricted and lacks a needed permission. Edit the key at "
                "https://elevenlabs.io/app/settings/api-keys and enable Text to Speech."
                + (f" ({message})" if message else ""))
    if code in ("invalid_api_key", "missing_api_key") or status == 401:
        return f"ElevenLabs rejected the API key (401). Check {KEY_VAR} in .env or your environment."
    if status == 429:
        return ("ElevenLabs is rate-limiting requests (429). Wait a minute and re-run; "
                "clips already made are cached.")
    if code == "voice_not_found" or status == 404:
        return "ElevenLabs: voice not found. Check --voice-id (the default voice works on every plan)."
    if status == 403:
        return f"ElevenLabs refused access ({code or 403}): {message or 'check the voice and model for your plan'}"
    if status is not None:
        return f"ElevenLabs API error {status}" + (f" ({code})" if code else "") + (f": {message}" if message else "")
    return f"Could not reach ElevenLabs ({type(e).__name__}). Check your internet connection."


class Synthesizer:
    """synthesize(text) -> MP3 bytes, with an on-disk cache. `client` is injectable for tests."""

    def __init__(self, voice_id=DEFAULT_VOICE, model_id=DEFAULT_MODEL, cache_dir=CACHE_DIR, use_cache=True,
                 client=None, env_file=ENV_FILE, retries=2, sleep=time.sleep):
        self.voice_id, self.model_id = voice_id, model_id
        self.cache_dir, self.use_cache = Path(cache_dir), use_cache
        self.env_file, self.retries, self.sleep = env_file, retries, sleep
        self._client = client
        self.stats = {"requests": 0, "cache_hits": 0, "characters": 0}

    def check(self):
        """Fail fast (before a long video pass) if the package or the key is missing."""
        self._get_client()

    def _get_client(self):
        if self._client is None:
            try:
                from elevenlabs.client import ElevenLabs
            except ImportError:
                raise TTSError(f"The 'elevenlabs' package is not installed.\n{SETUP_HELP}") from None
            self._client = ElevenLabs(api_key=api_key(self.env_file))
        return self._client

    def cache_path(self, text):
        return self.cache_dir / f"{cache_key(text, self.voice_id, self.model_id)}.mp3"

    def synthesize(self, text):
        text = text.strip()
        if not text:
            raise ValueError("nothing to synthesize")
        path = self.cache_path(text)
        if self.use_cache and path.exists():
            self.stats["cache_hits"] += 1
            return path.read_bytes()
        audio = self._request(text)
        if self.use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_suffix(".part")
            part.write_bytes(audio)
            part.replace(path)                          # atomic: the cache never holds a half-written clip
        return audio

    def _request(self, text):
        client = self._get_client()
        for attempt in range(self.retries + 1):
            try:
                audio = b"".join(client.text_to_speech.convert(
                    voice_id=self.voice_id, text=text, model_id=self.model_id, output_format=OUTPUT_FORMAT))
                break
            except Exception as e:                      # SDK raises ApiError; httpx raises transport errors
                if getattr(e, "status_code", None) == 429 and attempt < self.retries:
                    self.sleep(2 ** (attempt + 1))      # short back-off for concurrency limits
                    continue
                raise TTSError(describe_error(e)) from None
        if not audio:
            raise TTSError("ElevenLabs returned no audio.")
        self.stats["requests"] += 1
        self.stats["characters"] += len(text)
        return audio
