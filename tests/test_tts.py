"""tts.py with a fake ElevenLabs client: no network, no key, no elevenlabs package needed."""
import builtins
import os

import pytest

from aslcv import tts
from aslcv.tts import Synthesizer, TTSError, describe_error


class FakeTTS:
    def __init__(self, fail=None):
        self.calls, self.fail = [], list(fail or [])

    def convert(self, voice_id, *, text, model_id, output_format):
        self.calls.append((voice_id, text, model_id, output_format))
        if self.fail:
            raise self.fail.pop(0)
        return iter([b"ID3", text.encode()])               # the SDK returns an iterator of byte chunks


class FakeClient:
    def __init__(self, **kw):
        self.text_to_speech = FakeTTS(**kw)


class FakeApiError(Exception):
    def __init__(self, status_code, body=None):
        self.status_code, self.body = status_code, body


def synth(tmp_path, **kw):
    client = kw.pop("client", None) or FakeClient()
    return Synthesizer(cache_dir=tmp_path / "cache", client=client, sleep=lambda s: None, **kw), client


def test_synthesize_joins_chunks_and_passes_settings(tmp_path):
    s, c = synth(tmp_path, voice_id="v1", model_id="m1")
    assert s.synthesize(" hello ") == b"ID3hello"
    assert c.text_to_speech.calls == [("v1", "hello", "m1", tts.OUTPUT_FORMAT)]
    assert s.stats == {"requests": 1, "cache_hits": 0, "characters": 5}


def test_cache_hit_skips_the_api(tmp_path):
    s, c = synth(tmp_path)
    s.synthesize("campylobacter")
    s.synthesize("campylobacter")
    assert len(c.text_to_speech.calls) == 1
    assert s.stats["cache_hits"] == 1 and s.stats["characters"] == len("campylobacter")
    assert not list((tmp_path / "cache").glob("*.part"))


def test_cache_is_keyed_on_voice_and_model(tmp_path):
    a, ca = synth(tmp_path, voice_id="v1")
    b, cb = synth(tmp_path, voice_id="v2")
    a.synthesize("bed")
    b.synthesize("bed")                                    # different voice -> new request, not a hit
    assert len(cb.text_to_speech.calls) == 1 and b.stats["cache_hits"] == 0
    assert tts.cache_key("bed", "v1", "m") != tts.cache_key("bed", "v1", "m2")


def test_no_cache_always_requests(tmp_path):
    s, c = synth(tmp_path, use_cache=False)
    s.synthesize("book")
    s.synthesize("book")
    assert len(c.text_to_speech.calls) == 2 and not (tmp_path / "cache").exists()


def test_api_error_is_one_message_and_nothing_is_cached(tmp_path):
    s, _ = synth(tmp_path, client=FakeClient(fail=[FakeApiError(401, {"detail": {"code": "invalid_api_key"}})]))
    with pytest.raises(TTSError, match="rejected the API key"):
        s.synthesize("word")
    assert not (tmp_path / "cache").exists()


def test_rate_limit_is_retried(tmp_path):
    s, c = synth(tmp_path, client=FakeClient(fail=[FakeApiError(429), FakeApiError(429)]))
    assert s.synthesize("word") == b"ID3word"
    assert len(c.text_to_speech.calls) == 3


def test_rate_limit_gives_up_with_a_message(tmp_path):
    s, _ = synth(tmp_path, client=FakeClient(fail=[FakeApiError(429)] * 3))
    with pytest.raises(TTSError, match="rate-limiting"):
        s.synthesize("word")


@pytest.mark.parametrize("err, expected", [
    (FakeApiError(402, {"detail": {"code": "insufficient_credits"}}), "not enough credits"),
    (FakeApiError(401, {"detail": {"status": "quota_exceeded"}}), "not enough credits"),   # older error shape
    (FakeApiError(401, {"detail": {"status": "invalid_api_key", "message": "Invalid API key"}}), "rejected"),
    (FakeApiError(404, {"detail": {"code": "voice_not_found"}}), "voice not found"),
    (FakeApiError(401, {"detail": {"code": "unauthorized", "status": "missing_permissions",     # restricted key
                                   "message": "missing the permission text_to_speech"}}), "enable Text to Speech"),
    (FakeApiError(500, {"detail": {"message": "boom"}}), "API error 500: boom"),
    (ConnectionError("dns"), "Could not reach ElevenLabs"),
])
def test_describe_error(err, expected):
    assert expected in describe_error(err)


def test_missing_package_is_a_friendly_error(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def no_elevenlabs(name, *a, **kw):
        if name.startswith("elevenlabs"):
            raise ImportError(name)
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_elevenlabs)
    s = Synthesizer(cache_dir=tmp_path)
    with pytest.raises(TTSError, match="not installed") as e:
        s.check()
    assert "requirements-tts.txt" in str(e.value)


def test_missing_key_is_a_friendly_error(tmp_path, monkeypatch):
    monkeypatch.delenv(tts.KEY_VAR, raising=False)
    with pytest.raises(TTSError, match="is not set") as e:
        tts.api_key(env_file=tmp_path / "missing.env")
    assert ".env.example" in str(e.value)


def test_dotenv_loads_key_without_overriding_environment(tmp_path, monkeypatch):
    monkeypatch.delenv(tts.KEY_VAR, raising=False)
    monkeypatch.delenv("OTHER", raising=False)
    env = tmp_path / ".env"
    env.write_text(f"# comment\n{tts.KEY_VAR}='abc'\nexport OTHER=1\n")
    assert tts.api_key(env_file=env) == "abc"
    assert os.environ["OTHER"] == "1"
    monkeypatch.setenv(tts.KEY_VAR, "from-env")
    assert tts.api_key(env_file=env) == "from-env"


def test_dotenv_bare_line_is_reported_without_echoing_it(tmp_path):
    env = tmp_path / ".env"
    env.write_text("sk_secretvalue\n")
    with pytest.raises(TTSError) as e:
        tts.load_dotenv(env)
    assert "sk_secretvalue" not in str(e.value) and "line 1" in str(e.value)


@pytest.mark.skipif(not os.environ.get(tts.KEY_VAR), reason="integration test: set ELEVENLABS_API_KEY to run")
def test_integration_real_api_one_word(tmp_path):
    pytest.importorskip("elevenlabs")
    s = Synthesizer(cache_dir=tmp_path)
    audio = s.synthesize("hello")
    assert len(audio) > 1000 and (audio[:3] == b"ID3" or audio[0] == 0xFF)   # an MP3 header
    assert s.stats["characters"] == 5
