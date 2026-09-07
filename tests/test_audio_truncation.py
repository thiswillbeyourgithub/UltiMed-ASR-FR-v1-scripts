#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.31",
#   "jiwer>=3.0",
#   "tqdm>=4.66",
#   "loguru>=0.7",
#   "click>=8.1",
#   "soundfile>=0.12",
# ]
# ///
"""Truncation guard: a TTS clip that stopped for LENGTH must never be kept.

`/v1/audio/speech` has no `finish_reason`, so a generation cut off at the backend's
frame cap comes back as a normal 200 with audio that just ends mid-sentence. The only
observable is a duration pinned at the cap, which is what `is_truncated` checks. This
test covers both sides of that guard:

  * stage 05 (`01_generate_audio.py`): `synthesize` raises TruncatedAudioError and
    writes NOTHING, so the batch counts the row as failed and a resume retries it.
  * stage 06 (`01_recursive_improvement.py`): `_synthesize_draft` flags the draw and
    `generate_drafts` saves no sidecar for it, so a truncated draw can never be
    promoted over a real clip.

Network and audio decoding are faked (no server, no wheels beyond the module's own),
so this runs offline. Because the modules import soundfile at import time, run it
under uv rather than the stdlib `python tests/...` convention:

    uv run tests/test_audio_truncation.py

This file was written with Claude Code.
"""

from __future__ import annotations

import importlib.util
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
TTS_PATH = HERE.parent / "05_generate_audio" / "01_generate_audio.py"
IMPROVE_PATH = HERE.parent / "06_hotfixes" / "01_recursive_improvement.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_is_truncated() -> None:
    """The cap arithmetic and the tolerance window."""
    tts = load_module(TTS_PATH, "tts_under_test")

    # 12.5 frames per second: the packaged 2048 frames are the 163.84s cap that
    # truncated the long clips, the served 4096 the 327.68s one that replaced it.
    assert tts.frame_cap_seconds(2048) == 163.84
    assert tts.frame_cap_seconds() == 327.68

    # The default must track the SERVER's cap: flagging clips at the old 163.84s
    # ceiling would now reject perfectly good long audio.
    assert not tts.is_truncated(163.84, tts.frame_cap_seconds())

    cap = tts.frame_cap_seconds()
    assert tts.is_truncated(cap, cap)              # exactly at the cap
    assert tts.is_truncated(cap - 0.2, cap)        # inside the tolerance window
    assert tts.is_truncated(cap + 5.0, cap)        # a larger cap than the server's
    assert not tts.is_truncated(cap - 3.0, cap)    # stopped on its own, well short
    assert not tts.is_truncated(12.5, cap)
    assert not tts.is_truncated(None, cap)         # unparseable container: cannot judge
    assert not tts.is_truncated(cap, 0)            # 0 disables the check
    assert not tts.is_truncated(cap, None)
    print("OK test_is_truncated")


def _tts_args(tmp: Path, cap: float) -> SimpleNamespace:
    """Stage-05 argparse namespace with every knob synthesize()/build_body() reads."""
    return SimpleNamespace(
        format="flac", timeout=30.0, model=None, instruction=None, voice=None,
        seed=None, speed=None, temperature=None, top_p=None, top_k=None,
        repetition_penalty=None, do_sample=None, max_audio_seconds=cap,
    )


def test_stage05_rejects_a_truncated_clip() -> None:
    """A clip pinned at the cap is not written; a normal one is."""
    tts = load_module(TTS_PATH, "tts_under_test")
    cap = tts.frame_cap_seconds()
    tts.post_json = lambda url, body, timeout: (200, b"FAKEAUDIO", 0.5)

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "clip.flac"

        tts.payload_duration = lambda payload: cap
        try:
            tts.synthesize("http://x", "un texte beaucoup trop long", out, _tts_args(Path(td), cap))
        except tts.TruncatedAudioError as exc:
            assert "frame cap" in str(exc)
        else:
            raise AssertionError("a clip pinned at the frame cap must not be accepted")
        assert not out.exists(), "a truncated clip must not be written (a resume must retry it)"
        assert not list(Path(td).glob("*.tmp")), "no .tmp leftover either"

        # Same payload, a duration that stopped on its own: written normally.
        tts.payload_duration = lambda payload: 12.0
        elapsed, dur = tts.synthesize("http://x", "un texte court", out, _tts_args(Path(td), cap))
        assert dur == 12.0 and out.read_bytes() == b"FAKEAUDIO"

        # TruncatedAudioError is a RuntimeError, which is what run_batch catches per row
        # (count as failed, keep going) instead of aborting the whole batch.
        assert issubclass(tts.TruncatedAudioError, RuntimeError)
    print("OK test_stage05_rejects_a_truncated_clip")


def _improve_cfg(mod, cap: float) -> SimpleNamespace:
    """Stage-06 cfg with the fields _synthesize_draft / generate_drafts read."""
    return SimpleNamespace(
        tts_url="http://x", tts_voice=None, tts_speed=None, tts_model=None,
        tts_format="flac", tts_timeout=30.0, tts_max_retries=1,
        tts_sem=threading.Semaphore(1), start_seed=42, speed_jitter=0.0,
        cfg_alpha_start=None, cfg_alpha_step=0.0, n_improv=3,
        cer_threshold=0.15, max_audio_seconds=cap,
    )


def test_stage06_discards_a_truncated_draw() -> None:
    """A draw that hit the cap is flagged, and no sidecar draft is saved for it."""
    mod = load_module(IMPROVE_PATH, "recimp_under_test")
    cap = mod._tts.frame_cap_seconds()
    cfg = _improve_cfg(mod, cap)

    # Distinct bytes per seed so the draw is never taken for a duplicate.
    mod.tts_synthesize = lambda url, text, **kw: f"audio-{kw['seed']}".encode()

    # First: the draw stops at the cap.
    mod._tts.payload_duration = lambda payload: cap
    draft = mod._synthesize_draft(0, cfg, "clip", "texte", 99, "deadbeef", set())
    assert draft["truncated"] is True and draft["duplicate"] is False

    # Then: the same draw at a normal length is not flagged.
    mod._tts.payload_duration = lambda payload: 20.0
    draft = mod._synthesize_draft(1, cfg, "clip", "texte", 99, "deadbeef", set())
    assert draft["truncated"] is False

    with tempfile.TemporaryDirectory() as td:
        audio = Path(td) / "000001_0000_terme.flac"
        audio.write_bytes(b"the original clip")

        mod._tts.payload_duration = lambda payload: cap
        out = mod.generate_drafts(0, "clip", audio, "texte", 0.9, cfg)
        assert out["drafts"] == [], "a truncated draw must not be persisted as a candidate"
        assert not list(Path(td).glob("*.draft*")), "no draft sidecar on disk either"
        assert out["attempts"] == cfg.n_improv
        assert audio.read_bytes() == b"the original clip", "the original must be untouched"

        # Sanity: with normal-length draws the same call does save candidates.
        mod._tts.payload_duration = lambda payload: 20.0
        out = mod.generate_drafts(0, "clip", audio, "texte", 0.9, cfg)
        assert len(out["drafts"]) == cfg.n_improv
    print("OK test_stage06_discards_a_truncated_draw")


if __name__ == "__main__":
    test_is_truncated()
    test_stage05_rejects_a_truncated_clip()
    test_stage06_discards_a_truncated_draw()
    print("all OK")
