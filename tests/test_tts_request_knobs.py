#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests>=2.31","jiwer>=3.0","tqdm>=4.66","loguru>=0.7","click>=8.1","soundfile>=0.12"]
# ///
"""What each regeneration draw actually asks the TTS server for.

Run: `uv run tests/test_tts_request_knobs.py`

This exists because the knobs failed SILENTLY. `cfg_alpha` used to be sent nested in
`extra_params` against a server with no reader for it there, and the flat `seed` used
to be trusted to vary the audio, so five "different" draws were five identical requests
whose audio differed only through global-RNG drift. Nothing errored, nothing logged, and
the cfg ramp in `_synthesize_draft` had no effect at all. So the request BODY is pinned
here, not just the arithmetic that feeds it.

Both knobs are flat fields, which is a contract of the forked vllm-omni the CrispASR
voxtral-tts image is built from (`CrispASR/voxtral/vllm-omni`): it promotes `cfg_alpha`
to a top-level field and routes the flat `seed` into stage 0's `voxtral_noise_seed`.
Upstream drops the first and cannot act on the second. If these assertions ever need
changing, check the fork's `serving_speech.py::_apply_voxtral_request_knobs` first.

This file was written with Claude Code.
"""
from __future__ import annotations

import importlib.util
import re
import threading
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
IMPROVE_PATH = HERE.parent / "06_hotfixes" / "01_recursive_improvement.py"
DRIVER_PATH = HERE.parent / "06_hotfixes" / "driver.sh"

_spec = importlib.util.spec_from_file_location("improve_under_test", IMPROVE_PATH)
_imp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_imp)

_failures: list[str] = []


def eq(label: str, got, want) -> None:
    if got != want:
        _failures.append(f"{label}: got {got!r} want {want!r}")


def _capture_bodies(n_improv: int, **overrides) -> list[dict]:
    """Run n_improv drafts with the network stubbed, returning the request bodies."""
    bodies: list[dict] = []

    def fake_post(url, body, timeout):  # signature of the real post_json
        bodies.append(dict(body))
        return 200, b"FAKEAUDIO" + bytes([len(bodies)]), 0.5

    cfg = SimpleNamespace(
        start_seed=43, tts_speed=None, speed_jitter=0.0,
        cfg_alpha_start=1.0, cfg_alpha_step=1.0, n_improv=n_improv,
        tts_url="http://localhost:8003", tts_format="flac", tts_voice="fr_female",
        tts_model=None, tts_timeout=30.0, tts_max_retries=1,
        max_audio_seconds=327.68, tts_sem=threading.Semaphore(1),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)

    # Both live on the stage-05 module this stage imports as `_tts` (the HTTP call and
    # the frame-cap duration probe exist once, shared by the two stages).
    orig_post, orig_dur = _imp._tts.post_json, _imp._tts.payload_duration
    _imp._tts.post_json = fake_post
    _imp._tts.payload_duration = lambda payload: 12.0    # never near the frame cap
    try:
        for attempt in range(n_improv):
            _imp._synthesize_draft(attempt, cfg, "clip", "bonjour", 999, "nohash", set())
    finally:
        _imp._tts.post_json, _imp._tts.payload_duration = orig_post, orig_dur
    return bodies


# --- the guidance ladder ------------------------------------------------------------
# Driven by the DRIVER's defaults, not by a copy of them: the ladder is a tuning choice
# that lives in driver.sh, and a test that restated it here would keep passing after the
# driver changed. Parsed, fed to the client, and the rendered ladder is what is pinned.
def _driver_default(name: str) -> float:
    text = DRIVER_PATH.read_text(encoding="utf-8")
    m = re.search(rf'^{name}="\$\{{{name}:-([0-9.]+)\}}"', text, re.MULTILINE)
    assert m, f"{name} default not found in driver.sh"
    return float(m.group(1))


bodies = _capture_bodies(5, cfg_alpha_start=_driver_default("CFG_ALPHA_START"),
                         cfg_alpha_step=_driver_default("CFG_ALPHA_STEP"))
eq("one request per draw", len(bodies), 5)
# Near the server's tuned value (it boots at 1.3), not a wide sweep: the draws have to
# land where the model is usable, and the winner is picked on CER anyway.
eq("cfg_alpha ladder", [b.get("cfg_alpha") for b in bodies], [1.0, 1.5, 2.0, 2.5, 3.0])
eq("seed ladder", [b.get("seed") for b in bodies], [43, 44, 45, 46, 47])

# --- where the server actually reads them -------------------------------------------
# FLAT, both of them. Nesting cfg_alpha under extra_params is what made the ramp inert
# for a while: keep this assertion, it is the whole point of the file.
first = bodies[0]
eq("cfg_alpha is a top-level field", "cfg_alpha" in first, True)
eq("no extra_params nesting", first.get("extra_params"), None)
eq("text is sent as input", first.get("input"), "bonjour")
eq("voice is forwarded", first.get("voice"), "fr_female")

# --- unset knobs stay off -----------------------------------------------------------
# Same contract as stage 05: a knob nobody set is omitted so the server keeps its
# startup default, rather than being pinned to a client-side guess.
eq("speed omitted when jitter is off", "speed" in first, False)
eq("model omitted when unset", "model" in first, False)
# A negative start is the documented way to leave guidance entirely to the server.
off = _capture_bodies(2, cfg_alpha_start=None)
eq("cfg_alpha omitted when disabled", ["cfg_alpha" in b for b in off], [False, False])

# --- speed jitter, if someone turns it on -------------------------------------------
# It cycles around the base, but note it is NOT a second source of variation on this
# backend: vllm-omni applies speed as a phase vocoder over the finished waveform.
jit = _capture_bodies(3, tts_speed=1.0, speed_jitter=0.05)
eq("speed cycles around the base", [round(b["speed"], 4) for b in jit], [1.0, 1.05, 0.95])

if _failures:
    print("FAILED:")
    for f in _failures:
        print("  -", f)
    raise SystemExit(1)
print("ALL PASSED")
