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
"""The seed and cfg_alpha must travel NESTED in ``extra_params``, on both TTS stages.

This is a regression test for a bug that produced no error at all. vllm-omni's
``OpenAICreateSpeechRequest`` declares no ``model_config``, so pydantic v2's default
``extra='ignore'`` DROPS unknown top-level keys and still answers 200. A flat
``cfg_alpha`` was therefore discarded silently, and a flat ``seed``, while a real
field, only reaches the vLLM sampler, which Voxtral's stage 0 feeds a vocab-wide -inf
row with one finite entry: arithmetically inert. The visible symptom was nil, because
unseeded takes drift anyway (the flow-matching noise came from a global RNG), so five
"different" regeneration draws were five identical requests and nothing ever flagged it.

What the server reads is stage 0's ``SamplingParams.extra_args``, populated from
``extra_params``. Hence the assertions below: the nested keys are what matter, the flat
``seed`` stays for the backends that do read it (TADA, qwen3-tts), and both are omitted
entirely when unset so the server keeps its startup defaults.

No server and no network: only the request bodies are built.

Run: uv run tests/test_tts_wire_format.py

This file was written with Claude Code.
"""

from __future__ import annotations

import importlib.util
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


def _stage05_args(**overrides) -> SimpleNamespace:
    """Only the fields build_body reads."""
    base = dict(
        format="flac", model=None, instruction=None, voice=None, seed=None, speed=None,
        temperature=None, top_p=None, top_k=None, repetition_penalty=None, do_sample=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_stage05_nests_the_seed() -> None:
    tts = load_module(TTS_PATH, "tts_wire_under_test")

    body = tts.build_body("bonjour", _stage05_args(seed=43))
    assert body["seed"] == 43, "flat seed must stay for TADA / qwen3-tts"
    assert body["extra_params"] == {"voxtral_noise_seed": 43}, (
        "the seed must ALSO be nested, or it never reaches Voxtral's noise draw"
    )

    # Unset knobs stay off the wire so the server keeps its startup defaults.
    plain = tts.build_body("bonjour", _stage05_args())
    assert "seed" not in plain
    assert "extra_params" not in plain, (
        "an empty extra_params would override nothing but still change the request"
    )
    print("OK stage 05 build_body nests voxtral_noise_seed")


def test_stage06_nests_seed_and_cfg_alpha() -> None:
    improve = load_module(IMPROVE_PATH, "improve_wire_under_test")

    sent: dict = {}

    def fake_post_json(url, body, timeout):
        sent["url"] = url
        sent["body"] = body
        return 200, b"audio-bytes", 0.01

    original = improve._tts.post_json
    improve._tts.post_json = fake_post_json
    try:
        payload = improve.tts_synthesize(
            "http://127.0.0.1:8003", "bonjour",
            response_format="flac", seed=44, voice="fr_male", speed=None,
            model=None, timeout=5.0, cfg_alpha=1.6,
        )
    finally:
        improve._tts.post_json = original

    assert payload == b"audio-bytes"
    body = sent["body"]
    assert body["seed"] == 44, "flat seed must stay for TADA / qwen3-tts"
    assert body["extra_params"] == {"cfg_alpha": 1.6, "voxtral_noise_seed": 44}, (
        f"cfg_alpha and the seed must both be nested, got {body.get('extra_params')!r}"
    )
    assert "cfg_alpha" not in body, (
        "a flat cfg_alpha is silently dropped by the server; sending it hides the bug"
    )
    print("OK stage 06 tts_synthesize nests cfg_alpha and voxtral_noise_seed")


def test_stage06_omits_extra_params_when_nothing_is_set() -> None:
    improve = load_module(IMPROVE_PATH, "improve_wire_under_test_2")

    sent: dict = {}

    def fake_post_json(url, body, timeout):
        sent["body"] = body
        return 200, b"x", 0.01

    original = improve._tts.post_json
    improve._tts.post_json = fake_post_json
    try:
        improve.tts_synthesize(
            "http://127.0.0.1:8003", "bonjour",
            response_format="flac", seed=None, voice=None, speed=None,
            model=None, timeout=5.0, cfg_alpha=None,
        )
    finally:
        improve._tts.post_json = original

    assert "extra_params" not in sent["body"]
    print("OK stage 06 omits extra_params when no knob is set")


if __name__ == "__main__":
    test_stage05_nests_the_seed()
    test_stage06_nests_seed_and_cfg_alpha()
    test_stage06_omits_extra_params_when_nothing_is_set()
    print("OK all")
