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
"""Transcribe a dataset AND regenerate the worst TTS clips, in one combined pass.

This is a superset of ``01_compute_stt.py``: it runs the same STT + CER scoring
over the whole dataset, and interleaved with it (same parallel pool, one clip at a
time) it tries to repair the clips the TTS engine derailed on. Doing both together
means the "before" transcript needed to measure the TTS gain is computed right
where the improvement happens, no separate scoring run.

It reuses ``01_compute_stt.py``'s orchestration verbatim (its ``process_file``: the
dual clip / audio-seconds progress bars, resume, atomic flushing, thread pool)
through injected per-clip hooks, so none of that is duplicated. With ``--n-improv 0``
it degenerates to exactly ``01_compute_stt.py``.

STT and TTS run in parallel against their own servers. Each has an independent
concurrency cap (``--stt-parallel`` / ``--tts-parallel``): every transcription passes
through one semaphore, every regeneration through another, so a clip busy on TTS
never holds a transcription slot and both GPUs stay saturated. The worker pool is
sized ``stt-parallel + tts-parallel`` so both caps can be full at once.

``--mode`` picks how STT and TTS are combined, so this runs even when the two servers
cannot share VRAM:

* ``both`` (default): the pass above, both servers up, regeneration scores each draw
  inline and stops early once one clears the threshold.
* ``stt`` (only the STT server up): transcribe + score originals, flag the bad ones
  ``pending_tts``; AND score the candidates a prior ``tts`` pass persisted
  (``pending_stt``), promoting the best draw or exhausting the clip. Deletes the
  candidate files once picked. It only picks from a COMPLETE set: a clip whose
  candidates are short of ``--n-improv`` (the tts pass ran with a smaller one, or a
  sidecar has gone missing) goes back to ``pending_tts`` rather than being judged, and
  settled, on a partial set. "Complete" means the tts pass delivered every draw it was
  asked for, not one file per draw: a draw it skipped as a duplicate or as frame-cap
  truncated is skipped deterministically (the seed per attempt is fixed), so demanding
  a file for it would bounce the clip between the two passes forever.
* ``tts`` (only the TTS server up): for every ``pending_tts`` clip, synthesize all
  ``--n-improv`` candidates (no early stop, since scoring needs STT) and persist each
  as a ``<clip>.draft<seed>.<fmt>`` sidecar, marking the clip ``pending_stt``.

A driver alternates ``stt`` and ``tts`` runs (bringing the matching server up between
them); the state lives in the row's ``improvement.status`` so each pass resumes where
the last left off. Convergence is signalled by ``stt`` exiting 10 when no clip anywhere
still needs either pass. The split flow loses only ``both``'s per-clip early stop (the
whole candidate set is generated before any is scored), which is the right trade when
minimizing server swaps dominates.

Failures retry with exponential backoff (STT and TTS alike). Nothing that fails is
silently skipped: a request that keeps failing after every retry ABORTS the run.
Exit codes: 2 the seed had no effect (SeedDeterminismError), 3 a generation came
back empty / duration-0 (GenerationError: a real clip must never be replaced with
silence), 4 a server stayed unreachable (TranscriptionError / TTSRequestError), 5 a
dataset clip transcribed blank after every retry (EmptyTranscriptError: an empty
transcript is never written, it stops the run so the bad clip is surfaced), 10 (``stt``
mode only) a clean "no improvement work left, dataset converged" signal for the driver
loop, 11 (split modes) "this mode has run dry": the pass completed nothing, so repeating
it changes nothing, but clips are still waiting on the OTHER mode. 11 is what lets a
driver run one mode on its own (an STT-only night while the TTS server is down) and
still stop by itself. The one non-fatal case is a regenerated DRAW that transcribes blank: that seed is
discarded and the next is tried (the point of trying several seeds), the original clip
is untouched.

Progress is visible live across three bars: the clip bar (STT sweep) and the
audio-seconds bar are inherited from ``01_compute_stt.py``; a third TTS bar shows
the repair backlog, its total is the clips found bad (high CER) and it fills as they
are successfully improved, so the gap between the two reveals how far the slow TTS
engine is lagging the fast STT pass. The clip bar postfix shows the last clip's CER,
running improved / exhausted tallies, and a rolling STT real-time factor (rtf = STT
seconds / audio seconds, < 1 = faster than real time). Every scored clip logs one
CER line (with its STT time), and a clip over the threshold additionally dumps its
source / target / transcript, exactly like ``01_compute_stt.py``. Every clip being
redone also logs its per-attempt seed, CER and STT time above the bar (REDO /
IMPROVED / EXHAUSTED lines). Per-clip stt_seconds / stt_rtf are also stored in the
output jsonl. Every row also carries a top-level ``cfg_alpha`` recording the TTS
cfg_alpha behind its CURRENT audio: a regenerated clip stores its winning draw's
cfg_alpha, while a clip that keeps its original audio (passed under the threshold, or
exhausted) stores ``--original-cfg-alpha`` (1.3 by default, the value the whole dataset
was synthesized with per ``99_hf_release/README.md``). Audio is replaced through atomic
writes (temp + fsync + os.replace)
for both the ``.bak`` backup and the new clip, so an interrupted run can never
corrupt a clip.

Per clip (in the pool):

1. Transcribe the original audio (reusing a stored transcript on resume) and score
   it by CER only, against BOTH ``asr_training_source`` and ``asr_training_target``,
   taking the lower of the two (``min``). CER uses the exact same normalization as
   script 1 (NFC, lowercase, punctuation stripped but accents kept, whitespace
   collapsed), applied to source, target and transcript alike.
1a. Score the ENDING too. A whole-clip CER averages over the whole reference, so a
   defect confined to the last seconds of a long clip is diluted by everything that came
   out right: a chunk truncated at the TTS output cap keeps ~85% of its text and lands at
   CER ~0.13, under any sane gate. So every clip longer than 30s is also scored over its
   last ~30s (approximated by character count at the corpus median pace, see
   ``_stt.tail_cer``), stored as ``cer_tail`` on the transcription entry. A clip is bad
   when EITHER score is over its threshold (``--cer-threshold`` 0.08 /
   ``--tail-cer-threshold`` 0.12, the latter looser because the window boundary cuts
   mid-sentence), and the same double gate decides when a regenerated draw is good enough
   to promote.

1b. Triple-check a bad CER before trusting it. STT can hallucinate a wildly wrong
   transcript on perfectly fine audio (Whisper's "Sous-titrage ..." caption is the
   classic case), which scores a false-bad CER and would waste a regeneration. So while
   the CER is >= ``--cer-threshold`` and the clip has been read fewer than
   ``STT_CHECK_TARGET`` (3) times, re-transcribe it, each recheck one step further up the
   ``--stt-recheck-temperature`` ladder (default base 0.5, distinct from
   ``--stt-temperature``, so the two rechecks run at 0.5 then 1.0), and keep whichever
   reading scored the LOWEST CER. The reading count lives in ``n_stt_check`` on the
   transcription entry, so a clip still bad afterwards is never re-read again (it proceeds
   to step 2) and a resumed run only tops a partially checked clip back up to 3; a clip a
   recheck rescued drops the marker and is treated as a normal good clip (no regeneration).
2. If that best CER is >= ``--cer-threshold``, improve it: up to ``--n-improv``
   attempts, re-synthesize via the TTS server (seed incremented each attempt,
   TTS ``cfg_alpha`` ramped from ``--cfg-alpha-start`` by ``--cfg-alpha-step`` each
   attempt, optional ``--speed-jitter``), re-transcribe, re-score. The goal is a draw with
   CER BELOW ``--cer-threshold``: stop early the moment one is reached. If no draw
   gets under the threshold, keep the lowest-CER draw as long as it still beats the
   original by ``--min-improvement`` (otherwise keep the original untouched).
3. On a win: back the current clip up before overwriting (the first win writes the
   pristine original to ``<clip>.flac.bak``, later wins write the prior version to
   ``<clip>.flac.bak2`` / ``.bak3`` / ..., so every intermediate audio is kept),
   atomically overwrite the original with the better audio, re-probe and update the
   row's ``duration``,
   update the stored transcription, and append a record to ``<name>.improved.jsonl``.

A per-row ``improvement`` marker records the outcome so a resumed run skips clips
already improved / exhausted (override with ``--force``). Output mirrors script 1:
``<name>.stt.jsonl`` under ``--output``.

``--rescore`` re-derives every stored ``cer`` / ``cer_tail`` under the current scoring
rules and re-applies the gates. Changing ``normalize_for_scoring`` or a threshold makes
every score already on disk stale, while leaving the TRANSCRIPTS themselves perfectly
valid, so re-transcribing the corpus would be pure waste: the worker reuses the stored
transcript and only a row that now reads bad (and has fewer than ``STT_CHECK_TARGET``
readings) spends an STT call. It also re-judges the REGENERATION QUEUE the old rules
built: a ``pending_tts`` row that now reads clean drops its marker instead of costing a
regeneration, and one that stays bad has its ``orig_cer`` / ``orig_tail`` refreshed, so
a candidate is later compared against today's number and not a stale inflated one.

``<name>.improved.jsonl`` is a durable audit trail of every win: it is loaded and
merged (never deleted) on start, and each record keeps the before / after CER
(``orig_cer`` / ``new_cer`` / ``cer_delta``), the before / after transcript
(``orig_transcript`` / ``new_transcript``), the before / after duration
(``orig_duration`` / ``new_duration``), the row ``index`` and audio path, the
winning ``seed`` / ``speed`` / ``attempts``, the ``.bak`` path, and timestamps
(``first_ts`` / ``ts`` epoch plus ``*_iso``). The pristine baseline
(``orig_cer_pristine``) and ``first_ts`` survive re-improvement so history is never
lost.

Seed: retries walk seeds from ``--start-seed`` (default 43), incrementing per attempt.
It reaches Voxtral's flow-matching noise only on the forked vllm-omni the CrispASR image
is built from, which routes it to stage 0's ``voxtral_noise_seed``; on a stock server it
is accepted and inert (see ``tts_synthesize``). As a safety net, a regenerated clip
byte-identical to the original (checked by length then SHA1) means the seed had no
effect, so the run CRASHES immediately (SeedDeterminismError) instead of wasting a
whole run. ``--speed-jitter`` is NOT a second lever on this backend: vllm-omni applies
``speed`` after generation, as a torchaudio phase vocoder over the finished waveform,
so it time-stretches identical audio rather than drawing it differently. Leave it at 0.

In ``--mode both`` two servers must be up at once: the STT endpoint
(``--stt-endpoint``) and the TTS server (``--tts-url``, default
http://127.0.0.1:8003). The split ``stt`` / ``tts`` modes need only their own server
up (that is their whole point). Manifest audio paths are resolved via ``--audio-root``
(usually required), then absolute / CWD.

``--category`` (repeatable) restricts ALL work to rows whose ``category`` matches (e.g.
``--category parhaf``). Rows in every other category are passed through UNCHANGED: their
already-computed transcriptions / CER are preserved, never re-run, so you can focus the
expensive TTS regeneration on the one noisy subset now and resume the rest later without
losing anything. Convergence (the ``stt`` exit 10) is then judged over the selected
categories only, so the driver's alternating loop stops when just those subsets are done,
ignoring any pending clips a prior full run left in the other categories.

Usage (one command does STT + improvement over the whole manifest):

    uv run 01_recursive_improvement.py \
        --input ../99_hf_release/data/NeMO_files/full.jsonl \
        --output ./stt_out \
        --audio-root ../99_hf_release/data/NeMO_files \
        --stt-endpoint http://localhost:8000/v1/audio/transcriptions \
        --stt-model whisper-large-v3 \
        --tts-url http://127.0.0.1:8003 --tts-voice fr_male \
        --stt-parallel 8 --tts-parallel 8 --n-improv 5 --cer-threshold 0.08

When STT and TTS cannot share VRAM, alternate the split modes instead (a driver script
brings the matching server up before each), same flags plus ``--mode``:

    uv run 01_recursive_improvement.py ... --mode stt   # transcribe + score + promote
    uv run 01_recursive_improvement.py ... --mode tts   # generate candidates
    # repeat until the --mode stt run exits 10 (converged), or 11 (that mode ran dry)

This file was written with Claude Code.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import click
import soundfile as sf
from loguru import logger
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent


class SeedDeterminismError(RuntimeError):
    """Raised when a regenerated clip is byte-identical to the original, which
    means the request seed did not take effect. We abort the whole run rather than
    waste it producing no change."""

    def __init__(self, ident: str, seed: int):
        super().__init__(
            f"regenerated clip for {ident!r} is BYTE-IDENTICAL to the original "
            f"(same length and SHA1) with seed {seed}: the seed had no effect, so "
            f"retries cannot change anything. Check for a fixed seed env var on the "
            f"TTS server overriding the request, or bump --start-seed if the "
            f"originals were generated with this seed."
        )


class GenerationError(RuntimeError):
    """A TTS generation came back empty (0 bytes) or with duration 0. Raised so the
    run aborts rather than replacing a real clip with silence or nothing."""


class TTSRequestError(RuntimeError):
    """The TTS request kept failing (network / HTTP) after every retry. Raised so
    the run aborts rather than silently skipping the clip: a dead TTS server would
    otherwise make every regeneration a no-op with nobody noticing."""


def _load_module(path: Path, name: str):
    """Import a sibling script whose filename is not a valid module name (the
    ``01_`` prefix), so we can reuse its helpers instead of duplicating them.
    Importing runs the module body only; its ``main()`` is guarded by
    ``if __name__ == '__main__'`` so nothing executes on import."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reuse script 1's STT + scoring + io helpers and orchestration, and the stage-05
# audio client's HTTP layer, rather than copying them (the repo forbids duplication).
_stt = _load_module(_HERE / "01_compute_stt.py", "compute_stt_mod")
_tts = _load_module(
    _HERE.parent / "05_generate_audio" / "01_generate_audio.py", "generate_audio_mod"
)
# The stage-05 clip-name convention lives with the manifest builder that already reads it
# (stdlib-only import, so it costs nothing here).
sys.path.insert(0, str(_HERE.parent / "utils"))
from nemo_manifest import parse_clip_name  # noqa: E402


# How many independent STT readings a bad-CER clip gets before its CER is trusted: the
# first pass plus (STT_CHECK_TARGET - 1) rechecks at rising temperatures. 3 = the
# triple-check of step 1b above. The count is stored per clip in ``n_stt_check``.
STT_CHECK_TARGET = 3


def recheck_temperature(base: float, readings_done: int) -> float:
    """Temperature for the recheck that follows ``readings_done`` readings (>= 1): the
    ladder walks ``base``, ``2 * base``, ... capped at 1.0 (Whisper's own fallback
    ceiling), so each recheck samples the audio differently instead of repeating the
    previous reading. With the default base 0.5 the two rechecks run at 0.5 then 1.0."""
    return round(min(1.0, base * readings_done), 4)


def best_cer(transcript: str | None, source: str | None, target: str | None) -> float | None:
    """Lower of the CER against source and against target, both fully normalized
    by script 1's ``compute_metrics`` (which normalizes reference and hypothesis
    identically). None when there is nothing scorable."""
    if not transcript:
        return None
    vals: list[float] = []
    for ref in (source, target):
        if ref:
            _wer, cer = _stt.compute_metrics(ref, transcript)
            if cer is not None:
                vals.append(cer)
    return min(vals) if vals else None


def best_tail_cer(
    transcript: str | None, source: str | None, target: str | None, duration: float | None
) -> float | None:
    """``best_cer`` restricted to the last ~30s of the clip (see ``_stt.tail_cer``): the
    lower of the two tail CERs, or None on a clip too short to have a tail."""
    if not transcript:
        return None
    vals = [
        cer for ref in (source, target) if ref
        for cer in [_stt.tail_cer(ref, transcript, duration or 0.0)] if cer is not None
    ]
    return min(vals) if vals else None


def is_bad(cer: float | None, tail: float | None, cfg: SimpleNamespace) -> bool:
    """Whether a reading fails quality: the whole clip is off (``cer >= --cer-threshold``)
    OR just its ending is (``tail >= --tail-cer-threshold``). The tail gate is looser
    because its window is ~5x shorter, hence ~5x noisier, than the whole-clip score."""
    return ((cer is not None and cer >= cfg.cer_threshold)
            or (tail is not None and tail >= cfg.tail_cer_threshold))


def _fmt_scores(cer: float | None, tail: float | None) -> str:
    """`cer=0.123 tail=0.456` for the logs, with the tail omitted on short clips."""
    cer_s = f"{cer:.3f}" if cer is not None else "n/a"
    return f"cer={cer_s}" + (f" tail={tail:.3f}" if tail is not None else "")


def badness(cer: float | None, tail: float | None, cfg: SimpleNamespace) -> tuple[bool, float]:
    """Sort key picking the most favorable of several readings / candidate draws: one that
    passes both gates beats one that does not, and CER breaks the tie within each class.
    A reading whose only defect is a derailed tail therefore loses to a clean one, which is
    the whole point of re-reading a clip before condemning it."""
    return (is_bad(cer, tail, cfg), cer if cer is not None else float("inf"))


def flac_duration(path: Path) -> float | None:
    """Clip length in seconds from the audio header (no full decode)."""
    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate if info.samplerate else None
    except Exception as exc:  # noqa: BLE001 - any unreadable file, just skip refresh
        logger.warning(f"could not probe duration of {path.name}: {exc}")
        return None


def attempt_speed(base_speed: float | None, jitter: float, attempt: int) -> float | None:
    """Speed for a given retry: cycle 0, +j, -j, +2j, -2j ... around the base. Returns
    None (knob omitted, server default) when jitter is off and no base speed is set.

    NOT a source of generation variation on the Voxtral backend: vllm-omni applies
    ``speed`` to the finished waveform (a torchaudio phase vocoder), so jitter yields
    time-stretched copies of the SAME draw, with vocoder artifacts on top. Off by
    default (``--speed-jitter 0``) and best left there; the seed is the real lever."""
    if jitter <= 0:
        return base_speed
    base = base_speed if base_speed is not None else 1.0
    step = (attempt + 1) // 2
    sign = 1 if attempt % 2 == 1 else -1
    return round(base + sign * step * jitter, 4)


def tts_synthesize(
    tts_url: str,
    text: str,
    *,
    response_format: str,
    seed: int | None,
    voice: str | None,
    speed: float | None,
    model: str | None,
    timeout: float,
    max_retries: int = 4,
    cfg_alpha: float | None = None,
) -> bytes:
    """POST one synthesis request and return the raw audio bytes. Unset knobs are
    omitted so the server keeps its startup defaults (same contract as stage 05).

    REQUIRES the forked vllm-omni that the CrispASR ``voxtral-tts`` image is built from
    (``CrispASR/voxtral/vllm-omni``). Both per-draw knobs are sent flat, which is a
    contract of that fork, not of upstream:

    * ``cfg_alpha`` is a top-level field there, promoted precisely so a request that
      sets it cannot silently do nothing. Upstream has no such field and its request
      model declares no ``model_config``, so pydantic's default ``extra=ignore`` DROPS
      a flat one without an error, and every draw runs at the server's startup value.
    * ``seed`` exists upstream but cannot reach Voxtral's audio: stage 0 hands vLLM's
      sampler ``fake_logits_for_audio_tokens()`` (a vocab-wide -inf row with one finite
      entry), so sampling params are arithmetically inert there, and the only randomness
      is the flow-matching Gaussian off the global torch RNG. The fork routes the flat
      seed into stage 0's ``extra_args["voxtral_noise_seed"]``, which does pin it.

    So against a STOCK vllm-omni both knobs are silently ignored and the draws differ
    only by global-RNG drift, which is what this stage used to do without knowing it.
    ``extra_params`` remains the escape hatch and wins on conflict, so nothing here
    needs to nest. One more server-side requirement: stage 0's deploy YAML must keep a
    default ``cfg_alpha`` under ``default_sampling_params.extra_args``, or vllm-omni
    sets ``has_sampling_extra_args=False`` and drops every per-request extra.

    Retries transient failures (connection errors, 429, 5xx) with exponential
    backoff, mirroring the STT client. A non-retryable status (e.g. 4xx) stops
    early. On exhaustion raises TTSRequestError so the run aborts rather than
    silently skipping the clip."""
    body: dict = {"input": text, "response_format": response_format}
    for key, value in (("seed", seed), ("voice", voice), ("speed", speed),
                       ("model", model), ("cfg_alpha", cfg_alpha)):
        if value is not None:
            body[key] = value

    last: Exception | None = None
    for attempt in range(max_retries):
        try:
            code, payload, elapsed = _tts.post_json(f"{tts_url}/v1/audio/speech", body, timeout)
        except Exception as exc:  # noqa: BLE001 - connection/timeout, retry with backoff
            last = exc
        else:
            if code == 200:
                return payload
            last = RuntimeError(
                f"TTS server returned {code} after {elapsed:.2f}s: "
                f"{payload[:300].decode('utf-8', 'replace')}"
            )
            if code not in (429, 500, 502, 503, 504):
                break  # not transient (bad request etc): retrying will not help
        if attempt < max_retries - 1:
            backoff = 2.0 * (2 ** attempt)
            logger.debug(f"TTS retry {attempt + 1}/{max_retries} after {backoff}s: {last}")
            time.sleep(backoff)
    raise TTSRequestError(f"TTS failed after {max_retries} attempts: {last}")


def _synthesize_draft(
    attempt: int, cfg: SimpleNamespace, ident: str, text: str,
    orig_len: int, orig_hash: str, seen_hashes: set[str],
) -> dict:
    """Synthesize ONE candidate for the given attempt index (the TTS-only half of a
    regeneration, shared by ``both`` mode's inline loop and ``tts`` mode's draft
    generation). Returns ``{"payload", "seed", "speed", "cfg_alpha", "dur",
    "duplicate", "truncated"}``: ``duplicate`` is True when the draw byte-repeats an
    earlier one (seeds collided), ``truncated`` when it stopped on the backend's frame
    cap rather than at the end of the text; either way the caller skips the draw.

    Only the network call is gated by ``tts_sem`` so a clip busy on STT never holds a
    TTS slot. ``tts_synthesize`` retries transient failures itself and raises
    ``TTSRequestError`` on exhaustion, which we let propagate to abort the run (never
    skip silently). Empty / duration-0 audio raises ``GenerationError``; a draw
    byte-identical to the original raises ``SeedDeterminismError`` (the seed had no
    effect), both aborting the run rather than replacing a real clip with silence or
    wasting the whole run producing no change."""
    seed = cfg.start_seed + attempt
    speed = attempt_speed(cfg.tts_speed, cfg.speed_jitter, attempt)
    # Nudge the TTS guidance up on each re-attempt: cfg_alpha starts at
    # --cfg-alpha-start on the first re-attempt and grows by --cfg-alpha-step each
    # attempt, pushing the model harder to follow the text when the first draws stay
    # bad. None -> knob omitted (server keeps its default).
    cfg_alpha = (
        None if cfg.cfg_alpha_start is None
        else round(cfg.cfg_alpha_start + cfg.cfg_alpha_step * attempt, 4)
    )
    with cfg.tts_sem:
        payload = tts_synthesize(
            cfg.tts_url, text,
            response_format=cfg.tts_format, seed=seed, voice=cfg.tts_voice,
            speed=speed, model=cfg.tts_model, timeout=cfg.tts_timeout,
            max_retries=cfg.tts_max_retries, cfg_alpha=cfg_alpha,
        )

    # A generation must never be empty or duration-0: that would mean replacing a
    # real clip with silence / nothing. Abort the run so the user notices.
    if not payload:
        raise GenerationError(f"{ident}: TTS returned empty audio (0 bytes) at seed {seed}")
    dur = _tts.payload_duration(payload)
    if dur is not None and dur <= 0:
        raise GenerationError(f"{ident}: TTS returned duration-0 audio at seed {seed}")

    # Stopped for length, not at the end of the text: /v1/audio/speech has no
    # finish_reason, so a draw pinned at the backend's frame cap is the only sign that
    # generation was cut off mid-sentence. Such a draw can never be a valid replacement,
    # so report it and let the caller skip it (every draw of an over-long text truncates,
    # so the clip ends up EXHAUSTED and keeps its original, which is the honest outcome:
    # the fix is re-chunking the text, not another seed).
    truncated = _tts.is_truncated(dur, cfg.max_audio_seconds)

    # Seed check: only bother hashing when the length matches the original (a cheap
    # gate). A byte-identical regen means the seed did not take effect, so we crash
    # the whole run rather than waste it producing no change.
    h = hashlib.sha1(payload).hexdigest()
    if len(payload) == orig_len and h == orig_hash:
        raise SeedDeterminismError(ident, seed)
    duplicate = h in seen_hashes
    if not duplicate:
        seen_hashes.add(h)
    return {"payload": payload, "seed": seed, "speed": speed, "cfg_alpha": cfg_alpha,
            "dur": dur, "duplicate": duplicate, "truncated": truncated}


def _score_audio_file(
    audio_file: Path, cfg: SimpleNamespace, source: str | None, target: str | None,
    duration: float | None = None,
) -> dict | None:
    """Transcribe an audio file and score its best CER, plus the tail CER of its last
    ~30s when ``duration`` says the draw is long enough (the STT-only half of a
    regeneration, shared by ``both`` mode's inline loop and ``tts`` mode's draft
    scoring). Returns ``{"transcript", "cer", "tail", "stt_seconds"}``, or None when the
    clip transcribed blank.

    A blank transcription of a fresh DRAW is a bad draw, not the dataset clip: the
    caller discards it and tries the next (the whole point of trying several seeds).
    Persistent network failure is a ``TranscriptionError``, which propagates and
    aborts the run; a blank ORIGINAL clip is fatal, handled in ``combined_worker``.
    Only the call is gated by ``stt_sem`` so a clip busy on TTS never holds an STT
    slot."""
    try:
        with cfg.stt_sem:
            transcript, stt_seconds = _stt.transcribe_timed(
                audio_file, cfg.stt_endpoint, cfg.stt_token, cfg.stt_model, cfg.stt_temperature,
                cfg.stt_language, cfg.stt_response_format, cfg.stt_extra_params,
                cfg.stt_timeout, cfg.stt_max_retries,
            )
    except _stt.EmptyTranscriptError:
        return None
    return {"transcript": transcript, "cer": best_cer(transcript, source, target),
            "tail": best_tail_cer(transcript, source, target, duration),
            "stt_seconds": stt_seconds}


def run_regeneration(
    pos: int, ident: str, audio_path: Path, text: str,
    source: str | None, target: str | None, orig_cer: float, cfg: SimpleNamespace,
    orig_tail: float | None = None,
) -> dict:
    """Regenerate one bad clip, trying up to --n-improv draws to clear BOTH gates (CER
    under --cer-threshold and tail CER under --tail-cer-threshold), stopping early the
    moment a draw does. If no draw clears them, keep the lowest-CER draw when it still
    beats the original by --min-improvement, otherwise keep the original. Returns a dict;
    when a draw wins it carries the winning ``payload`` (bytes) + metadata, else
    ``payload`` is None. Raises SeedDeterminismError if a regen comes back byte-identical
    to the original."""
    try:
        orig_bytes = audio_path.read_bytes()
    except OSError as exc:
        logger.warning(f"{ident}: cannot read {audio_path} to regenerate: {exc}")
        return {"payload": None, "attempts": 0, "all_duplicates": False}
    orig_len = len(orig_bytes)
    orig_hash = hashlib.sha1(orig_bytes).hexdigest()

    seen_hashes = {orig_hash}
    dup_count = 0
    trunc_count = 0
    attempts = 0
    best = None  # best draw seen so far (kept if none clears the gates)
    logger.info(
        f"REDO {ident}: {_fmt_scores(orig_cer, orig_tail)} over threshold "
        f"({cfg.cer_threshold} / tail {cfg.tail_cer_threshold}), up to {cfg.n_improv} attempt(s)"
    )
    for attempt in range(cfg.n_improv):
        attempts = attempt + 1
        # Synthesize one candidate (TTS only). Empty / duration-0 audio and a draw
        # byte-identical to the original abort the run (GenerationError /
        # SeedDeterminismError); a draw that merely repeats an earlier one is skipped.
        draft = _synthesize_draft(attempt, cfg, ident, text, orig_len, orig_hash, seen_hashes)
        seed, speed, cfg_alpha, dur = draft["seed"], draft["speed"], draft["cfg_alpha"], draft["dur"]
        alpha_s = f" alpha={cfg_alpha}" if cfg_alpha is not None else ""
        if draft["truncated"]:
            trunc_count += 1
            logger.warning(
                f"  {ident} attempt {attempt + 1}/{cfg.n_improv} seed={seed}{alpha_s} -> "
                f"{dur:.2f}s, stopped at the TTS frame cap ({cfg.max_audio_seconds:.2f}s): "
                f"cut off mid-sentence, discarding draw"
            )
            continue
        if draft["duplicate"]:
            dup_count += 1
            logger.debug(f"{ident}: attempt {attempt + 1} repeated an earlier draw, skipped")
            continue

        # Score the fresh clip (STT): write to a temp file (transcribe reads a path),
        # transcribe, delete. A unique temp name keeps concurrent clips isolated. A
        # blank transcript is a bad draw (discard and try the next), never fatal here.
        tmp = Path(tempfile.gettempdir()) / f"improve_{os.getpid()}_{pos}_{attempt}.{cfg.tts_format}"
        try:
            tmp.write_bytes(draft["payload"])
            scored = _score_audio_file(tmp, cfg, source, target, dur)
        finally:
            tmp.unlink(missing_ok=True)
        if scored is None:
            logger.info(
                f"  {ident} attempt {attempt + 1}/{cfg.n_improv} seed={seed}{alpha_s} -> transcribed empty, discarding draw"
            )
            continue

        transcript, stt_seconds, cand_cer = scored["transcript"], scored["stt_seconds"], scored["cer"]
        cand_tail = scored["tail"]
        # Real-time factor of this draw's transcription (STT seconds / clip seconds).
        rtf_s = f" stt={stt_seconds:.2f}s"
        if dur and dur > 0:
            rtf_s += f" rtf={stt_seconds / dur:.3f}"
        if cand_cer is None:
            logger.info(f"  {ident} attempt {attempt + 1}/{cfg.n_improv} seed={seed}{alpha_s} -> cer=n/a{rtf_s}")
            continue
        # Keep the best draw so far (one that clears both gates beats one that does not,
        # CER breaks the tie), so we can fall back to it if none clears them.
        is_best = best is None or badness(cand_cer, cand_tail, cfg) < badness(
            best["new_cer"], best["new_tail"], cfg)
        if is_best:
            best = {"payload": draft["payload"], "new_transcript": transcript, "new_cer": cand_cer,
                    "new_tail": cand_tail, "seed": seed, "speed": speed, "cfg_alpha": cfg_alpha}
        below = not is_bad(cand_cer, cand_tail, cfg)
        if below:
            note = f" (under both thresholds, done)"
        elif is_best:
            note = f" (new best, still over threshold, trying next)"
        else:
            note = " (not better, trying next)"
        logger.info(
            f"  {ident} attempt {attempt + 1}/{cfg.n_improv} seed={seed}{alpha_s} -> "
            f"{_fmt_scores(cand_cer, cand_tail)}{rtf_s}{note}"
        )
        if below:
            # Goal reached: both gates cleared. Stop early with this draw (which is also
            # the best so far, since the original failed at least one gate).
            logger.info(
                f"IMPROVED {ident} {_fmt_scores(orig_cer, orig_tail)} -> "
                f"{_fmt_scores(cand_cer, cand_tail)} "
                f"(attempt {attempt + 1}, seed {seed}, under both thresholds)"
            )
            return {**best, "attempts": attempts, "all_duplicates": False}

    # Exhausted every attempt without clearing the gates. Keep the best draw if it still
    # beats the original by --min-improvement, otherwise leave the original untouched.
    if best is not None and best["new_cer"] <= orig_cer - cfg.min_improvement:
        logger.info(
            f"IMPROVED {ident} {_fmt_scores(orig_cer, orig_tail)} -> "
            f"{_fmt_scores(best['new_cer'], best['new_tail'])} "
            f"(best of {attempts}, still over threshold)"
        )
        return {**best, "attempts": attempts, "all_duplicates": False}

    logger.info(
        f"EXHAUSTED {ident}: kept {_fmt_scores(orig_cer, orig_tail)} after {attempts} attempt(s)"
        + (f"; best draw {_fmt_scores(best['new_cer'], best['new_tail'])} did not beat it"
           if best is not None else "")
    )
    if trunc_count:
        logger.warning(
            f"{ident}: {trunc_count}/{attempts} draw(s) hit the TTS frame cap "
            f"({cfg.max_audio_seconds:.2f}s). Its text is too long for the backend to "
            f"speak in full, so no seed can fix this clip: re-chunk the source text."
        )
    return {"payload": None, "attempts": attempts,
            "all_duplicates": bool(dup_count and dup_count == attempts)}


def draft_path_for(audio_path: Path, seed: int, fmt: str) -> Path:
    """Sidecar path for one persisted TTS candidate, deterministic by seed so a
    re-run of the tts pass overwrites the same file instead of piling up. Lives next
    to the clip, like the ``.bak`` backups."""
    return audio_path.with_name(f"{audio_path.name}.draft{seed}.{fmt}")


def clear_drafts(audio_path: Path) -> None:
    """Delete any candidate sidecars next to a clip (leftovers from an interrupted or
    superseded tts pass) so drafts never accumulate or get mistaken for fresh ones."""
    for p in audio_path.parent.glob(f"{audio_path.name}.draft*"):
        try:
            p.unlink()
        except OSError:
            pass


def generate_drafts(
    pos: int, ident: str, audio_path: Path, text: str, orig_cer: float, cfg: SimpleNamespace,
) -> dict:
    """TTS-only half of a split regeneration (``--mode tts``): synthesize up to
    ``--n-improv`` candidates for one bad clip and persist each as a sidecar draft
    file, WITHOUT scoring (the STT server is down in this mode). Returns
    ``{"drafts": [...], "attempts": n, "all_duplicates": bool}`` where each draft is
    ``{"path", "seed", "speed", "cfg_alpha", "dur"}``; the next ``--mode stt`` pass
    scores them and promotes the best. Raises SeedDeterminismError / GenerationError
    exactly like the inline loop (byte-identical draw / empty audio abort the run)."""
    try:
        orig_bytes = audio_path.read_bytes()
    except OSError as exc:
        logger.warning(f"{ident}: cannot read {audio_path} to regenerate: {exc}")
        return {"drafts": [], "attempts": 0, "all_duplicates": False}
    orig_len = len(orig_bytes)
    orig_hash = hashlib.sha1(orig_bytes).hexdigest()

    # Drop any stale candidates from a previous, interrupted tts pass before writing a
    # fresh set (draft names are seed-deterministic, but a smaller --n-improv than last
    # time would otherwise orphan the higher-seed files).
    clear_drafts(audio_path)

    seen_hashes = {orig_hash}
    dup_count = 0
    trunc_count = 0
    attempts = 0
    drafts: list[dict] = []
    logger.info(
        f"REDO {ident}: cer={orig_cer:.3f} >= {cfg.cer_threshold}, generating up to {cfg.n_improv} candidate(s)"
    )
    for attempt in range(cfg.n_improv):
        attempts = attempt + 1
        draft = _synthesize_draft(attempt, cfg, ident, text, orig_len, orig_hash, seen_hashes)
        seed, cfg_alpha = draft["seed"], draft["cfg_alpha"]
        alpha_s = f" alpha={cfg_alpha}" if cfg_alpha is not None else ""
        if draft["truncated"]:
            trunc_count += 1
            logger.warning(
                f"  {ident} candidate {attempt + 1}/{cfg.n_improv} seed={seed}{alpha_s} -> "
                f"{draft['dur']:.2f}s, stopped at the TTS frame cap "
                f"({cfg.max_audio_seconds:.2f}s): cut off mid-sentence, not saved"
            )
            continue
        if draft["duplicate"]:
            dup_count += 1
            logger.debug(f"{ident}: attempt {attempt + 1} repeated an earlier draw, skipped")
            continue
        draft_path = draft_path_for(audio_path, seed, cfg.tts_format)
        _stt.atomic_write_bytes(draft_path, draft["payload"])
        drafts.append({"path": str(draft_path), "seed": seed, "speed": draft["speed"],
                       "cfg_alpha": cfg_alpha, "dur": draft["dur"]})
        logger.info(
            f"  {ident} candidate {attempt + 1}/{cfg.n_improv} seed={seed}{alpha_s} -> saved {draft_path.name}"
        )
    if trunc_count:
        logger.warning(
            f"{ident}: {trunc_count}/{attempts} candidate(s) hit the TTS frame cap "
            f"({cfg.max_audio_seconds:.2f}s). Its text is too long for the backend to "
            f"speak in full, so no seed can fix this clip: re-chunk the source text."
        )
    return {"drafts": drafts, "attempts": attempts,
            "all_duplicates": bool(dup_count and dup_count == attempts and not drafts)}


def score_and_pick(
    pos: int, ident: str, drafts: list[dict],
    source: str | None, target: str | None, orig_cer: float, cfg: SimpleNamespace,
    orig_tail: float | None = None,
) -> dict:
    """STT-only half of a split regeneration (``--mode stt``): transcribe every
    candidate a prior ``--mode tts`` pass persisted, score each, and pick the best.
    No early stop (all candidates already exist), so we keep the best draw (one clearing
    both the CER and tail gates beats one that does not, CER breaks the tie) and promote
    it when it clears them, or when it still beats the original by ``--min-improvement``,
    else keep the original. Returns a dict shaped like ``run_regeneration``'s: a winner
    carries ``payload`` (read back from its draft file) + metadata, else ``payload`` is
    None. The draft files are removed by ``apply_result`` once the pick is applied."""
    best = None
    n = len(drafts)
    logger.info(
        f"SCORE {ident}: {n} candidate(s) from the tts pass, "
        f"original {_fmt_scores(orig_cer, orig_tail)}"
    )
    for i, d in enumerate(drafts):
        draft_path = Path(d["path"])
        seed, cfg_alpha, dur = d.get("seed"), d.get("cfg_alpha"), d.get("dur")
        alpha_s = f" alpha={cfg_alpha}" if cfg_alpha is not None else ""
        if not draft_path.is_file():
            logger.warning(f"  {ident} candidate {i + 1}/{n} seed={seed}: draft file missing, skipping")
            continue
        scored = _score_audio_file(draft_path, cfg, source, target, dur)
        if scored is None:
            logger.info(f"  {ident} candidate {i + 1}/{n} seed={seed}{alpha_s} -> transcribed empty, discarding draw")
            continue
        transcript, stt_seconds, cand_cer = scored["transcript"], scored["stt_seconds"], scored["cer"]
        cand_tail = scored["tail"]
        rtf_s = f" stt={stt_seconds:.2f}s"
        if dur and dur > 0:
            rtf_s += f" rtf={stt_seconds / dur:.3f}"
        if cand_cer is None:
            logger.info(f"  {ident} candidate {i + 1}/{n} seed={seed}{alpha_s} -> cer=n/a{rtf_s}")
            continue
        is_best = best is None or badness(cand_cer, cand_tail, cfg) < badness(
            best["new_cer"], best["new_tail"], cfg)
        if is_best:
            best = {"path": str(draft_path), "new_transcript": transcript, "new_cer": cand_cer,
                    "new_tail": cand_tail, "seed": seed, "speed": d.get("speed"),
                    "cfg_alpha": cfg_alpha}
        below = not is_bad(cand_cer, cand_tail, cfg)
        note = (" (under both thresholds)" if below
                else (" (new best, still over threshold)" if is_best else " (not better)"))
        logger.info(
            f"  {ident} candidate {i + 1}/{n} seed={seed}{alpha_s} -> "
            f"{_fmt_scores(cand_cer, cand_tail)}{rtf_s}{note}"
        )

    # Pick the best candidate: promote it when it clears both gates, or when it still
    # beats the original by --min-improvement; otherwise keep the original.
    if best is not None and (not is_bad(best["new_cer"], best["new_tail"], cfg)
                             or best["new_cer"] <= orig_cer - cfg.min_improvement):
        reached = not is_bad(best["new_cer"], best["new_tail"], cfg)
        logger.info(
            f"IMPROVED {ident} {_fmt_scores(orig_cer, orig_tail)} -> "
            f"{_fmt_scores(best['new_cer'], best['new_tail'])} "
            + (f"(best of {n}, under both thresholds)" if reached
               else f"(best of {n}, still over threshold)")
        )
        return {**best, "payload": Path(best["path"]).read_bytes(), "attempts": n,
                "all_duplicates": False}

    logger.info(
        f"EXHAUSTED {ident}: kept {_fmt_scores(orig_cer, orig_tail)} after {n} candidate(s)"
        + (f"; best draw {_fmt_scores(best['new_cer'], best['new_tail'])} did not beat it"
           if best is not None else "")
    )
    return {"payload": None, "attempts": n, "all_duplicates": False}


def _transcribe_original(audio_path: Path, cfg: SimpleNamespace, temperature: float) -> tuple[str, float]:
    """One STT pass over a DATASET clip at ``temperature``, gated by ``stt_sem`` so a clip
    busy on TTS never holds an STT slot, and timed inside the semaphore so the RTF excludes
    the queue wait. Shared by the first transcription and the bad-CER recheck (only the
    temperature differs). EmptyTranscriptError (a dataset clip blank after every retry) and
    TranscriptionError (a dead server) propagate: both are fatal, we never skip a clip we
    cannot transcribe. The recheck caller catches EmptyTranscriptError to fall back to the
    first reading instead of aborting."""
    with cfg.stt_sem:
        return _stt.transcribe_timed(
            audio_path, cfg.stt_endpoint, cfg.stt_token, cfg.stt_model, temperature,
            cfg.stt_language, cfg.stt_response_format, cfg.stt_extra_params,
            cfg.stt_timeout, cfg.stt_max_retries,
        )


def combined_worker(pos: int, rec: dict, cfg: SimpleNamespace, model_key: str, input_file: Path) -> dict:
    """Thread-side per-clip work: transcribe the original (reusing a stored
    transcript on resume), score it, and if it is bad regenerate it. Returns the
    transcription entry plus any improvement payload; the main thread applies the
    disk changes. Raises only to abort the whole run on a fatal condition:
    SeedDeterminismError, GenerationError, or a persistent server failure
    (TranscriptionError / TTSRequestError after backoff)."""
    ident = _stt.short_ident(rec, cfg.audio_key)
    entry: dict = {"text": None, "wer": None, "cer": None}
    # ``action`` tells apply_result which disk mutation to run: "scored" (store a
    # transcription, maybe regenerate inline in both mode / flag pending_tts in stt
    # mode), "drafted" (a tts pass persisted candidates), "picked" (an stt pass scored
    # candidates and chose a winner), or None (nothing to do this pass).
    out = {"pos": pos, "entry": entry, "improvement": None, "orig_cer": None,
           "audio_path": None, "action": None, "drafts": None}

    audio_path = _stt.resolve_audio(rec, input_file, cfg.audio_key, cfg.audio_root)
    if audio_path is None:
        entry["error"] = f"audio not found for {rec.get(cfg.audio_key)!r}"
        logger.warning(f"{ident}: {entry['error']} (try --audio-root)")
        return out
    out["audio_path"] = str(audio_path)

    source = rec.get("asr_training_source")
    target = rec.get("asr_training_target")
    text = source or target  # what the TTS was fed is asr_training_source
    imp_state = rec.get("improvement") or {}
    status = imp_state.get("status")

    # --- Split mode: tts (generate candidates only, STT server down) ----------
    # Only clips a prior --mode stt pass flagged bad (pending_tts) do work here; we
    # never transcribe, just synthesize and persist candidates for the next stt pass.
    if cfg.mode == "tts":
        if status == "pending_tts" and cfg.n_improv > 0 and text:
            out["action"] = "drafted"
            out["orig_cer"] = imp_state.get("orig_cer")
            out["orig_tail"] = imp_state.get("orig_tail")
            out["improvement"] = generate_drafts(
                pos, ident, audio_path, text, imp_state.get("orig_cer") or 0.0, cfg
            )
        return out

    # --- Split mode: stt, candidates waiting to be scored (TTS server down) ----
    # A prior tts pass left persisted candidates (pending_stt); score them and pick
    # the winner. The original transcript / cer are already on the row from the pass
    # that first flagged this clip, so the original is not re-transcribed.
    if cfg.mode == "stt" and status == "pending_stt":
        drafts = imp_state.get("drafts") or []
        # Only judge a COMPLETE candidate set. Picking the best of 3 when 5 were asked
        # for settles for a draw that may not be the best available, and the clip is
        # then marked improved / exhausted and never revisited. So a set left short by
        # the tts pass (it ran with a smaller --n-improv, or its sidecars have since
        # gone missing) goes BACK to that pass to be made in full.
        # The bar is "the tts pass delivered everything it was asked for", not
        # "len(drafts) == --n-improv": a draw it skipped as a duplicate or as frame-cap
        # truncated is skipped deterministically (the seed of each attempt is fixed), so
        # demanding a file for it would bounce the clip between the two passes forever.
        missing = [d for d in drafts if not Path(str(d.get("path") or "")).is_file()]
        asked = int(imp_state.get("attempts") or 0)
        if (not drafts) or missing or asked < cfg.n_improv:
            out["action"] = "requeue_tts"
            out["orig_cer"] = imp_state.get("orig_cer")
            out["orig_tail"] = imp_state.get("orig_tail")
            why = (f"{len(missing)} of its {len(drafts)} candidate file(s) are gone"
                   if missing else
                   f"only {asked} draw(s) were tried, --n-improv is {cfg.n_improv}"
                   if asked < cfg.n_improv else "it has no candidate at all")
            logger.warning(
                f"{ident}: incomplete candidate set ({why}), back to the tts pass "
                f"instead of picking from a partial set"
            )
            return out
        out["action"] = "picked"
        out["orig_cer"] = imp_state.get("orig_cer")
        out["orig_tail"] = imp_state.get("orig_tail")
        out["drafts"] = drafts
        out["improvement"] = score_and_pick(
            pos, ident, out["drafts"], source, target, imp_state.get("orig_cer") or 0.0, cfg,
            orig_tail=imp_state.get("orig_tail"),
        )
        return out

    # Reuse a good transcript already stored on this row (resume); else transcribe.
    # A blank stored transcript is treated as missing so it is redone, never kept.
    prior = (rec.get("transcriptions") or {}).get(model_key) or {}
    transcript = prior.get("text")
    if prior.get("error") or not (transcript and transcript.strip()):
        transcript = None
    stt_seconds = None
    # How many STT requests this clip actually costs this pass. 0 means the score was
    # re-derived from the stored transcript alone (a rescore), which is what the
    # RESCORED tag below reports: it is the cheap, offline half of a --rescore run, and
    # seeing it on a clip you expected to be transcribed means something is wrong.
    stt_calls = 0
    if transcript is None:
        if cfg.rescore_only:
            # Offline pass: never open a connection. needs_work already filters these
            # out, so this is just a belt-and-braces guard.
            return out
        # Both fatal, on purpose: EmptyTranscriptError (a dataset clip that stays blank
        # through every retry) and TranscriptionError (a persistent network failure) abort
        # the run. We never skip past a clip we cannot transcribe.
        transcript, stt_seconds = _transcribe_original(audio_path, cfg, cfg.stt_temperature)
        stt_calls += 1

    duration = rec.get("duration") or 0.0
    orig_cer = best_cer(transcript, source, target)
    orig_tail = best_tail_cer(transcript, source, target, duration)

    # Triple-check a bad CER at DIFFERENT STT temperatures before trusting it: a
    # hallucinated transcript (e.g. Whisper's "Sous-titrage ...") scores a spuriously high
    # CER on audio that is actually fine, which would waste a regeneration on a good clip.
    # While the clip reads bad, re-transcribe it until it has been read STT_CHECK_TARGET
    # times in total (the first pass plus whatever a previous run already did, per
    # n_stt_check), each recheck one step up the temperature ladder, keeping whichever
    # reading scored the LOWEST CER. A clip still bad at the end records its reading count
    # so it is never re-read again (it now genuinely goes to regeneration); one a recheck
    # rescued drops the marker and is treated as the good clip it is.
    checks = max(int(prior.get("n_stt_check") or 0), 1)
    while (is_bad(orig_cer, orig_tail, cfg)
           and checks < STT_CHECK_TARGET and cfg.stt_recheck_temperature is not None
           # An offline rescore never re-reads a clip: a newly-bad score is left on the
           # row and the next normal stt pass does the triple-check.
           and not cfg.rescore_only):
        temp = recheck_temperature(cfg.stt_recheck_temperature, checks)
        stt_calls += 1
        try:
            re_transcript, re_seconds = _transcribe_original(audio_path, cfg, temp)
        except _stt.EmptyTranscriptError:
            # A blank recheck is inconclusive (an earlier pass already produced text), not a
            # broken dataset clip: keep the reading we have rather than abort the run.
            re_transcript, re_seconds = None, None
        checks += 1
        re_cer = best_cer(re_transcript, source, target)
        re_tail = best_tail_cer(re_transcript, source, target, duration)
        re_s = _fmt_scores(re_cer, re_tail)
        if re_transcript and badness(re_cer, re_tail, cfg) < badness(orig_cer, orig_tail, cfg):
            logger.info(
                f"{ident}  RECHECK {checks}/{STT_CHECK_TARGET} temp={temp}: "
                f"{_fmt_scores(orig_cer, orig_tail)} -> {re_s} (kept the better reading)"
            )
            transcript, orig_cer, orig_tail, stt_seconds = (
                re_transcript, re_cer, re_tail, re_seconds)
        else:
            logger.info(
                f"{ident}  RECHECK {checks}/{STT_CHECK_TARGET} temp={temp}: {re_s} "
                f"no better than {_fmt_scores(orig_cer, orig_tail)} (kept the previous reading)"
            )

    entry["text"] = transcript
    entry["cer"] = orig_cer
    if orig_tail is not None:
        entry["cer_tail"] = orig_tail
    out["orig_cer"] = orig_cer
    out["orig_tail"] = orig_tail
    # A clip still bad records how many readings it has had, so it is not re-read past
    # STT_CHECK_TARGET (and a later run tops up a partially checked one); a clip that came
    # back good drops the marker.
    if is_bad(orig_cer, orig_tail, cfg):
        entry["n_stt_check"] = checks
    # Real-time factor for this clip (STT seconds / audio duration), when we actually
    # transcribed it this run (not on a resumed transcript). process_file rolls these
    # up into the bar's rtf= readout.
    dur = rec.get("duration") or 0
    rtf_s = ""
    if stt_seconds is not None:
        entry["stt_seconds"] = round(stt_seconds, 3)
        rtf_s = f" stt={stt_seconds:.2f}s"
        if dur and dur > 0:
            rtf = stt_seconds / dur
            entry["stt_rtf"] = round(rtf, 3)
            rtf_s += f" rtf={rtf:.3f}"
    elif stt_calls == 0:
        # Reused transcript (a rescore): the timing already on the row measured THIS
        # transcript, so carry it forward. The entry is rebuilt from scratch each pass,
        # so without this a rescore would silently erase the run's RTF telemetry from
        # every row it touched.
        for key in ("stt_seconds", "stt_rtf"):
            if prior.get(key) is not None:
                entry[key] = prior[key]

    # One log line per clip scored (like 01_compute_stt.py), so every STT result and
    # its CER is visible above the bars; a clip over the threshold additionally dumps
    # its texts so the derailed ones are readable. Runs in a pool thread; loguru +
    # tqdm.write keep this from tearing the progress bars.
    # RESCORED = this clip cost no STT request, its score was re-derived from the
    # transcript already stored on the row. Tagged in the line itself so a rescore pass
    # is distinguishable from a transcription pass at a glance.
    out["rescored"] = stt_calls == 0
    scores_s = _fmt_scores(orig_cer, orig_tail)
    tag = "RESCORED " if out["rescored"] else ""
    logger.info(f"{ident}  {tag}{scores_s}{rtf_s}")
    if is_bad(orig_cer, orig_tail, cfg):
        logger.warning(
            f"LOW QUALITY {ident} {scores_s}\n"
            f"  source    : {source}\n"
            f"  target    : {target}\n"
            f"  transcript: {transcript}"
        )

    # We transcribed / scored the original this pass. apply_result stores the entry
    # and (both mode) regenerates inline, or (stt mode) flags a bad clip pending_tts.
    out["action"] = "scored"
    if (is_bad(orig_cer, orig_tail, cfg) and cfg.n_improv > 0 and text
            and cfg.mode == "both" and not cfg.rescore_only):
        out["improvement"] = run_regeneration(
            pos, ident, audio_path, text, source, target, orig_cer, cfg, orig_tail=orig_tail)
    return out


def next_backup_path(audio_path: Path) -> Path:
    """The ``.bak`` path to write for THIS overwrite, keeping every prior version.

    The first improvement stores the pristine original at ``<clip>.bak``. Every
    later improvement stores the current (about-to-be-overwritten) clip at
    ``<clip>.bak2``, ``<clip>.bak3``, ... (first free slot), so no intermediate
    version is ever lost across repeated ``--force`` runs."""
    base = audio_path.with_suffix(audio_path.suffix + ".bak")
    if not base.exists():
        return base
    n = 2
    while True:
        cand = audio_path.with_suffix(audio_path.suffix + f".bak{n}")
        if not cand.exists():
            return cand
        n += 1


def _apply_improvement(
    row: dict, result: dict, imp: dict, model_key: str, improved_records: dict, cfg: SimpleNamespace
) -> None:
    """Single-threaded disk mutation for a winning clip: back up, overwrite,
    refresh duration, update the row, record the win. Both the backup and the
    replacement are written atomically (temp + fsync + os.replace) so an interrupted
    run can never leave a half-written or corrupted clip."""
    audio_path = Path(result["audio_path"])
    # Always back up the current clip before overwriting it: the first win writes the
    # pristine original to .bak, later wins write the prior version to .bak2/.bak3/...
    # so every intermediate audio is kept.
    backup = next_backup_path(audio_path)
    try:
        _stt.atomic_write_bytes(backup, audio_path.read_bytes())
    except OSError as exc:
        logger.error(f"{audio_path.name}: could not write backup, skipping replace: {exc}")
        return

    # Capture the pre-replacement transcript / duration for the audit record before
    # we overwrite them below.
    orig_transcript = row["transcriptions"][model_key].get("text")
    orig_duration = row.get("duration")

    _stt.atomic_write_bytes(audio_path, imp["payload"])

    if cfg.refresh_duration:
        new_dur = flac_duration(audio_path)
        if new_dur is not None:
            row["duration"] = round(new_dur, 4)

    row["transcriptions"][model_key]["text"] = imp["new_transcript"]
    row["transcriptions"][model_key]["cer"] = imp["new_cer"]
    # The winning draw's tail score, or no key at all when it is too short to have one
    # (a regenerated clip can be shorter than the tail window the original needed).
    if imp.get("new_tail") is not None:
        row["transcriptions"][model_key]["cer_tail"] = imp["new_tail"]
    else:
        row["transcriptions"][model_key].pop("cer_tail", None)
    # The bad-CER recheck-count marker described the ORIGINAL reading; the clip now has
    # fresh, good audio + transcript, so drop it (harmless if it was never set).
    row["transcriptions"][model_key].pop("n_stt_check", None)
    # Record the cfg_alpha behind the clip's CURRENT (just-overwritten) audio at the
    # row level, so every clip carries the value that synthesized its shipping audio.
    # A regeneration draw's cfg_alpha is authoritative here (may be None if the knob
    # was omitted); clips that keep their original audio get --original-cfg-alpha in
    # apply_result instead.
    row["cfg_alpha"] = imp.get("cfg_alpha")
    row["improvement"] = {
        "status": "improved",
        "attempts": imp["attempts"],
        "orig_cer": result["orig_cer"],
        "orig_tail": result.get("orig_tail"),
        "best_cer": imp["new_cer"],
        "best_tail": imp.get("new_tail"),
        "seed": imp["seed"],
        "speed": imp["speed"],
        "cfg_alpha": imp.get("cfg_alpha"),
    }

    now = time.time()
    now_iso = datetime.now(timezone.utc).isoformat()
    orig_cer = result["orig_cer"]
    cer_delta = round(orig_cer - imp["new_cer"], 4) if orig_cer is not None else None

    key = row.get(cfg.audio_key) or result["audio_path"]
    # Merge onto any prior win for this clip so the pristine baseline and first-seen
    # time survive re-improvement (--force), while this run's before/after is recorded.
    prev = improved_records.get(key) or {}
    index, term = clip_identity(row, cfg.audio_key)
    improved_records[key] = {
        "audio_filepath": row.get(cfg.audio_key),
        "audio_path": result["audio_path"],
        "index": index,
        "term": term,
        "orig_cer": orig_cer,
        "new_cer": imp["new_cer"],
        "cer_delta": cer_delta,
        "orig_transcript": orig_transcript,
        "new_transcript": imp["new_transcript"],
        "orig_duration": orig_duration,
        "new_duration": row.get("duration"),
        "seed": imp["seed"],
        "speed": imp["speed"],
        "cfg_alpha": imp.get("cfg_alpha"),
        "attempts": imp["attempts"],
        "backup": str(backup),
        # Preserved across reruns so the audit trail is never lost:
        "orig_cer_pristine": prev.get("orig_cer_pristine", orig_cer),
        "n_improvements": prev.get("n_improvements", 0) + 1,
        "first_ts": prev.get("first_ts", now),
        "first_ts_iso": prev.get("first_ts_iso", now_iso),
        "ts": now,
        "ts_iso": now_iso,
    }


def load_existing_improved(path: Path) -> dict[str, dict]:
    """Prior ``.improved.jsonl`` keyed by audio path, so re-runs merge wins."""
    out: dict[str, dict] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = rec.get("audio_filepath") or rec.get("audio_path")
            if key:
                out[key] = rec
    return out


def clip_identity(row: dict, audio_key: str = "audio_filepath") -> tuple:
    """The ``(index, term)`` pair naming a clip in the dataset it came from.

    A stage's own ``generated_dataset.jsonl`` carries both fields and they are used as-is.
    The NeMo manifests stage 99 builds (what the driver actually runs on) do NOT: they keep
    ``group_id`` / ``item_index`` and drop the term, so reading ``row["term"]`` there wrote
    a null into every audit record. Fall back to the clip's own name, which stage 05 built
    as ``{term_index}_{variant_index}_{slug}`` and is the one per-clip identity a manifest
    row still carries. For PARHAF / PARROT the slug is the document + chunk id rather than
    a term, those stages having none, but it is still what names the clip.
    """
    index, term = row.get("index"), row.get("term")
    if index is not None and term is not None:
        return index, term
    parsed = parse_clip_name(Path(row.get(audio_key) or "").name)
    if parsed is None:
        return index, term
    term_index, _variant_index, slug = parsed
    return (index if index is not None else term_index,
            term if term is not None else (slug or None))


def backfill_audit_identity(improved_file: Path, improved_records: dict) -> int:
    """Fill ``index`` / ``term`` on audit records written before ``clip_identity`` existed,
    which stored two nulls for every manifest-driven win. Derived from each record's own
    ``audio_filepath``, so it needs no join with the scored rows. Returns how many records
    were repaired; the file is rewritten only if that is non-zero (the pass would write it
    at its next flush anyway, but a converged run never flushes)."""
    repaired = 0
    for rec in improved_records.values():
        if rec.get("index") is not None and rec.get("term") is not None:
            continue
        index, term = clip_identity(rec)
        if (index, term) == (rec.get("index"), rec.get("term")):
            continue
        rec["index"], rec["term"] = index, term
        repaired += 1
    if repaired:
        _stt.atomic_write_jsonl(improved_file, list(improved_records.values()))
    return repaired


def backfill_cfg_alpha(out_file: Path, improved_records: dict, cfg: SimpleNamespace) -> int:
    """Add the top-level ``cfg_alpha`` to any row in a prior ``.stt.jsonl`` written before
    the field existed, so a plain re-run populates it without needing ``--force``. Returns
    the number of rows backfilled (0 if the file is absent or every row already has it).

    Runs before ``process_file`` on the same output file, so the value it writes is loaded
    back by ``load_existing`` and preserved on skipped (already-resolved) rows. Resolves
    each clip's value the same way the live paths do:

    * a clip whose audio was replaced by a win carries that win's cfg_alpha, taken from
      the ``improvement`` marker (status ``improved``) or, when a later ``exhausted`` marker
      hides it, from the ``.improved.jsonl`` audit (which tracks the audio actually on
      disk); both may legitimately be ``None`` if the knob was omitted at generation;
    * every other clip still has its original audio, so it gets ``--original-cfg-alpha``.
    """
    if not out_file.is_file():
        return 0
    rows: list[dict] = []
    changed = 0
    with out_file.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"backfill: skipping unreadable line {i} in {out_file.name}")
                continue
            rows.append(rec)
            if "cfg_alpha" in rec:
                continue
            imp = rec.get("improvement") or {}
            key = rec.get(cfg.audio_key)
            audit = improved_records.get(key) if key else None
            # A win's cfg_alpha (marker or audit) means the audio on disk is that draw;
            # anything else still has the original audio -> --original-cfg-alpha.
            if imp.get("cfg_alpha") is not None:
                rec["cfg_alpha"] = imp["cfg_alpha"]
            elif audit is not None:
                rec["cfg_alpha"] = audit.get("cfg_alpha")
            else:
                rec["cfg_alpha"] = cfg.original_cfg_alpha
            changed += 1
    if changed:
        _stt.atomic_write_jsonl(out_file, rows)
    return changed


def count_pending(out_file: Path, categories: set[str] | None = None) -> tuple[int, int]:
    """(pending_tts, pending_stt) counts in a written ``.stt.jsonl``, so the driver /
    main can tell when the alternating passes have fully converged (both zero). A
    pending_tts clip still needs a tts pass to draft candidates; a pending_stt clip
    still needs an stt pass to score them.

    When ``categories`` is given, only rows in those categories are counted, so a
    category-restricted run converges on its own subset and ignores pending clips a
    prior full run left in other categories (which this run never touches)."""
    n_tts = n_stt = 0
    if not out_file.is_file():
        return 0, 0
    with out_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if categories and (rec.get("category") not in categories):
                continue
            status = (rec.get("improvement") or {}).get("status")
            if status == "pending_tts":
                n_tts += 1
            elif status == "pending_stt":
                n_stt += 1
    return n_tts, n_stt


def make_hooks(cfg: SimpleNamespace, model_key: str, input_file: Path,
               improved_file: Path, improved_records: dict):
    """Build the (worker, apply_result, needs_work, on_flush) hooks that plug this
    stage's transcribe+improve behavior into 01_compute_stt.py's process_file."""
    # `queued` counts the rows this pass picked up, `done` the ones it actually
    # completed. A split-mode pass that completes nothing has made no progress, which
    # is how the driver knows an stt-only (or tts-only) loop has run dry: waiting for
    # the OTHER mode is not a reason to keep spinning.
    stats = {"improved": 0, "exhausted": 0, "deterministic": 0, "queued": 0, "done": 0,
             # rescored = scored from the transcript already on the row (no STT request);
             # transcribed = a real server round trip. rescored_bad is the subset that now
             # reads bad under the current rules, i.e. what a rescore actually surfaced.
             # unflagged / requeued = pending_tts rows a rescore re-judged: the queue was
             # built under the old rules, so some entries no longer belong in it.
             "rescored": 0, "transcribed": 0, "rescored_bad": 0,
             "unflagged": 0, "requeued": 0,
             # incomplete_drafts = pending_stt clips handed BACK to the tts pass because
             # their candidate set was short of --n-improv (see combined_worker).
             "incomplete_drafts": 0}

    # Third progress bar, the TTS repair backlog: its total is the clips found bad
    # (high CER, sent to regeneration) and it fills as they are successfully improved.
    # The total grows as the STT sweep discovers bad clips, so the gap between n and
    # total shows how far the slow TTS engine is lagging the fast STT pass. Sits below
    # process_file's clip (pos 0) and audio-seconds (pos 1) bars. Only meaningful in
    # 'both' mode (one pass discovers and repairs); the split modes each do only half
    # the work per pass, so the backlog is tracked across passes by the driver instead.
    tts_bar = (
        tqdm(total=0, desc=f"{input_file.name} tts", unit="clip", position=2)
        if cfg.n_improv > 0 and cfg.mode == "both" else None
    )

    def worker(pos, rec):
        return combined_worker(pos, rec, cfg, model_key, input_file)

    def apply_result(pos, result, results):
        stats["done"] += 1
        row = results[pos]
        action = result.get("action")
        imp = result.get("improvement") or {}
        orig_cer = result.get("orig_cer")
        # Carried alongside orig_cer through every state hop, so the tts pass and the
        # candidate scoring know a clip was flagged for its ENDING, not its whole CER.
        orig_tail = result.get("orig_tail")

        # tts pass: candidates were generated for a bad clip. Record them (await the
        # next stt pass to score) and do NOT touch the stored transcription.
        if action == "drafted":
            drafts = imp.get("drafts") or []
            if drafts:
                row["improvement"] = {"status": "pending_stt", "orig_cer": orig_cer,
                                      "orig_tail": orig_tail,
                                      "attempts": imp.get("attempts", 0), "drafts": drafts}
            else:
                # No usable candidate (unreadable clip / every draw a duplicate): give
                # up on this clip so the alternation can still converge.
                if imp.get("all_duplicates"):
                    stats["deterministic"] += 1
                stats["exhausted"] += 1
                row["improvement"] = {"status": "exhausted", "orig_cer": orig_cer,
                                      "orig_tail": orig_tail,
                                      "attempts": imp.get("attempts", 0)}
            return result["entry"]

        # stt pass: the candidate set is short of what --n-improv asks for (see
        # combined_worker), so the clip goes back to the tts pass to have it made in
        # full rather than being judged on a partial set. The leftover sidecars stay
        # where they are: the next tts pass clears them before regenerating.
        if action == "requeue_tts":
            row["improvement"] = {"status": "pending_tts", "orig_cer": orig_cer,
                                  "orig_tail": orig_tail}
            stats["incomplete_drafts"] += 1
            return result["entry"]

        # stt pass: candidates from a prior tts pass were scored. Promote the winner or
        # exhaust, then delete every candidate file (win or lose).
        if action == "picked":
            # Kept-original fallback: unless a candidate wins (below), the clip still
            # carries the audio the original dataset synthesized at --original-cfg-alpha.
            row.setdefault("cfg_alpha", cfg.original_cfg_alpha)
            if imp.get("payload") is not None:
                _apply_improvement(row, result, imp, model_key, improved_records, cfg)
                stats["improved"] += 1
            else:
                stats["exhausted"] += 1
                row["improvement"] = {"status": "exhausted",
                                      "attempts": imp.get("attempts", 0), "orig_cer": orig_cer,
                                      "orig_tail": orig_tail}
            for d in result.get("drafts") or []:
                Path(d["path"]).unlink(missing_ok=True)
            return result["entry"]

        # scored (both / stt original transcription): store the transcription entry.
        entry = result["entry"]
        row.setdefault("transcriptions", {})[model_key] = entry
        # Split the tally by what the clip actually cost: a score re-derived from the
        # stored transcript (no server round trip) versus a real transcription.
        if result.get("rescored"):
            stats["rescored"] += 1
            if is_bad(orig_cer, orig_tail, cfg):
                stats["rescored_bad"] += 1
        else:
            stats["transcribed"] += 1
        # Every clip records the cfg_alpha behind its CURRENT audio. A clip that keeps
        # its original audio (passed with low CER, exhausted, or flagged pending_tts)
        # carries the dataset's original synthesis cfg_alpha (--original-cfg-alpha, 1.3
        # per the release README); a win overwrites this with the draw's cfg_alpha in
        # _apply_improvement. setdefault so a prior improved value is never clobbered.
        row.setdefault("cfg_alpha", cfg.original_cfg_alpha)

        # A regeneration queue built under the OLD scoring rules is stale: this row was
        # flagged pending_tts by a threshold or a normalization that no longer applies,
        # and the orig_cer stored beside that flag is the number a candidate would later
        # have to beat. Nothing else ever revisits it (an stt pass hands pending_tts to
        # the tts pass and moves on), so a rescore settles it here: a row that now reads
        # clean leaves the queue instead of costing a regeneration, and one that stays bad
        # carries the refreshed score with it, so the promote comparison is against
        # today's number instead of a stale inflated one that any draw would beat.
        # Skipped when this pass regenerates inline (both mode), which resolves the row
        # itself just below.
        regenerates_inline = (cfg.mode == "both" and cfg.n_improv > 0
                              and not cfg.rescore_only)
        # --retry-exhausted opens the same door for the clips that gave up: they are the
        # only other rows holding a score computed under the old rules that nothing would
        # revisit, and the whole point of retrying them is to re-judge first (the ones a
        # scoring change makes clean cost no regeneration at all) and redraw second.
        reopened = ("pending_tts", "exhausted") if cfg.retry_exhausted else ("pending_tts",)
        if (cfg.rescore and not regenerates_inline
                and (row.get("improvement") or {}).get("status") in reopened):
            if is_bad(orig_cer, orig_tail, cfg):
                # Back to the queue rather than left resolved, so the next tts pass draws
                # for it again. The attempt count starts over: what matters downstream is
                # the score a new candidate has to beat, and that is refreshed here.
                row["improvement"] = {"status": "pending_tts", "orig_cer": orig_cer,
                                      "orig_tail": orig_tail}
                stats["requeued"] += 1
            else:
                row.pop("improvement", None)
                stats["unflagged"] += 1
            return entry

        if imp.get("payload") is not None:
            # both mode won inline.
            _apply_improvement(row, result, imp, model_key, improved_records, cfg)
            stats["improved"] += 1
            if tts_bar is not None:
                tts_bar.total += 1   # a bad clip, now counted in the backlog...
                tts_bar.update(1)    # ...and successfully repaired, so fill it
        elif is_bad(orig_cer, orig_tail, cfg) and cfg.n_improv > 0 and not cfg.rescore_only:
            if cfg.mode == "both":
                # both mode tried and exhausted inline.
                if imp.get("all_duplicates"):
                    stats["deterministic"] += 1
                stats["exhausted"] += 1
                if tts_bar is not None:
                    tts_bar.total += 1   # bad clip that could not be repaired: total only
                    tts_bar.refresh()
                row["improvement"] = {"status": "exhausted", "orig_cer": orig_cer,
                                      "orig_tail": orig_tail,
                                      "attempts": imp.get("attempts", 0)}
            else:
                # stt mode: flag the bad clip for the next tts pass to regenerate.
                row["improvement"] = {"status": "pending_tts", "orig_cer": orig_cer,
                                      "orig_tail": orig_tail}
        elif (cfg.retry_exhausted and not is_bad(orig_cer, orig_tail, cfg)
              and (row.get("improvement") or {}).get("status") == "exhausted"):
            # Reopened, re-read, and it clears the gates now (a scoring change, or a
            # regenerated clip whose promotion the old rules undersold). Nothing is left
            # to retry, so drop the marker instead of leaving a stale "gave up" on a
            # clip that passes.
            row.pop("improvement", None)
            stats["unflagged"] += 1
        return entry

    def needs_work(out_rec):
        # Category restriction: rows outside the selected categories are never queued,
        # so they pass through untouched (existing transcription / CER preserved). This
        # is what lets a parhaf-only run keep the dictionary/drugs work already computed.
        if cfg.categories and (out_rec.get("category") not in cfg.categories):
            return False

        # --rescore-only: an OFFLINE pass that re-derives stored scores and opens no
        # connection at all, so it can run with both servers down and finishes in
        # minutes. Only a row that already carries a usable transcript qualifies;
        # everything a normal pass would have to transcribe or synthesize (never
        # transcribed, blank, errored, awaiting a tts/stt half) is deliberately left
        # alone for that pass to pick up afterwards from the refreshed score.
        if cfg.rescore_only:
            # A row anywhere in the improvement state machine belongs to the normal
            # passes: pending_stt would drag us into scoring persisted candidates (an
            # STT call), and improved / exhausted are resolved. pending_tts is the one
            # exception: it is a QUEUE ENTRY written under the old rules, its original
            # clip already has a transcript, so refreshing it here costs nothing and
            # keeps the tts pass from regenerating clips that now read clean.
            # --retry-exhausted adds `exhausted` to that exception, for the same reason:
            # its stored score is old-rules too, and re-judging it offline is free.
            reopened = ((None, "pending_tts", "exhausted") if cfg.retry_exhausted
                        else (None, "pending_tts"))
            if (out_rec.get("improvement") or {}).get("status") not in reopened:
                return False
            prior = (out_rec.get("transcriptions") or {}).get(model_key) or {}
            text = prior.get("text")
            if prior.get("error") or not (text and text.strip()):
                return False
            return _stt.resolve_audio(
                out_rec, input_file, cfg.audio_key, cfg.audio_root
            ) is not None

        imp = out_rec.get("improvement") or {}
        status = imp.get("status")

        # tts pass: only clips a prior stt pass flagged bad and awaiting candidates.
        if cfg.mode == "tts":
            return status == "pending_tts" and cfg.n_improv > 0

        # stt (and both) passes below.
        # Candidates a prior tts pass persisted always need this stt pass to score them.
        if cfg.mode == "stt" and status == "pending_stt":
            return True
        # Resolved clips are skipped unless --force, or, for the ones that gave up
        # without a good draw, unless --retry-exhausted reopens them specifically.
        resolved = ("improved",) if cfg.retry_exhausted else ("improved", "exhausted")
        if status in resolved and not cfg.force:
            return False
        # A clip already handed to the tts pass has nothing to do in an stt pass until
        # candidates exist (the tts pass owns it), except under --rescore: the flag that
        # queued it was computed under the old rules, so it is exactly what a rescore
        # exists to re-derive (see the pending_tts branch in apply_result).
        if cfg.mode == "stt" and status == "pending_tts" and not cfg.force and not cfg.rescore:
            return False
        prior = (out_rec.get("transcriptions") or {}).get(model_key)
        if prior is None or prior.get("error"):
            return True  # not transcribed yet -> transcribe (and maybe flag / improve)
        text = prior.get("text")
        if not (text and text.strip()):
            return True  # blank transcript -> redo it, never keep empty
        # --rescore: re-derive every stored score under the CURRENT scoring rules. A
        # change to normalize_for_scoring or to a threshold makes every cer / cer_tail
        # already on disk stale, and nothing else would ever revisit a row that still
        # reads clean. This costs no STT call: the worker reuses the stored transcript
        # (only a row that now reads BAD and has fewer than STT_CHECK_TARGET readings
        # spends a recheck, which is real work either way).
        # Gated on the clip still being on disk: a row whose audio is gone comes back
        # from the worker as an error entry, which would REPLACE a perfectly good stored
        # transcript with nothing. That is fine when the audio is genuinely missing and
        # you are transcribing, but a rescore must never destroy the very transcripts it
        # exists to preserve (e.g. a subset whose clips are being regenerated).
        if cfg.rescore:
            return _stt.resolve_audio(
                out_rec, input_file, cfg.audio_key, cfg.audio_root
            ) is not None
        # Already transcribed: only revisit if it is bad (both mode regenerates inline;
        # stt mode flags it pending_tts). The stored score is READ BACK, not re-derived:
        # it is exactly what this gate would compute (the worker stores best_cer /
        # best_tail_cer), and re-deriving it costs ~3 ms per row (normalize + jiwer,
        # against source and target), i.e. ~22 minutes of silence on a 600k-row corpus
        # before a single clip is transcribed. Scores going stale under a scoring-rule
        # change is what --rescore above is for.
        # The tail is the one thing still derived here, and only when it is MISSING on a
        # clip long enough to have one: a row scored before the tail gate existed must
        # not escape it just because the field was never written.
        cer = prior.get("cer")
        if cer is None:
            cer = best_cer(text, out_rec.get("asr_training_source"),
                           out_rec.get("asr_training_target"))
        tail = prior.get("cer_tail")
        if tail is None and (out_rec.get("duration") or 0.0) > _stt.TAIL_SECONDS:
            tail = best_tail_cer(text, out_rec.get("asr_training_source"),
                                 out_rec.get("asr_training_target"),
                                 out_rec.get("duration") or 0.0)
        return is_bad(cer, tail, cfg)

    def counting_needs_work(out_rec):
        """Tally what this pass picks up, so main can tell a pass that found nothing
        to do from one that did work (see the `queued` / `done` note above)."""
        if needs_work(out_rec):
            stats["queued"] += 1
            return True
        return False

    def on_flush():
        _stt.atomic_write_jsonl(improved_file, list(improved_records.values()))

    def postfix_fn(entry):
        # Live readout on the bar: the just-finished clip's CER plus running tallies.
        cer = entry.get("cer")
        cer_s = f"{cer:.3f}" if cer is not None else "n/a"
        return f"cer={cer_s} improved={stats['improved']} exhausted={stats['exhausted']}"

    return worker, apply_result, counting_needs_work, on_flush, postfix_fn, stats, tts_bar


@click.command(context_settings={"show_default": True})
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path),
              help="Manifest jsonl (or a dir searched recursively for *.jsonl, skipping *.stt.jsonl).")
@click.option("--output", "output_root", required=True, type=click.Path(path_type=Path),
              help="Output root; mirrors input hierarchy, files get .stt.jsonl (same as 01_compute_stt.py).")
@click.option("--audio-root", default=None, type=click.Path(path_type=Path),
              help="Base dir to resolve each row's audio path against. Usually required.")
@click.option("--audio-key", default="audio_filepath", help="JSONL key holding the audio path.")
@click.option("--category", "categories", multiple=True,
              help="Restrict work to rows whose `category` is one of these (repeatable, e.g. "
                   "--category parhaf). Rows in other categories are passed through UNCHANGED "
                   "(existing transcriptions / CER preserved, never re-run), so a focused run "
                   "does not lose the rest of the dataset's work. Convergence (exit 10) is then "
                   "judged over the selected categories only. Default: all categories.")
# STT.
@click.option("--stt-endpoint", required=True, help="OpenAI-compatible /v1/audio/transcriptions URL.")
@click.option("--stt-api-token", default="", envvar="STT_API_TOKEN", help="Bearer token; falls back to $STT_API_TOKEN.")
@click.option("--stt-model", default=None, help="STT model name; also the transcriptions key (keyed 'default' when unset).")
@click.option("--stt-temperature", default=0.0, type=float, help="STT temperature.")
@click.option("--stt-recheck-temperature", default=0.5, type=float, help="Base temperature of the bad-CER recheck ladder (default 0.5, distinct from --stt-temperature). A bad clip is re-transcribed until it has been read 3 times in total, at this temperature then twice it (capped at 1.0), keeping the LOWEST CER: a hallucinated transcript at one temperature (e.g. Whisper's 'Sous-titrage ...') scores a false-bad CER on audio that is actually fine, which would waste a regeneration. A clip still bad after its rechecks records its reading count in `n_stt_check` and is never re-read again; one that came back good drops the marker. Pass a negative value to disable the rechecks.")
@click.option("--stt-language", default="fr", help="STT language hint (empty to omit).")
@click.option("--stt-response-format", default="json", help="STT response_format (json / verbose_json / text).")
@click.option("--stt-param", "stt_params", multiple=True, help="Extra STT form field KEY=VALUE (repeatable).")
@click.option("--stt-timeout", default=120.0, type=float, help="STT per-request timeout (seconds).")
@click.option("--stt-max-retries", default=4, type=int, help="STT retries on transient errors.")
# TTS (should match how the dataset audio was originally synthesized).
@click.option("--tts-url", default="http://127.0.0.1:8003", help="Base URL of the TTS server.")
@click.option("--tts-voice", default=None, help="TTS voice (match the original run for a consistent dataset). Unset keeps the server default.")
@click.option("--tts-speed", default=None, type=float, help="Base TTS speed (unset = server default).")
@click.option("--tts-model", default=None, help="TTS `model` field (e.g. mistralai/Voxtral-4B-TTS-2603 if vLLM rejects a modelless request).")
@click.option("--tts-format", default="flac", help="TTS response format / clip extension (should match the dataset, default flac).")
@click.option("--tts-timeout", default=300.0, type=float, help="TTS per-request timeout (seconds).")
@click.option("--tts-max-retries", default=4, type=int, help="TTS retries on transient errors (connection / 429 / 5xx) before the run aborts.")
@click.option("--max-audio-seconds", default=_tts.frame_cap_seconds(), type=float,
              help="TTS frame cap in seconds: a draw landing within half a second of it stopped for LENGTH (cut off mid-sentence) rather than at the end of the text, since /v1/audio/speech has no finish_reason. Such draws are discarded, never promoted. Default is the served 4096 frames at 12.5 Hz; must match the server's actual VOXTRAL_MAX_TOKENS. 0 disables the check.")
# Improvement policy.
@click.option("--mode", type=click.Choice(["both", "stt", "tts"]), default="both",
              help="both = STT+TTS in one pass (needs both servers up, default). Split modes for when the two servers cannot share VRAM: 'stt' transcribes originals, scores any candidates generated by a prior 'tts' pass, and promotes/exhausts them (only STT server up); 'tts' generates candidates for the clips a prior 'stt' pass flagged bad (only TTS server up). Alternate stt/tts runs until 'stt' reports no work left (exit code 10).")
@click.option("--n-improv", default=5, type=int, help="Max regeneration attempts per bad clip (0 = transcribe only, i.e. behave like 01_compute_stt.py).")
@click.option("--cer-threshold", default=0.08, type=float, help="A clip is 'bad' (regenerated) when its best CER (vs source or target) is >= this; regeneration also aims to get a draw BELOW this (stops early once one is). This is a SWEEP DEPTH, not a quality verdict: the corpus was first taken to 0.15 (~P99.95, 71 clips of 601k), then to 0.10 (487), and 0.08 is the next ~1550 clips. Below it the diffs are increasingly things no seed can fix (French agreement homophony, Whisper's spelling of rare terms, notation the scorer does not fold), so pair a low gate with --min-improvement or the winner is picked on judge noise rather than on the audio.")
@click.option("--tail-cer-threshold", default=0.12, type=float, help="Second gate, on the CER of the clip's LAST ~30s only (stored as `cer_tail`, computed on clips longer than that): a clip is 'bad' when EITHER its whole-clip CER reaches --cer-threshold OR its tail CER reaches this. Catches endings a whole-clip average dilutes (truncated at the TTS output cap, derailed, looped). Looser than --cer-threshold on purpose (the window boundary cuts mid-sentence, which adds noise) but only 1.5x: the window is ~500 characters, the size of a whole PARHAF chunk, not a fifth of one. Measured over 23.6k long clips: tail P95 0.076, P99 0.193, so 0.12 sits around P98. Pass a negative value to disable the tail gate.")
@click.option("--min-improvement", default=0.02, type=float, help="When no draw reaches --cer-threshold, keep the lowest-CER draw only if it beats the original CER by at least this. A draw below the threshold is always kept. Non-zero matters once the gate is low: a clip flagged at 0.09 whose best draw lands at 0.088 was not repaired, it was fitted to the judge, and a clean clip's CER moves that much on Whisper's own spelling choices. Pass 0 to accept any gain.")
@click.option("--start-seed", default=43, type=int, help="First seed; each attempt uses start-seed + attempt. Always sent (Voxtral respects the request seed).")
@click.option("--speed-jitter", default=0.0, type=float, help="Optional: also vary speed by +/- this each attempt (e.g. 0.03). Not needed when the seed works.")
@click.option("--cfg-alpha-start", default=1.3, type=float, help="TTS cfg_alpha sent on the first re-attempt (raised by --cfg-alpha-step each attempt to push the model harder). Pass a negative value to omit cfg_alpha entirely.")
@click.option("--cfg-alpha-step", default=0.1, type=float, help="Amount added to cfg_alpha on each successive re-attempt.")
@click.option("--original-cfg-alpha", default=1.3, type=float, help="cfg_alpha the ORIGINAL dataset audio was synthesized with (1.3, per 99_hf_release/README.md). Written as each clip's `cfg_alpha` when it keeps its original audio (passed / exhausted); a regenerated clip records its winning draw's cfg_alpha instead.")
@click.option("--refresh-duration/--no-refresh-duration", default=True, help="Re-probe and update each row's `duration` when its audio is replaced (keeps stage-99 splits correct).")
@click.option("--force", is_flag=True, help="Re-process clips already marked improved/exhausted in a previous run.")
@click.option("--retry-exhausted", is_flag=True, help="Give the clips marked `exhausted` (flagged bad, no draw good enough) another run: they are treated as if they carried no marker, so a rescore can clear them and an stt pass can send the ones that still read bad back through the tts queue. Unlike --force this leaves the clips that already passed and the ones already improved alone, so a retry costs only the known-weak tail (173 clips after the 0.08 sweep) instead of re-transcribing the corpus. Use it after a scoring-rule change or with a different seed / cfg_alpha ladder.")
@click.option("--rescore", is_flag=True, help="Re-derive cer / cer_tail for EVERY row that already has a stored transcript, under the current scoring rules, and re-apply the gates. Use after changing normalize_for_scoring or a threshold, which makes every score already on disk stale while leaving the transcripts themselves perfectly valid. Costs no STT call (the stored transcript is reused); only a row that now reads bad and has fewer than 3 readings spends a recheck. Also re-judges the regeneration queue: a pending_tts row that now reads clean drops its marker, one that stays bad gets a refreshed orig_cer.")
@click.option("--rescore-only", is_flag=True, help="Offline rescore pass: like --rescore but makes NO request at all (no STT, no TTS), so it runs with both servers down. Only rows that already carry a transcript are touched: scores are refreshed in place, nothing new is flagged for regeneration and nothing is transcribed (rows the OLD rules already queued are re-judged, since that queue is exactly what a rules change invalidates). Run it once after a scoring-rule change, then run the normal passes, which pick up whatever now reads bad. Implies --rescore.")
# Runtime. STT and TTS run in parallel against their own servers: each has its own
# concurrency cap so a clip busy on TTS never blocks a transcription slot, and vice
# versa. The worker pool is sized stt-parallel + tts-parallel so both can be full.
@click.option("--stt-parallel", default=4, type=int, help="Max concurrent STT (transcription) requests in flight. Size to saturate the STT server.")
@click.option("--tts-parallel", default=4, type=int, help="Max concurrent TTS (regeneration) requests in flight, independent of STT. Size to saturate the TTS server.")
@click.option("--shuffle/--no-shuffle", default=False, help="Randomize processing order (not output order) so the ETA reflects a representative clip-length mix early.")
@click.option("--flush-every", default=50, type=int, help="Atomically write outputs every N processed clips (0 = disable this cap).")
@click.option("--flush-interval", default=0.0, type=float, help="Also flush every this many seconds (e.g. 300 = every 5 min); 0 = disabled.")
@click.option("--limit", default=0, type=int, help="Process at most this many lines per file (0 = all), for testing.")
def main(
    input_path, output_root, audio_root, audio_key, categories, stt_endpoint, stt_api_token, stt_model,
    stt_temperature, stt_recheck_temperature, stt_language, stt_response_format, stt_params,
    stt_timeout, stt_max_retries, tts_url, tts_voice, tts_speed, tts_model, tts_format, tts_timeout,
    tts_max_retries, max_audio_seconds, mode, n_improv, cer_threshold, tail_cer_threshold, min_improvement,
    start_seed, speed_jitter,
    cfg_alpha_start, cfg_alpha_step, original_cfg_alpha, refresh_duration, force, retry_exhausted,
    rescore, rescore_only,
    stt_parallel, tts_parallel, shuffle, flush_every, flush_interval, limit,
):
    """Transcribe a dataset and regenerate the worst TTS clips, in one combined pass."""
    logger.remove()
    logger.add(lambda m: tqdm.write(m, end=""), colorize=True,
               format="<level>{level: <8}</level> | {message}", level="INFO")

    cfg = SimpleNamespace(
        audio_root=audio_root, audio_key=audio_key,
        # Empty tuple -> None so "no filter" is a simple falsy check everywhere.
        categories=(set(categories) or None),
        stt_endpoint=stt_endpoint, stt_token=stt_api_token, stt_model=stt_model,
        stt_temperature=stt_temperature,
        # A negative recheck temperature disables the bad-CER triple-check.
        stt_recheck_temperature=(None if stt_recheck_temperature < 0 else stt_recheck_temperature),
        stt_language=stt_language,
        stt_response_format=stt_response_format,
        stt_extra_params=_stt._parse_params(stt_params),
        stt_timeout=stt_timeout, stt_max_retries=stt_max_retries,
        tts_url=tts_url.rstrip("/"), tts_voice=tts_voice, tts_speed=tts_speed,
        tts_model=tts_model, tts_format=tts_format, tts_timeout=tts_timeout,
        max_audio_seconds=max_audio_seconds,
        tts_max_retries=tts_max_retries, mode=mode,
        n_improv=n_improv, cer_threshold=cer_threshold,
        # A negative tail threshold disables the tail gate (inf never compares true).
        tail_cer_threshold=(float("inf") if tail_cer_threshold < 0 else tail_cer_threshold),
        min_improvement=min_improvement,
        start_seed=start_seed, speed_jitter=speed_jitter,
        # cfg_alpha ramps up on each re-attempt; a negative start disables the knob.
        cfg_alpha_start=(None if cfg_alpha_start < 0 else cfg_alpha_start),
        cfg_alpha_step=cfg_alpha_step,
        # cfg_alpha the original dataset audio was made with, written on every clip that
        # keeps its original audio (99_hf_release/README.md documents it as 1.3).
        original_cfg_alpha=original_cfg_alpha,
        refresh_duration=refresh_duration,
        force=force,
        retry_exhausted=retry_exhausted,
        # --rescore-only is --rescore plus "never open a connection", so it implies it.
        rescore=(rescore or rescore_only),
        rescore_only=rescore_only,
        # Independent concurrency caps: STT calls pass through stt_sem, TTS through
        # tts_sem, so the two servers saturate without waiting on each other.
        stt_sem=threading.Semaphore(stt_parallel), tts_sem=threading.Semaphore(tts_parallel),
    )
    # One thread pool feeds both semaphores; size it so stt_parallel STT calls and
    # tts_parallel TTS calls can be in flight at the same time.
    pool_parallel = stt_parallel + tts_parallel
    model_key = stt_model or "default"

    input_root, input_files = _stt.discover_inputs(input_path)
    if not input_files:
        logger.warning(f"no *.jsonl files found under {input_path}")
        return
    if audio_root is None:
        logger.warning("no --audio-root given; audio will only resolve via absolute / CWD paths")
    if shuffle:
        import random
        random.shuffle(input_files)
    logger.info(
        f"{len(input_files)} input file(s); stt_model={model_key} tts_url={cfg.tts_url} "
        f"n_improv={n_improv} (0 = transcribe only); "
        f"parallel: {stt_parallel} STT + {tts_parallel} TTS = {pool_parallel} workers"
    )
    if cfg.categories:
        logger.info(
            f"category filter: only {sorted(cfg.categories)} "
            f"(rows in other categories are passed through unchanged)"
        )
    if cfg.rescore_only:
        if mode == "tts":
            raise click.UsageError("--rescore-only is an stt-side pass; use --mode stt (or both).")
        logger.info(
            "--rescore-only: offline pass, no STT or TTS request will be made. Only rows "
            "with a stored transcript are rescored; nothing is transcribed, flagged or "
            "regenerated. Run the normal passes afterwards."
        )

    # Split modes only: how much improvement work is still outstanding across all
    # files, so the driver knows when the alternating stt/tts passes have converged.
    total_pending_tts = total_pending_stt = 0
    # ... and how much THIS mode picked up and completed, so a driver running one mode
    # on its own (no TTS server up yet, say) can stop when that mode has run dry
    # instead of looping on work only the other mode can clear.
    total_queued = total_done = 0

    for input_file in input_files:
        out_file = _stt.output_path_for(input_file, input_root, output_root)
        improved_file = out_file.with_name(out_file.name.replace(".stt.jsonl", "") + ".improved.jsonl")
        improved_records = load_existing_improved(improved_file)
        # Backfill cfg_alpha onto any rows an earlier (pre-field) run resolved, so a plain
        # re-run populates them without --force. Runs before process_file on the same file,
        # so load_existing carries the values forward onto the skipped rows.
        n_backfilled = backfill_cfg_alpha(out_file, improved_records, cfg)
        if n_backfilled:
            logger.info(f"{out_file.name}: backfilled cfg_alpha on {n_backfilled} pre-existing row(s)")
        n_named = backfill_audit_identity(improved_file, improved_records)
        if n_named:
            logger.info(f"{improved_file.name}: filled index/term on {n_named} audit record(s)")
        worker, apply_result, needs_work, on_flush, postfix_fn, stats, tts_bar = make_hooks(
            cfg, model_key, input_file, improved_file, improved_records
        )
        try:
            _stt.process_file(
                input_file, out_file,
                endpoint=stt_endpoint, token=stt_api_token, model=stt_model,
                temperature=stt_temperature, language=stt_language,
                response_format=stt_response_format, extra_params=cfg.stt_extra_params,
                timeout=stt_timeout, max_retries=stt_max_retries, audio_key=audio_key,
                reference_key="", audio_root=audio_root, wer_threshold=1.0,
                cer_threshold=cer_threshold, flush_every=flush_every,
                flush_interval=flush_interval, limit=limit, n_parallel=pool_parallel,
                shuffle=shuffle,
                worker=worker, apply_result=apply_result, needs_work=needs_work, on_flush=on_flush,
                postfix_fn=postfix_fn,
            )
        except SeedDeterminismError as exc:
            logger.error(str(exc))
            raise SystemExit(2)
        except GenerationError as exc:
            logger.error(str(exc))
            raise SystemExit(3)
        except (_stt.TranscriptionError, TTSRequestError) as exc:
            # A server stayed unreachable through every retry / backoff: abort rather
            # than grind out a run full of skipped clips against a dead endpoint.
            logger.error(f"aborting: {exc}")
            raise SystemExit(4)
        except _stt.EmptyTranscriptError as exc:
            # A dataset clip transcribed blank on every retry: fatal so a silent /
            # broken clip is surfaced, never written as an empty transcript.
            logger.error(f"aborting: clip transcribed empty after every retry: {exc}")
            raise SystemExit(5)
        finally:
            if tts_bar is not None:
                tts_bar.close()

        if stats["deterministic"]:
            logger.warning(
                f"{stats['deterministic']} clip(s) drew the SAME audio on every attempt (seeds "
                f"collided) yet none matched the original: the seed space may be too small. "
                f"Raise --n-improv, change --start-seed, or add --speed-jitter."
            )
        # Split by cost, so a pass that was supposed to be a cheap rescore and instead
        # transcribed thousands of clips (or the reverse) is obvious from the summary.
        if stats["rescored"] or stats["transcribed"]:
            logger.info(
                f"{out_file.name}: {stats['rescored']} rescored from stored transcripts "
                f"(no STT call), {stats['transcribed']} transcribed"
            )
        if cfg.rescore_only and stats["rescored_bad"]:
            logger.info(
                f"{out_file.name}: {stats['rescored_bad']} clip(s) now read bad under the "
                f"current rules; they are NOT flagged here, the next stt pass picks them up"
            )
        # The other half of a rules change: entries the OLD rules queued for the tts pass.
        if stats["unflagged"] or stats["requeued"]:
            logger.info(
                f"{out_file.name}: regeneration queue re-judged, {stats['unflagged']} clip(s) "
                f"left it (they read clean now, no regeneration needed), "
                f"{stats['requeued']} stayed with a refreshed score"
            )
        if stats["incomplete_drafts"]:
            logger.info(
                f"{out_file.name}: {stats['incomplete_drafts']} clip(s) went back to the tts "
                f"pass, their candidate set was short of --n-improv ({cfg.n_improv}); they are "
                f"scored once it has been made in full"
            )
        logger.success(
            f"{out_file.name}: {stats['improved']} improved, {stats['exhausted']} exhausted; "
            f"{len(improved_records)} total wins in {improved_file.name}"
        )
        total_queued += stats["queued"]
        total_done += stats["done"]
        if cfg.mode != "both":
            n_tts, n_stt = count_pending(out_file, cfg.categories)
            total_pending_tts += n_tts
            total_pending_stt += n_stt
            if n_tts or n_stt:
                logger.info(f"{out_file.name}: {n_tts} pending_tts, {n_stt} pending_stt remaining")

    # Split flow: the driver alternates stt/tts passes. Signal convergence from the stt
    # pass (the only one that resolves clips) so the driver loop can stop: exit 10 when
    # no clip anywhere still needs a tts or stt pass.
    if cfg.mode in ("stt", "tts"):
        logger.info(
            f"pending across all files: {total_pending_tts} pending_tts (need a tts pass), "
            f"{total_pending_stt} pending_stt (need an stt pass)"
        )
    if cfg.mode == "stt" and total_pending_tts == 0 and total_pending_stt == 0:
        logger.success("no improvement work left: dataset has converged")
        raise SystemExit(10)
    # This mode alone has run dry: it either queued nothing or completed nothing, so
    # repeating it cannot change anything. Distinct from 10, because clips are still
    # waiting on the OTHER mode. A driver running a single mode stops here; the
    # alternating driver treats it as "the work is on the other side" and runs the
    # other pass (an stt pass legitimately has nothing to do whenever a previous run
    # stopped right after flagging clips pending_tts), and only stops when BOTH
    # passes report it in one round.
    if cfg.mode in ("stt", "tts") and total_done == 0:
        logger.success(
            f"no {cfg.mode} work left ({total_queued} queued, 0 completed): "
            f"{total_pending_tts} clip(s) still need a tts pass, "
            f"{total_pending_stt} an stt pass"
        )
        raise SystemExit(11)


if __name__ == "__main__":
    main()
