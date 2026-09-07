#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["loguru", "soundfile", "tqdm"]
# ///
"""Simple client for the local TTS server (POST /v1/audio/speech).

Both TTS services bind :8003 by design (only one up at a time) and this
client covers whichever answers: `crispasr-tts` (TADA backend),
`voxtral-tts` (vLLM + vllm-omni serving mistralai/Voxtral-4B-TTS-2603),
or a qwen3-tts VoiceDesign deployment (which is what --instruction is
for). Against Voxtral, `input` / `voice` / `response_format` / `speed` are
honoured per request, and so is `--seed`, but only through the nested
`extra_params.voxtral_noise_seed` this client also sends (see build_body);
its sampling knobs really are startup-time env vars. Pass --voice one of
the built-in embedding names in VOXTRAL_VOICES and, if a modelless request
is rejected by vLLM, add --model mistralai/Voxtral-4B-TTS-2603.

Two modes, selected by what --input is:

* literal text: one synthesis, --output is the audio file to write
  (default tts_out.wav).
* path to a .jsonl file: batch mode. Each row's `asr_training_source` is
  synthesized and written into the --output folder as
  {term_index:06d}_{variant_index:04d}_{normalized_term}.{format} (the two
  indices are zero-padded so the files sort in order across hundreds of
  thousands of clips; --format defaults to flac). A clip already present
  is skipped unless --overwrite is given, matched by the index/term stem
  regardless of extension, so switching --format (e.g. wav to flac) does
  not re-synthesize clips you already have and an interrupted run resumes
  by re-running the same command. Rows are synthesized
  --concurrency at a time (default 4): a vLLM/Voxtral server batches those
  concurrent requests server-side, so raising it speeds the run up, while
  the single-context TADA/crispasr backend wants 1. A tqdm bar tracks the
  remaining work; per-row failures are logged and skipped, and the run
  exits non-zero if any row failed.

Every knob (--speed, the talker sampling flags, --instruction, --voice,
--seed, --model) is sent only when set, so the server keeps its startup
defaults otherwise (same per-request contract as the full CrispASR client
this is simplified from, and what keeps TADA-only fields out of Voxtral
requests).

Truncation guard: /v1/audio/speech has no `finish_reason`, so a generation
that stops on the backend's token limit instead of at the end of the text
still answers 200 with a clip cut off mid-sentence. Any clip landing within
half a second of the frame cap (--max-audio-seconds, default 327.68s = the
served 4096 frames at 12.5 Hz) is therefore rejected as "stopped for
length": nothing is written, the row counts as failed, and the run exits
non-zero. Keep --max-audio-seconds in step with the server's actual
max_new_tokens, and shorten the input text rather than retrying, since the
same text truncates every time.

Logs go to stderr and are appended to a local file (see --log-file).

Simplified from CrispASR/tts.py (which absorbed the old voxtral query
client). Written with Claude Code.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf
from loguru import logger
from tqdm import tqdm

DEFAULT_URL = "http://127.0.0.1:8003"
DEFAULT_LOG_FILE = Path(__file__).resolve().parent / "tts_generate.log"

# Built-in voice embeddings shipped with mistralai/Voxtral-4B-TTS-2603, only
# relevant when voxtral-tts is the service on :8003. DELIBERATE COPY of
# VOXTRAL_VOICES in CrispASR/tts.py (separate repo, standalone tool); if the
# served voices change, update both sides. Reference list, not argparse
# choices, since --voice is backend-dependent free text.
VOXTRAL_VOICES = [
    "ar_male",
    "casual_female",
    "casual_male",
    "cheerful_female",
    "de_female",
    "de_male",
    "es_female",
    "es_male",
    "fr_female",
    "fr_male",
    "hi_female",
    "hi_male",
    "it_female",
    "it_male",
    "neutral_female",
    "neutral_male",
    "nl_female",
    "nl_male",
    "pt_female",
    "pt_male",
]

# Voxtral decodes audio as discrete frames at a fixed 12.5 Hz and stops when it hits the
# server's max_new_tokens, WITHOUT saying so: /v1/audio/speech answers a plain 200 with a
# clip that simply ends mid-sentence (there is no `finish_reason` in the audio API, unlike
# the chat API where we guard on it). The only observable of a "length" stop is a duration
# pinned at the frame cap, which is what is_truncated() checks. DELIBERATE COPY of the
# serving constants (frame rate, VOXTRAL_MAX_TOKENS) from the voxtral-tts service in the
# separate CrispASR repo: if the server's cap changes, pass the matching
# --max-audio-seconds here (or update this default).
#
# The served value is 4096 frames (327.68 s), NOT the packaged default of 2048 (163.84 s):
# the cap was raised precisely because 2048 was cutting long clips off mid-sentence, and
# the stage-0 context was raised alongside it (VOXTRAL_MAX_MODEL_LEN=6144) since prompt
# tokens and generated frames share it, so 4096 frames are actually reachable.
VOXTRAL_FRAME_RATE_HZ = 12.5
VOXTRAL_DEFAULT_MAX_NEW_TOKENS = 4096  # -> 327.68 s of audio
TRUNCATION_TOLERANCE_S = 0.5


class TruncatedAudioError(RuntimeError):
    """The backend stopped on its token limit instead of at the end of the text, so the
    clip is cut short. Subclasses RuntimeError so batch mode counts it as a per-row
    failure (nothing written, a resume retries) rather than aborting the run."""


def frame_cap_seconds(max_new_tokens: int = VOXTRAL_DEFAULT_MAX_NEW_TOKENS) -> float:
    """Longest clip the backend can emit under a given frame budget, in seconds."""
    return max_new_tokens / VOXTRAL_FRAME_RATE_HZ


def is_truncated(duration: float | None, cap_seconds: float | None,
                 tolerance: float = TRUNCATION_TOLERANCE_S) -> bool:
    """Whether a clip looks cut off at the backend's frame cap. A generation that stops
    naturally lands wherever the text ends; one that stops on the token limit lands on the
    cap itself, so a duration within `tolerance` of it means "stopped for length".
    A non-positive cap disables the check (and an unparseable duration cannot be judged)."""
    if duration is None or not cap_seconds or cap_seconds <= 0:
        return False
    return duration >= cap_seconds - tolerance


def post_json(url: str, body: dict, timeout: float) -> tuple[int, bytes, float]:
    """POST `body` as JSON, return (status, payload, elapsed seconds)."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read(), time.perf_counter() - t0


def build_body(text: str, args: argparse.Namespace) -> dict:
    """Request body for /v1/audio/speech; unset knobs are omitted so the
    server falls back to its startup defaults."""
    body: dict = {"input": text, "response_format": args.format}
    optional = {
        "model": args.model,
        "instructions": args.instruction,
        "voice": args.voice,
        "seed": args.seed,
        "speed": args.speed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "do_sample": args.do_sample,
    }
    body.update({k: v for k, v in optional.items() if v is not None})
    # The seed is sent a SECOND time, nested, because the flat field above cannot
    # reach Voxtral: it lands in the vLLM sampler's params, and Voxtral's stage 0
    # feeds that sampler a vocab-wide -inf row with one finite entry, so the token
    # is forced whatever the seed says. The audio's only randomness is the
    # flow-matching Gaussian, which the CrispASR voxtral-tts service draws per
    # request from extra_params.voxtral_noise_seed. Backends that read the flat
    # field (TADA, qwen3-tts) keep working, and a server with no reader for the
    # nested key merges and ignores it, so sending both is always safe. Same wire
    # format as CrispASR/tts.py and stage 06's tts_synthesize.
    if args.seed is not None:
        body["extra_params"] = {"voxtral_noise_seed": args.seed}
    return body


def normalize_term(term: str) -> str:
    """Filesystem-safe slug: accents stripped, lowercased, runs of anything
    non-alphanumeric collapsed to a single underscore."""
    ascii_term = unicodedata.normalize("NFKD", term).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_term.lower()).strip("_")
    return (slug or "term")[:80]


def payload_duration(payload: bytes) -> float | None:
    """Duration in seconds of an in-memory audio payload, read from its header (no
    full decode). None when libsndfile cannot parse the container (raw pcm/f32, and
    mp3/aac depending on the local libsndfile build)."""
    try:
        info = sf.info(io.BytesIO(payload))
        return info.frames / info.samplerate if info.samplerate else None
    except Exception:  # noqa: BLE001 - any unparseable container, just report unknown
        return None


def synthesize(base_url: str, text: str, out: Path,
               args: argparse.Namespace) -> tuple[float, float | None]:
    """Synthesize `text` into `out`; return (backend elapsed seconds,
    clip duration seconds or None when unparseable).

    Raises RuntimeError on a non-200 response and lets URLError propagate
    (an unreachable server should abort a batch, not fail every row).
    The audio is written to a .tmp sibling then renamed, so a killed run
    never leaves a truncated file that a resume would skip as done.

    Raises TruncatedAudioError (a RuntimeError) when the clip comes back pinned at the
    backend's frame cap, i.e. generation stopped for length rather than at the end of
    the text. Nothing is written in that case, so the row stays "to generate" and a
    resume retries it; the real fix is shorter input text, since the same text will
    truncate again.
    """
    code, payload, elapsed = post_json(f"{base_url}/v1/audio/speech", build_body(text, args), args.timeout)
    if code != 200:
        raise RuntimeError(f"server returned {code} after {elapsed:.2f}s: "
                           f"{payload[:500].decode('utf-8', 'replace')}")
    dur = payload_duration(payload)
    if is_truncated(dur, args.max_audio_seconds):
        raise TruncatedAudioError(
            f"clip stopped at the backend frame cap ({dur:.2f}s >= "
            f"{args.max_audio_seconds:.2f}s): generation stopped for length, not at the "
            f"end of the text, so the audio is cut short. Not written. Shorten the input "
            f"({len(text)} chars) or raise the server's max_new_tokens and "
            f"--max-audio-seconds together."
        )
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(out)
    return elapsed, dur


def cmd_check(base_url: str, args: argparse.Namespace) -> int:
    """Probe /health, then synthesize a tiny clip on the server's defaults."""
    logger.info("GET {}/health", base_url)
    req = urllib.request.Request(f"{base_url}/health", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5.0) as r:
            code, body = r.status, r.read()
    except urllib.error.HTTPError as e:
        code, body = e.code, e.read()
    except urllib.error.URLError as e:
        logger.error("unreachable: {}", e)
        return 2
    logger.info("  -> {}: {}", code, body[:200].decode("utf-8", "replace"))
    if code != 200:
        return 1

    logger.info("POST {}/v1/audio/speech (smoke test)", base_url)
    smoke = {"input": "Test.", "response_format": "wav"}
    if args.model:
        smoke["model"] = args.model
    code, payload, elapsed = post_json(f"{base_url}/v1/audio/speech", smoke, timeout=args.timeout)
    if code != 200:
        logger.error("  -> {}: {}", code, payload[:500].decode("utf-8", "replace"))
        return 1
    logger.info("  -> {}, {} bytes; model OK, backend took {:.2f}s", code, len(payload), elapsed)
    return 0


def run_single(base_url: str, args: argparse.Namespace) -> int:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        elapsed, dur = synthesize(base_url, args.input, out, args)
    except urllib.error.URLError as e:
        logger.error("unreachable: {}", e)
        return 2
    except RuntimeError as e:
        logger.error("{}", e)
        return 1
    dur_info = f", {dur:.2f}s of audio" if dur is not None else ""
    logger.info("wrote {}{} (backend took {:.2f}s)", out.resolve(), dur_info, elapsed)
    return 0


def run_batch(base_url: str, jsonl_path: Path, args: argparse.Namespace) -> int:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Existing clips are matched by their index/term stem, not their full
    # filename, so a clip already synthesized as .wav counts as done even when
    # this run writes .flac. Scanned once up front (a single directory read) so
    # the resume check stays O(rows) instead of a glob per row. A leftover
    # ".flac.tmp" has stem "....flac" and so never masks a real stem.
    existing_stems = {p.stem for p in out_dir.iterdir() if p.is_file()}

    todo: list[tuple[Path, str]] = []
    planned: set[str] = set()
    skipped_existing = bad_rows = 0
    with jsonl_path.open() as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [k for k in ("term_index", "variant_index", "term", "asr_training_source")
                       if k not in row]
            if missing:
                logger.error("{}:{} missing {}, row skipped", jsonl_path, lineno, missing)
                bad_rows += 1
                continue
            stem = (f"{int(row['term_index']):06d}_{int(row['variant_index']):04d}_"
                    f"{normalize_term(row['term'])}")
            if stem in planned:
                logger.warning("{}:{} duplicate target {}, row skipped", jsonl_path, lineno, stem)
                bad_rows += 1
                continue
            planned.add(stem)
            if stem in existing_stems and not args.overwrite:
                skipped_existing += 1
                continue
            todo.append((out_dir / f"{stem}.{args.format}", row["asr_training_source"]))

    logger.info("{}: {} rows planned, {} already done (skipped), {} to generate",
                jsonl_path, len(planned), skipped_existing, len(todo))

    failed = generated = 0
    total_audio = 0.0
    aborted = False
    # Fan out over a thread pool: each synthesize() call is an independent HTTP
    # request writing its own deterministic filename, so it is thread-safe. Running
    # several in flight lets a vLLM/Voxtral server batch them together server-side
    # (big speedup); for the single-context TADA/crispasr backend pass
    # --concurrency 1. as_completed yields in finish order, so log lines interleave,
    # but each row still lands in its own file.
    pool = ThreadPoolExecutor(max_workers=max(1, args.concurrency))
    futures = {pool.submit(synthesize, base_url, text, out, args): out
               for out, text in todo}
    try:
        for fut in tqdm(as_completed(futures), total=len(futures), unit="clip", desc="TTS"):
            out = futures[fut]
            try:
                elapsed, dur = fut.result()
            except urllib.error.URLError as e:
                logger.error("server unreachable ({}), aborting; re-run to resume", e)
                aborted = True
                break
            except RuntimeError as e:
                logger.error("{}: {}", out.name, e)
                failed += 1
                continue
            generated += 1
            if dur is not None:
                total_audio += dur
            logger.debug("{} ({}{:.2f}s backend)", out.name,
                         f"{dur:.2f}s audio, " if dur is not None else "", elapsed)
    finally:
        # On abort, drop queued-but-unstarted work so an unreachable server fails
        # fast instead of grinding through every remaining row; in-flight requests
        # are left to finish. On success, wait for the pool normally.
        pool.shutdown(wait=not aborted, cancel_futures=aborted)
    if aborted:
        # Everything not written counts as failed so the run exits non-zero; a
        # re-run resumes cheaply via the skip-existing logic above.
        failed += len(todo) - generated - failed

    audio_info = f", {total_audio / 60:.1f} min of audio" if total_audio else ""
    logger.info("done: {} generated{}, {} failed, {} skipped as existing, {} bad rows",
                generated, audio_info, failed, skipped_existing, bad_rows)
    return 1 if (failed or bad_rows) else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default=DEFAULT_URL,
                   help=f"Base URL of the TTS server (default: {DEFAULT_URL}).")
    p.add_argument("--check", action="store_true",
                   help="Probe /health and run a tiny synthesis to confirm the model is loaded.")
    p.add_argument("--input",
                   help="Text to read aloud, or a path to a .jsonl file whose rows "
                        "carry asr_training_source (batch mode).")
    p.add_argument("--output", default="tts_out.flac",
                   help="Output audio file (text mode, default tts_out.flac) or output "
                        "folder (jsonl batch mode).")
    p.add_argument("--overwrite", action="store_true",
                   help="Batch mode: regenerate audio files that already exist "
                        "(default: skip them).")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Batch mode: number of requests in flight at once "
                        "(default 4). A vLLM/Voxtral server batches concurrent "
                        "requests server-side, so >1 is a big speedup there; for "
                        "the single-context TADA/crispasr backend use 1.")
    p.add_argument("--instruction", default=None,
                   help="Voice-direction prose (qwen3-tts VoiceDesign only; TADA and "
                        "Voxtral ignore it). Not sent unless given.")
    p.add_argument("--voice", default=None,
                   help="Voice selector, passed through verbatim (unset keeps the "
                        "server's current voice). TADA: a tada-ref-*.gguf path or "
                        "registry name. Voxtral: a built-in embedding name, e.g. "
                        "fr_male, fr_female, neutral_female (see VOXTRAL_VOICES in "
                        "this file).")
    p.add_argument("--model", default=None,
                   help="OpenAI `model` field (default: not sent). CrispASR does not "
                        "need it; vLLM validates it, so against voxtral-tts pass "
                        "mistralai/Voxtral-4B-TTS-2603 if a modelless request is rejected.")
    p.add_argument("--seed", type=int, default=None,
                   help="Integer seed for reproducible synthesis (default: not sent). "
                        "Sent both flat (TADA, qwen3-tts) and nested as "
                        "extra_params.voxtral_noise_seed, the only route that reaches "
                        "Voxtral's flow-matching noise. NOTE in batch mode every clip "
                        "gets this same seed: identical text would then give identical "
                        "audio, which is the point for reproducibility, not a bug.")
    p.add_argument("--speed", type=float, default=None,
                   help="Speech speed, 0.25 to 4.0 (default: not sent, server default "
                        "applies). Honoured per request by both TADA and Voxtral.")
    p.add_argument("--format", default="flac",
                   choices=("wav", "pcm", "f32", "flac", "mp3", "aac", "opus"),
                   help="Audio response format and file extension (default: flac, "
                        "lossless and about half the size of wav at the same rate). "
                        "CrispASR accepts wav/pcm/f32; Voxtral (vLLM) accepts "
                        "wav/flac/mp3/aac/opus/pcm, so against a wav-only backend "
                        "generate wav and convert afterwards.")
    p.add_argument("--timeout", type=float, default=300.0,
                   help="HTTP timeout in seconds (default: 300).")
    p.add_argument("--max-audio-seconds", type=float, default=frame_cap_seconds(),
                   help="Backend frame cap in seconds: a clip landing within "
                        f"{TRUNCATION_TOLERANCE_S}s of it stopped for length (truncated "
                        "mid-sentence) and is rejected instead of written. Default "
                        f"{frame_cap_seconds():.2f} (the served "
                        f"{VOXTRAL_DEFAULT_MAX_NEW_TOKENS} frames at "
                        f"{VOXTRAL_FRAME_RATE_HZ} Hz); must match the server's actual "
                        "VOXTRAL_MAX_TOKENS. 0 disables the check.")
    p.add_argument("--log-file", default=str(DEFAULT_LOG_FILE),
                   help=f"Loguru log file, appended to (default: {DEFAULT_LOG_FILE}).")

    sampling = p.add_argument_group(
        "talker sampling (TADA only)",
        "Per-request talker knobs; unset = server startup default "
        "(do_sample on, temperature 0.6, top_p 0.9, top_k 0, repetition_penalty 1.1). "
        "Leave unset against Voxtral: its sampling is startup-time only and the "
        "vllm-omni request schema does not define these fields.")
    sampling.add_argument("--temperature", type=float, default=None,
                          help="Talker sampling temperature (0 = greedy).")
    sampling.add_argument("--top-p", type=float, default=None,
                          help="Talker nucleus-sampling cutoff.")
    sampling.add_argument("--top-k", type=int, default=None,
                          help="Talker top-k cutoff (0 = disabled).")
    sampling.add_argument("--repetition-penalty", type=float, default=None,
                          help="Talker repetition penalty (1.0 = none).")
    sampling.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=None,
                          help="Enable talker sampling; --no-do-sample forces greedy decode.")
    args = p.parse_args()

    # stderr through tqdm.write so log lines don't mangle the progress bar.
    logger.remove()
    logger.add(lambda m: tqdm.write(m, end="", file=sys.stderr), level="INFO", colorize=True)
    logger.add(args.log_file, level="DEBUG")

    base_url = args.url.rstrip("/")
    if args.check:
        return cmd_check(base_url, args)
    if not args.input:
        p.error("--input is required unless --check is given")

    in_path = Path(args.input)
    if in_path.suffix == ".jsonl" and in_path.is_file():
        return run_batch(base_url, in_path, args)
    return run_single(base_url, args)


if __name__ == "__main__":
    sys.exit(main())
