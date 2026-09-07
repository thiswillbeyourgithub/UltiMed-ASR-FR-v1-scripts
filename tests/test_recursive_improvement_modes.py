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
"""State-machine test for 01_recursive_improvement.py's --mode split.

Drives the full stt -> tts -> stt alternation over one bad clip with the TTS and STT
network calls faked (distinct bytes per seed; one designated seed transcribes to the
target, so it "improves"), and asserts every state transition the driver relies on:

  * mode stt (round 1): a bad original is transcribed, scored, flagged pending_tts.
  * mode tts:           pending_tts -> pending_stt, N candidate sidecar files written.
  * mode stt (round 2): candidates scored, best promoted, original overwritten,
                        .bak holds the pristine clip, every draft file deleted.
  * convergence:        count_pending reports (0, 0) so --mode stt would exit 10.

Also checks needs_work gates each pass correctly and that clear_drafts / count_pending
behave. Runs a real jiwer CER through the module (no scoring shortcut). Because the
module imports the sibling stage scripts and soundfile at import time, this runs under
uv, not the stdlib `python tests/...` convention:

    uv run tests/test_recursive_improvement_modes.py

This file was written with Claude Code.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
MOD_PATH = HERE.parent / "06_hotfixes" / "01_recursive_improvement.py"

# The clip that "improves": the fake STT maps this candidate's bytes to the exact
# target text (CER 0), every other audio to garbage (high CER).
GOOD_SEED = 44
TARGET = "bonjour le patient prend du paracetamol"
BAD_TRANSCRIPT = "zzzz zzzz zzzz zzzz"


def load_module():
    spec = importlib.util.spec_from_file_location("recimp_under_test", MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def install_fakes(mod):
    """Replace the two network calls with deterministic in-memory fakes."""

    def fake_tts(url, text, *, response_format, seed, voice, speed, model,
                 timeout, max_retries, cfg_alpha):
        # Distinct bytes per seed (so no dedup / seed-determinism trip) and never
        # equal to the original clip bytes (b"ORIG").
        return f"AUD{seed}".encode()

    def fake_transcribe(path, *args, **kwargs):
        data = Path(path).read_bytes()
        transcript = TARGET if data == f"AUD{GOOD_SEED}".encode() else BAD_TRANSCRIPT
        return transcript, 0.1  # (text, stt_seconds)

    mod.tts_synthesize = fake_tts
    mod._stt.transcribe_timed = fake_transcribe


def make_cfg(mode, audio_root, n_improv=3, stt_recheck_temperature=None):
    return SimpleNamespace(
        audio_root=audio_root, audio_key="audio_filepath",
        stt_endpoint="http://stt.invalid/x", stt_token="", stt_model=None,
        stt_temperature=0.0, stt_recheck_temperature=stt_recheck_temperature,
        stt_language="fr", stt_response_format="json",
        stt_extra_params={}, stt_timeout=1.0, stt_max_retries=1,
        tts_url="http://tts.invalid", tts_voice="fr_female", tts_speed=None,
        tts_model=None, tts_format="flac", tts_timeout=1.0, tts_max_retries=1,
        # None = no --category filter, the whole dataset (needs_work reads this).
        categories=None,
        mode=mode, n_improv=n_improv, cer_threshold=0.4, tail_cer_threshold=0.3,
        min_improvement=0.0,
        start_seed=43, speed_jitter=0.0, cfg_alpha_start=1.3, cfg_alpha_step=0.1,
        original_cfg_alpha=1.3,
        refresh_duration=True, force=False, retry_exhausted=False,
        rescore=False, rescore_only=False,
        # Fake audio has no readable duration, so the frame-cap guard never fires here;
        # test_audio_truncation.py is what exercises it.
        max_audio_seconds=163.84,
        stt_sem=threading.Semaphore(1), tts_sem=threading.Semaphore(1),
    )


def run_pass(mod, cfg, row, improved_records, input_file, improved_file):
    """Run one file's worth of work for a single row through the real hooks, the way
    process_file would, and return whether the row needed work this pass."""
    worker, apply_result, needs_work, on_flush, _postfix, _stats, tts_bar = mod.make_hooks(
        cfg, "default", input_file, improved_file, improved_records
    )
    did_work = needs_work(row)
    if did_work:
        result = worker(0, row)
        apply_result(0, result, [row])
    if tts_bar is not None:
        tts_bar.close()
    return did_work


def assert_eq(actual, expected, label):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def main():
    mod = load_module()
    install_fakes(mod)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        clip = root / "clip.flac"
        clip.write_bytes(b"ORIG")
        input_file = root / "manifest.jsonl"  # only used for audio-path fallback
        improved_file = root / "manifest.improved.jsonl"
        improved_records: dict = {}

        row = {
            "audio_filepath": "clip.flac",
            "asr_training_source": TARGET,
            "asr_training_target": TARGET,
            "index": 7,
            "term": "paracetamol",
            "transcriptions": {},
        }

        # --- Pass 1: mode stt, transcribe original, flag it bad ------------------
        cfg = make_cfg("stt", root)
        assert run_pass(mod, cfg, row, improved_records, input_file, improved_file), \
            "pass1: fresh row should need work"
        assert_eq(row["improvement"]["status"], "pending_tts", "pass1 status")
        assert row["improvement"]["orig_cer"] >= cfg.cer_threshold, "pass1 orig_cer should be bad"
        # A bad clip still has its ORIGINAL audio, so it carries --original-cfg-alpha.
        assert_eq(row["cfg_alpha"], cfg.original_cfg_alpha, "pass1 cfg_alpha keeps original")
        assert_eq(count_pending_rows(row), (1, 0), "pass1 pending counts")

        # A second stt pass with no tts in between must NOT re-touch a pending_tts clip.
        assert not run_pass(mod, make_cfg("stt", root), row, improved_records,
                            input_file, improved_file), \
            "pending_tts clip should be idle in an stt pass (tts pass owns it)"

        # --- Pass 2: mode tts, generate candidates -------------------------------
        cfg = make_cfg("tts", root)
        assert run_pass(mod, cfg, row, improved_records, input_file, improved_file), \
            "pass2: pending_tts row should need work in tts mode"
        assert_eq(row["improvement"]["status"], "pending_stt", "pass2 status")
        drafts = row["improvement"]["drafts"]
        assert_eq(len(drafts), 3, "pass2 draft count")
        seeds = sorted(d["seed"] for d in drafts)
        assert_eq(seeds, [43, 44, 45], "pass2 draft seeds")
        for d in drafts:
            assert Path(d["path"]).is_file(), f"draft file missing: {d['path']}"
        assert_eq(count_pending_rows(row), (0, 1), "pass2 pending counts")

        # --- Pass 3: mode stt, score candidates, promote the winner --------------
        cfg = make_cfg("stt", root)
        assert run_pass(mod, cfg, row, improved_records, input_file, improved_file), \
            "pass3: pending_stt row should need work in stt mode"
        assert_eq(row["improvement"]["status"], "improved", "pass3 status")
        # The winning candidate (GOOD_SEED) was atomically promoted over the original.
        assert_eq(clip.read_bytes(), f"AUD{GOOD_SEED}".encode(), "promoted clip bytes")
        # The pristine original is preserved as .bak.
        bak = clip.with_suffix(clip.suffix + ".bak")
        assert bak.is_file() and bak.read_bytes() == b"ORIG", "pristine .bak"
        # Every candidate sidecar is cleaned up after the pick.
        leftover = list(root.glob("clip.flac.draft*"))
        assert not leftover, f"draft files not cleaned: {leftover}"
        # The row's stored transcript is now the improved (target) one.
        assert_eq(row["transcriptions"]["default"]["text"], TARGET, "pass3 stored transcript")
        # The improved clip now carries the WINNING draw's cfg_alpha, not the original.
        # GOOD_SEED=44 is attempt 1 (start_seed 43), so cfg_alpha = 1.3 + 0.1*1 = 1.4.
        won_alpha = round(cfg.cfg_alpha_start + cfg.cfg_alpha_step * (GOOD_SEED - cfg.start_seed), 4)
        assert_eq(row["cfg_alpha"], won_alpha, "pass3 cfg_alpha is the winning draw's")
        assert row["cfg_alpha"] != cfg.original_cfg_alpha, "improved cfg_alpha must differ from original"
        assert_eq(row["improvement"]["cfg_alpha"], won_alpha, "pass3 marker cfg_alpha")
        # An audit record was written for the win.
        assert improved_records, "no improved record recorded"
        rec = next(iter(improved_records.values()))
        assert_eq(rec["seed"], GOOD_SEED, "audit seed")
        assert_eq(rec["cfg_alpha"], won_alpha, "audit cfg_alpha is the winning draw's")
        assert rec["new_cer"] < cfg.cer_threshold, "audit new_cer should clear threshold"

        # --- Convergence: a resolved clip is idle and count_pending is (0, 0) -----
        assert not run_pass(mod, make_cfg("stt", root), row, improved_records,
                            input_file, improved_file), \
            "improved clip should be idle without --force"
        assert_eq(count_pending_rows(row), (0, 0), "converged pending counts")

        # count_pending over a written file (what main() uses for exit 10).
        conv = root / "conv.stt.jsonl"
        conv.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        assert_eq(mod.count_pending(conv), (0, 0), "count_pending on converged file")

    print("test_recursive_improvement_modes: OK")


def count_pending_rows(*rows):
    """(pending_tts, pending_stt) over in-memory rows, mirroring mod.count_pending."""
    n_tts = n_stt = 0
    for r in rows:
        status = (r.get("improvement") or {}).get("status")
        if status == "pending_tts":
            n_tts += 1
        elif status == "pending_stt":
            n_stt += 1
    return n_tts, n_stt


def test_backfill(mod):
    """backfill_cfg_alpha promotes a win's cfg_alpha to the row level and defaults
    every clip that kept its original audio to --original-cfg-alpha, without --force."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        out_file = root / "d.stt.jsonl"
        rows = [
            # never improved, no cfg_alpha -> original 1.3
            {"audio_filepath": "a.flac", "asr_training_target": "x"},
            # improved, the marker carries cfg_alpha -> promoted to the row top level
            {"audio_filepath": "b.flac", "improvement": {"status": "improved", "cfg_alpha": 1.6}},
            # already has a top-level cfg_alpha -> left untouched (not counted)
            {"audio_filepath": "c.flac", "cfg_alpha": 1.9},
            # improved-then-exhausted: the exhausted marker hides cfg_alpha, but the audit
            # (.improved.jsonl) still holds the win behind the on-disk audio -> from audit
            {"audio_filepath": "d.flac", "improvement": {"status": "exhausted"}},
        ]
        out_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        improved_records = {"d.flac": {"audio_filepath": "d.flac", "cfg_alpha": 1.7}}
        cfg = SimpleNamespace(audio_key="audio_filepath", original_cfg_alpha=1.3)

        n = mod.backfill_cfg_alpha(out_file, improved_records, cfg)
        assert_eq(n, 3, "backfill count (c already had cfg_alpha)")
        by = {json.loads(line)["audio_filepath"]: json.loads(line)
              for line in out_file.read_text().splitlines() if line.strip()}
        assert_eq(by["a.flac"]["cfg_alpha"], 1.3, "a -> original cfg_alpha")
        assert_eq(by["b.flac"]["cfg_alpha"], 1.6, "b -> marker win cfg_alpha")
        assert_eq(by["c.flac"]["cfg_alpha"], 1.9, "c untouched")
        assert_eq(by["d.flac"]["cfg_alpha"], 1.7, "d -> audit win cfg_alpha")
        # Idempotent: a second pass sees every row populated and rewrites nothing.
        assert_eq(mod.backfill_cfg_alpha(out_file, improved_records, cfg), 0, "backfill idempotent")
        # A missing output file is a no-op (fresh run, nothing to backfill).
        assert_eq(mod.backfill_cfg_alpha(root / "nope.stt.jsonl", improved_records, cfg), 0,
                  "backfill on missing file")
    print("test_backfill: OK")


def test_recheck_temperature(mod):
    """The recheck ladder walks base, 2 * base, ... capped at Whisper's 1.0 ceiling, so
    each of the triple-check's readings samples the audio differently."""
    assert_eq(mod.STT_CHECK_TARGET, 3, "triple-check target")
    assert_eq(mod.recheck_temperature(0.5, 1), 0.5, "first recheck = base")
    assert_eq(mod.recheck_temperature(0.5, 2), 1.0, "second recheck = 2 * base")
    assert_eq(mod.recheck_temperature(0.2, 2), 0.4, "ladder scales with the base")
    assert_eq(mod.recheck_temperature(0.7, 2), 1.0, "capped at 1.0")
    print("test_recheck_temperature: OK")


def test_tail_gate(mod):
    """A long clip whose ENDING is wrong is flagged even when its whole-clip CER passes:
    the tail score is stored as cer_tail, gates on --tail-cer-threshold, and is skipped
    entirely on clips at or under 30s."""
    long_ref = ("bonjour le patient prend du paracetamol tous les matins avant le repas. " * 60).strip()
    derailed = long_ref[:-600] + " zzzz" * 120   # last ~10% of the reading is garbage

    # The metric itself: no tail on short clips, a clean reading scores ~0.
    assert_eq(mod._stt.tail_cer(long_ref, long_ref, 30.0), None, "no tail at exactly 30s")
    assert_eq(mod._stt.tail_cer(long_ref, long_ref, 12.0), None, "no tail on a short clip")
    clean = mod._stt.tail_cer(long_ref, long_ref, 120.0)
    assert clean is not None and clean < 1e-9, f"identical text should score 0, got {clean}"
    derailed_tail = mod._stt.tail_cer(long_ref, derailed, 120.0)
    assert derailed_tail > 0.3, f"derailed ending should score high, got {derailed_tail}"

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "clip.flac").write_bytes(b"ORIG")
        input_file = root / "manifest.jsonl"
        cfg = make_cfg("stt", root, n_improv=5, stt_recheck_temperature=-1.0)
        cfg.stt_recheck_temperature = None   # isolate the gate from the recheck ladder

        def row(duration):
            return {"audio_filepath": "clip.flac", "asr_training_source": long_ref,
                    "asr_training_target": long_ref, "duration": duration, "index": 1,
                    "term": "paracetamol", "transcriptions": {}}

        mod._stt.transcribe_timed = lambda path, *a, **k: (derailed, 0.1)
        out = mod.combined_worker(0, row(120.0), cfg, "default", input_file)
        whole = out["orig_cer"]
        assert whole < cfg.cer_threshold, f"whole-clip CER should PASS the gate, got {whole}"
        assert out["entry"]["cer_tail"] > cfg.tail_cer_threshold, "tail CER should be stored and bad"
        assert_eq(out["orig_tail"], out["entry"]["cer_tail"], "tail carried on the result")
        assert_eq(out["entry"]["n_stt_check"], 1, "tail-only failure still counts as bad")

        # Same reading on a short clip: no tail key, and the clip is not flagged.
        out = mod.combined_worker(0, row(12.0), cfg, "default", input_file)
        assert "cer_tail" not in out["entry"], "no tail key under 30s"
        assert "n_stt_check" not in out["entry"], "short clip passes on its whole-clip CER"

        # A clean reading of the same long clip is not flagged at all.
        mod._stt.transcribe_timed = lambda path, *a, **k: (long_ref, 0.1)
        out = mod.combined_worker(0, row(120.0), cfg, "default", input_file)
        assert out["entry"]["cer_tail"] < cfg.tail_cer_threshold, "clean tail stored and good"
        assert "n_stt_check" not in out["entry"], "clean long clip is not flagged"

        # badness(): a reading with a clean tail beats one whose only defect is the tail.
        assert mod.badness(0.02, 0.9, cfg) > mod.badness(0.05, 0.01, cfg), \
            "a derailed tail must lose to a slightly worse but clean reading"
    print("test_tail_gate: OK")


def test_needs_work_rescans_tails_and_counts(mod):
    """A clip scored BEFORE the tail gate existed carries no cer_tail, so re-queueing it
    has to recompute the tail from the stored transcript: gating on the whole-clip CER
    alone would leave every such derailed clip invisible forever. Also locks the
    queued/done tally the driver's single-mode loop stops on."""
    long_ref = ("bonjour le patient prend du paracetamol tous les matins avant le repas. " * 60).strip()
    derailed = long_ref[:-600] + " zzzz" * 120

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        input_file = root / "manifest.jsonl"
        improved_file = root / "manifest.improved.jsonl"
        # --rescore only re-queues a row whose clip is still on disk, so the fixture
        # needs one. Existence is all resolve_audio checks; nothing decodes it here.
        (root / "clip.flac").write_bytes(b"not really audio")
        cfg = make_cfg("stt", root, n_improv=5)

        def scored_row(transcript, duration):
            # No cer_tail key: exactly the shape a pre-tail-gate run left behind.
            cer = mod.best_cer(transcript, long_ref, long_ref)
            assert cer < cfg.cer_threshold, f"whole-clip CER must pass the gate, got {cer}"
            return {"audio_filepath": "clip.flac", "asr_training_source": long_ref,
                    "asr_training_target": long_ref, "duration": duration, "index": 1,
                    "term": "paracetamol",
                    "transcriptions": {"default": {"text": transcript, "cer": cer}}}

        _w, _a, needs_work, _f, _p, stats, tts_bar = mod.make_hooks(
            cfg, "default", input_file, improved_file, {}
        )
        if tts_bar is not None:
            tts_bar.close()
        assert needs_work(scored_row(derailed, 120.0)), \
            "a stored transcript that derails at the end must be re-queued"
        assert not needs_work(scored_row(derailed, 12.0)), \
            "under 30s there is no tail, so the same reading passes"
        assert not needs_work(scored_row(long_ref, 120.0)), \
            "a clean long clip stays done"
        assert_eq(stats["queued"], 1, "only the derailed long clip was queued")
        assert_eq(stats["done"], 0, "nothing was completed, no worker ran")

        # --rescore re-queues even the clip that reads clean, which is the only way a
        # score computed under older normalization rules ever gets refreshed.
        cfg.rescore = True
        _w2, _a2, needs_work2, _f2, _p2, stats2, tts_bar2 = mod.make_hooks(
            cfg, "default", input_file, improved_file, {}
        )
        if tts_bar2 is not None:
            tts_bar2.close()
        assert needs_work2(scored_row(long_ref, 120.0)), \
            "--rescore must revisit a clip whose stored score is stale but clean"
        assert_eq(stats2["queued"], 1, "the clean clip was queued for rescoring")

        # --rescore-only is the OFFLINE variant: same refresh, but a row that would
        # cost a server round trip must be left alone, which is what makes the pass
        # safe to run with both servers down.
        cfg.rescore_only = True
        _w3, _a3, needs_work3, _f3, _p3, stats3, tts_bar3 = mod.make_hooks(
            cfg, "default", input_file, improved_file, {}
        )
        if tts_bar3 is not None:
            tts_bar3.close()
        assert needs_work3(scored_row(long_ref, 120.0)), \
            "--rescore-only still refreshes a row that has a transcript"
        never = scored_row(long_ref, 120.0)
        never["transcriptions"] = {}
        assert not needs_work3(never), \
            "--rescore-only must NOT queue a row that has never been transcribed"
        blank = scored_row(long_ref, 120.0)
        blank["transcriptions"]["default"]["text"] = "   "
        assert not needs_work3(blank), "a blank transcript would need a real STT call"
        errored = scored_row(long_ref, 120.0)
        errored["transcriptions"]["default"] = {"error": "audio not found"}
        assert not needs_work3(errored), "an errored row would need a real STT call"
        pending = scored_row(long_ref, 120.0)
        pending["improvement"] = {"status": "pending_stt", "drafts": [{"path": "x", "seed": 1}]}
        assert not needs_work3(pending), \
            "candidates waiting to be scored need the STT server, not an offline pass"
        for st in ("improved", "exhausted"):
            row = scored_row(long_ref, 120.0)
            row["improvement"] = {"status": st}
            assert not needs_work3(row), \
                f"a row in state {st} is resolved, the offline rescore leaves it alone"
        # pending_tts is the exception: it is a queue entry the OLD rules wrote, and its
        # original clip already has a transcript, so re-judging it costs no request. See
        # test_rescore_rejudges_the_tts_queue for what the pass then does with it.
        queued_row = scored_row(long_ref, 120.0)
        queued_row["improvement"] = {"status": "pending_tts", "orig_cer": 0.9}
        assert needs_work3(queued_row), \
            "a rescore re-judges the regeneration queue the old rules built"
        assert_eq(stats3["queued"], 2, "the scored row and the queued one")
    print("test_needs_work_rescans_tails_and_counts: OK")


def test_recheck(mod):
    """A bad CER is re-transcribed until the clip has had 3 readings, at rising
    temperatures, and the lowest reading wins: a rescued clip drops n_stt_check (no
    regeneration), a confirmed-bad clip records all 3 readings, a clip a previous run
    only read twice gets its third, and a fully checked one is never re-transcribed."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "clip.flac").write_bytes(b"ORIG")
        input_file = root / "manifest.jsonl"

        def base_row():
            return {"audio_filepath": "clip.flac", "asr_training_source": TARGET,
                    "asr_training_target": TARGET, "index": 1, "term": "paracetamol",
                    "transcriptions": {}}

        calls = [0]

        def temp_fake(by_temp):
            """Fake STT keyed on the request temperature (positional arg 3), counting calls."""
            def fake(path, *args, **kwargs):
                calls[0] += 1
                temp = args[3] if len(args) > 3 else kwargs.get("temperature")
                return by_temp.get(temp, BAD_TRANSCRIPT), 0.1
            return fake

        # --- A: the first recheck rescues a false-bad clip (bad at 0.0, good at 0.5) --
        cfg = make_cfg("both", root, n_improv=0, stt_recheck_temperature=0.5)
        mod._stt.transcribe_timed = temp_fake({0.0: BAD_TRANSCRIPT, 0.5: TARGET})
        calls[0] = 0
        out = mod.combined_worker(0, base_row(), cfg, "default", input_file)
        assert_eq(calls[0], 2, "A: transcribed twice (first + one recheck), then stopped")
        assert_eq(out["entry"]["text"], TARGET, "A: kept the rescued transcript")
        assert out["orig_cer"] < cfg.cer_threshold, "A: rescued CER should clear threshold"
        assert "n_stt_check" not in out["entry"], "A: rescued clip drops n_stt_check"
        assert out["improvement"] is None, "A: rescued clip is not regenerated"

        # --- A2: the SECOND recheck (top of the ladder) still rescues the clip -----
        mod._stt.transcribe_timed = temp_fake({0.0: BAD_TRANSCRIPT, 0.5: BAD_TRANSCRIPT,
                                               1.0: TARGET})
        calls[0] = 0
        out = mod.combined_worker(0, base_row(), cfg, "default", input_file)
        assert_eq(calls[0], 3, "A2: all three readings used")
        assert_eq(out["entry"]["text"], TARGET, "A2: kept the rescued transcript")
        assert "n_stt_check" not in out["entry"], "A2: rescued clip drops n_stt_check"
        assert out["improvement"] is None, "A2: rescued clip is not regenerated"

        # --- B: the rechecks confirm a genuinely bad clip (bad at every temperature) --
        mod._stt.transcribe_timed = temp_fake({0.0: BAD_TRANSCRIPT, 0.5: BAD_TRANSCRIPT,
                                               1.0: BAD_TRANSCRIPT})
        calls[0] = 0
        out = mod.combined_worker(0, base_row(), cfg, "default", input_file)
        assert_eq(calls[0], 3, "B: transcribed three times (first + two rechecks)")
        assert out["orig_cer"] >= cfg.cer_threshold, "B: confirmed CER stays bad"
        assert_eq(out["entry"]["n_stt_check"], 3, "B: confirmed-bad clip records 3 readings")

        # --- C: a clip an older run only read twice gets its third reading ---------
        calls[0] = 0
        row = base_row()
        row["transcriptions"]["default"] = {"text": BAD_TRANSCRIPT, "cer": 0.9, "n_stt_check": 2}
        out = mod.combined_worker(0, row, cfg, "default", input_file)
        assert_eq(calls[0], 1, "C: reused stored transcript, topped up with one recheck")
        assert out["orig_cer"] >= cfg.cer_threshold, "C: still bad"
        assert_eq(out["entry"]["n_stt_check"], 3, "C: reading count reaches the target")

        # --- C2: a fully checked clip is NOT re-transcribed again ------------------
        calls[0] = 0
        row = base_row()
        row["transcriptions"]["default"] = {"text": BAD_TRANSCRIPT, "cer": 0.9, "n_stt_check": 3}
        out = mod.combined_worker(0, row, cfg, "default", input_file)
        assert_eq(calls[0], 0, "C2: reused stored transcript, no recheck")
        assert_eq(out["entry"]["n_stt_check"], 3, "C2: reading count preserved")

        # --- D: recheck disabled (negative temp -> None) never re-transcribes ------
        cfg_off = make_cfg("both", root, n_improv=0, stt_recheck_temperature=None)
        mod._stt.transcribe_timed = temp_fake({0.0: BAD_TRANSCRIPT})
        calls[0] = 0
        out = mod.combined_worker(0, base_row(), cfg_off, "default", input_file)
        assert_eq(calls[0], 1, "D: only the first transcription, rechecks disabled")
        assert_eq(out["entry"]["n_stt_check"], 1, "D: records the single reading it had")
    print("test_recheck: OK")


def test_rescore_keeps_the_rows_data(mod):
    """A rescore rebuilds the transcription entry from scratch, so anything the previous
    pass measured has to be carried across explicitly: the timing telemetry
    (stt_seconds / stt_rtf) describes the very transcript being reused, and dropping it
    would quietly gut the run statistics of every row a rescore touches. Also locks the
    RESCORED marker, which is how a no-network pass is told apart from a real one."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        input_file = root / "manifest.jsonl"
        (root / "clip.flac").write_bytes(b"not really audio")
        cfg = make_cfg("stt", root, n_improv=0)
        cfg.rescore = cfg.rescore_only = True

        def stored_row():
            return {"audio_filepath": "clip.flac", "asr_training_source": TARGET,
                    "asr_training_target": TARGET, "index": 1, "term": "paracetamol",
                    "duration": 9.75,
                    "transcriptions": {"default": {"text": TARGET, "wer": None, "cer": 0.0,
                                                   "stt_seconds": 4.046, "stt_rtf": 0.415}}}

        def boom(*a, **k):
            raise AssertionError("--rescore-only must never call the STT server")

        mod._stt.transcribe_timed = boom
        out = mod.combined_worker(0, stored_row(), cfg, "default", input_file)
        entry = out["entry"]
        assert out["rescored"] is True, "no STT call was made, so the row is a rescore"
        assert_eq(entry["text"], TARGET, "the stored transcript is reused verbatim")
        assert_eq(entry["stt_seconds"], 4.046, "timing of the reused transcript survives")
        assert_eq(entry["stt_rtf"], 0.415, "and so does its real-time factor")

        # A row with no transcript is skipped rather than transcribed, which is what
        # makes the pass safe to run with the servers down.
        empty = stored_row()
        empty["transcriptions"] = {}
        out = mod.combined_worker(0, empty, cfg, "default", input_file)
        assert_eq(out["action"], None, "nothing to rescore -> no action, no call")

        # A real transcription is NOT tagged as a rescore.
        cfg.rescore = cfg.rescore_only = False
        mod._stt.transcribe_timed = lambda *a, **k: (TARGET, 0.25)
        out = mod.combined_worker(0, empty, cfg, "default", input_file)
        assert out["rescored"] is False, "a transcribed clip must not claim to be rescored"
        assert_eq(out["entry"]["stt_seconds"], 0.25, "fresh timing is recorded")
    print("test_rescore_keeps_the_rows_data: OK")


def test_needs_work_reads_the_stored_score(mod):
    """needs_work runs on EVERY row of the manifest before the pass transcribes anything,
    so what it costs per row is what the run costs in silence up front: re-deriving the
    CER there is ~3 ms a row (normalize + jiwer, against source and target), i.e. ~22
    minutes on a 600k-row corpus. The stored cer / cer_tail are exactly what the gate
    would recompute, so a plain pass reads them back. Locked with a poisoned best_cer:
    if this fails, the scan phase went slow again."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "clip.flac").write_bytes(b"not really audio")
        _w, _a, needs_work, _f, _p, _s, bar = mod.make_hooks(
            make_cfg("stt", root, n_improv=5), "default",
            root / "manifest.jsonl", root / "improved.jsonl", {})
        if bar is not None:
            bar.close()

        def poisoned(*a, **k):
            raise AssertionError("needs_work must not re-derive a score already on the row")

        def row(cer, tail, duration=120.0):
            entry = {"text": TARGET, "cer": cer}
            if tail is not None:
                entry["cer_tail"] = tail
            return {"audio_filepath": "clip.flac", "asr_training_source": TARGET,
                    "asr_training_target": TARGET, "duration": duration,
                    "transcriptions": {"default": entry}}

        keep = (mod.best_cer, mod.best_tail_cer)
        mod.best_cer = mod.best_tail_cer = poisoned
        try:
            assert not needs_work(row(0.01, 0.01)), "a clean stored score is trusted"
            assert needs_work(row(0.9, 0.01)), "a bad stored CER re-queues"
            assert needs_work(row(0.01, 0.9)), "a bad stored tail re-queues"
            # A clip too short to have a tail has nothing missing, so nothing to derive.
            assert not needs_work(row(0.01, None, duration=5.0)), \
                "a short clip has no tail to look for"
        finally:
            mod.best_cer, mod.best_tail_cer = keep
    print("test_needs_work_reads_the_stored_score: OK")


def test_rescore_rejudges_the_tts_queue(mod):
    """A scoring-rule change invalidates the QUEUE the old rules built, not only the
    scores on disk. A pending_tts row carries the orig_cer a candidate will later have to
    beat, and a plain stt pass hands those rows straight to the tts pass without looking
    at them, so nothing else would ever revisit them: the tts pass would regenerate clips
    that now read clean, and promote candidates that merely beat a stale, inflated
    number. A rescore therefore re-judges the queue too."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        input_file = root / "manifest.jsonl"
        improved_file = root / "improved.jsonl"
        (root / "clean.flac").write_bytes(b"ORIG")
        (root / "bad.flac").write_bytes(b"ORIG")

        def queued(name, transcript):
            """A row the OLD rules flagged: orig_cer 0.9 is the number to beat."""
            return {"audio_filepath": name, "asr_training_source": TARGET,
                    "asr_training_target": TARGET, "duration": 9.75,
                    "transcriptions": {"default": {"text": transcript, "cer": 0.9,
                                                   "stt_seconds": 4.0, "stt_rtf": 0.4,
                                                   "n_stt_check": 3}},
                    "improvement": {"status": "pending_tts", "orig_cer": 0.9,
                                    "orig_tail": None, "attempts": 0}}

        def boom(*a, **k):
            raise AssertionError("--rescore-only must never call the STT server")

        mod._stt.transcribe_timed = boom
        cfg = make_cfg("stt", root, n_improv=3)
        cfg.rescore = cfg.rescore_only = True
        worker, apply_result, needs_work, _flush, _pf, stats, tts_bar = mod.make_hooks(
            cfg, "default", input_file, improved_file, {}
        )
        rows = [queued("clean.flac", TARGET), queued("bad.flac", BAD_TRANSCRIPT)]
        for pos, row in enumerate(rows):
            assert needs_work(row), f"{row['audio_filepath']}: a rescore must re-judge it"
            apply_result(pos, worker(pos, row), rows)
        if tts_bar is not None:
            tts_bar.close()

        assert "improvement" not in rows[0], "a clip that now reads clean leaves the queue"
        assert_eq(rows[0]["transcriptions"]["default"]["cer"], 0.0,
                  "and keeps its refreshed score")
        assert_eq(rows[1]["improvement"]["status"], "pending_tts", "a still-bad clip stays")
        assert_eq(rows[1]["improvement"]["orig_cer"],
                  rows[1]["transcriptions"]["default"]["cer"],
                  "the number to beat is the score derived under the CURRENT rules")
        assert rows[1]["improvement"]["orig_cer"] != 0.9, "not the stale flagged value"
        assert_eq(stats["unflagged"], 1, "one clip left the queue")
        assert_eq(stats["requeued"], 1, "one stayed in it with a fresh score")

        # Without --rescore the ownership rule is unchanged: the stt pass leaves a queued
        # row alone and the tts pass claims it.
        plain = queued("bad.flac", BAD_TRANSCRIPT)
        _w, _a, needs_plain, _f, _p, _s, bar = mod.make_hooks(
            make_cfg("stt", root, n_improv=3), "default", input_file, improved_file, {})
        if bar is not None:
            bar.close()
        assert not needs_plain(plain), "a plain stt pass still leaves the queue to tts"
        _w, _a, needs_tts, _f, _p, _s, bar = mod.make_hooks(
            make_cfg("tts", root, n_improv=3), "default", input_file, improved_file, {})
        if bar is not None:
            bar.close()
        assert needs_tts(plain), "the tts pass still owns a queued row"
    print("test_rescore_rejudges_the_tts_queue: OK")


def test_retry_exhausted(mod):
    """`exhausted` means "flagged bad, no draw was good enough", and it is final: nothing
    revisits the clip, so its score stays frozen under the rules of the run that gave up.
    --retry-exhausted reopens exactly those rows (not the ones that passed, not the ones
    already improved, which is what --force would cost) so a rescore can clear the ones a
    scoring change makes clean and send the rest back through the tts queue."""
    install_fakes(mod)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        input_file = root / "manifest.jsonl"
        improved_file = root / "improved.jsonl"

        def exhausted(name, audio, stored):
            """A clip a prior run flagged and then gave up on, with the transcript that
            run stored (what an offline rescore re-judges). `stored=None` leaves the row
            without one, which is what makes a pass actually re-read the audio."""
            (root / name).write_bytes(audio)
            return {"audio_filepath": name, "asr_training_source": TARGET,
                    "asr_training_target": TARGET, "index": 1, "term": "paracetamol",
                    "transcriptions": ({"default": {"text": stored, "cer": 0.9}}
                                       if stored is not None else {}),
                    "improvement": {"status": "exhausted", "orig_cer": 0.9,
                                    "orig_tail": None, "attempts": 5}}

        # --- without the flag, an exhausted clip is idle in every pass ---------------
        for mode in ("stt", "tts"):
            row = exhausted("idle.flac", b"ORIG", BAD_TRANSCRIPT)
            assert not run_pass(mod, make_cfg(mode, root), row, {}, input_file, improved_file), \
                f"exhausted clip should be idle in a plain {mode} pass"
            assert_eq(row["improvement"]["status"], "exhausted", f"plain {mode} leaves the marker")

        # --- offline rescore + retry: still bad -> back to the queue -----------------
        cfg = make_cfg("stt", root)
        cfg.rescore = cfg.rescore_only = cfg.retry_exhausted = True
        row = exhausted("bad.flac", b"ORIG", BAD_TRANSCRIPT)
        assert run_pass(mod, cfg, row, {}, input_file, improved_file), \
            "rescore + retry should reopen an exhausted clip"
        assert_eq(row["improvement"]["status"], "pending_tts", "still bad -> requeued")
        assert row["improvement"]["orig_cer"] >= cfg.cer_threshold, "requeued with a fresh score"
        assert "attempts" not in row["improvement"], "the requeued marker starts over"

        # --- same pass, a clip the new rules now read as clean -----------------------
        row = exhausted("clean.flac", b"ORIG", TARGET)
        assert run_pass(mod, cfg, row, {}, input_file, improved_file), \
            "rescore + retry should reopen this one too"
        assert "improvement" not in row, "a now-clean clip should lose its marker entirely"

        # --- an improved clip stays resolved: only --force reopens those -------------
        row = exhausted("won.flac", b"ORIG", BAD_TRANSCRIPT)
        row["improvement"] = {"status": "improved", "orig_cer": 0.9, "new_cer": 0.01}
        assert not run_pass(mod, cfg, row, {}, input_file, improved_file), \
            "--retry-exhausted must not reopen an improved clip"

        # --- online retry (no rescore): a clip with no stored transcript is re-read ---
        cfg = make_cfg("stt", root)
        cfg.retry_exhausted = True
        # Audio the fake STT transcribes exactly, i.e. a clip that now passes.
        row = exhausted("reread.flac", f"AUD{GOOD_SEED}".encode(), None)
        assert run_pass(mod, cfg, row, {}, input_file, improved_file), "retry reopens it"
        assert "improvement" not in row, "re-read clean -> marker dropped"
        row = exhausted("reread_bad.flac", b"ORIG", None)
        assert run_pass(mod, cfg, row, {}, input_file, improved_file), "retry reopens it"
        assert_eq(row["improvement"]["status"], "pending_tts", "re-read bad -> requeued")
    print("test_retry_exhausted: OK")


def test_incomplete_candidate_set_goes_back_to_tts(mod):
    """The stt pass must not settle for the best of 3 when --n-improv asks for 5: it
    picks a winner (or exhausts the clip) once and never revisits it, so judging a
    partial set locks in a draw that may not be the best available. A short set goes
    back to the tts pass instead. Short means "the tts pass did not deliver every draw
    it was asked for" (raised --n-improv, sidecar gone missing), NOT "fewer files than
    --n-improv": a draw skipped as a duplicate or as frame-cap truncated is skipped
    deterministically, so demanding a file for it would ping-pong forever."""
    install_fakes(mod)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        input_file = root / "manifest.jsonl"
        improved_file = root / "improved.jsonl"
        clip = root / "clip.flac"

        def flagged():
            """A clip a prior stt pass found bad and queued for regeneration."""
            clip.write_bytes(b"ORIG")
            return {"audio_filepath": "clip.flac", "asr_training_source": TARGET,
                    "asr_training_target": TARGET, "index": 3,
                    "transcriptions": {"default": {"text": BAD_TRANSCRIPT, "cer": 0.9,
                                                   "n_stt_check": 3}},
                    "improvement": {"status": "pending_tts", "orig_cer": 0.9,
                                    "orig_tail": None}}

        def do(mode, row, n_improv, records=None):
            return run_pass(mod, make_cfg(mode, root, n_improv=n_improv), row,
                            records if records is not None else {}, input_file, improved_file)

        # --- --n-improv raised between the two passes ----------------------------
        row = flagged()
        do("tts", row, 3)
        assert_eq(row["improvement"]["status"], "pending_stt", "3 candidates drafted")
        assert_eq(len(row["improvement"]["drafts"]), 3, "draft count")
        assert do("stt", row, 5), "a pending_stt row is always picked up by an stt pass"
        assert_eq(row["improvement"]["status"], "pending_tts", "a short set goes back to tts")
        assert "drafts" not in row["improvement"], "the partial set is dropped with it"
        assert_eq(row["improvement"]["orig_cer"], 0.9, "the number to beat is carried over")
        assert_eq(row["transcriptions"]["default"]["text"], BAD_TRANSCRIPT,
                  "the stored transcript is untouched")
        assert_eq(count_pending_rows(row), (1, 0), "it is queued for the tts pass again")
        # The tts pass makes the full set, and only then is the clip judged.
        do("tts", row, 5)
        assert_eq(len(row["improvement"]["drafts"]), 5, "full candidate set")
        do("stt", row, 5, records={})
        assert_eq(row["improvement"]["status"], "improved", "a complete set is scored")

        # --- a sidecar has gone missing ------------------------------------------
        row = flagged()
        do("tts", row, 3)
        Path(row["improvement"]["drafts"][0]["path"]).unlink()
        do("stt", row, 3)
        assert_eq(row["improvement"]["status"], "pending_tts",
                  "a candidate file that is gone sends the clip back to tts")

        # --- a draw the tts pass skipped is NOT a shortfall -----------------------
        # Seeds 43 and 45 draw the same audio here, so 3 attempts leave 2 sidecars.
        # Regenerating would skip it again (the seed of each attempt is fixed), so this
        # set is as complete as it will ever be and must be judged, not requeued.
        def dup_tts(url, text, *, response_format, seed, voice, speed, model,
                    timeout, max_retries, cfg_alpha):
            return f"AUD{43 if seed == 45 else seed}".encode()

        mod.tts_synthesize = dup_tts
        row = flagged()
        do("tts", row, 3)
        install_fakes(mod)
        assert_eq(len(row["improvement"]["drafts"]), 2, "the duplicate draw was skipped")
        assert_eq(row["improvement"]["attempts"], 3, "but all three draws were tried")
        do("stt", row, 3, records={})
        assert_eq(row["improvement"]["status"], "improved",
                  "a set the tts pass could not fill is still judged")
    print("test_incomplete_candidate_set_goes_back_to_tts: OK")


def test_resume_drops_rows_whose_text_changed(mod):
    """Re-chunking a source rebuilds the manifest, and a clip filename is only unique
    within one chunking: `PARHAF/000000_0000_x_00001_t0_c0.flac` exists in the old and
    the new manifest with DIFFERENT text. The stored transcription, CER and improvement
    state then describe audio that no longer exists, and inheriting them would score the
    new clip against the old label (and regenerate audio for a sentence the manifest no
    longer holds). A row's identity is its path AND its text, so a changed text drops the
    stored work. A row whose text is unchanged must still resume normally."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        input_file = root / "manifest.jsonl"
        out_file = root / "out.jsonl"
        rows = [
            {"audio_filepath": "rechunked.flac", "text": "le nouveau texte", "duration": 9.0},
            {"audio_filepath": "unchanged.flac", "text": "toujours le meme", "duration": 8.0},
        ]
        input_file.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        stored = [
            {"audio_filepath": "rechunked.flac", "text": "l ancien texte, autre decoupage",
             "duration": 163.84,
             "transcriptions": {"default": {"text": "l ancien texte", "cer": 0.01}},
             "improvement": {"status": "exhausted", "orig_cer": 0.9}},
            {"audio_filepath": "unchanged.flac", "text": "toujours le meme", "duration": 8.0,
             "transcriptions": {"default": {"text": "toujours le meme", "cer": 0.0}}},
        ]
        out_file.write_text("".join(json.dumps(r) + "\n" for r in stored), encoding="utf-8")

        # A row that lost its stored transcription is exactly what needs_work queues, and
        # the pass only rewrites the output file once something has been processed.
        seen = []
        mod._stt.process_file(
            input_file, out_file,
            endpoint="http://stt.invalid/x", token="", model=None, temperature=0.0,
            language="fr", response_format="json", extra_params={}, timeout=1.0,
            max_retries=1, audio_key="audio_filepath", reference_key="text",
            audio_root=root, wer_threshold=1.0, cer_threshold=0.4,
            flush_every=1, flush_interval=0.0, limit=0, n_parallel=1, shuffle=False,
            worker=lambda pos, rec: seen.append(rec.get("audio_filepath")),
            apply_result=lambda pos, result, results: {},
            needs_work=lambda out_rec: not out_rec["transcriptions"],
        )
        assert_eq(seen, ["rechunked.flac"], "only the re-chunked row is queued for work")
        written = {json.loads(line)["audio_filepath"]: json.loads(line)
                   for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()}
        fresh = written["rechunked.flac"]
        assert_eq(fresh["text"], "le nouveau texte", "the input's text wins over the stored one")
        assert_eq(fresh["duration"], 9.0, "and so does its duration")
        assert_eq(fresh["transcriptions"], {}, "the old audio's transcription is dropped")
        assert "improvement" not in fresh, "and so is its improvement state"
        kept = written["unchanged.flac"]
        assert_eq(kept["transcriptions"]["default"]["cer"], 0.0,
                  "an unchanged row still resumes from its stored transcription")
    print("test_resume_drops_rows_whose_text_changed: OK")


def test_flush_backoff(mod):
    """Every flush rewrites the WHOLE output, so its cost scales with the manifest, not
    with the clips just done. On the 600k-row release manifest a rewrite is ~10 s, and
    obeying --flush-every 50 literally meant a pass with nothing to wait for (a rescore)
    spent all of its time writing: measured at 5 clips/s with a 20 h ETA. The cadence
    therefore follows the measured cost, keeping the rewrite overhead near 5%."""
    backoff = mod._stt.flush_backoff_seconds
    # Nothing written yet, or an instant write: no hold-off at all, so small files and
    # the first flush of any run behave exactly as they did before.
    assert_eq(backoff(0.0), 0.0, "no measured cost means no hold-off")
    assert_eq(backoff(-1.0), 0.0, "a nonsense cost cannot introduce a delay")
    # 10 s to rewrite at a 5% duty cycle: one flush per 200 s, so at most 200 s of work
    # is at risk, against 10 s spent writing.
    assert_eq(backoff(10.0), 200.0, "an expensive rewrite has to be earned")
    assert abs(backoff(0.01) - 0.2) < 1e-9, "a cheap rewrite barely delays the next one"
    assert_eq(backoff(10.0, duty_cycle=0.5), 20.0, "a laxer duty cycle flushes sooner")
    assert_eq(backoff(10.0, duty_cycle=0.0), 0.0, "duty cycle 0 disables the hold-off")
    print("test_flush_backoff: OK")


if __name__ == "__main__":
    main()
    test_flush_backoff(load_module())
    test_backfill(load_module())
    test_recheck_temperature(load_module())
    test_tail_gate(load_module())
    test_needs_work_rescans_tails_and_counts(load_module())
    test_recheck(load_module())
    test_rescore_keeps_the_rows_data(load_module())
    test_needs_work_reads_the_stored_score(load_module())
    test_rescore_rejudges_the_tts_queue(load_module())
    test_retry_exhausted(load_module())
    test_incomplete_candidate_set_goes_back_to_tts(load_module())
    test_resume_drops_rows_whose_text_changed(load_module())
