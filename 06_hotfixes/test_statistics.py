#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click>=8.1","loguru>=0.7","tqdm>=4.66"]
# ///
"""Tests for 02_statistics.py: the audio-pace metric, the triple-check marker and the
shared quantile / table plumbing. Run: `uv run test_statistics.py` (needs the module's
deps to import it, hence the uv header). Written with Claude Code.

Locks three behaviors:
* pace = clip duration / UNNORMALIZED source character count (nothing folded or stripped),
* a high-CER clip counts as checked only at STT_CHECK_TARGET (3) readings, not 2,
* every metric is reported per category AND overall, as a summary and as quantiles."""
import importlib.util
from collections import Counter
from pathlib import Path

_spec = importlib.util.spec_from_file_location("statistics_mod", Path(__file__).with_name("02_statistics.py"))
_st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_st)

_failures = []


def eq(label, got, want):
    if got != want:
        _failures.append(f"{label}: got {got!r} want {want!r}")


def close(label, got, want, tol=1e-9):
    if got is None or abs(got - want) > tol:
        _failures.append(f"{label}: got {got!r} want ~{want!r}")


def has(label, haystack, needle):
    if needle not in haystack:
        _failures.append(f"{label}: {needle!r} missing from the report")


# --- source_char_count: raw characters, nothing normalized ---
eq("raw count keeps accents/case/punct", _st.source_char_count({"asr_training_source": "Été, à 5 %."}), 11)
eq("inner whitespace counted", _st.source_char_count({"asr_training_source": "a b"}), 3)
eq("outer whitespace stripped", _st.source_char_count({"asr_training_source": "  abc \n"}), 3)
eq("falls back to the label", _st.source_char_count({"text": "abcd"}), 4)
eq("source wins over the label", _st.source_char_count({"asr_training_source": "ab", "text": "abcdef"}), 2)
eq("no text -> None", _st.source_char_count({"duration": 1.0}), None)
eq("blank text -> None", _st.source_char_count({"asr_training_source": "   "}), None)

# --- pace_of: seconds per source character ---
close("pace = duration / chars", _st.pace_of({"asr_training_source": "abcde", "duration": 1.0}), 0.2)
eq("no duration -> None", _st.pace_of({"asr_training_source": "abcde"}), None)
eq("zero duration -> None", _st.pace_of({"asr_training_source": "abcde", "duration": 0}), None)
eq("negative duration -> None", _st.pace_of({"asr_training_source": "abcde", "duration": -3}), None)
eq("non-numeric duration -> None", _st.pace_of({"asr_training_source": "abc", "duration": "1.0"}), None)
eq("bool duration -> None", _st.pace_of({"asr_training_source": "abc", "duration": True}), None)
eq("no text -> None", _st.pace_of({"duration": 1.0}), None)

# --- tail CER key ---
eq("tail read from the entry", _st.resolve_tail({"cer": 0.1, "cer_tail": 0.42}), 0.42)
eq("no tail key -> None", _st.resolve_tail({"cer": 0.1}), None)
eq("errored entry -> None", _st.resolve_tail({"error": "boom", "cer_tail": 0.42}), None)

# --- triple-check marker: 2 readings is no longer enough ---
eq("2 readings is not fully checked", _st.is_fully_checked({"n_stt_check": 2}), False)
eq("3 readings is fully checked", _st.is_fully_checked({"n_stt_check": 3}), True)
eq("more than 3 still counts", _st.is_fully_checked({"n_stt_check": 4}), True)
eq("no marker -> not checked", _st.is_fully_checked({"cer": 0.9}), False)
eq("garbage marker -> not checked", _st.is_fully_checked({"n_stt_check": "many"}), False)
eq("target is 3 (triple-check)", _st.STT_CHECK_TARGET, 3)

# --- quantiles / numeric stats ---
q = _st.quantile_points(0.1)
eq("11 deciles", len(q), 11)
eq("deciles span 0..1", (q[0], q[-1]), (0.0, 1.0))
vals = [float(i) for i in range(11)]  # 0..10
close("P0 = min", _st.percentile(vals, 0.0), 0.0)
close("P50 = median", _st.percentile(vals, 0.5), 5.0)
close("P100 = max", _st.percentile(vals, 1.0), 10.0)
close("interpolated P95", _st.percentile(vals, 0.95), 9.5)
s = _st.numeric_stats(vals, q)
eq("n", s["n"], 11)
close("median", s["median"], 5.0)
close("min", s["min"], 0.0)
close("max", s["max"], 10.0)
eq("empty group", _st.numeric_stats([], q), {"n": 0})

# --- group_stats: bad rate + triple-check coverage ---
samples = [(0.01, False), (0.2, True), (0.5, False)]
g = _st.group_stats(samples, q, 0.15)
eq("bad count at 0.15", g["bad_n"], 2)
close("bad fraction", g["bad_frac"], 2 / 3)
eq("only the checked bad clip counts", g["bad_checked"], 1)
eq("empty group has no bad clips", _st.group_stats([], q, 0.15)["bad_n"], 0)

# --- report: every metric per category AND overall ---
rows = [
    {"category": "dictionary", "asr_training_source": "abcdefghij", "duration": 0.6,
     "transcriptions": {"whisper-1": {"cer": 0.02}}},
    {"category": "dictionary", "asr_training_source": "abcde", "duration": 0.3,
     "transcriptions": {"whisper-1": {"cer": 0.9, "n_stt_check": 3}}},
    {"category": "parhaf", "asr_training_source": "abcdefghij", "duration": 5.0,
     "transcriptions": {}},  # not transcribed yet: pace still counts it
]
samples_by_cat = {"dictionary": [(0.02, False), (0.9, True)]}
pace_by_cat = {"dictionary": [_st.pace_of(rows[0]), _st.pace_of(rows[1])],
               "parhaf": [_st.pace_of(rows[2])]}
# (whole-clip cer, tail cer): the second one passes on CER but fails on its ending.
tails_by_cat = {"dictionary": [(0.02, 0.05), (0.13, 0.80)]}
# Durations feed the truncation audit only. dictionary tails off normally; parhaf is the
# shape a token-capped TTS run leaves behind, every clip pinned on the same 163.84 s.
CAP = 163.84
durations_by_cat = {"dictionary": [3.0, 7.5, 12.25, 19.0],
                    "parhaf": [40.0, CAP, CAP, CAP, CAP - 0.04]}
md = _st.build_markdown(
    Path("x.stt.jsonl"), "whisper-1", Counter({"whisper-1": 2}), samples_by_cat,
    tails_by_cat, pace_by_cat, durations_by_cat, Counter({"dictionary": 2, "parhaf": 1}),
    q, 0.15, 0.3, Counter(), "cer",
)
has("CER summary section", md, "## CER summary")
has("CER quantiles section", md, "## CER quantiles")
has("tail section", md, "## Tail CER (last ~30s")
has("tail quantiles section", md, "### Tail CER quantiles")
has("tail-only column", md, "Flagged by the tail ONLY")
has("pace section", md, "## Audio pace (seconds per source character)")
has("pace quantiles section", md, "### Audio pace quantiles")
has("triple-check section", md, "## Triple-check coverage of high-CER clips")
has("threshold echoed", md, "CER>=0.15")
# The truncation audit: a capped category must be named, and a healthy one left alone.
# Without this the only symptom of a token-capped TTS run is a high CER, which reads as
# "regenerate it" when the real fix is re-chunking the text.
has("ceiling section", md, "## Duration ceiling")
has("capped category flagged", md, "| parhaf | 163.84 s | 4 |")
has("ceiling warning", md, "> **Warning:** parhaf sits on a duration ceiling")
has("healthy category quiet", md, "| dictionary | 19.00 s | 1 | 25.0% | no ceiling |")
# Both quantile tables must carry a row per dataset subset plus the bold Overall row.
# coverage + cer summary + triple-check + cer quantiles + tail summary + tail-only
# + tail quantiles + pace summary + pace quantiles + duration ceiling
eq("every table has an Overall row", md.count("| **Overall** |"), 10)
for cat in ("dictionary", "parhaf"):
    if md.count(f"| {cat} |") < 10:
        _failures.append(f"category {cat} missing from a table")
# Pace covers the untranscribed parhaf row (0.5 s/char), which no CER table can show.
has("untranscribed row paced", md, "0.50000")

if _failures:
    print("FAILED:")
    for f in _failures:
        print("  -", f)
    raise SystemExit(1)
print("ALL PASSED")
