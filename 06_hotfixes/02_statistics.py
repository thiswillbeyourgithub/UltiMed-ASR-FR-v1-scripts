#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "click>=8.1",
#   "loguru>=0.7",
#   "tqdm>=4.66",
# ]
# ///
"""Summarize the CER of a stt.jsonl into a Markdown report, by category and overall.

Consumes the ``<name>.stt.jsonl`` written by ``01_recursive_improvement.py`` (or
``01_compute_stt.py``): every manifest row is present in input order, and its per-model
CER lives at ``transcriptions[<model>].cer``. A row that has not been transcribed yet
carries an empty ``transcriptions: {}``, so this script works on a PARTIAL file while the
STT/TTS run is still going: it treats "row exists but has no numeric CER" as not-yet-done
and reports a COVERAGE estimate (scored rows / total rows, overall and per category) next
to the statistics, so you know how much of the dataset the numbers are based on.

Because ``01_compute_stt.py`` writes the whole row list on every atomic flush (unscored
rows included), reading the live ``.stt.jsonl`` gives a consistent, complete-manifest
snapshot: total row counts are exact, and coverage is exactly the fraction scored so far.

For each group (each ``category``, i.e. each dataset subset, plus an ``Overall`` row) it
reports over the numeric CER values:

* count, total rows, coverage %,
* mean, median, sample standard deviation,
* min / max,
* the fraction at or above ``--threshold`` (the "bad clip" rate, 0.08 by default, matching
  the regeneration threshold used by ``01_recursive_improvement.py``),
* quantiles at every ``--quantile-step`` (default 0.1, i.e. the deciles P0, P10, ... P100),
  computed by linear interpolation on the sorted values.

It then reports the TAIL CER the same way (summary + quantiles, per category and overall):
``01_compute_stt.py`` scores the last ~30 seconds of every clip longer than 30 s on its own
(the last ``TAIL_CHARS`` normalized characters of reference and hypothesis, extrapolated
from the corpus median pace) and stores it as ``cer_tail`` next to ``cer``. A long clip that
loops, derails or gets truncated at the end can still have an acceptable whole-clip CER
simply because the correct beginning outweighs the broken ending, so the tail is scored
separately against its own, looser ``--tail-threshold`` (0.12 by default: the window
boundary cuts mid-sentence, so its noise floor is higher than the whole clip's). The report also
counts, per category, the long clips flagged by the TAIL ONLY, i.e. those the whole-clip
threshold would have let through.

It then reports the AUDIO PACE the same way (summary + quantiles, per category and
overall): each clip's duration divided by the number of UNNORMALIZED characters of the
text it was synthesized from (``asr_training_source`` as stored, nothing folded or
stripped). The TTS voice speaks at a near-constant rate, so information density per second
should be near-constant too, which makes an unexpectedly long clip (stall, repetition,
trailing silence) or an unexpectedly short one (truncated / skipped reading) stand out in
the tails of that distribution. Pace needs no transcription, so it covers every row with a
duration, even on a partially scored file.

It also reports the TRIPLE-CHECK coverage of those high-CER clips: before trusting a bad
CER, ``01_recursive_improvement.py`` re-transcribes the clip at rising STT temperatures
until it has been read 3 times, and records that reading count in ``n_stt_check`` on the
transcription entry (it drops the marker when a recheck rescues the clip or a regeneration
wins). So of the clips currently at CER >= ``--threshold``, this counts how many have had
all 3 readings, i.e. how many of the still-bad clips are fully checked versus still
awaiting a reading.

The report is written to ``--output`` (default ``<input>.statistics.md``) and, unless
``--no-stdout`` is passed, also echoed to the terminal.

Usage:

    uv run 02_statistics.py --input improved/full.stt.jsonl
    uv run 02_statistics.py --input improved/full.stt.jsonl --threshold 0.08 --model whisper-1

This file was written with Claude Code.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import click
from loguru import logger
from tqdm import tqdm

# The truncation audit is shared with 99_hf_release/scripts/get_statistics.py, so it
# lives in utils/ and both statistics scripts ask the question the same way.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from nemo_manifest import CEILING_TOL_S, duration_ceiling  # noqa: E402

# Row-level field holding the subset name (dictionary / drugs / parhaf / parrot).
CATEGORY_KEY = "category"
# Value used when a row has no category, so those rows are still counted somewhere
# instead of being silently dropped.
UNKNOWN_CATEGORY = "(no category)"
# Field 01_recursive_improvement.py stamps on a transcription entry of a high-CER clip:
# how many independent STT readings it has had (the first pass plus each recheck at a
# higher temperature). It reaches STT_CHECK_TARGET (3) when the clip is fully triple-checked
# and still bad; the marker is dropped when a recheck rescues the clip or a regeneration
# wins, so its presence means "still bad AND already re-read that many times".
STT_CHECK_KEY = "n_stt_check"
STT_CHECK_TARGET = 3
# Fields the audio-pace statistic reads: the text the clip was synthesized from (the TTS
# input, falling back to the written label when a row carries only that) and the clip
# length in seconds. Both live on the manifest row itself, so pace is available for every
# row, transcribed or not.
SOURCE_TEXT_KEYS = ("asr_training_source", "asr_training_target", "text")
DURATION_KEY = "duration"
# Per-entry CER of the clip's last ~30s, written by 01_compute_stt.py /
# 01_recursive_improvement.py on clips longer than that (see _stt.tail_cer). Absent on
# short clips and on anything scored before the key existed.
TAIL_CER_KEY = "cer_tail"


def percentile(sorted_vals: list[float], q: float) -> float | None:
    """Linear-interpolation percentile (numpy's default method) of an ALREADY sorted,
    non-empty list. ``q`` is a fraction in [0, 1]: 0.0 -> min, 0.5 -> median, 1.0 -> max."""
    n = len(sorted_vals)
    if n == 0:
        return None
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def quantile_points(step: float) -> list[float]:
    """The quantile fractions to report: 0.0, step, 2*step, ... up to and including 1.0.
    With the default 0.1 step this is the 11 deciles [0.0, 0.1, ..., 1.0]."""
    if step <= 0 or step > 1:
        raise click.BadParameter("--quantile-step must be in (0, 1]")
    pts: list[float] = []
    i = 0
    while True:
        q = round(i * step, 6)
        if q >= 1.0:
            break
        pts.append(q)
        i += 1
    pts.append(1.0)
    return pts


def resolve_cer(entry: dict | None, cer_field: str) -> float | None:
    """The numeric CER of one transcription entry, or None when it is missing / non-numeric
    (a not-yet-scored or unscorable clip). An entry carrying an ``error`` is treated as
    unscored."""
    if not isinstance(entry, dict) or entry.get("error"):
        return None
    val = entry.get(cer_field)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        f = float(val)
        if math.isnan(f):
            return None
        return f
    return None


def resolve_tail(entry: dict | None) -> float | None:
    """The stored tail CER of one transcription entry, or None when the clip was too short
    to have one (or predates the key)."""
    return resolve_cer(entry, TAIL_CER_KEY)


def is_fully_checked(entry: dict | None) -> bool:
    """True when a transcription entry has had all ``STT_CHECK_TARGET`` STT readings, i.e.
    the clip was re-transcribed at every recheck temperature and stayed bad (see
    ``STT_CHECK_KEY``)."""
    if not isinstance(entry, dict):
        return False
    try:
        return int(entry.get(STT_CHECK_KEY) or 0) >= STT_CHECK_TARGET
    except (TypeError, ValueError):
        return False


def source_char_count(rec: dict) -> int | None:
    """Number of UNNORMALIZED characters of the text this clip was synthesized from: the
    raw string as stored (only surrounding whitespace stripped), with no lowercasing,
    accent folding or punctuation stripping. None when the row carries no usable text."""
    for key in SOURCE_TEXT_KEYS:
        val = rec.get(key)
        if isinstance(val, str) and val.strip():
            return len(val.strip())
    return None


def pace_of(rec: dict) -> float | None:
    """Seconds of audio per source character for one row, or None when the row lacks a
    usable duration or source text.

    The TTS voice speaks at a near-constant rate, so information density per second should
    be near-constant too: a clip far off the median pace is suspicious (audio much longer
    than its text = a stall / repetition / trailing silence, much shorter = a truncated or
    skipped reading)."""
    n = source_char_count(rec)
    if not n:
        return None
    dur = rec.get(DURATION_KEY)
    if not isinstance(dur, (int, float)) or isinstance(dur, bool):
        return None
    dur = float(dur)
    if math.isnan(dur) or dur <= 0:
        return None
    return dur / n


def scan(
    input_file: Path, model: str | None, cer_field: str
) -> tuple[
    dict[str, list[tuple[float, bool]]], dict[str, list[tuple[float, float]]],
    dict[str, list[float]], dict[str, list[float]], Counter, Counter, str | None, Counter,
]:
    """One streaming pass over the jsonl. Returns:

    * ``samples``    : {category -> [(cer, fully_checked), ...]} for the CHOSEN model,
    * ``tails``      : {category -> [(cer, tail cer), ...]} for the CHOSEN model, over the
      long clips that carry a tail score (see ``TAIL_CER_KEY``),
    * ``pace``       : {category -> [seconds per source character, ...]} (model-independent,
      so it covers every row with a duration, transcribed or not),
    * ``durations``  : {category -> [clip duration, ...]}, for the truncation audit,
    * ``total_rows`` : {category -> total rows seen} (model-independent),
    * ``model_rows`` : {model_key -> rows carrying that model} (to auto-pick / report),
    * ``chosen``     : the model actually used,
    * ``bad_lines``  : {reason -> count} for unreadable / skipped lines.

    Each sample pairs the numeric CER with whether the clip has had all its STT readings
    (see ``is_fully_checked``), so the report can count triple-check coverage of the
    high-CER clips without a second pass.

    The model is resolved up front only if ``--model`` was given; otherwise we accumulate
    samples for EVERY model during the single pass and pick the most common one at the end,
    so no second read of a large file is needed."""
    total_rows: Counter = Counter()
    model_rows: Counter = Counter()
    bad_lines: Counter = Counter()
    pace: dict[str, list[float]] = defaultdict(list)
    durations: dict[str, list[float]] = defaultdict(list)
    # {model_key -> {category -> [(cer, checked), ...]}} filled for whichever models appear;
    # with one model (the usual case) this holds a single inner dict.
    per_model: dict[str, dict[str, list[tuple[float, bool]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    # Same shape, holding (whole-clip cer, tail cer) for the clips long enough to have one.
    per_model_tail: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    with input_file.open(encoding="utf-8") as fh:
        for line in tqdm(fh, desc="scanning", unit=" rows"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A partial run can be mid-write; tolerate a torn trailing line instead
                # of crashing so partial stats still come out.
                bad_lines["json_decode_error"] += 1
                continue
            cat = rec.get(CATEGORY_KEY) or UNKNOWN_CATEGORY
            total_rows[cat] += 1
            p = pace_of(rec)
            if p is not None:
                pace[cat].append(p)
            dur = rec.get(DURATION_KEY)
            if isinstance(dur, (int, float)) and dur > 0:
                durations[cat].append(float(dur))
            transcripts = rec.get("transcriptions")
            if not isinstance(transcripts, dict):
                continue
            for mk, entry in transcripts.items():
                model_rows[mk] += 1
                # Only accumulate samples for the model(s) we might report on: the pinned
                # one, or all of them when auto-picking.
                if model is not None and mk != model:
                    continue
                cer = resolve_cer(entry, cer_field)
                if cer is not None:
                    per_model[mk][cat].append((cer, is_fully_checked(entry)))
                    tail = resolve_tail(entry)
                    if tail is not None:
                        per_model_tail[mk][cat].append((cer, tail))

    if model is not None:
        chosen: str | None = model
    elif model_rows:
        chosen = model_rows.most_common(1)[0][0]
    else:
        chosen = None

    samples: dict[str, list[tuple[float, bool]]] = {}
    tails: dict[str, list[tuple[float, float]]] = {}
    if chosen is not None:
        samples = {cat: vals for cat, vals in per_model.get(chosen, {}).items()}
        tails = {cat: vals for cat, vals in per_model_tail.get(chosen, {}).items()}

    return (samples, tails, dict(pace), dict(durations),
            total_rows, model_rows, chosen, bad_lines)


def numeric_stats(vals: list[float], q_points: list[float]) -> dict:
    """Count / mean / median / stdev / min / max / quantiles of one group of numbers.
    Shared by every reported metric (CER, pace, ...) so the maths lives in one place."""
    n = len(vals)
    if n == 0:
        return {"n": 0}
    sv = sorted(vals)
    return {
        "n": n,
        "mean": statistics.fmean(sv),
        "median": statistics.median(sv),
        "std": statistics.stdev(sv) if n >= 2 else 0.0,
        "min": sv[0],
        "max": sv[-1],
        "quantiles": [percentile(sv, q) for q in q_points],
    }


def group_stats(samples: list[tuple[float, bool]], q_points: list[float], threshold: float) -> dict:
    """All reported statistics for one group's (cer, fully_checked) samples.

    Adds, on top of the shared numeric statistics, the bad-clip rate and ``bad_checked``
    (high-CER clips that have had all their STT readings) so the report can show how many
    of the still-bad clips have already been triple-checked."""
    s = numeric_stats([c for c, _ in samples], q_points)
    bad = sum(1 for c, _ in samples if c >= threshold)
    s["bad_n"] = bad
    s["bad_checked"] = sum(1 for c, chk in samples if c >= threshold and chk)
    if s["n"]:
        s["bad_frac"] = bad / s["n"]
    return s


def fmt(x: float | None, nd: int = 4) -> str:
    """Fixed-decimal formatting, with a dash for missing values."""
    if x is None:
        return "-"
    return f"{x:.{nd}f}"


def pct(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{100.0 * x:.1f}%"


def bold(cells: list[str]) -> list[str]:
    """Bold every cell of a row (used for the Overall line)."""
    return [f"**{c}**" for c in cells]


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    """A Markdown table: label column left-aligned, every numeric column right-aligned."""
    sep = "|---|" + "|".join(["--:"] * (len(header) - 1)) + "|"
    return (
        ["| " + " | ".join(header) + " |", sep]
        + ["| " + " | ".join(r) + " |" for r in rows]
    )


# Header of the shared summary block, reused by every metric's summary table.
SUMMARY_HEADER = ["Category", "N", "Mean", "Median", "Std", "Min", "Max"]


def summary_cells(label: str, s: dict, nd: int) -> list[str]:
    """The shared N / mean / median / std / min / max cells of one summary row."""
    if s["n"] == 0:
        return [label] + ["0"] + ["-"] * (len(SUMMARY_HEADER) - 2)
    return [
        label, str(s["n"]), fmt(s["mean"], nd), fmt(s["median"], nd), fmt(s["std"], nd),
        fmt(s["min"], nd), fmt(s["max"], nd),
    ]


def quantile_cells(label: str, s: dict, nd: int, n_q: int) -> list[str]:
    """The N + one-cell-per-quantile cells of one quantile row."""
    if s["n"] == 0:
        return [label, "0"] + ["-"] * n_q
    return [label, str(s["n"])] + [fmt(q, nd) for q in s["quantiles"]]


def build_markdown(
    input_file: Path,
    chosen_model: str | None,
    model_rows: Counter,
    samples: dict[str, list[tuple[float, bool]]],
    tails: dict[str, list[tuple[float, float]]],
    pace: dict[str, list[float]],
    durations: dict[str, list[float]],
    total_rows: Counter,
    q_points: list[float],
    threshold: float,
    tail_threshold: float,
    bad_lines: Counter,
    cer_field: str,
) -> str:
    """Assemble the whole report as a Markdown string. Every metric is reported per
    category (per dataset subset) AND overall, both as a summary and as quantiles."""
    categories = sorted(samples.keys() | total_rows.keys() | pace.keys() | durations.keys())
    all_samples: list[tuple[float, bool]] = [s for lst in samples.values() for s in lst]
    all_tails: list[tuple[float, float]] = [t for lst in tails.values() for t in lst]
    all_pace: list[float] = [p for lst in pace.values() for p in lst]
    total_all = sum(total_rows.values())
    scored_all = len(all_samples)
    # Every table below is driven by these, so each metric is computed once per group.
    cer_stats = {cat: group_stats(samples.get(cat, []), q_points, threshold) for cat in categories}
    cer_overall = group_stats(all_samples, q_points, threshold)
    tail_stats = {cat: numeric_stats([t for _, t in tails.get(cat, [])], q_points)
                  for cat in categories}
    tail_overall = numeric_stats([t for _, t in all_tails], q_points)
    pace_stats = {cat: numeric_stats(pace.get(cat, []), q_points) for cat in categories}
    pace_overall = numeric_stats(all_pace, q_points)
    q_labels = [f"P{int(round(q * 100))}" for q in q_points]
    q_step = q_points[1] if len(q_points) > 1 else 1.0

    lines: list[str] = []
    lines.append("# CER statistics")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Input: `{input_file}`")
    lines.append(f"- CER field: `{cer_field}`")
    if chosen_model is not None:
        lines.append(f"- Model: `{chosen_model}` (n={model_rows.get(chosen_model, 0)} rows)")
    else:
        lines.append("- Model: none found (no transcriptions in the file yet)")
    others = [f"`{m}` (n={c})" for m, c in model_rows.most_common() if m != chosen_model]
    if others:
        lines.append(f"- Other models present: {', '.join(others)}")
    lines.append(f"- Bad-clip threshold (CER >=): {threshold}")
    if bad_lines:
        detail = ", ".join(f"{k}={v}" for k, v in bad_lines.items())
        lines.append(f"- Skipped unreadable lines: {detail}")
    lines.append("")

    # --- Coverage -----------------------------------------------------------------
    lines.append("## Coverage")
    lines.append("")
    lines.append(
        "How much of the manifest has a numeric CER so far. On a still-running job this is "
        "below 100%; the statistics below are computed over the scored rows only."
    )
    lines.append("")
    lines.append("| Category | Scored | Total | Coverage |")
    lines.append("|---|--:|--:|--:|")
    for cat in categories:
        scored = len(samples.get(cat, []))
        total = total_rows.get(cat, 0)
        cov = scored / total if total else None
        lines.append(f"| {cat} | {scored} | {total} | {pct(cov)} |")
    overall_cov = scored_all / total_all if total_all else None
    lines.append(f"| **Overall** | **{scored_all}** | **{total_all}** | **{pct(overall_cov)}** |")
    lines.append("")

    # --- Truncation audit ---------------------------------------------------------
    # A clip cut off by the TTS token limit stops mid-sentence, so its transcript can
    # never match: it is bad audio no seed can fix, and the fix is re-chunking the text.
    # It is invisible in the CER tables (it just reads as a high CER), but obvious in the
    # duration distribution, where truncated clips pile up on the cap to the frame.
    lines.append("## Duration ceiling")
    lines.append("")
    lines.append(
        f"`At max` counts the clips within one 12.5 Hz frame ({CEILING_TOL_S} s) of the "
        "longest one in the category. More than a couple means the audio was generated "
        "against a token cap, not to the end of its text: those clips are cut off "
        "mid-sentence and no regeneration seed can fix them, so the source text has to be "
        "re-chunked."
    )
    lines.append("")
    lines.append("| Category | Longest clip | At max | Share | Verdict |")
    lines.append("|---|--:|--:|--:|---|")
    capped_cats: list[str] = []
    all_durs = [d for lst in durations.values() for d in lst]
    for cat in categories + ["**Overall**"]:
        durs = all_durs if cat == "**Overall**" else durations.get(cat, [])
        c = duration_ceiling(durs)
        if c is None:
            lines.append(f"| {cat} | - | - | - | no duration on any row |")
            continue
        if c["capped"] and cat != "**Overall**":
            capped_cats.append(cat)
        verdict = "**CEILING**" if c["capped"] else "no ceiling"
        lines.append(f"| {cat} | {c['max']:.2f} s | {c['at_max']} | {pct(c['share'])} | "
                     f"{verdict} |")
    lines.append("")
    if capped_cats:
        lines.append(
            f"> **Warning:** {', '.join(capped_cats)} sits on a duration ceiling. Those "
            "clips are truncated, so their transcript does not describe the whole audio. "
            "Re-chunk the source text and regenerate them; scoring them again will not "
            "help."
        )
        lines.append("")

    # --- Summary statistics -------------------------------------------------------
    lines.append("## CER summary")
    lines.append("")

    def bad_cell(s: dict) -> str:
        """The bad-clip rate cell: fraction at or above the threshold, with the count."""
        return "-" if s["n"] == 0 else f"{pct(s['bad_frac'])} ({s['bad_n']})"

    rows = [summary_cells(cat, cer_stats[cat], 4) + [bad_cell(cer_stats[cat])]
            for cat in categories]
    rows.append(bold(summary_cells("Overall", cer_overall, 4) + [bad_cell(cer_overall)]))
    lines.extend(table(SUMMARY_HEADER + [f"CER>={threshold}"], rows))
    lines.append("")

    # --- Triple-check coverage of high-CER clips ----------------------------------
    lines.append("## Triple-check coverage of high-CER clips")
    lines.append("")
    lines.append(
        f"Of the clips at CER >= {threshold}, how many carry `{STT_CHECK_KEY}` >= "
        f"{STT_CHECK_TARGET}, i.e. have already been re-transcribed at every recheck "
        "temperature and stayed bad. \"Not yet\" clips are high-CER but still short of "
        f"{STT_CHECK_TARGET} readings (awaiting a recheck, or the rechecks were disabled). "
        "A clip a recheck rescued is no longer high-CER and is not counted here."
    )
    lines.append("")

    def checked_cells(label: str, s: dict) -> list[str]:
        bad_n = s["bad_n"]
        checked = s["bad_checked"]
        cov = checked / bad_n if bad_n else None
        return [label, str(bad_n), str(checked), str(bad_n - checked), pct(cov)]

    rows = [checked_cells(cat, cer_stats[cat]) for cat in categories]
    rows.append(bold(checked_cells("Overall", cer_overall)))
    lines.extend(table(
        ["Category", "High-CER (bad)", "Triple-checked", "Not yet", "Coverage"], rows))
    lines.append("")

    # --- CER quantiles ------------------------------------------------------------
    lines.append(f"## CER quantiles (step {q_step:g})")
    lines.append("")
    rows = [quantile_cells(cat, cer_stats[cat], 4, len(q_labels)) for cat in categories]
    rows.append(bold(quantile_cells("Overall", cer_overall, 4, len(q_labels))))
    lines.extend(table(["Category", "N"] + q_labels, rows))
    lines.append("")

    # --- Tail CER -----------------------------------------------------------------
    lines.append(f"## Tail CER (last ~30s of clips longer than that)")
    lines.append("")
    lines.append(
        f"`{TAIL_CER_KEY}` scores the END of a clip on its own, because a whole-clip CER "
        "averages a late defect away: a chunk truncated at the TTS output cap keeps most "
        "of its text and lands around 0.13, under the gate. A clip counts as bad when "
        f"EITHER its CER reaches {threshold} OR its tail CER reaches {tail_threshold} "
        "(looser: the tail window is ~5x shorter, so ~5x noisier). Clips at or under 30s "
        "carry no tail score and are not counted here."
    )
    lines.append("")
    rows = [summary_cells(cat, tail_stats[cat], 4) for cat in categories]
    rows.append(bold(summary_cells("Overall", tail_overall, 4)))
    lines.extend(table(SUMMARY_HEADER, rows))
    lines.append("")

    def tail_only_cells(label: str, pairs: list[tuple[float, float]]) -> list[str]:
        """Clips the tail gate adds: whole-clip CER passes, tail CER does not."""
        n = len(pairs)
        added = sum(1 for c, t in pairs if c < threshold and t >= tail_threshold)
        both = sum(1 for c, t in pairs if c >= threshold and t >= tail_threshold)
        return [label, str(n), str(added), str(both),
                pct(added / n) if n else "-"]

    rows = [tail_only_cells(cat, tails.get(cat, [])) for cat in categories]
    rows.append(bold(tail_only_cells("Overall", all_tails)))
    lines.extend(table(
        ["Category", "Long clips scored", "Flagged by the tail ONLY", "Bad on both",
         "Tail-only rate"], rows))
    lines.append("")

    lines.append(f"### Tail CER quantiles (step {q_step:g})")
    lines.append("")
    rows = [quantile_cells(cat, tail_stats[cat], 4, len(q_labels)) for cat in categories]
    rows.append(bold(quantile_cells("Overall", tail_overall, 4, len(q_labels))))
    lines.extend(table(["Category", "N"] + q_labels, rows))
    lines.append("")

    # --- Audio pace ---------------------------------------------------------------
    lines.append("## Audio pace (seconds per source character)")
    lines.append("")
    lines.append(
        "Clip duration divided by the number of UNNORMALIZED characters of the text it was "
        f"synthesized from (`{SOURCE_TEXT_KEYS[0]}`, raw, whitespace and punctuation "
        "included). The TTS voice speaks at a near-constant rate, so information density "
        "per second should be near-constant too: clips far from the median pace are "
        "suspicious. A HIGH value means the audio is much longer than its text (a stall, a "
        "repetition, trailing silence), a LOW one means it is too short (a truncated or "
        "skipped reading). This is model-independent, so it covers every row carrying a "
        "duration and a source text, transcribed or not."
    )
    lines.append("")
    rows = [summary_cells(cat, pace_stats[cat], 5) for cat in categories]
    rows.append(bold(summary_cells("Overall", pace_overall, 5)))
    lines.extend(table(SUMMARY_HEADER, rows))
    lines.append("")

    lines.append(f"### Audio pace quantiles (step {q_step:g})")
    lines.append("")
    lines.append(
        "The same seconds-per-character value at each quantile, so the suspicious tails "
        "are readable: compare P0 / P10 and P90 / P100 against the median."
    )
    lines.append("")
    rows = [quantile_cells(cat, pace_stats[cat], 5, len(q_labels)) for cat in categories]
    rows.append(bold(quantile_cells("Overall", pace_overall, 5, len(q_labels))))
    lines.extend(table(["Category", "N"] + q_labels, rows))
    lines.append("")

    return "\n".join(lines)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--input", "input_file", required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the .stt.jsonl written by 01_recursive_improvement.py / 01_compute_stt.py.",
)
@click.option(
    "--output", "output_file", default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Markdown report path. Default: <input> with a .statistics.md suffix.",
)
@click.option(
    "--model", default=None,
    help="Which transcriptions[<model>] key to read CER from. Default: the most common one.",
)
@click.option(
    "--cer-field", default="cer", show_default=True,
    help="Field inside each transcription entry to summarize (use 'wer' to summarize WER).",
)
@click.option(
    "--tail-threshold", default=0.12, show_default=True, type=float,
    help=("Gate on the stored tail CER (last ~30s of a long clip): a clip is bad when "
          "EITHER its CER reaches --threshold OR its tail CER reaches this. Looser on "
          "purpose, the tail window is ~5x shorter and so ~5x noisier. Matches "
          "01_recursive_improvement.py's --tail-cer-threshold."),
)
@click.option(
    "--threshold", default=0.08, show_default=True, type=float,
    help="Report the fraction of clips at or above this CER (the bad-clip rate).",
)
@click.option(
    "--quantile-step", default=0.1, show_default=True, type=float,
    help="Quantile spacing as a fraction: 0.1 gives the deciles P0, P10, ... P100.",
)
@click.option(
    "--no-stdout", is_flag=True, default=False,
    help="Only write the .md file, do not also echo the report to the terminal.",
)
def main(
    input_file: Path,
    output_file: Path | None,
    model: str | None,
    cer_field: str,
    threshold: float,
    tail_threshold: float,
    quantile_step: float,
    no_stdout: bool,
) -> None:
    """Compute CER statistics (mean / median / stdev / deciles) by category and overall
    from a stt.jsonl, with a coverage estimate, and write them as Markdown."""
    if output_file is None:
        # full.stt.jsonl -> full.stt.statistics.md
        stem = input_file.name
        if stem.endswith(".jsonl"):
            stem = stem[: -len(".jsonl")]
        output_file = input_file.with_name(f"{stem}.statistics.md")

    q_points = quantile_points(quantile_step)

    logger.info(f"scanning {input_file} ...")
    samples, tails, pace, durations, total_rows, model_rows, chosen, bad_lines = scan(
        input_file, model, cer_field
    )

    if model is not None and model not in model_rows and sum(total_rows.values()):
        logger.warning(
            f"--model {model!r} not found in the file; models present: "
            f"{', '.join(model_rows) or '(none)'}"
        )

    total_all = sum(total_rows.values())
    scored_all = sum(len(v) for v in samples.values())
    logger.info(
        f"model={chosen!r}  scored={scored_all}  total_rows={total_all}  "
        f"coverage={(100.0 * scored_all / total_all) if total_all else 0:.1f}%"
    )

    md = build_markdown(
        input_file, chosen, model_rows, samples, tails, pace, durations, total_rows,
        q_points, threshold, tail_threshold, bad_lines, cer_field,
    )
    output_file.write_text(md, encoding="utf-8")
    logger.info(f"wrote {output_file}")

    if not no_stdout:
        click.echo("")
        click.echo(md)


if __name__ == "__main__":
    main()
