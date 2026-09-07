#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru"]
# ///
"""Carry stage 06's quality-control results into the NeMo manifests.

Run AFTER an improvement run and BEFORE ``scripts/build_parquet.py``:

    uv run 03_sync_hotfix_results.py            # dry run, reports what it would change
    uv run 03_sync_hotfix_results.py --apply

Stage 06 regenerates bad clips by OVERWRITING the audio in place, so the clips on the
SSD are already the good ones, but the manifests built before that run still describe the
old audio: a regenerated draw is a different length (deltas up to ~115 s observed, and a
handful of clips cross the trainer's 45 s ``max_duration`` in one direction or the other).
Rebuilding the manifests instead of patching them would not fix that by itself either,
because ``probe_durations`` caches by FILE NAME with no mtime or size, so a rebuild
happily reuses every stale value.

So this reads stage 06's scored output (which already carries the re-probed duration and
the transcript of the audio that actually shipped) and, joining on the resolved absolute
audio path, writes into every manifest row:

    duration          re-probed after the swap
    cer               CER of the shipped audio against its label
    cer_tail          CER of the last ~30 s, null on clips shorter than that
    stt_transcript    what the STT model heard, so every CER above is auditable
    n_stt_check       how many readings the clip needed (null unless it was rechecked)
    stt_model         which model produced them, spelled out rather than an API alias
    cfg_alpha         TTS guidance the shipped audio was drawn at
    regenerated       true if stage 06 replaced this clip
    qc_status         improved / exhausted / original (see below)

``qc_status`` is ``improved`` when the clip was flagged and a better draw replaced it,
``exhausted`` when it was flagged and no draw was good enough (the ORIGINAL audio ships,
so these are the known-weak clips), and ``original`` when it never tripped a gate.

Idempotent: re-run it after a rescore or another improvement run and it simply re-copies
the current values. It refuses to run while a stage-06 pass is alive, since that pass
rewrites its own output file wholesale at every flush.

This file was written with Claude Code.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "utils"))
from nemo_manifest import read_jsonl, write_jsonl  # noqa: E402

# The STT side of the corpus, named as it actually is rather than as the OpenAI-compatible
# alias the client sends (`whisper-1`). Confirmed against the serving endpoint itself,
# which reports `{"id": "/cache/ggml-large-v3-turbo.bin", "backend": "whisper"}`, and
# against the CrispASR compose that starts it (`--hf-repo
# ggerganov/whisper.cpp:ggml-large-v3-turbo.bin`).
STT_MODEL = "whisper.cpp/ggml-large-v3-turbo"
# Key under which stage 06 stored that model's readings (the alias, again).
STT_KEY = "whisper-1"

# Stage 06's scored outputs. Each was produced from one manifest, so its rows' relative
# audio paths resolve against that manifest's directory.
HOTFIX_SOURCES = (
    ("../06_hotfixes/improved/full.stt.jsonl", "data/NeMO_files"),
    ("../06_hotfixes/improved_parrot/full.stt.jsonl", "data/NeMO_files/PARROT"),
)
# Every manifest the release path reads or rebuilds from, top level and per dataset.
MANIFEST_GLOBS = ("data/NeMO_files/*.jsonl", "data/NeMO_files/*/*.jsonl")

QC_FIELDS = ("cer", "cer_tail", "stt_transcript", "n_stt_check", "stt_model",
             "cfg_alpha", "regenerated", "qc_status")


PASS_SCRIPT = "01_recursive_improvement.py"


def is_pass_cmdline(cmdline: str) -> bool:
    """True when this /proc cmdline IS the stage-06 pass, not merely a mention of it.

    /proc cmdlines are NUL-separated, so the pass itself always has one ARGUMENT whose
    basename is the script (`uv run .../01_recursive_improvement.py --mode stt`). Anything
    that just names it carries the name inside a longer argument instead: a shell running
    `-c '... 01_recursive_improvement.py ...'`, a `grep`, or a wait loop polling
    `pgrep -f "01_recursive_improvement.py --mode stt"`. Substring matching counted all of
    those as a live pass, and a wait loop matches its OWN pattern, so one stale loop
    blocked this script indefinitely with no pass running anywhere.
    """
    return any(os.path.basename(arg) == PASS_SCRIPT for arg in cmdline.split("\0"))


def running_passes() -> list[int]:
    """PIDs of any live stage-06 pass, which would overwrite what we read."""
    me = os.getpid()
    pids: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        if is_pass_cmdline(cmdline):
            pids.append(int(entry.name))
    return sorted(pids)


def resolve(base: Path, rel: str) -> str:
    """A row's audio path as one absolute string, so manifests that store it relative to
    different directories still join onto each other."""
    return os.path.normpath(os.path.join(base, rel))


def load_hotfix(root: Path, stt_model: str) -> dict[str, dict]:
    """Everything stage 06 knows, keyed by absolute audio path."""
    index: dict[str, dict] = {}
    for rel_source, rel_base in HOTFIX_SOURCES:
        source = (root / rel_source).resolve()
        if not source.is_file():
            logger.warning(f"{source} not found, skipping (was that dataset never run?)")
            continue
        base = root / rel_base
        n = pending = 0
        for row in read_jsonl(source):
            reading = (row.get("transcriptions") or {}).get(STT_KEY) or {}
            imp = row.get("improvement") or {}
            status = imp.get("status")
            index[resolve(base, row["audio_filepath"])] = {
                "duration": row.get("duration"),
                "cer": reading.get("cer"),
                "cer_tail": reading.get("cer_tail"),
                "stt_transcript": (reading.get("text") or "").strip() or None,
                "n_stt_check": reading.get("n_stt_check"),
                "stt_model": stt_model,
                "cfg_alpha": row.get("cfg_alpha"),
                "regenerated": status == "improved",
                # pending_* means a run was interrupted mid-flight: the clip is whatever
                # is on disk right now, which is the original, so it ships as one.
                "qc_status": status if status in ("improved", "exhausted") else "original",
            }
            n += 1
            if status in ("pending_tts", "pending_stt"):
                pending += 1
        logger.info(f"{source.name}: {n} scored rows (base {rel_base})")
        if pending:
            # These ship their ORIGINAL audio and are labelled `original`, which is true
            # of the audio but hides that the clip is known-weak: stage 06 flagged it and
            # was interrupted before resolving it. Finish the improvement run first, or
            # the release calls a bad clip a clean one.
            logger.warning(
                f"{source.name}: {pending} row(s) are still mid-flight (pending_tts / "
                f"pending_stt) and would ship as `original`. Finish the stage-06 run "
                f"before releasing, or they lose their `exhausted` label."
            )
    return index


def sync_manifest(path: Path, hotfix: dict[str, dict], apply: bool) -> tuple[int, int, int, int]:
    """Returns (rows, matched, duration changes, field changes) for one manifest."""
    rows = read_jsonl(path)
    matched = redur = changed = 0
    for row in rows:
        rel = row.get("audio_filepath")
        if rel is None:
            continue
        qc = hotfix.get(resolve(path.parent, rel))
        if qc is None:
            continue
        matched += 1
        touched = False
        if qc["duration"] is not None and abs((row.get("duration") or 0.0) - qc["duration"]) > 5e-4:
            row["duration"] = qc["duration"]
            redur += 1
            touched = True
        for key in QC_FIELDS:
            # Written even when the value is null (a clip too short for a tail score, a
            # clip never rechecked), so every row declares the same keys and the parquet
            # schema is not inferred from whichever rows happen to have them.
            if key not in row or row[key] != qc[key]:
                row[key] = qc[key]
                touched = True
        changed += int(touched)
    if apply and changed:
        write_jsonl(path, rows)
    return len(rows), matched, redur, changed


@click.command()
@click.argument("manifests", nargs=-1, type=click.Path(path_type=Path))
@click.option("--root", default=str(_HERE), show_default="99_hf_release/",
              help="Release stage root, holding the `data` symlink.")
@click.option("--stt-model", default=STT_MODEL, show_default=True,
              help="Model name written to every row. The default is what the serving "
                   "endpoint reports, not the `whisper-1` alias the client sends.")
@click.option("--apply", is_flag=True, help="Write. Without it this is a dry run.")
def main(manifests: tuple[Path, ...], root: str, stt_model: str, apply: bool) -> None:
    """Refresh manifest durations and attach stage 06's QC columns."""
    root_path = Path(root).resolve()
    if apply:
        pids = running_passes()
        if pids:
            raise SystemExit(
                f"a stage-06 pass is running (pid {pids}); let it finish first, or it "
                "will keep writing the very file this reads"
            )

    hotfix = load_hotfix(root_path, stt_model)
    if not hotfix:
        raise SystemExit("no stage-06 output found, nothing to sync")
    logger.info(f"{len(hotfix)} scored clips, {sum(1 for q in hotfix.values() if q['regenerated'])} "
                f"regenerated, {sum(1 for q in hotfix.values() if q['qc_status'] == 'exhausted')} exhausted")

    targets = [Path(m) for m in manifests]
    if not targets:
        for pattern in MANIFEST_GLOBS:
            targets.extend(sorted(root_path.glob(pattern)))
    total_rows = total_matched = total_changed = 0
    for path in targets:
        if not path.is_file() or path.name.startswith("."):
            continue
        rows, matched, redur, changed = sync_manifest(path, hotfix, apply)
        total_rows += rows
        total_matched += matched
        total_changed += changed
        gap = rows - matched
        logger.info(
            f"{path.relative_to(root_path)}: {rows} rows, {matched} matched, "
            f"{redur} duration(s) refreshed, {changed} row(s) updated"
            + (f", {gap} WITHOUT a stage-06 score" if gap else "")
        )
    logger.info(f"total: {total_matched}/{total_rows} rows matched, {total_changed} updated")
    if not apply:
        logger.warning("dry run, nothing written (pass --apply)")


if __name__ == "__main__":
    main()
