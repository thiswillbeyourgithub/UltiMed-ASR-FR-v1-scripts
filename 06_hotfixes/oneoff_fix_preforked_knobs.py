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
"""ONE-OFF migration: make the recorded ``seed`` / ``cfg_alpha`` of already-improved
clips describe the audio that actually exists, not the request the client sent.

Run it once, from this folder, with no improvement pass running:

    uv run oneoff_fix_preforked_knobs.py            # dry run, reports what it would do
    uv run oneoff_fix_preforked_knobs.py --apply    # rewrites the files

Why it is needed
----------------
Until 2026-07-31 the two per-draw knobs were sent to a server that could not read
them, and nothing errored:

* ``cfg_alpha`` went nested under ``extra_params``. Upstream vllm-omni's
  ``OpenAICreateSpeechRequest`` is a pydantic model with the default
  ``extra='ignore'``, so the whole field was dropped before anything looked at it.
  Every draw therefore ran at the server's STARTUP guidance, ``VOXTRAL_CFG_ALPHA``,
  which the CrispASR compose has pinned at 1.3 since 2026-07-06.
* ``seed`` went flat, where it reached vLLM's sampler. That is inert for this model:
  stage 0 is fed ``fake_logits_for_audio_tokens()``, so no sampling decision depends
  on it, and the flow-matching noise is drawn from the global torch RNG. Draws
  differed only through RNG drift.

So a stored ``cfg_alpha`` of 1.4 to 1.7 and a stored ``seed`` of 43 to 47 record what
the client ASKED for. The audio on disk was made at 1.3, unseeded. The forked
vllm-omni the ``voxtral-tts`` image now builds from (CrispASR ``b5ff2f3``, image built
2026-07-31 18:26 UTC) routes both knobs for real, and the client sends them flat since
``5aacb18``, so anything DRAWN after that is recorded correctly and is left alone here.
Mind the difference between when a draw was made and when it was recorded: see the
``FORK_CUTOFF_ISO`` note below, which is why the default cutoff is not the image build.

What it writes
--------------
For every clip in ``<name>.improved.jsonl`` whose win predates the cutoff:

* ``cfg_alpha`` -> ``--server-cfg-alpha`` (1.3), in the audit record, in the row's
  top-level ``cfg_alpha`` column and in the row's ``improvement`` marker;
* ``seed`` -> ``null``, same three places. Not the draw index it used to hold: every
  draw of one redo went out with identical effective parameters, so which of them won
  says nothing about how to reproduce it, and the clip is not reproducible anyway;
* ``knobs_effective: false`` on the audit record, so the audit still says why.

The rewrite is idempotent (a second run finds nothing to do) and atomic (temp file +
``os.replace``, reusing ``01_compute_stt.atomic_write_jsonl``). ``--backup`` copies the
scored file next to itself first; delete the copy once you are satisfied.

This file was written with Claude Code.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent

# Reuse the improvement script's io helpers (and, through it, the scorer's atomic
# writer) instead of copying them: the audit file's key rule lives in exactly one place.
_spec = importlib.util.spec_from_file_location(
    "improve_mod", _HERE / "01_recursive_improvement.py"
)
_imp = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_imp)  # type: ignore[union-attr]
_stt = _imp._stt

# The cutoff is on the WIN's timestamp, but what matters is when the AUDIO was drawn, and
# those are not the same moment: the tts pass persists candidates and a later stt pass
# scores and promotes them. The run interrupted at 18:27 +0200 left 60 candidates already
# synthesized against the old server; the restarted driver scored them at 23:27 to 23:28
# and recorded the OLD ladder (cfg_alpha 1.3 to 1.7) on pre-fork audio. The first fresh
# post-fork draws were promoted at 00:23 +0200 with the new ladder (1.0 / 1.5 / 2.0 / 2.5
# / 3.0), and the two clusters do not overlap, so the cutoff sits between them rather than
# at the image build (2026-07-31 20:26 +0200, CrispASR b5ff2f3).
FORK_CUTOFF_ISO = "2026-07-31T22:00:00+00:00"
# docker-compose.yml: VOXTRAL_CFG_ALPHA=1.3, unchanged since 2026-07-06 (79960e4). This
# is what stage 0 actually used on every pre-fork draw, whatever the request said.
SERVER_CFG_ALPHA = 1.3

DEFAULT_FILES = ("improved/full.stt.jsonl", "improved_parrot/full.stt.jsonl")


def running_passes() -> list[int]:
    """PIDs of any live scoring / improvement pass. Rewriting these files underneath one
    would lose the rewrite: a pass loads the whole dataset at start and writes it back
    wholesale at every flush, so its in-memory copy would win the race."""
    me = os.getpid()
    pids: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        if "01_recursive_improvement.py" in cmdline or "01_compute_stt.py" in cmdline:
            pids.append(int(entry.name))
    return sorted(pids)


def audit_path_for(stt_path: Path) -> Path:
    """The ``.improved.jsonl`` beside a ``.stt.jsonl``, named as the pass names it."""
    return stt_path.with_name(stt_path.name.replace(".stt.jsonl", "") + ".improved.jsonl")


def stale_knobs(rec: dict, server_cfg_alpha: float) -> bool:
    """Does this record still describe the request rather than the audio?"""
    return rec.get("seed") is not None or rec.get("cfg_alpha") != server_cfg_alpha


def fix_audit(records: dict[str, dict], cutoff: float, server_cfg_alpha: float
              ) -> tuple[set[str], int, int]:
    """Rewrite the pre-cutoff records in place. Returns (affected keys, changed, kept)."""
    affected: set[str] = set()
    changed = post = 0
    for key, rec in records.items():
        ts = rec.get("ts") or rec.get("first_ts")
        if ts is None:
            logger.warning(f"no timestamp on {key}, leaving it alone")
            continue
        if ts >= cutoff:
            post += 1
            continue
        affected.add(key)
        if not stale_knobs(rec, server_cfg_alpha):
            continue
        rec["seed"] = None
        rec["cfg_alpha"] = server_cfg_alpha
        rec["knobs_effective"] = False
        changed += 1
    return affected, changed, post


def fix_rows(rows: list[dict], affected: set[str], audio_key: str,
             server_cfg_alpha: float) -> int:
    """Point the scored rows of the affected clips at the same values. Returns the count
    of rows touched."""
    changed = 0
    for row in rows:
        if row.get(audio_key) not in affected:
            continue
        touched = False
        if row.get("cfg_alpha") != server_cfg_alpha:
            row["cfg_alpha"] = server_cfg_alpha
            touched = True
        imp = row.get("improvement") or {}
        # A win can be hidden behind a later marker (an exhausted re-improvement), but
        # the audio on disk is still that draw, so its knobs are corrected either way.
        if "cfg_alpha" in imp and imp.get("cfg_alpha") != server_cfg_alpha:
            imp["cfg_alpha"] = server_cfg_alpha
            touched = True
        if imp.get("seed") is not None:
            imp["seed"] = None
            touched = True
        changed += int(touched)
    return changed


def read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.error(f"unreadable line {i} in {path.name}, aborting")
                raise
    return rows


def verify(stt_path: Path, audio_key: str, server_cfg_alpha: float) -> None:
    """Re-read what was just written and confirm no stale knob survived on an affected
    clip. Cheap insurance on a file that costs GPU-hours to rebuild."""
    records = _imp.load_existing_improved(audit_path_for(stt_path))
    bad_audit = [k for k, r in records.items()
                 if r.get("knobs_effective") is False and stale_knobs(r, server_cfg_alpha)]
    fixed = {k for k, r in records.items() if r.get("knobs_effective") is False}
    bad_rows = []
    for row in read_rows(stt_path):
        key = row.get(audio_key)
        if key not in fixed:
            continue
        imp = row.get("improvement") or {}
        if (row.get("cfg_alpha") != server_cfg_alpha
                or imp.get("seed") is not None
                or ("cfg_alpha" in imp and imp["cfg_alpha"] != server_cfg_alpha)):
            bad_rows.append(key)
    if bad_audit or bad_rows:
        raise SystemExit(
            f"verification FAILED for {stt_path}: {len(bad_audit)} audit records and "
            f"{len(bad_rows)} rows still carry the request values"
        )
    logger.success(f"{stt_path.name}: verified, {len(fixed)} clips describe their audio")


@click.command()
@click.argument("files", nargs=-1, type=click.Path(path_type=Path))
@click.option("--cutoff", default=FORK_CUTOFF_ISO, show_default=True,
              help="ISO timestamp the forked TTS server went live. Wins recorded before "
                   "it are corrected, wins after it are left alone.")
@click.option("--server-cfg-alpha", default=SERVER_CFG_ALPHA, show_default=True,
              help="The startup VOXTRAL_CFG_ALPHA the pre-fork draws really ran at.")
@click.option("--audio-key", default="audio_filepath", show_default=True,
              help="Row key holding the audio path (the join onto the audit file).")
@click.option("--apply", is_flag=True, help="Write. Without it this is a dry run.")
@click.option("--backup/--no-backup", default=True, show_default=True,
              help="Copy each .stt.jsonl to .stt.jsonl.prefork.bak before rewriting it.")
def main(files: tuple[Path, ...], cutoff: str, server_cfg_alpha: float, audio_key: str,
         apply: bool, backup: bool) -> None:
    """Rewrite the recorded per-draw knobs of pre-fork wins to what the server used."""
    targets = [Path(f) for f in files] or [_HERE / name for name in DEFAULT_FILES]
    cutoff_ts = datetime.fromisoformat(cutoff).timestamp()
    logger.info(f"cutoff {cutoff} ({cutoff_ts:.0f}), server cfg_alpha {server_cfg_alpha}")

    if apply:
        pids = running_passes()
        if pids:
            raise SystemExit(
                f"a scoring / improvement pass is running (pid {pids}); stop it first, "
                "or its in-memory copy of the dataset will overwrite this rewrite"
            )

    for stt_path in targets:
        if not stt_path.is_file():
            logger.warning(f"{stt_path} not found, skipping")
            continue
        audit = audit_path_for(stt_path)
        records = _imp.load_existing_improved(audit)
        if not records:
            logger.info(f"{stt_path.parent.name}: no wins recorded, nothing to do")
            continue

        # Sorted defensively: a partly migrated file mixes None seeds with int ones.
        before = sorted({(r.get("cfg_alpha"), r.get("seed")) for r in records.values()},
                        key=lambda pair: tuple((v is None, v) for v in pair))
        affected, changed, post = fix_audit(records, cutoff_ts, server_cfg_alpha)
        logger.info(
            f"{stt_path.parent.name}: {len(records)} wins, {len(affected)} pre-fork "
            f"({changed} to correct), {post} after the cutoff left alone"
        )
        logger.info(f"  recorded (cfg_alpha, seed) pairs before: {before}")
        if not changed:
            logger.info("  already correct, skipping this dataset")
            continue

        rows = read_rows(stt_path)
        n_rows = fix_rows(rows, affected, audio_key, server_cfg_alpha)
        logger.info(f"  {n_rows} of {len(rows)} scored rows would change")
        if not apply:
            logger.warning("  dry run, nothing written (pass --apply)")
            continue

        if backup:
            dest = stt_path.with_name(stt_path.name + ".prefork.bak")
            logger.info(f"  backing up to {dest.name}")
            shutil.copy2(stt_path, dest)
        _stt.atomic_write_jsonl(audit, list(records.values()))
        _stt.atomic_write_jsonl(stt_path, rows)
        verify(stt_path, audio_key, server_cfg_alpha)


if __name__ == "__main__":
    main()
