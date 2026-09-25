#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru"]
# ///
"""Drop the stage-06 ``exhausted`` clips from every NeMo manifest of the release.

Run AFTER ``03_sync_hotfix_results.py --apply`` (which is what writes ``qc_status``
into the manifests) and BEFORE ``scripts/build_parquet.py``:

    uv run 04_drop_exhausted.py            # dry run, reports what would be dropped
    uv run 04_drop_exhausted.py --apply

A clip is ``exhausted`` when stage 06 flagged it (its STT reading disagreed too much
with its label) and no regenerated draw was good enough, so the ORIGINAL, known-bad
audio is still what sits on disk. UltiMed v1 shipped those rows, which was a mistake:
a clip whose audio does not say its label is label noise in train and a wrong
reference in val/test. This removes them from the manifests instead of patching the
audio, because by definition no good audio exists for them.

Every manifest is filtered independently, including the down-sampled ``*.down-N``
fixtures, since each is a copy of rows and the parquet / trainer read them directly.
Rows are matched on their own ``qc_status`` field, not by joining on audio path
against another manifest, so a manifest that was never synced (no ``qc_status`` at
all) is reported rather than silently treated as clean.

The ``.flac`` files are NOT deleted: nothing references them once the manifests are
filtered (``build_parquet.py`` embeds only manifest rows), and keeping them lets a
later stage-06 retry still reach them.

Idempotent: a second run finds nothing left to drop.

This file was written by Claude Code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "utils"))
from nemo_manifest import read_jsonl, write_jsonl  # noqa: E402

# Same manifest set 03_sync_hotfix_results.py refreshes, so every manifest that got a
# `qc_status` is also one this filters.
MANIFEST_GLOBS = ("data/NeMO_files/*.jsonl", "data/NeMO_files/*/*.jsonl")

EXHAUSTED = "exhausted"


def filter_rows(rows: list[dict]) -> tuple[list[dict], list[dict], int]:
    """Split manifest rows into kept and dropped ones.

    Parameters
    ----------
    rows : list[dict]
        NeMo manifest rows, as read by ``read_jsonl``.

    Returns
    -------
    kept : list[dict]
        Rows whose ``qc_status`` is anything but ``exhausted`` (order preserved).
    dropped : list[dict]
        The ``exhausted`` rows.
    n_unsynced : int
        How many rows carry no ``qc_status`` key at all, i.e. were never touched by
        ``03_sync_hotfix_results.py``: their status is unknown, so they are kept, but
        the caller should warn since an exhausted clip could hide among them.
    """
    kept: list[dict] = []
    dropped: list[dict] = []
    n_unsynced = 0
    for row in rows:
        if "qc_status" not in row:
            n_unsynced += 1
        if row.get("qc_status") == EXHAUSTED:
            dropped.append(row)
        else:
            kept.append(row)
    return kept, dropped, n_unsynced


@click.command()
@click.argument("manifests", nargs=-1, type=click.Path(path_type=Path))
@click.option("--root", default=str(_HERE), show_default="99_hf_release/",
              help="Release stage root, holding the `data` symlink.")
@click.option("--apply", is_flag=True, help="Write. Without it this is a dry run.")
def main(manifests: tuple[Path, ...], root: str, apply: bool) -> None:
    """Remove stage-06 exhausted clips from the release manifests."""
    root_path = Path(root).resolve()
    targets = [Path(m) for m in manifests]
    if not targets:
        for pattern in MANIFEST_GLOBS:
            targets.extend(sorted(root_path.glob(pattern)))
    if not targets:
        raise SystemExit(f"no manifest found under {root_path} (is the `data` symlink mounted?)")

    total_dropped = 0
    for path in targets:
        if not path.is_file() or path.name.startswith("."):
            continue
        rows = read_jsonl(path)
        kept, dropped, n_unsynced = filter_rows(rows)
        label = path.relative_to(root_path) if path.is_relative_to(root_path) else path
        if n_unsynced:
            logger.warning(f"{label}: {n_unsynced}/{len(rows)} rows WITHOUT qc_status, run "
                           f"03_sync_hotfix_results.py --apply first or they cannot be judged")
        if not dropped:
            continue
        total_dropped += len(dropped)
        logger.info(f"{label}: dropping {len(dropped)}/{len(rows)} exhausted rows")
        for row in dropped:
            logger.info(f"    {row.get('audio_filepath')} (cer={row.get('cer')})")
        if apply:
            write_jsonl(path, kept)

    logger.info(f"total: {total_dropped} exhausted rows across all manifests")
    if total_dropped and not apply:
        logger.warning("dry run, nothing written (pass --apply)")
    elif total_dropped:
        logger.info("now rebuild the parquet: uv run scripts/build_parquet.py")


if __name__ == "__main__":
    main()
