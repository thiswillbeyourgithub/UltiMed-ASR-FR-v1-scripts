#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru"]
# ///
"""Drop the rows that must not ship from every NeMo manifest of the release.

Run AFTER ``03_sync_hotfix_results.py --apply`` (which is what writes ``qc_status``
into the manifests) and BEFORE ``05_normalize_text.py`` and ``scripts/build_parquet.py``:

    uv run 04_drop_bad_rows.py            # dry run, reports what would be dropped
    uv run 04_drop_bad_rows.py --apply

Two kinds of rows are dropped:

1. **Exhausted clips.** A clip is ``exhausted`` when stage 06 flagged it (its STT
   reading disagreed too much with its label) and no regenerated draw was good
   enough, so the ORIGINAL, known-bad audio is still what sits on disk. UltiMed v1
   shipped those rows, which was a mistake: a clip whose audio does not say its label
   is label noise in train and a wrong reference in val/test. They are removed rather
   than patched because by definition no good audio exists for them. Matched on the
   row's own ``qc_status`` field, not by joining against another manifest, so a
   manifest that was never synced (no ``qc_status`` at all) is reported rather than
   silently treated as clean.

2. **Eval clips whose text is also a training text.** The release-wide split
   (``02_combine_nemo_manifests.py``) groups rows by document / term, not by text, so
   two PARHAF documents that happen to contain the same sentence can put it in train
   AND in val/test (UltiMed v1 had one such sentence in val and one in test, differing
   only in punctuation or case). The eval side is dropped, since train loses nothing
   and the eval set would otherwise score a sentence the model was trained on.
   Texts are compared after ``normalize_text`` (NFKC, lowercase, punctuation to space,
   whitespace collapsed), the same key the trainer's ``data_leak_check`` uses (see
   below). Fixing it here rather than in the splitter keeps the v1 split stable: a
   splitter change would reshuffle every row.

A dropped clip is dropped from EVERY manifest (matched on its resolved audio path),
including the per-source ``<source>/*.jsonl`` partitions and the down-sampled
``*.down-N`` fixtures, since each is a copy of rows and the parquet / trainer read
them directly. The ``.flac`` files are NOT deleted: nothing references them once the
manifests are filtered (``build_parquet.py`` embeds only manifest rows), and keeping
them lets a later stage-06 retry still reach them.

Idempotent: a second run finds nothing left to drop.

This file was written by Claude Code.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from collections.abc import Collection
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "utils"))
from nemo_manifest import read_jsonl, resolve_audio, write_jsonl  # noqa: E402

# Same manifest set 03_sync_hotfix_results.py refreshes, so every manifest that got a
# `qc_status` is also one this filters.
MANIFEST_GLOBS = ("data/NeMO_files/*.jsonl", "data/NeMO_files/*/*.jsonl")

# The release-wide split the parquet and the trainer read. PARROT is eval-only and
# lives outside it, but it must not repeat a training text either.
TRAIN_MANIFEST = "data/NeMO_files/train.jsonl"
EVAL_MANIFESTS = ("data/NeMO_files/val.jsonl", "data/NeMO_files/test.jsonl",
                  "data/NeMO_files/PARROT/full.jsonl")

EXHAUSTED = "exhausted"
DUPLICATE = "duplicate-of-train"

_PUNCT = re.compile(r"[^\w\s]")


def normalize_text(text: str) -> str:
    """Comparison key for "same sentence": NFKC, lowercase, punctuation to space,
    whitespace collapsed, so ``Sur le plan cardiovasculaire, il n'y a pas...`` and
    ``sur le plan cardiovasculaire il n'y a pas...`` collide.

    Deliberate cross-repo copy of ``normalize_text`` in the NeMo fork's
    ``nemo/collections/asr/parts/utils/data_leak_check.py`` (a separate repo, so it
    cannot be imported). Keep the two identical: if this one is weaker, the trainer
    refuses a release this script called clean.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(_PUNCT.sub(" ", text).split())


def eval_duplicates(train_rows: list[dict],
                    eval_manifests: dict[Path, list[dict]]) -> dict[str, str]:
    """Find the eval rows whose normalized text also appears in train.

    Parameters
    ----------
    train_rows : list[dict]
        Rows of the release-wide train manifest.
    eval_manifests : dict[Path, list[dict]]
        Eval manifest path -> its rows (the path resolves relative audio paths).

    Returns
    -------
    dict[str, str]
        Resolved audio path of each eval clip to drop -> its text, for the log.
    """
    train_keys = {normalize_text(r["text"]) for r in train_rows}
    return {resolve_audio(path.parent, r["audio_filepath"]): r["text"]
            for path, rows in eval_manifests.items() for r in rows
            if normalize_text(r["text"]) in train_keys}


def filter_rows(rows: list[dict], manifest_dir: Path,
                drop_audio: Collection[str] = ()) -> tuple[list[dict], list[tuple[dict, str]], int]:
    """Split manifest rows into kept and dropped ones.

    Parameters
    ----------
    rows : list[dict]
        NeMo manifest rows, as read by ``read_jsonl``.
    manifest_dir : Path
        Directory of the manifest, which relative ``audio_filepath`` values resolve against.
    drop_audio : Collection[str]
        Resolved audio paths to drop whatever their status (from ``eval_duplicates``).

    Returns
    -------
    kept : list[dict]
        Rows that stay (order preserved).
    dropped : list[tuple[dict, str]]
        Each dropped row with its reason, ``EXHAUSTED`` or ``DUPLICATE``.
    n_unsynced : int
        How many rows carry no ``qc_status`` key at all, i.e. were never touched by
        ``03_sync_hotfix_results.py``: their status is unknown, so they are kept, but
        the caller should warn since an exhausted clip could hide among them.
    """
    kept: list[dict] = []
    dropped: list[tuple[dict, str]] = []
    n_unsynced = 0
    for row in rows:
        if "qc_status" not in row:
            n_unsynced += 1
        if row.get("qc_status") == EXHAUSTED:
            dropped.append((row, EXHAUSTED))
        elif drop_audio and resolve_audio(manifest_dir, row["audio_filepath"]) in drop_audio:
            dropped.append((row, DUPLICATE))
        else:
            kept.append(row)
    return kept, dropped, n_unsynced


@click.command()
@click.argument("manifests", nargs=-1, type=click.Path(path_type=Path))
@click.option("--root", default=str(_HERE), show_default="99_hf_release/",
              help="Release stage root, holding the `data` symlink.")
@click.option("--apply", is_flag=True, help="Write. Without it this is a dry run.")
def main(manifests: tuple[Path, ...], root: str, apply: bool) -> None:
    """Remove exhausted clips and eval duplicates of training texts from the release manifests."""
    root_path = Path(root).resolve()
    targets = [Path(m) for m in manifests]
    if not targets:
        for pattern in MANIFEST_GLOBS:
            targets.extend(sorted(root_path.glob(pattern)))
    if not targets:
        raise SystemExit(f"no manifest found under {root_path} (is the `data` symlink mounted?)")

    duplicates = eval_duplicates(
        read_jsonl(root_path / TRAIN_MANIFEST),
        {root_path / m: read_jsonl(root_path / m) for m in EVAL_MANIFESTS if (root_path / m).is_file()},
    )
    for audio, text in duplicates.items():
        logger.info(f"eval clip repeats a training text: {audio} ({text[:80]!r})")

    total_dropped = 0
    for path in targets:
        if not path.is_file() or path.name.startswith("."):
            continue
        rows = read_jsonl(path)
        kept, dropped, n_unsynced = filter_rows(rows, path.parent, duplicates)
        label = path.relative_to(root_path) if path.is_relative_to(root_path) else path
        if n_unsynced:
            logger.warning(f"{label}: {n_unsynced}/{len(rows)} rows WITHOUT qc_status, run "
                           f"03_sync_hotfix_results.py --apply first or they cannot be judged")
        if not dropped:
            continue
        total_dropped += len(dropped)
        logger.info(f"{label}: dropping {len(dropped)}/{len(rows)} rows")
        for row, reason in dropped:
            logger.info(f"    {reason}: {row.get('audio_filepath')} (cer={row.get('cer')})")
        if apply:
            write_jsonl(path, kept)

    logger.info(f"total: {total_dropped} rows dropped across all manifests")
    if total_dropped and not apply:
        logger.warning("dry run, nothing written (pass --apply)")
    elif total_dropped:
        logger.info("next: uv run 05_normalize_text.py --apply, then uv run scripts/build_parquet.py")


if __name__ == "__main__":
    main()
