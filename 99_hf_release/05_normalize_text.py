#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru"]
# ///
"""Normalize the transcripts of every NeMo manifest of the release.

Run AFTER ``04_drop_bad_rows.py --apply`` and BEFORE ``scripts/build_parquet.py``:

    uv run 05_normalize_text.py            # dry run, reports counts + examples
    uv run 05_normalize_text.py --apply

Each row carries two texts, and they are normalized differently because they serve
different readers:

- ``text`` is the ASR label, what the model learns to WRITE. A percentage is written
  ``%`` there: ``95 pour cent`` and ``95 pourcent`` both become ``95 %``. UltiMed v1
  mixed the three spellings (16.6k ``pour cent``, 0.4k ``pourcent``, 11.6k ``%``), so
  the model learnt no consistent form.
- ``asr_training_source`` is what the TTS READ. There ``pour cent`` becomes the
  single-word ``pourcent``. The audio itself is unchanged: both spellings are spoken
  identically, so this only makes the shipped source text consistent with its audio.

Both rewrites live in ``utils/percent_normalize.py``, which documents the guards that
keep a "per hundred X" (``106 garçons pour cent filles``, ``pour cent millilitres``) and
the vol% / g% units. The text generators apply the label rewrite themselves when they
parse the LLM output, so a fresh generation already writes ``%``; 05 exists for the v1
manifests, generated before that. Rows left with a "pour cent" are reported so they can
be eyeballed (26 in UltiMed v1, all "per hundred X" or units).

``text`` also goes through ``ParakeetTokenizer.clean_label`` (``utils/parakeet_tokenizer.py``),
which collapses whitespace runs (6 v1 labels held a raw newline, an artefact of the source
document) and replaces every character the Parakeet vocab does not cover AS WRITTEN by
its NFKC form: ``CO₂`` -> ``CO2``, ``Brª`` -> ``Bra``, a decomposed ``c`` + combining
cedilla -> ``ç``, with trademark-like signs dropped (``CUBE™`` -> ``CUBE``). None of these
ever produced ``<unk>`` (SentencePiece NFKC-folds them at training time), but the label
then showed a character the model is never taught. The generators apply it at parse
time too.
``asr_training_source`` is left alone there: the TTS read it, and Parakeet never does.

Every manifest is rewritten independently, like 04, and the run is idempotent.

This file was written by Claude Code.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "utils"))
from nemo_manifest import read_jsonl, write_jsonl  # noqa: E402
from parakeet_tokenizer import ParakeetTokenizer  # noqa: E402
from percent_normalize import LEFTOVER_RE, percent_to_one_word, percent_to_symbol  # noqa: E402

# Same manifest set 03 and 04 process.
MANIFEST_GLOBS = ("data/NeMO_files/*.jsonl", "data/NeMO_files/*/*.jsonl")

def normalize_row(row: dict, tok: ParakeetTokenizer) -> dict:
    """Return a copy of ``row`` with ``text`` and ``asr_training_source`` normalized.

    ``tok`` is a ``ParakeetTokenizer``; its ``clean_label`` checks coverage raw, so
    its ``nfkc`` setting does not matter here.
    """
    out = dict(row)
    out["text"] = percent_to_symbol(tok.clean_label(row["text"]))
    if "asr_training_source" in row:
        # One space between words here too, without clean_label's character fold.
        out["asr_training_source"] = " ".join(
            percent_to_one_word(row["asr_training_source"]).split())
    return out


@click.command()
@click.argument("manifests", nargs=-1, type=click.Path(path_type=Path))
@click.option("--root", default=str(_HERE), show_default="99_hf_release/",
              help="Release stage root, holding the `data` symlink.")
@click.option("--apply", is_flag=True, help="Write. Without it this is a dry run.")
def main(manifests: tuple[Path, ...], root: str, apply: bool) -> None:
    """Normalize percent spellings, whitespace and uncovered characters in the manifests."""
    root_path = Path(root).resolve()
    targets = [Path(m) for m in manifests]
    if not targets:
        for pattern in MANIFEST_GLOBS:
            targets.extend(sorted(root_path.glob(pattern)))
    if not targets:
        raise SystemExit(f"no manifest found under {root_path} (is the `data` symlink mounted?)")

    tok = ParakeetTokenizer(nfkc=False)
    for path in targets:
        if not path.is_file() or path.name.startswith("."):
            continue
        label = path.relative_to(root_path) if path.is_relative_to(root_path) else path
        rows = read_jsonl(path)
        new_rows = [normalize_row(r, tok) for r in rows]
        changed = Counter()
        leftovers, still_bad = [], Counter()
        for old, new in zip(rows, new_rows):
            for key in ("text", "asr_training_source"):
                if old.get(key) != new.get(key):
                    changed[key] += 1
            if LEFTOVER_RE.search(new["text"]):
                leftovers.append(new["text"])
            for ch in tok.offending_chars(new["text"]):
                still_bad[ch] += 1
        if not changed:
            continue
        logger.info(f"{label}: {len(rows)} rows, text changed in {changed['text']}, "
                    f"asr_training_source changed in {changed['asr_training_source']}")
        if leftovers:
            logger.info(f"    {len(leftovers)} labels keep a non-percent 'pour cent', e.g. "
                        f"{leftovers[0][:120]!r}")
        if still_bad:
            logger.warning(f"    still uncovered after the fix: {dict(still_bad)}")
        if apply:
            write_jsonl(path, new_rows)
    if not apply:
        logger.warning("dry run, nothing written (pass --apply)")
    else:
        logger.info("now rebuild the parquet: uv run scripts/build_parquet.py")


if __name__ == "__main__":
    main()
