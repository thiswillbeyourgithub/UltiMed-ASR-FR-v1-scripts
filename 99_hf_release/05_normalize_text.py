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
  ``%`` there: ``95 pour cent`` and ``95 pourcent`` both become ``95 %`` (the space
  before ``%`` is kept, which is French typography and what the ~11.6k rows that
  already used ``%`` do). UltiMed v1 mixed the three spellings (16.6k ``pour cent``,
  0.4k ``pourcent``, 11.6k ``%``), so the model learnt no consistent form.
- ``asr_training_source`` is what the TTS READ. There ``pour cent`` becomes the
  single-word ``pourcent``, the form ``voxtral_normalize`` should produce. The audio
  itself is unchanged: both spellings are spoken identically, so this only makes the
  shipped source text consistent with its audio.

"pour cent" is only a percentage after a quantity. The rewrite skips it when the
preceding word is not a number (``106 garçons pour cent filles``, ``2 grammes pour
cent grammes``) and when it is followed by what "cent" is counting (``pour cent
millilitres``, ``3 cas pour cent mille habitants``, ``1,70 m pour cent vingt
kilogrammes``). ``6 volumes pour cent`` and ``0,10 gramme pour cent`` are also left
alone: they are the vol% and g% concentration units, and ``volumes %`` would be no
better a label. Rows left with a "pour cent" are reported so they can be eyeballed
(26 in UltiMed v1, all of the kinds above).

``text`` also goes through ``ParakeetTokenizer.clean_label`` (``utils/parakeet_tokenizer.py``),
which collapses whitespace runs (6 v1 labels held a raw newline, an artefact of the source
document) and replaces every character the Parakeet vocab does not cover AS WRITTEN by
its NFKC form: ``CO₂`` -> ``CO2``, ``Brª`` -> ``Bra``, a decomposed ``c`` + combining
cedilla -> ``ç``, with trademark-like signs dropped (``CUBE™`` -> ``CUBE``). None of these
ever produced ``<unk>`` (SentencePiece NFKC-folds them at training time), but the label
then showed a character the model is never taught. The text generators now apply the
same method when they parse the LLM output, so a fresh generation no longer needs this
step for it; 05 applies it to the v1 manifests, which were generated before that fix.
``asr_training_source`` is left alone there: the TTS read it, and Parakeet never does.

Every manifest is rewritten independently, like 04, and the run is idempotent.

This file was written by Claude Code.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "utils"))
from nemo_manifest import read_jsonl, write_jsonl  # noqa: E402
from parakeet_tokenizer import ParakeetTokenizer  # noqa: E402

# Same manifest set 03 and 04 process.
MANIFEST_GLOBS = ("data/NeMO_files/*.jsonl", "data/NeMO_files/*/*.jsonl")

# French number words that can end the quantity before "pour cent". Hyphenated
# compounds (quatre-vingt-dix-sept) match on their last part; "demi" covers
# "sept et demi pour cent", "quelques" covers "quelques pour cent".
_NUM_WORD = (r"(?:z[ée]ro|une?|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze"
             r"|treize|quatorze|quinze|seize|vingts?|trente|quarante|cinquante|soixante"
             r"|cents?|mille|demi|quelques)")
# What "cent" counts when "pour cent" means "per hundred X" rather than "percent"
# (pour cent grammes, pour cent millilitres, pour cent filles, pour cent mille).
_COUNTED = (r"(?:\w*grammes?|\w*litres?|\w*mètres?|filles|femmes|garçons|hommes|habitants"
            r"|patients|personnes|naissances|enfants|cas|mille|millions?)")
# "cent" completed by a number word is a body weight (un mètre soixante-dix pour cent
# vingt kilogrammes). Only kilograms qualify: after a percentage, a number word plus a
# unit is the dose that follows it (glucosé à cinq pour cent un litre, Fungizone 10
# pour cent quarante millilitres), which must still become %.
_WEIGHT = rf"{_NUM_WORD}(?:-\w+)*\s+kilo(?:gramme)?s?"
_PERCENT_RE = re.compile(
    rf"(\d|\b{_NUM_WORD})(\s+)(?:pour\s+cent|pourcent)\b"
    rf"(?!\s+(?:{_COUNTED}|{_WEIGHT})\b)",
    re.IGNORECASE,
)
_LEFTOVER_RE = re.compile(r"\bpour\s+cent\b|\bpourcent\b", re.IGNORECASE)

def percent_to_symbol(text: str) -> str:
    """ASR label form: ``95 pour cent`` / ``95 pourcent`` -> ``95 %``."""
    return _PERCENT_RE.sub(r"\1\2%", text)


def percent_to_one_word(text: str) -> str:
    """TTS source form: ``95 pour cent`` -> ``95 pourcent``."""
    return _PERCENT_RE.sub(r"\1\2pourcent", text)


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
            if _LEFTOVER_RE.search(new["text"]):
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
