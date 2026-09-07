#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "click",
#     "loguru",
# ]
# ///
"""Turn the raw PARHAF documents jsonl into coherent longish French text entries.

AUDIT-ONLY as of the LLM rewrite pivot. The dataset text for this stage is now
produced by the rewrite path (``03_chunk_for_rewrite.py`` then
``04_generate_texts.py``), which rewrites each raw document into one faithful
paragraph via the shared engine. This deterministic line cleaner is kept for its
alphanumeric-loss accounting and character audit (which rules cut the most, what
still maps to ``<unk>``); ``02_parhaf_texts.jsonl`` is a diagnostic artefact, no
longer the input to audio.

Reads ``01_parhaf_documents.jsonl`` (produced by ``01_parquet_to_jsonl.py``) and
emits one clean entry per coherent block of text. The cleaning is line based:

1. Split every ``documents.text`` string on newlines, keep only non-empty lines.
2. Normalize each line: fold ``‐`` (U+2010) to an ASCII hyphen, strip
   ``​`` (zero width space), drop any parenthesized span ``(...)``, replace
   ``°`` with "degrés", ``+`` with "plus", ``×`` (and ``*`` between two digits
   or after ``g`` before a digit, and the letter ``x`` inside a cm/mm
   measurement) with "fois", strip ``"``, and spell a ``>``/``<`` sitting right
   before a number or ``à`` as "supérieur"/"inférieur", then collapse the
   whitespace.
3. Drop a line unless it:
   - carries none of the untokenizable ``IGNORE_CHARS`` we do not rewrite,
   - ends with ``.`` or ``?``,
   - contains no ``:`` anywhere (drops labels like "Patient : ..." and
     enumeration headers/intros),
   - is at least MIN_LINE_CHARS long,
   - is not fully uppercase (drops shouting headers),
   - starts with a capital letter followed by a non-capital char (drops
     all-caps headers and dash/number-led list items).
4. Join surviving lines that were *contiguous* (no dropped non-empty line
   between them) with a single space; a dropped line starts a new block.
5. Drop any resulting block shorter than MIN_TEXT_CHARS, one that has two or
   more dots in a row (".."), or one that still carries a Parakeet-
   untokenizable character outside the ``KEEP_CHARS`` whitelist (checked with
   the shared ``parakeet_tokenizer`` coverage).

Each emitted entry gets an id encoding its provenance:
``{original_id}-t{text_index}-s{split_index}`` so it is ordered, unique, and
says which split of which text-list entry of which original document it came
from. Every entry also carries ``category = "parhaf"``.

Written with the help of Claude Code.
"""
import json
import re
import sys
from pathlib import Path

# The shared Parakeet coverage detector lives in utils/; import it rather than
# duplicating a hardcoded list of what maps to <unk>.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from parakeet_tokenizer import ParakeetTokenizer  # noqa: E402
from cleanup_stats import CleanupStats, respace_after_stop, split_blocks  # noqa: E402

SRC = Path("01_parhaf_documents.jsonl")
DST = Path("02_parhaf_texts.jsonl")

# Innermost balanced parenthesized span, applied repeatedly to handle nesting.
_PAREN = re.compile(r"\([^()]*\)")

# A measurement written with the letter "x" as multiplication and ending in a
# cm/mm unit ("20x16mm", "6,5 x 2,5 cm", "5 x 3 x 2 cm"): each "x" is spelled
# "fois". The trailing unit keeps this from touching a stray letter "x".
_MEASURE_X = re.compile(r"\d[\d.,]*(?:\s*[xX]\s*\d[\d.,]*)+\s*(?:cm|mm)\b")

# Hardcoded thresholds, tweak here.
MIN_LINE_CHARS = 15  # drop lines shorter than this before joining
MIN_TEXT_CHARS = 80  # drop joined blocks shorter than this

CATEGORY = "parhaf"

# Untokenizable characters we choose to drop the whole line on rather than
# rewrite: any line carrying one of these is discarded by ``keep_line``.
# ``*``/``>``/``<`` are here too, but ``clean_line`` runs first, so only the
# leftover ones (not spelled out as fois/supérieur/inférieur) reach this.
IGNORE_CHARS = frozenset("~;#±≤≥_&€@$©®⁄^=*<>")

# Untokenizable characters we deliberately keep in the output, even though
# Parakeet has no token for them. Any finished split text that still carries an
# untokenizable char NOT in this set is dropped whole (see ``split_blocks``).
# Adding a glyph here is how you whitelist it. Currently empty: every offender
# is either rewritten in ``clean_line`` or dropped via ``IGNORE_CHARS``.
KEEP_CHARS = frozenset()

_TOK = ParakeetTokenizer()


def drop_text(block: str) -> bool:
    """True if a finished block still carries an untokenizable char we do not keep."""
    return bool(_TOK.offending_chars(block) - KEEP_CHARS)


def clean_line(line: str) -> str:
    """Normalize a raw line: drop parenthesized spans and spell out ``°``."""
    line = line.replace("‐", "-").replace("‑", "-")  # U+2010/U+2011 -> ASCII hyphen
    line = line.replace("​", "")  # ZERO WIDTH SPACE -> removed
    prev = None
    while prev != line:  # repeat to peel nested/adjacent parentheses
        prev = line
        line = _PAREN.sub(" ", line)
    line = line.replace("(", " ").replace(")", " ")  # stray unbalanced parens
    line = line.replace("°", " degrés ")
    # Spoken rewrites; wrapped in spaces so the collapse below re-normalizes
    # whatever surrounded them (e.g. "3+3" and "3 + 3" both -> "3 plus 3").
    line = line.replace("+", " plus ").replace("×", " fois ").replace('"', "")
    # "*" between two digits is multiplication ("20*16" -> "20 fois 16"), and a
    # "*" after "g" before a digit is a dose frequency ("1g*3/j" ->
    # "1g fois 3/j", also catching the "g" of "mg"); a "*" elsewhere (footnote
    # markers) is left as-is (KEEP_CHARS).
    line = re.sub(r"(?<=\d)\s*\*\s*(?=\d)", " fois ", line)
    line = re.sub(r"(?<=g)\s*\*\s*(?=\d)", " fois ", line)
    # letter "x" as multiplication inside a cm/mm measurement -> "fois"
    line = _MEASURE_X.sub(lambda m: re.sub(r"\s*[xX]\s*", " fois ", m.group(0)), line)
    # ">"/"<" right before a number or "à" (optionally across a space) are
    # comparisons; spell them out. The "à" form keeps the "à" it precedes
    # ("< à 1 cm" -> "inférieur à 1 cm"). Other ">"/"<" stay as-is (KEEP_CHARS).
    line = re.sub(r">\s*(?=\d)", " supérieur à ", line)
    line = re.sub(r"<\s*(?=\d)", " inférieur à ", line)
    line = re.sub(r">\s*(?=[àÀ])", " supérieur ", line)
    line = re.sub(r"<\s*(?=[àÀ])", " inférieur ", line)
    line = re.sub(r"\s+", " ", line).strip()
    line = re.sub(r"\s+([.,;?!])", r"\1", line)  # no space before punctuation
    line = respace_after_stop(line)  # add missing space in "gauche.Persistance"
    return line


def main() -> None:
    if not SRC.exists():
        sys.exit(f"missing {SRC}")
    n_in = 0
    n_out = 0
    stats = CleanupStats()
    with SRC.open(encoding="utf-8") as fin, DST.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            row = json.loads(line)
            orig_id = row["id"]
            texts = row.get("documents", {}).get("text", []) or []
            for text_index, text in enumerate(texts):
                if not isinstance(text, str):
                    continue
                text_blocks = split_blocks(
                    text,
                    stats,
                    clean_line=clean_line,
                    ignore_chars=IGNORE_CHARS,
                    min_line_chars=MIN_LINE_CHARS,
                    min_text_chars=MIN_TEXT_CHARS,
                    drop_text=drop_text,
                )
                for split_index, block in enumerate(text_blocks):
                    entry = {
                        "id": f"{orig_id}-t{text_index}-s{split_index}",
                        "category": CATEGORY,
                        "text": block,
                    }
                    fout.write(json.dumps(entry, ensure_ascii=False))
                    fout.write("\n")
                    n_out += 1
    print(f"read {n_in} documents, wrote {n_out} text entries to {DST}")
    print(
        f"kept {stats.kept:,} / {stats.raw:,} alphanumeric chars "
        f"({stats.cut_pct():.1f}% cut)"
    )
    print(stats.render())


if __name__ == "__main__":
    main()
