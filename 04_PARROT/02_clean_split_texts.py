#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "click",
#   "loguru",
# ]
# ///
"""Turn French PARROT radiology reports into coherent longish French text entries.

AUDIT-ONLY as of the LLM rewrite pivot. The dataset text for this stage is now
produced by the rewrite path (``03_chunk_for_rewrite.py`` then
``04_generate_texts.py``), which rewrites each raw report into one faithful
paragraph via the shared engine. This deterministic line cleaner is kept for its
alphanumeric-loss accounting and character audit; ``02_parrot_texts.jsonl`` is a
diagnostic artefact, no longer the input to audio.

This is the PARROT twin of ``03_PARHAF/02_clean_split_texts.py`` and uses the
exact same strategy: normalize each line, keep only clean sentence-like lines,
join contiguous survivors into blocks, drop blocks that are too short or still
carry an untokenizable char. Only the input format and a handful of
radiology-specific quirks differ, so the two stay deliberately parallel; if you
change the shared behaviour, change both.

Reads ``PARROT_v1_0_french.jsonl`` (produced by ``01_filter_french.py``) and
emits one clean entry per coherent block of text into ``02_parrot_texts.jsonl``.

Line-based cleaning:

1. Split every ``report`` string on newlines, keep only non-empty lines.
2. Normalize each line (``clean_line``): NFC-normalize; fold the French curly
   apostrophes ``' '`` (U+2019/U+2018) to ASCII ``'``, the guillemets ``« »`` and
   typographic quotes to nothing, the en dash ``-`` and U+2010/U+2011 to an ASCII
   hyphen; turn the non-breaking space ``\\xa0`` (pervasive in PARROT) and the
   zero-width space into ordinary space; rewrite the enumeration semicolon ``;``
   (as in "séquences T1 ; T2") to a comma so the line survives; drop Word
   "Wingdings" bullet glyphs
   (private-use area U+F000..U+F0FF); strip leading list markers (``- • * ·``);
   drop any parenthesized span ``(...)``; spell the gradient-echo sequences
   ``T2*``/``T1*`` as ``T2 étoile``/``T1 étoile``; spell ``°`` as "degrés" (angles
   and temperatures), ``+`` as "plus", ``×``/``*``-between-digits and ``x`` inside
   a cm/mm measurement as "fois", ``>``/``<`` before a number/``à`` as
   "supérieur"/"inférieur"; collapse whitespace.
3. Drop a line unless it (``line_reject_reason``, identical to PARHAF): carries none of the
   untokenizable ``IGNORE_CHARS`` we do not rewrite, ends with ``.`` or ``?``,
   contains no ``:`` anywhere (drops "Indication : ...", "Renseignements
   cliniques : ..." and other labelled/enumeration headers), is at least
   MIN_LINE_CHARS long, is not fully uppercase (drops the shouting section
   headers "IRM DU BASSIN", "CONCLUSION", "RESULTATS"), and starts with a capital
   letter followed by a non-capital char.
4. Join surviving lines that are *contiguous* (no dropped non-empty line between
   them) with a single space; a dropped line starts a new block. Because most
   PARROT findings are bullet lines, this is what re-assembles a section's
   enumeration into one readable paragraph.
5. Drop any resulting block shorter than MIN_TEXT_CHARS, containing two dots in a
   row (".."), or still carrying an untokenizable char not in ``KEEP_CHARS``.

Each emitted entry is ``{"id": "parrot-{no}-s{split_index}", "category":
"parrot", "text": block}``, matching the PARHAF output shape.

This file was written with the help of Claude Code.
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

# The shared Parakeet coverage detector lives in utils/; import it rather than
# duplicating the hardcoded list of glyphs that map to <unk>.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from parakeet_tokenizer import ParakeetTokenizer  # noqa: E402
from cleanup_stats import CleanupStats, respace_after_stop, split_blocks  # noqa: E402

SRC = Path("PARROT_v1_0_french.jsonl")
DST = Path("02_parrot_texts.jsonl")

# Innermost balanced parenthesized span, applied repeatedly to handle nesting.
_PAREN = re.compile(r"\([^()]*\)")

# Word "Wingdings" bullet/arrow glyphs land in the Unicode private-use area when
# a .docx is exported to text; they are meaningless list markers here, so wipe
# them (they also count as untokenizable, so a leftover would drop the block).
_PUA = re.compile(r"[-]")

# Leading list markers on a finding line ("- ...", "• ...", "** ...", "· ...").
# Stripped so the finding sentence underneath survives ``keep_line`` (PARROT is
# bullet-structured, unlike PARHAF prose). "–" is already folded to "-" first.
_LEAD = re.compile(r"^[\s\-•*·]+")

# "x" measurement multiplication where a cm/mm unit appears somewhere in the run.
# One regex covers every shape radiology uses:
#   - shared trailing unit ("20x16mm", "6,5 x 2,5 cm", "5 x 3 x 2 cm")
#   - a unit repeated on every factor ("20mm x 20mm", "15 mm x 12 mm")
#   - 3D with a mix / abbreviated middle unit ("59mmx45mx27mm")
# The unit after each factor is optional (so it may sit only on the last one);
# the ``_spell_measure_x`` guard requires a real cm/mm before converting, so a
# bare "3 x 4" without any unit is left alone. The unit must be followed by the
# "x"/"X" operator or a non-letter, which stops the "m" alternative from eating
# the leading "m" of a following word while still allowing the no-space form
# "59mmx45m" (where the operator "x" directly follows the unit).
_MEASURE_UNIT = r"(?:(?:cm|mm|m)(?:(?=[xX])|(?![a-zA-Z])))?"
_MEASURE_X = re.compile(
    r"\d[\d.,]*\s*" + _MEASURE_UNIT + r"(?:\s*[xX]\s*\d[\d.,]*\s*" + _MEASURE_UNIT + r")+"
)


def _spell_measure_x(m: "re.Match") -> str:
    """Spell the "x" of a measurement as "fois".

    Convert when the run carries a cm/mm unit ("20 x 16 mm") OR has 3+ factors
    ("12 x 9 x7", "64x32x35") which is always a dimension measurement even with
    the unit detached ("... de diamètre") or absent. A bare 2-factor run with no
    unit ("revu dans 3 x 4 semaines") is left alone.
    """
    span = m.group(0)
    n_sep = span.count("x") + span.count("X")
    if n_sep < 2 and "mm" not in span and "cm" not in span:
        return span
    return re.sub(r"\s*[xX]\s*", " fois ", span)

# Hardcoded thresholds, tweak here.
MIN_LINE_CHARS = 15  # a line shorter than this never joins a block
MIN_TEXT_CHARS = 80  # drop joined blocks shorter than this

CATEGORY = "parrot"

# Untokenizable characters we do NOT rewrite: a line carrying one is discarded by
# ``keep_line``. PARROT adds ``[]{}`` to the PARHAF set (they show up as stray
# report artifacts). ``*``/``>``/``<`` appear here too, but ``clean_line`` runs
# first, so only leftover ones (not spelled out) reach this.
IGNORE_CHARS = frozenset("~;#±≤≥_&€@$©®⁄^=*<>[]{}")

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
    """Normalize a raw PARROT line: fold typography, drop parens, spell out quirks."""
    line = unicodedata.normalize("NFC", line)
    # Fold PARROT typography to the ASCII/spoken forms Parakeet can tokenize.
    line = line.replace("’", "'").replace("‘", "'")  # curly apostrophes -> ASCII
    line = line.replace("«", " ").replace("»", " ")  # guillemets -> space
    line = line.replace("“", "").replace("”", "").replace('"', "")  # quotes dropped
    line = line.replace("–", "-").replace("‐", "-").replace("‑", "-")  # dashes -> hyphen
    line = line.replace("\xa0", " ").replace("​", "")  # NBSP -> space, ZWSP -> removed
    line = line.replace(";", ",")  # semicolon enumerations ("T1 ; T2") -> comma
    line = _PUA.sub(" ", line)  # Word Wingdings bullet/arrow glyphs -> space
    line = _LEAD.sub("", line)  # strip leading list markers ("- ", "• ", "** ")
    prev = None
    while prev != line:  # repeat to peel nested/adjacent parentheses
        prev = line
        line = _PAREN.sub(" ", line)
    line = line.replace("(", " ").replace(")", " ")  # stray unbalanced parens
    # Gradient-echo sequences "T2*"/"T1*" -> the spoken French "T2 étoile". Done
    # before the generic "*"-between-digits rule so the sequence name wins.
    line = re.sub(r"\b([Tt][12])\s*\*", r"\1 étoile", line)
    line = line.replace("°", " degrés ")
    # Spoken rewrites; wrapped in spaces so the collapse below re-normalizes
    # whatever surrounded them (e.g. "3+3" and "3 + 3" both -> "3 plus 3").
    line = line.replace("+", " plus ").replace("×", " fois ")
    # "*" between two digits is multiplication ("20*16" -> "20 fois 16"), and a
    # "*" after "g" before a digit is a dose frequency ("1g*3/j" -> "1g fois 3/j",
    # also catching the "g" of "mg"); a "*" elsewhere is left as-is (IGNORE_CHARS).
    line = re.sub(r"(?<=\d)\s*\*\s*(?=\d)", " fois ", line)
    line = re.sub(r"(?<=g)\s*\*\s*(?=\d)", " fois ", line)
    # letter "x" as multiplication inside a cm/mm measurement -> "fois" (covers
    # 2D/3D, shared-unit, repeated-unit and abbreviated-middle-unit shapes).
    line = _MEASURE_X.sub(_spell_measure_x, line)
    # ">"/"<" right before a number or "à" (optionally across a space) are
    # comparisons; spell them out. The "à" form keeps the "à" it precedes
    # ("< à 1 cm" -> "inférieur à 1 cm"). Other ">"/"<" stay as-is (IGNORE_CHARS).
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
            report = row.get("report")
            if not isinstance(report, str):
                continue
            no = row.get("no", n_in)
            report_blocks = split_blocks(
                report,
                stats,
                clean_line=clean_line,
                ignore_chars=IGNORE_CHARS,
                min_line_chars=MIN_LINE_CHARS,
                min_text_chars=MIN_TEXT_CHARS,
                drop_text=drop_text,
            )
            for split_index, block in enumerate(report_blocks):
                entry = {
                    "id": f"parrot-{no}-s{split_index}",
                    "category": CATEGORY,
                    "text": block,
                }
                fout.write(json.dumps(entry, ensure_ascii=False))
                fout.write("\n")
                n_out += 1
    print(f"read {n_in} reports, wrote {n_out} text entries to {DST}")
    print(
        f"kept {stats.kept:,} / {stats.raw:,} alphanumeric chars "
        f"({stats.cut_pct():.1f}% cut)"
    )
    print(stats.render())


if __name__ == "__main__":
    main()
