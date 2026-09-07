# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "click",
#   "loguru",
# ]
# ///

"""
Normalize dictionary terms and gate the rewrite on Parakeet TDT v3 ``<unk>``.

Goal: take the stage-1 scored file (``original_dictionnary.scored.jsonl``) and
produce a corrected copy whose ``term`` field contains *only* characters the
Parakeet TDT v3 tokenizer can represent (no ``<unk>``); every other field,
including ``score``, passes through unchanged. The script is meant to be run
repeatedly while tuning the rewrite
rules: it applies the character-rewrite rules in ``NORMALIZATION_RULES`` to each
term, then checks every normalized term for untokenizable characters using the
shared :class:`parakeet_tokenizer.ParakeetTokenizer`.

The output file is written **only if not a single term contains an ``<unk>``**.
Otherwise nothing is written and the offending terms / characters are reported,
so you can add one more rule to ``NORMALIZATION_RULES`` and re-run, tightening
the rules one at a time until the whole dictionary is clean.

Only the ``term`` field is touched; every other field of each record is passed
through unchanged. The source file is never overwritten: output goes to a
separate ``--output`` path (default ``original_dictionnary.scored.normalized.jsonl``).

Note on the ``’/‘ -> '`` rule: ``create_french_medical.py`` applies the same apostrophe
folding in its own ``_normalize``. The rule is duplicated here on purpose
because the two scripts have different jobs (this one is a tunable per-character
tokenization gate over the raw dictionary; ``create_french_medical`` builds the flat,
glyph-stripped biasing list). If the rule set here grows to mirror that logic,
consider factoring the shared normalization into one place.

Run with:
    uv run 01_dictionnary/02_token_check.py                 # check + write if clean
    uv run 01_dictionnary/02_token_check.py --no-nfkc       # inspect raw chars

Created with assistance from Claude Code.
"""

import json
import re
import sys
from pathlib import Path

import click
from loguru import logger

# utils/ holds parakeet_tokenizer.py (the shared <unk> detector) and
# parakeet_vocab.txt. This script lives one level down in 01_dictionnary/, so we
# add utils/ to sys.path to import the shared module regardless of CWD.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "utils"))

from parakeet_tokenizer import ParakeetTokenizer, describe, excerpt  # noqa: E402

# ---------------------------------------------------------------------------
# SHARED GLYPH RULES -- KEEP IN SYNC WITH
# ../../parakeet_web_phrase_boosting/create_french_medical.py
# ---------------------------------------------------------------------------
# NORMALIZATION_RULES, EXCLUDE_CHARS and the "+"/"*" spoken rewrites below are a
# DELIBERATE COPY of the same rules in create_french_medical.py (_GLYPH_REWRITES /
# _DISCARD_CHARS / _PLUS_RE / _STAR_RE). The two scripts keep private copies on
# purpose (each is a standalone `uv run` tool), but the copies MUST agree or the
# biasing list and this <unk> gate disagree about which terms are tokenizable --
# exactly the drift this comment exists to prevent. If you change a rule here,
# ASK THE USER whether the matching rule in create_french_medical.py should change too
# (and vice-versa); do not silently edit one side.

# Ordered character-rewrite rules applied to every term before the <unk> check.
# Add rules here one at a time as you discover untokenizable characters in the
# report, until the whole dictionary passes. Each entry is (find, replace) and
# is applied with str.replace in order. Keep a comment per rule explaining why
# the replacement is the correct spoken-form substitution.
NORMALIZATION_RULES = [
    # En-dash and Unicode hyphen -> plain ASCII hyphen. The source uses – and ‐
    # as a hyphen in compound terms, but the model only knows the ASCII -. Done
    # first so later rules and the <unk> check see the normalized hyphen.
    ("–", "-"),
    ("‐", "-"),
    # Latin small capital OE ligature -> the regular œ the model expects.
    ("ɶ", "œ"),
    # Macron O -> plain O (the model has no macron vowels).
    ("Ō", "O"),
    # G with breve -> plain g (Turkish names like "Sağiroğlu").
    ("ğ", "g"),
    # S with cedilla -> plain s (Turkish names).
    ("ş", "s"),
    # Typographic single quotes -> straight apostrophe. French elision ("l'os")
    # is written with ’/‘ in the source but the model expects the plain '.
    ("’", "'"),
    ("‘", "'"),
    # French quotation marks (guillemets) around a quoted span. They are not
    # spoken, so drop the glyph together with the inner space the source pairs
    # it with ("« mot »" -> "mot").
    ("« ", ""),
    (" »", ""),
    # Catch any guillemet not paired with the adjacent space handled above.
    ("«", ""),
    ("»", ""),
    # Trademark/copyright glyphs are not spoken; strip them from brand-name
    # terms ("Doliprane®" -> "Doliprane").
    ("®", ""),
    ("©", ""),
    # Straight double quotes wrap quoted spans but are not spoken; drop them.
    ('"', ""),
    # Typographic double quotation marks, same treatment as the straight ".
    ("“", ""),
    ("”", ""),
    # Low / reversed typographic double quotes (German-style), also not spoken.
    ("„", ""),
    ("‟", ""),
]

# "+" and "*" have no standalone vocab token, so rather than drop the term we
# speak them: "+" -> "plus" ("CD4+" -> "CD4 plus") and "*" -> "etoile"
# ("T2*" -> "T2 etoile"). Each absorbs any adjacent whitespace / hyphen so no
# dangling separator is left behind. Applied in normalize_term after
# NORMALIZATION_RULES. KEEP IN SYNC with create_french_medical.py (_PLUS_RE / _STAR_RE).
_PLUS_RE = re.compile(r"[\s-]*\+[\s-]*")
_STAR_RE = re.compile(r"[\s-]*\*[\s-]*")

# Characters that mark a term as a compound/formula-style entry (e.g. drug
# combinations) with no clean spoken form. A term containing any of these is
# dropped wholesale rather than rewritten. Checked on the raw term, before
# NORMALIZATION_RULES.
EXCLUDE_CHARS = "[]→°ʋ\uf077;"

# Pipeline stage 2 reads the stage-1 scored file (01_llm_scoring.py output) and
# writes the scored+normalized file that stage 3 (03_generate_texts.py) consumes
# by default. Only the `term` field is rewritten; the `score` (and every other
# field) passes through unchanged, so the output is both scored and tokenizable.
DEFAULT_INPUT = SCRIPT_DIR / "original_dictionnary.scored.jsonl"
DEFAULT_OUTPUT = SCRIPT_DIR / "original_dictionnary.scored.normalized.jsonl"

# Eponym reorder: dictionary entries sometimes front-load the proper noun and
# park the descriptor in parentheses, e.g. "Bowen (maladie de) multicentrique".
# When the parenthetical ends in a linking word ("de"/"du", its elided "d'", or
# an article "le"/"la"), spoken French puts the descriptor first: "maladie de
# Bowen multicentrique", "coeur (le)" -> "le coeur". This regex moves such a
# parenthetical in front of the single token that precedes it and drops the
# parentheses. The trailing "\b(?:de|du|le|la)|\bd['’]" anchored just before ")"
# both selects only these linking parentheticals and (thanks to the leading word
# boundary) avoids matching words that merely end in those letters ("grande",
# "muscle").
PAREN_DE_RE = re.compile(r"(\S+)\s+\(([^()]*(?:\b(?:de|du|le|la)|\bd['’]))\s*\)", re.IGNORECASE)


def _reorder_sub(match: "re.Match[str]") -> str:
    """Rebuild a matched eponym parenthetical in spoken word order."""
    head, descriptor = match.group(1), match.group(2)
    # Elided "d'" glues straight onto the next word ("seuil d'" + "apnée"),
    # whereas a full "de"/"du" keeps the space ("maladie de" + "Bowen").
    joiner = "" if descriptor.endswith(("'", "’")) else " "
    return f"{descriptor}{joiner}{head}"


def strip_trailing_open_paren(term: str) -> str:
    """
    Drop a dangling "(" left at the end of a term and re-strip.

    Some scraped entries keep an opening "(" whose content was lost entirely,
    leaving the glyph hanging at the end ("débit ventilatoire de repos ("). With
    nothing to reorder, the only sensible fix is to remove the "(" (and the
    surrounding whitespace) so we are left with "débit ventilatoire de repos".
    Run before :func:`close_dangling_paren` so the empty "(" is removed rather
    than turned into "()".

    Parameters
    ----------
    term : str
        The raw term, possibly ending in a content-less "(".

    Returns
    -------
    str
        The term with any trailing "(" removed and the ends stripped.
    """
    term = term.rstrip()
    while term.endswith("("):
        term = term[:-1].rstrip()
    return term


def close_dangling_paren(term: str) -> str:
    """
    Re-add a closing parenthesis scraped off the end of a term.

    Some source entries lost their trailing ")" during scraping, leaving an
    open "(" with no match (e.g. "apnée (seuil d'"). We assume the ")" sat at
    the very end and append the missing one(s) so :func:`reorder_parenthetical`
    can fire ("apnée (seuil d')" -> "seuil d'apnée").

    Parameters
    ----------
    term : str
        The raw term, possibly missing trailing ")".

    Returns
    -------
    str
        The term with any unbalanced "(" closed at the end (unchanged if
        already balanced).
    """
    missing = term.count("(") - term.count(")")
    if missing > 0:
        term = term + ")" * missing
    return term


def reorder_parenthetical(term: str) -> str:
    """
    Move a trailing-"de"/"du"/"d'"/"le"/"la" parenthetical in front of its word.

    Examples
    --------
    "Bowen (maladie de) multicentrique" -> "maladie de Bowen multicentrique"
    "apnée (seuil d')"                  -> "seuil d'apnée"
    "coeur (le)"                        -> "le coeur"

    Parameters
    ----------
    term : str
        The raw term, possibly containing an eponym parenthetical.

    Returns
    -------
    str
        The term with every matching parenthetical reordered (unchanged if no
        parenthetical matched).
    """
    return PAREN_DE_RE.sub(_reorder_sub, term)


def normalize_term(term: str) -> str:
    """
    Reorder eponym parentheticals, then apply the ordered char-rewrite rules.

    Paren fixups (drop a trailing content-less "(", then close a scraped-off
    ")") and the structural :func:`reorder_parenthetical` pass run first so that
    any "(... de|du|d')" descriptor is flattened into spoken word order before
    the per-character :data:`NORMALIZATION_RULES` (and the downstream <unk>
    check) see the term.

    Parameters
    ----------
    term : str
        The raw term from the dictionary record.

    Returns
    -------
    str
        The term with the reorder pass and every rewrite rule applied
        (unchanged if nothing matched).
    """
    term = strip_trailing_open_paren(term)
    term = close_dangling_paren(term)
    term = reorder_parenthetical(term)
    for find, replace in NORMALIZATION_RULES:
        term = term.replace(find, replace)
    # "+"/"*" are spoken, not dropped: "CD4+" -> "CD4 plus", "T2*" -> "T2 etoile"
    # (each absorbing any adjacent whitespace / hyphen). KEEP IN SYNC with
    # create_french_medical.py.
    term = _PLUS_RE.sub(" plus ", term)
    term = _STAR_RE.sub(" étoile ", term)
    # Several rules above delete glyphs and can leave runs of whitespace behind
    # ("« mot »" stripping, etc.). Collapse them to a single space and trim the
    # ends as the final step.
    term = re.sub(r"\s+", " ", term).strip()
    return term


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_INPUT,
    show_default=True,
    help="Source dictionary JSONL (read-only; never overwritten).",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT,
    show_default=True,
    help="Where to write the corrected JSONL (only if zero <unk>).",
)
@click.option(
    "--nfkc/--no-nfkc",
    default=True,
    show_default=True,
    help="NFKC-normalize terms before the <unk> check (matches the tokenizer).",
)
@click.option(
    "--max-report",
    default=40,
    show_default=True,
    help="Max offending terms to log before truncating the per-term list.",
)
def main(input_path: Path, output_path: Path, nfkc: bool, max_report: int) -> None:
    """
    Normalize every term, then write the corrected file only if no term has <unk>.

    Parameters
    ----------
    input_path : Path
        Source dictionary JSONL.
    output_path : Path
        Destination for the corrected JSONL (written only when fully clean).
    nfkc : bool
        Whether to NFKC-normalize terms before the <unk> check.
    max_report : int
        Cap on the number of offending terms printed individually.
    """
    tok = ParakeetTokenizer(nfkc=nfkc)

    # Hold all normalized records in memory so we can gate the write on a clean
    # pass over the *entire* file (the dictionary is ~60k short records, so this
    # is cheap and keeps the "write only if all valid" contract simple).
    records = []
    offenders = []  # (line_no, normalized_term, offending_chars)
    all_bad = set()
    excluded = 0  # terms dropped for containing an EXCLUDE_CHARS glyph
    excluded_parens = 0  # terms dropped for still having () after normalization

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            term = record.get("term")
            if isinstance(term, str):
                # Terms carrying an EXCLUDE_CHARS glyph (e.g. '+', '*') are
                # compound/formula-style entries with no clean spoken form; drop
                # them rather than try to rewrite the glyph.
                if any(ch in term for ch in EXCLUDE_CHARS):
                    excluded += 1
                    continue
                term = normalize_term(term)
                # If a parenthesis survives normalization the reorder rules did
                # not recognize the structure; these are rare outliers, so drop
                # them rather than ship a term with raw "()" in it.
                if "(" in term or ")" in term:
                    excluded_parens += 1
                    continue
                record["term"] = term
                bad = tok.offending_chars(term)
                if bad:
                    offenders.append((line_no, term, bad))
                    all_bad |= bad
            records.append(record)

    logger.info(f"Read {len(records)} record(s) from {input_path}")
    if excluded:
        logger.info(f"Excluded {excluded} term(s) containing one of {EXCLUDE_CHARS!r}.")
    if excluded_parens:
        logger.info(f"Excluded {excluded_parens} term(s) still containing '(' or ')' after normalization.")

    if offenders:
        logger.warning(
            f"{len(offenders)} term(s) still contain untokenizable characters "
            f"after normalization. Nothing written."
        )
        for line_no, term, bad in offenders[:max_report]:
            logger.warning(f"[line {line_no}] UNK chars {' '.join(sorted(bad))}: {term!r}")
        if len(offenders) > max_report:
            logger.warning(f"... and {len(offenders) - max_report} more (raise --max-report to see).")
        logger.error("Distinct offending characters (add a rule for each, then re-run):")
        # One example term per offending char, so each char is shown in context.
        char_example = {}
        for _line_no, term, bad in offenders:
            for ch in bad:
                char_example.setdefault(ch, term)
        for ch in sorted(all_bad):
            logger.error(f"  {describe(ch)}  e.g. {excerpt(char_example[ch], ch)!r}")
        sys.exit(1)

    # Fully clean: every normalized term is tokenizable, so commit the rewrite.
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.success(
        f"All {len(records)} terms are tokenizable (no <unk>). "
        f"Wrote corrected dictionary to {output_path}"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
