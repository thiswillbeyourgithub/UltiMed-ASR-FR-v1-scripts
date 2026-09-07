# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "click",
#   "loguru",
# ]
# ///

"""
Reusable Parakeet TDT v3 ``<unk>`` detection, built from ``vocab.txt`` alone.

This module is meant to be *imported* by other scripts in the project (e.g.
``01_dictionnary/02_token_check.py``) but also works as a standalone CLI. It is
the shared, source-agnostic character-audit tool every stage points at to see
which characters still map to ``<unk>`` (so they can be normalized away):

    uv run utils/parakeet_tokenizer.py --input terms.txt                 # one term per line
    uv run utils/parakeet_tokenizer.py --input data.jsonl --field text   # a JSONL field
    uv run utils/parakeet_tokenizer.py --text "Ωμέγα 𝕏"                   # inline

With ``--field`` the input is read as JSONL and that field is pulled from every
record; the report groups the distinct offending characters by how many entries
they hit, with an example each, so you know what to handle next.

What it does (and does not do)
------------------------------
It answers exactly one question: "would the Parakeet TDT 0.6B v3 tokenizer map
this text to the ``<unk>`` token?". It does **not** reproduce the SentencePiece
segmentation (token ids / piece boundaries); that needs the original ``.model``
file, which is not shipped with the onnx-asr export. For ``<unk>`` detection the
segmentation is irrelevant: only character coverage matters, and that is fully
encoded in ``vocab.txt``. See ``TOKENIZE.md`` for the full reasoning.

Why ``vocab.txt`` is sufficient
-------------------------------
1. This vocab has *no* byte-fallback tokens (no ``<0xXX>`` pieces). Without byte
   fallback, any character outside the model's coverage is emitted as ``<unk>``,
   so "produces ``<unk>``" is equivalent to "character not covered".
2. SentencePiece guarantees a single-character piece for every covered
   character, and no multi-character piece can contain an uncovered character.
   So the union of characters across the vocab's normal (non-special) pieces
   *is* the coverage set.

Importing
---------
>>> from parakeet_tokenizer import ParakeetTokenizer
>>> tok = ParakeetTokenizer()              # loads ./parakeet_vocab.txt once
>>> tok.has_unk("café")                    # False (covered)
>>> tok.offending_chars("Ωμέγα 𝕏")          # {'𝕏'} (the rest is covered)

Created with assistance from Claude Code.
"""

import json
import sys
import unicodedata
from collections import Counter, namedtuple
from pathlib import Path

import click
from loguru import logger

# SentencePiece space marker (U+2581). In the vocab it stands for a word
# boundary, so a literal space in the input is always representable.
SPACE_MARKER = "▁"  # ▁

# Default vocab location: this file lives in utils/ next to the vocab, so
# importing scripts in subfolders get a working default without configuring a
# path. Resolve relative to this file (not the CWD) so it works from anywhere.
DEFAULT_VOCAB_PATH = Path(__file__).resolve().parent / "parakeet_vocab.txt"


def is_special(piece: str) -> bool:
    """
    Return whether a vocab piece is a special / control token, not real text.

    Special tokens look like ``<unk>``, ``<pad>``, ``<blk>`` or the language /
    task control tokens ``<|fr|>``, ``<|pnc|>``, ``<|spk0|>``, etc. They must be
    excluded from the coverage set, otherwise the characters ``<``, ``|``, ``>``
    and the language codes inside them would pollute it.

    Parameters
    ----------
    piece : str
        A piece (the token text) from a line of ``vocab.txt``.

    Returns
    -------
    bool
        True if the piece is a special token wrapped in angle brackets.
    """
    return piece.startswith("<") and piece.endswith(">")


def load_coverage(vocab_path: Path = DEFAULT_VOCAB_PATH) -> frozenset:
    """
    Build the set of characters the Parakeet TDT v3 tokenizer can represent.

    Each line of ``vocab.txt`` is ``<piece> <id>``; the split is on the *last*
    space because the piece may itself be the space marker ``▁``. Special tokens
    are skipped, the space marker is mapped to a literal space, and common
    whitespace is added (harmless inside terms).

    Parameters
    ----------
    vocab_path : Path, optional
        Path to ``vocab.txt`` (defaults to ``parakeet_vocab.txt`` next to this
        module).

    Returns
    -------
    frozenset of str
        The coverage set: every character the tokenizer can emit without
        falling back to ``<unk>``.
    """
    coverage = set()
    for raw in Path(vocab_path).read_text(encoding="utf-8").splitlines():
        if not raw:
            continue
        # Split on the LAST space: '<piece> <id>'. Splitting on the first space
        # would corrupt the bare '▁' piece, which is itself a single space char.
        piece, _, _id = raw.rpartition(" ")
        if piece == "":  # malformed / empty-line guard
            continue
        if is_special(piece):
            continue
        coverage.update(piece)
    coverage.discard(SPACE_MARKER)
    coverage.add(" ")  # the space marker means a literal space is representable
    coverage.update("\t\n\r")  # whitespace inside terms is harmless to intent
    return frozenset(coverage)


def describe(ch: str) -> str:
    """
    Return a human-readable description of a single character for reporting.

    Parameters
    ----------
    ch : str
        A single character.

    Returns
    -------
    str
        A string like ``U+2019 RIGHT SINGLE QUOTATION MARK ('’')``.
    """
    try:
        name = unicodedata.name(ch)
    except ValueError:
        name = "<no name>"
    return f"U+{ord(ch):04X} {name} ({ch!r})"


def excerpt(text: str, ch: str, width: int = 80, nfkc: bool = True) -> str:
    """
    Return a ``width``-char window of ``text`` centered on ``ch``.

    The point is to keep the offending character visible even when it sits far
    from the start of a long text (a plain ``text[:width]`` would hide it).

    ``ch`` may be an NFKC artifact (e.g. U+2044 produced from ``¼``) that is not
    literally present in ``text``; in that case the raw character whose NFKC
    expansion contains ``ch`` is located instead. Falls back to the head of the
    text when the character cannot be located. ``...`` marks a clipped edge.

    Parameters
    ----------
    text : str
        The full text the character was found in.
    ch : str
        The offending character to center the window on.
    width : int, optional
        Total window size in characters.
    nfkc : bool, optional
        Whether to also locate ``ch`` via each raw character's NFKC expansion
        (matches the tokenizer's own NFKC pass).

    Returns
    -------
    str
        The windowed snippet, with ``...`` prepended/appended when clipped.
    """
    pos = text.find(ch)
    if pos < 0 and nfkc:
        for i, c in enumerate(text):
            if ch in unicodedata.normalize("NFKC", c):
                pos = i
                break
    if pos < 0:
        pos = 0
    start = max(0, pos - width // 2)
    end = start + width
    return ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")


# Aggregate result of auditing a batch of texts: `total` texts seen, `flagged`
# of them carrying at least one uncovered character, `char_entries` a Counter of
# how many texts each offending char appears in, `char_example` one example text
# per offending char, and `per_text` the (index, text, offending-set) detail rows.
AuditResult = namedtuple(
    "AuditResult", ["total", "flagged", "char_entries", "char_example", "per_text"]
)


def audit_texts(texts, tok: "ParakeetTokenizer") -> AuditResult:
    """
    Audit an iterable of strings for characters that map to ``<unk>``.

    Shared by the CLI here and reusable by any stage that has a batch of strings
    (dictionary terms, PARHAF texts, drug names, ...) and wants to know which
    characters are not yet tokenizable.

    Parameters
    ----------
    texts : Iterable[str]
        The strings to check.
    tok : ParakeetTokenizer
        A loaded tokenizer/coverage detector.

    Returns
    -------
    AuditResult
        Aggregated counts, per-character examples and per-text detail.
    """
    char_entries: Counter = Counter()
    char_example: dict = {}
    per_text: list = []
    total = 0
    flagged = 0
    for i, text in enumerate(texts, 1):
        total += 1
        bad = tok.offending_chars(text)
        if bad:
            flagged += 1
            per_text.append((i, text, bad))
            for ch in bad:
                char_entries[ch] += 1
                char_example.setdefault(ch, text)
    return AuditResult(total, flagged, char_entries, char_example, per_text)


class ParakeetTokenizer:
    """
    Coverage-based ``<unk>`` detector for the Parakeet TDT v3 tokenizer.

    Load the vocabulary once, then call :meth:`has_unk` / :meth:`offending_chars`
    repeatedly. This is the object other scripts import.

    Parameters
    ----------
    vocab_path : Path, optional
        Path to ``vocab.txt`` (defaults to ``parakeet_vocab.txt`` next to this
        module).
    nfkc : bool, optional
        If True (default), NFKC-normalize text before checking, matching what
        SentencePiece does internally. This avoids false positives on composed /
        fullwidth characters the real tokenizer would fold into a covered one.
        Set to False to inspect the raw, un-normalized characters instead.

    Attributes
    ----------
    coverage : frozenset of str
        The character coverage set loaded from the vocab.
    nfkc : bool
        Whether input is NFKC-normalized before checking.
    """

    def __init__(self, vocab_path: Path = DEFAULT_VOCAB_PATH, nfkc: bool = True):
        self.coverage = load_coverage(vocab_path)
        self.nfkc = nfkc

    def _prepare(self, text: str) -> str:
        """NFKC-normalize the text if configured, else return it unchanged."""
        return unicodedata.normalize("NFKC", text) if self.nfkc else text

    def offending_chars(self, text: str) -> set:
        """
        Return the distinct characters in ``text`` that are not covered.

        These are exactly the characters that would collapse to ``<unk>``.

        Parameters
        ----------
        text : str
            The text to check.

        Returns
        -------
        set of str
            The offending characters (empty if the text is fully tokenizable).
        """
        return {ch for ch in self._prepare(text) if ch not in self.coverage}

    def has_unk(self, text: str) -> bool:
        """
        Return whether ``text`` contains any character that maps to ``<unk>``.

        Parameters
        ----------
        text : str
            The text to check.

        Returns
        -------
        bool
            True if at least one character is uncovered.
        """
        return any(ch not in self.coverage for ch in self._prepare(text))


@click.command()
@click.option(
    "--vocab",
    "vocab_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_VOCAB_PATH,
    show_default=True,
    help="Path to vocab.txt.",
)
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Text file to check, one term/line.",
)
@click.option("--text", default=None, help="Inline text to check.")
@click.option(
    "--field",
    default=None,
    help="Treat --input as JSONL and audit this field of every record (e.g. 'text').",
)
@click.option(
    "--nfkc/--no-nfkc",
    default=True,
    show_default=True,
    help="NFKC-normalize input first (matches SentencePiece normalization).",
)
@click.option(
    "--max-report",
    default=40,
    show_default=True,
    help="Max offending entries to print individually before truncating.",
)
def main(vocab_path: Path, input_path: Path, text: str, field: str, nfkc: bool, max_report: int) -> None:
    """
    Standalone CLI: flag entries containing characters that map to ``<unk>``.

    Provide exactly one of ``--input`` or ``--text``. Without ``--field`` the
    input is one term per line; with ``--field`` it is JSONL and that field is
    pulled from each record. Exits 1 if anything is untokenizable, 0 otherwise,
    so it slots into a CI gate.
    """
    if (input_path is None) == (text is None):
        raise click.UsageError("Provide exactly one of --input or --text.")
    if field is not None and input_path is None:
        raise click.UsageError("--field only applies to --input (JSONL).")

    tok = ParakeetTokenizer(vocab_path=vocab_path, nfkc=nfkc)

    if text is not None:
        texts = [text]
    elif field is not None:
        texts = []
        with input_path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                value = json.loads(raw).get(field)
                if isinstance(value, str):
                    texts.append(value)
    else:
        texts = input_path.read_text(encoding="utf-8").splitlines()

    result = audit_texts(texts, tok)

    for i, entry, bad in result.per_text[:max_report]:
        logger.warning(f"[entry {i}] UNK chars {' '.join(sorted(bad))}: {entry!r}")
    if result.flagged > max_report:
        logger.warning(f"... and {result.flagged - max_report} more (raise --max-report to see).")

    logger.info(f"{result.flagged}/{result.total} entr(ies) contain untokenizable characters.")
    if result.char_entries:
        logger.error("Distinct offending characters (most entries first; add a rewrite for each):")
        for ch, n in result.char_entries.most_common():
            example = result.char_example[ch]
            logger.error(f"  {describe(ch)}  in {n} entr(ies), e.g. {excerpt(example, ch)!r}")
        sys.exit(1)
    logger.success("All input is tokenizable (no <unk>).")
    sys.exit(0)


if __name__ == "__main__":
    main()
