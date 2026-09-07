"""Deterministic text preprocessing for the document-rewrite stages (stdlib only).

Turns a raw clinical document / radiology report into rewrite-ready chunks:

* ``strip_parens``: drop every parenthesised span (nested included), because the
  user wants everything in parentheses removed before the LLM sees the text.
* ``strip_table_markup``: drop ASCII-table scaffolding (box borders, rulers,
  cell-separator pipes) so a pipe/box table export never reaches the rewrite LLM
  as if it were prose. Applied inside ``chunk_text``.
* ``chunk_text``: split the de-parenthesised text into coherent chunks capped at
  ``DEFAULT_MAX_CHARS`` on paragraph then sentence boundaries, so a short report
  stays one chunk (one paragraph out) and a long PARHAF document becomes a few.
  Chunks below ``DEFAULT_MIN_CHARS`` are rebalanced against a neighbour so no
  stray fragment becomes its own tiny clip, but never past the cap: one chunk is
  one spoken clip, and the ASR trainer silently drops clips over its own
  ``max_duration``, so the cap is a hard one.

Kept free of the LLM stack so the per-stage ``03_chunk_for_rewrite.py``
preprocessors can import it cheaply; ``text_rewrite`` re-exports it for the
adapter side.

Written with the help of Claude Code.
"""

from __future__ import annotations

import re

# Innermost balanced parenthesised span, applied repeatedly to peel nested ones.
# Same shape the cleaners use; the rewrite input strips parentheses entirely
# because the user wants "everything that's in parenthesis" removed before the
# LLM ever sees the text (asides, abbreviations, measurement notes).
_PAREN = re.compile(r"\([^()]*\)")

# Sentence boundary: a stop char followed by whitespace. Used only to split a
# single over-long paragraph into packable pieces; the LLM re-joins them anyway.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# One chunk becomes one spoken clip, so this cap is really a CLIP-LENGTH cap and
# the binding constraint is the ASR trainer, not the LLM context: the fine-tuning
# config (NeMo/perso/training_config.yaml, train_ds) sets max_duration 45.0, and
# every clip past that is silently dropped at cache-build/train time. Measured on
# stage-05 audio, a raw chunk is spoken at 0.065 s per character (p99 0.093), so
# 500 chars lands at a median of 27 s, p95 38 s, p99 43 s: under the trainer's
# ceiling with margin. The old 4000 produced 138 s clips, of which 89% were
# unusable for training and 42% were truncated mid-sentence by the TTS server's
# own 163.84 s output cap.
DEFAULT_MAX_CHARS = 500

# A short trailing/leading chunk (e.g. a 25-char "Pertes sanguines 300 cc" left
# after a long report split) makes an unnaturally tiny clip and a noisy length
# ratio, so chunks under this are merged into a neighbour. Roughly 20% of the cap.
DEFAULT_MIN_CHARS = 100

# ASCII-table scaffolding glyphs: box borders (+ |) and ruler runs (- = _).
# A line built only from these (plus spaces) is drawing, not prose.
_TABLE_DRAW_CHARS = set("+|-=_ ")
# Four or more ruler glyphs in a row inside an otherwise-textual line: an inline
# ruler / underline. Collapsed to a space so the surrounding words survive. A
# single "_" (identifier) or a lone ">" (genetics notation) is left untouched;
# those are a tokenizer / forbidden-char concern, not table structure.
_INLINE_RULER = re.compile(r"[-=_]{4,}")


def strip_parens(text: str) -> str:
    """Remove every parenthesised span (nested included) and tidy whitespace.

    Peels innermost ``(...)`` spans repeatedly so nested parentheses go too, then
    drops any stray unbalanced parenthesis and collapses the whitespace the
    removals leave behind. Newlines are preserved (``chunk_text`` needs them to
    find paragraph boundaries); only runs of spaces/tabs are collapsed.
    """
    prev = None
    while prev != text:
        prev = text
        text = _PAREN.sub(" ", text)
    text = text.replace("(", " ").replace(")", " ")  # stray unbalanced parens
    # Collapse spaces/tabs but keep newlines so paragraph structure survives.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_table_markup(text: str) -> str:
    """Drop ASCII-table scaffolding so a table export never reaches the LLM.

    Clinical exports embed pipe/box tables (gene panels, technique grids) that
    are not prose and cannot be faithfully spoken. Operates line by line on the
    raw (still newline-bearing) document, before ``chunk_text`` flattens it:

    * a line made only of table-drawing glyphs (``+ | - = _`` and spaces) is a
      ruler / border and is dropped entirely;
    * on every surviving line, the cell-separator ``|`` becomes a space and any
      inline ruler run (four or more of ``- = _``) collapses to a space, so the
      textual content survives without the scaffolding.

    In-content glyphs that merely resemble table junk (a single ``_`` inside an
    identifier, a ``>`` in a genetics notation) are left untouched; those are a
    tokenizer / forbidden-char concern handled elsewhere, not table structure.
    """
    out_lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and set(stripped) <= _TABLE_DRAW_CHARS:
            continue  # pure ruler / border line, nothing but drawing glyphs
        line = line.replace("|", " ")
        line = _INLINE_RULER.sub(" ", line)
        out_lines.append(line)
    return "\n".join(out_lines)


def _split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Split one paragraph longer than ``max_chars`` into sentence-packed pieces.

    Greedily packs whole sentences up to the cap. A single sentence longer than
    the cap (rare in clinical text) is hard-split on whitespace as a last resort
    so nothing is silently dropped.
    """
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT.split(paragraph):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            words = sentence.split(" ")
            buf = ""
            for w in words:
                if buf and len(buf) + 1 + len(w) > max_chars:
                    pieces.append(buf)
                    buf = w
                else:
                    buf = f"{buf} {w}".strip()
            if buf:
                current = buf
            continue
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def _split_in_two(text: str, max_chars: int) -> tuple[str, str] | None:
    """Cut ``text`` into two pieces of at most ``max_chars``, as evenly as the
    text allows. Prefers a sentence boundary, falls back to a word boundary, and
    returns ``None`` when neither exists inside the legal window (or when the
    text simply does not fit in two pieces)."""
    n = len(text)
    if n > 2 * max_chars:
        return None
    # The first piece must end at or before max_chars, and leave at most
    # max_chars behind, so only cuts inside [n - max_chars, max_chars] are legal.
    low, high = max(1, n - max_chars), min(n - 1, max_chars)
    if low > high:
        return None
    middle = n // 2
    sentence_cuts = [m.start() for m in _SENTENCE_SPLIT.finditer(text)]
    word_cuts = [i for i, ch in enumerate(text) if ch == " "]
    for candidates in (sentence_cuts, word_cuts):
        legal = [c for c in candidates if low <= c <= high]
        if legal:
            cut = min(legal, key=lambda c: abs(c - middle))
            return text[:cut].strip(), text[cut:].strip()
    return None


def _rebalance_short_chunks(chunks: list[str], min_chars: int,
                            max_chars: int) -> list[str]:
    """Remove stray fragments without ever breaching ``max_chars``.

    Whenever the running chunk or the next one is below ``min_chars``, the two
    are joined; if the join fits under the cap it stays one chunk, otherwise the
    joined text is re-split down the middle so the fragment's neighbour donates
    it some text instead. Plain forward merging cannot do this job under a hard
    cap: the packer has already joined everything that fits, so any surviving
    fragment is by definition one the cap forbids merging. The cap wins over the
    floor because one chunk is one spoken clip and the ASR trainer silently drops
    clips past its ``max_duration``, while a short clip is merely low-value. A
    pair that cannot be re-split (no boundary in the legal window) is left as
    packed. A whole document that is itself one short chunk has nothing to
    rebalance and is returned as-is. ``min_chars <= 0`` disables this entirely.
    """
    if min_chars <= 0 or len(chunks) <= 1:
        return chunks
    out = [chunks[0]]
    for c in chunks[1:]:
        if len(out[-1]) >= min_chars and len(c) >= min_chars:
            out.append(c)
            continue
        combined = f"{out[-1]} {c}".strip()
        if len(combined) <= max_chars:
            out[-1] = combined
            continue
        halves = _split_in_two(combined, max_chars)
        if halves is None:
            out.append(c)
        else:
            out[-1], nxt = halves
            out.append(nxt)
    return out


def chunk_text(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[str]:
    """Split one document into coherent chunks no longer than ``max_chars``.

    Drops ASCII-table scaffolding (``strip_table_markup``), then splits on
    blank-line paragraph boundaries and greedily packs whole paragraphs together
    up to the cap, so a short report stays a single chunk (one paragraph out) and
    a long document becomes a few. A paragraph that is itself over the cap is
    split on sentence boundaries. Any chunk below ``min_chars`` is then joined
    with a neighbour so no stray fragment becomes its own tiny clip, and the pair
    is re-split evenly when the join would not fit (pass ``min_chars=0`` to
    disable). No returned chunk ever exceeds ``max_chars``, because one chunk is
    one spoken clip and the trainer drops clips over its own duration ceiling.
    Every returned chunk is a single-line string
    (internal newlines collapsed to spaces) ready to hand to the rewrite prompt
    as the ``<source>``.
    """
    text = text.strip()
    if not text:
        return []
    text = strip_table_markup(text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for p in paragraphs:
        p = re.sub(r"\s+", " ", p)
        if len(p) > max_chars:
            units.extend(_split_long_paragraph(p, max_chars))
        else:
            units.append(p)
    chunks: list[str] = []
    current = ""
    for u in units:
        if current and len(current) + 1 + len(u) > max_chars:
            chunks.append(current)
            current = u
        else:
            current = f"{current} {u}".strip()
    if current:
        chunks.append(current)
    return _rebalance_short_chunks(chunks, min_chars, max_chars)
