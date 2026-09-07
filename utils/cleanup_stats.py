"""Shared line-splitting + alphanumeric-loss accounting for the text cleaners.

Both ``03_PARHAF/02_clean_split_texts.py`` and
``04_PARROT/02_clean_split_texts.py`` call ``split_blocks`` here (injecting their
own ``clean_line``, ``drop_text`` and thresholds) so the orchestration and the
diagnostics live in one place, not two parallel copies.

``CleanupStats`` attributes every alphanumeric character to the specific rule
that removed it, so you can see which filter cuts the most (and where a "trick"
would buy back the most text). Only alphanumeric characters are counted
(``str.isalnum``, Unicode-aware so accented French letters count); whitespace and
punctuation are ignored.

The buckets reconcile exactly to the raw total::

    raw = clean_norm + sum(line_*) + sum(block_*) + kept

so ``CleanupStats.check()`` can assert nothing was lost in the accounting. A
dropped line is charged to the FIRST rule that rejected it (same short-circuit
order as ``line_reject_reason``), so the numbers are an upper bound on what
relaxing that one rule would recover (a line an early rule drops might still be
caught by a later one).

For the ``no_end_punct`` rule (a line not ending in ``.``/``?``) it also records,
per dropped line, whether the next source line starts with a capital letter
(a complete enumeration item / sentence, so safe to auto-terminate, versus a
mid-sentence wrap whose next line is lowercase) and the line's position in its
text as a decile of alphanumeric chars, since a missing period at the very end
of a report is not the same as one inside an enumeration.

Written with the help of Claude Code.
"""
import re
from collections import Counter

# A sentence stop or comma glued directly to a capitalized word ("gauche.Persistance",
# "injection.L'hypophyse", "gauche,Cette") is a source artifact: a space is missing
# after the punctuation. Matches the stop + the capital; the trailing char is a
# lookahead so we only split when the capital is followed by a NON-capital (a normal
# Capitalized word), never inside an all-caps acronym (".FLAIR") or a decimal (",5",
# excluded because the char after the stop must be a letter, not a digit).
_STOP_GLUED = re.compile(r"([.,?!])([^\W\d_])(?=(.?))")


def respace_after_stop(line: str) -> str:
    """Insert the missing space after ``. , ? !`` when glued to a Capitalized word."""

    def repl(m):
        stop, first, nxt = m.group(1), m.group(2), m.group(3)
        if first.isupper() and not (nxt and nxt.isupper()):
            return f"{stop} {first}"
        return m.group(0)

    return _STOP_GLUED.sub(repl, line)

# Ordered line-reject reasons, mirroring keep_line's checks (first that fires).
LINE_REASONS = (
    "too_short",
    "ignore_char",
    "colon",
    "no_end_punct",
    "all_caps",
    "not_cap_start",
)
# Block-level drops applied after contiguous lines are joined.
BLOCK_REASONS = ("too_short", "double_dot", "untokenizable")

POS_BINS = 10  # deciles for the no_end_punct position histogram


def alnum_len(s: str) -> int:
    """Count only alphanumeric characters (ignores whitespace and punctuation)."""
    return sum(1 for c in s if c.isalnum())


def starts_capital(s: str) -> bool:
    """True if the first alphabetic char of ``s`` is uppercase (skips markers)."""
    for ch in s:
        if ch.isalpha():
            return ch.isupper()
    return False


def line_reject_reason(line, ignore_chars, min_line_chars):
    """Return the first per-line rule that rejects ``line``, or None if it is kept.

    Identical in effect to the cleaners' old ``keep_line`` (None <=> kept True);
    it just names the offending rule so the loss can be attributed.
    """
    if len(line) < min_line_chars:
        return "too_short"
    if ignore_chars.intersection(line):
        return "ignore_char"  # untokenizable chars we do not rewrite
    if ":" in line:
        return "colon"  # labels / enumeration headers
    if not line.endswith((".", "?")):
        return "no_end_punct"
    if line.isupper():
        return "all_caps"
    if len(line) < 2:
        return "too_short"  # dead while min_line_chars >= 2, kept for fidelity
    if not (line[0].isupper() and not line[1].isupper()):
        return "not_cap_start"
    return None


def split_blocks(
    text,
    stats,
    *,
    clean_line,
    ignore_chars,
    min_line_chars,
    min_text_chars,
    drop_text,
):
    """Split one raw text into coherent blocks of joined contiguous lines.

    Shared by the PARHAF and PARROT cleaners; the per-stage bits (``clean_line``,
    the untokenizable ``drop_text`` and the thresholds / ignore set) are injected.
    Empty lines never break contiguity. A non-empty line that
    ``line_reject_reason`` rejects breaks the current block. Every alphanumeric
    char is charged to ``stats`` (normalization, first line rule, block rule, or
    kept), and each ``no_end_punct`` line also records whether the next source
    line starts with a capital and its position in the text.
    """
    lines = [(raw, alnum_len(raw)) for raw in text.split("\n") if raw.strip()]
    total = sum(ra for _, ra in lines) or 1  # total alnum chars of this text
    blocks: list[str] = []
    current: list[str] = []
    pos = 0  # alnum chars seen so far (position of the current line's start)
    for i, (raw, raw_alnum) in enumerate(lines):
        stats.add_raw(raw_alnum)
        cleaned = clean_line(raw)
        stats.add_clean_norm(raw_alnum - alnum_len(cleaned))  # chars clean_line cut
        pos_frac = pos / total
        pos += raw_alnum
        if not cleaned:
            continue
        reason = line_reject_reason(cleaned, ignore_chars, min_line_chars)
        if reason is None:
            current.append(cleaned)
            continue
        a = alnum_len(cleaned)
        stats.drop_line(reason, a)
        if reason == "no_end_punct":
            next_cap = i + 1 < len(lines) and starts_capital(lines[i + 1][0])
            stats.note_no_end_punct(a, pos_frac, next_cap)
        if current:
            blocks.append(" ".join(current))
            current = []
    if current:
        blocks.append(" ".join(current))
    kept: list[str] = []
    for b in blocks:
        a = alnum_len(b)
        if len(b) < min_text_chars:
            stats.drop_block("too_short", a)
        elif ".." in b:
            stats.drop_block("double_dot", a)
        elif drop_text(b):
            stats.drop_block("untokenizable", a)
        else:
            stats.keep(a)
            kept.append(b)
    return kept


class CleanupStats:
    """Accumulate alphanumeric-char losses per rule across a whole cleanup run."""

    def __init__(self) -> None:
        self.c: Counter = Counter()
        # no_end_punct detail: next-line-capital split and position histogram.
        self.nep_cap = Counter()  # keys "lines"/"chars", next line starts capital
        self.nep_low = Counter()  # keys "lines"/"chars", next line lower/absent
        self.nep_pos_chars = [0] * POS_BINS  # char-weighted position deciles
        self.nep_pos_lines = [0] * POS_BINS

    def add_raw(self, n: int) -> None:
        self.c["raw"] += n

    def add_clean_norm(self, n: int) -> None:
        """Chars removed (or added, if negative) by clean_line before filtering."""
        self.c["clean_norm"] += n

    def drop_line(self, reason: str, n: int) -> None:
        self.c[f"line_{reason}"] += n

    def drop_block(self, reason: str, n: int) -> None:
        self.c[f"block_{reason}"] += n

    def keep(self, n: int) -> None:
        self.c["kept"] += n

    def note_no_end_punct(self, alnum: int, pos_frac: float, next_cap: bool) -> None:
        bucket = min(POS_BINS - 1, int(pos_frac * POS_BINS))
        self.nep_pos_chars[bucket] += alnum
        self.nep_pos_lines[bucket] += 1
        side = self.nep_cap if next_cap else self.nep_low
        side["lines"] += 1
        side["chars"] += alnum

    @property
    def raw(self) -> int:
        return self.c["raw"]

    @property
    def kept(self) -> int:
        return self.c["kept"]

    def cut_pct(self) -> float:
        return (1 - self.kept / self.raw) * 100 if self.raw else 0.0

    def _loss_rows(self):
        rows = [("clean_norm (parens/markers/normalization)", self.c["clean_norm"])]
        rows += [(f"line: {r}", self.c[f"line_{r}"]) for r in LINE_REASONS]
        rows += [(f"block: {r}", self.c[f"block_{r}"]) for r in BLOCK_REASONS]
        rows.sort(key=lambda kv: kv[1], reverse=True)
        return rows

    def check(self) -> bool:
        """True if every counted char is accounted for (buckets sum to raw)."""
        accounted = self.kept + self.c["clean_norm"]
        accounted += sum(self.c[f"line_{r}"] for r in LINE_REASONS)
        accounted += sum(self.c[f"block_{r}"] for r in BLOCK_REASONS)
        return accounted == self.raw

    def _render_no_end_punct(self, out: list) -> None:
        total = self.c["line_no_end_punct"]
        if not total:
            return
        denom = total
        n_lines = sum(self.nep_pos_lines)
        out.append("")
        out.append(f"no_end_punct detail ({total:,} chars, {n_lines:,} lines):")
        cap, low = self.nep_cap, self.nep_low
        out.append(
            f"  next line starts capital : {cap['lines']:>6,} lines  "
            f"{cap['chars']:>10,} chars  {cap['chars'] / denom * 100:5.1f}%"
        )
        out.append(
            f"  next line lower / absent : {low['lines']:>6,} lines  "
            f"{low['chars']:>10,} chars  {low['chars'] / denom * 100:5.1f}%"
        )
        out.append("  position in source text (char-weighted decile):")
        peak = max(self.nep_pos_chars) or 1
        for i in range(POS_BINS):
            lo, hi = i * 10, i * 10 + 10
            pc = self.nep_pos_chars[i]
            bar = "#" * int(round(pc / peak * 24))
            out.append(f"    {lo:3d}-{hi:3d}%  {pc:>10,}  {pc / denom * 100:5.1f}%  {bar}")

    def render(self) -> str:
        """A sorted, percentage-annotated breakdown of where the raw chars went."""
        denom = self.raw or 1
        out = [f"alphanumeric char accounting (of {self.raw:,} raw):"]
        out.append(f"  {'kept':<42}{self.kept:>12,}  {self.kept / denom * 100:5.1f}%")
        for label, n in self._loss_rows():
            out.append(f"  {label:<42}{n:>12,}  {n / denom * 100:5.1f}%")
        if not self.check():
            out.append("  WARNING: buckets do not reconcile to raw (accounting bug)")
        self._render_no_end_punct(out)
        return "\n".join(out)
