"""One written form for titles, dates, clock times and drug names in the ASR label.

Like ``percent_normalize.py``, this canonicalizes what the LLM wrote AFTER the LLM:
UltiMed v1 labels spell the same spoken thing several ways, so the model learns no
consistent form. Each rule below picks the form the corpus already uses most and
rewrites the minority spellings to it:

- **Titles** (``expand_titles``). ``M. Dupont`` (1.6k) -> ``Monsieur Dupont`` (92k),
  ``Mme`` -> ``Madame``, ``Mlle`` -> ``Mademoiselle``, ``Dr`` / ``Pr`` -> ``docteur`` /
  ``professeur``. Casing follows the corpus majority mid-sentence: ``Monsieur`` /
  ``Madame`` / ``Mademoiselle`` are capitalized before a name (``Monsieur Dupont``,
  65k against 1.3k ``monsieur Dupont``), ``docteur`` / ``professeur`` are not (``le
  docteur Martin``, 5k against 0.5k ``le Docteur Martin``); every title is capitalized
  at a sentence start. ``M.`` is only a title before a capitalized name that is not
  itself a sentence opener, and not after another title (``Madame M. présente``,
  ``PCR M. tuberculosis``, ``stade M. Le patient`` are untouched), and a lowercase
  ``monsieur`` / ``madame`` without a name (``le monsieur du lit 4``) stays lowercase.
- **Dates** (``normalize_dates``). Days and years in a date are digits (22.7k ``le 15
  mars`` against 1.2k ``le quinze mars``), and the first of the month is ``1er`` (0.7k
  against 0.5k ``premier mars``). A spelled-out year (1900 to 2099, every spelling:
  ``mille neuf cent``, ``mil neuf cent``, ``dix-neuf cent``, ``deux mille``, hyphenated
  or not, ``vingt et un`` or ``vingt-et-un``) becomes digits after a month, and after
  ``en`` / ``depuis`` when nothing but punctuation, the end, ``au``, ``et`` or ``puis``
  follows (``en deux mille dix-neuf.`` -> ``en 2019.``, but ``depuis deux mille ans``
  and ``mille neuf cent soixante-quatorze grammes`` are left alone). A spelled day
  (2 to 31) becomes digits before a month, or before ``au`` and a converted day
  (``du six au treize mai`` -> ``du 6 au 13 mai``); ``un`` is not converted (``un
  mars`` is not a date anyone dictates).
- **Clock times** (``normalize_clock``). ``14h30`` / ``14 h 30`` / ``09h45`` / ``20 h 00``
  (a few dozen) -> ``14 heures 30`` / ``9 heures 45`` / ``20 heures`` (47k labels write
  ``N heures``), ``1 heure`` in the singular. Durations (``24h``) get the same form,
  which is also how they are written elsewhere.
- **Drug names** (``DrugCaser``). One spelling per drug word, from the lexicon
  ``drug_casing.json`` built by ``02_drugs/02_build_drug_casing.py`` (see its
  docstring for how the canonical form is chosen): ``PRIMPERAN`` / ``Primperan`` ->
  ``Primpéran``, ``PARACETAMOL`` -> ``paracétamol``, ``UVEDOSE`` -> ``Uvedose``, with
  ALLCAPS acronyms (``LP``, ``BCG``) protected by the lexicon.

``apply_label_conventions`` runs all four. The text generators apply it in
``_pipeline_shared.parse_asr_training_target``, so every freshly generated label is
canonical (and so is the TTS source ``voxtral_normalize`` derives from it);
``99_hf_release/05_normalize_text.py`` applies it to the v1 manifests, to both the
label and the TTS source: every rewrite here is spoken identically before and after
(``M.`` was read ``monsieur``, ``2019`` and ``deux mille dix-neuf`` are the same
words, casing is silent), so the audio still says the rewritten text.

Stdlib only, so both the LLM stack and the light release scripts can import it.
Tested by ``tests/test_label_conventions.py``.

This file was written by Claude Code.
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

# One alphabetic word (Unicode letters only, so digits, "×" and apostrophes split).
WORD_RE = re.compile(r"[^\W\d_]+")


def fold_key(word: str) -> str:
    """Accent- and case-insensitive key: ``Primpéran`` -> ``PRIMPERAN``."""
    return "".join(c for c in unicodedata.normalize("NFD", word)
                   if unicodedata.category(c) != "Mn").upper()


def at_sentence_start(text: str, pos: int) -> bool:
    """True when ``pos`` starts a sentence: the start of the text, or after ".", "!" or "?"
    (spaces allowed). A colon does not end a sentence."""
    prefix = text[:pos].rstrip()
    return not prefix or prefix[-1] in ".!?"


def _cap(word: str) -> str:
    return word[:1].upper() + word[1:]


# ---------------------------------------------------------------------------
# Titles
# ---------------------------------------------------------------------------

# A capitalized word: a name, when it is not one of the words that open a sentence
# (then the "M." before it is a sentence-final initial or TNM stage, not "Monsieur").
_NAME = r"[A-ZÀ-Ý][\w'-]*"
_OPENERS = frozenset("""
    Le La Les L Un Une Des De Du D Il Elle Ils Elles On Nous Vous Je Ce Cet Cette Ces
    Son Sa Ses Leur Leurs Au Aux À En Et Ou Mais Donc Pas Par Pour Sur Sous Dans Avec
    Sans Après Avant Depuis Pendant Lors Il Y Absence Présence Pas Aucun Aucune Tout
    Toute Tous Toutes Cela Ceci Selon Suite Chez Quant Puis Enfin Ainsi Alors
""".split())
# A title word right before "M." makes that "M." an initial (Madame M. Dupont).
_TITLE_BEFORE = re.compile(r"(?:Monsieur|Madame|Mademoiselle|Mme|Mlle|docteur|Docteur|Dr|"
                           r"professeur|Professeur|Pr)\s+$")
_M_AND_MME_RE = re.compile(r"(?<![\w.'-])M\.?\s+et\s+Mme\b\.?")
_M_DOT_RE = re.compile(rf"(?<![\w.'-])M\.(?=\s+{_NAME})")
_ABBREV_RE = re.compile(r"(?<![\w.'-])(Mmes|Mme|Mlles|Mlle)\b\.?")
_ABBREV_FULL = {"Mme": "Madame", "Mmes": "Mesdames", "Mlle": "Mademoiselle", "Mlles": "Mesdemoiselles"}
_DR_PR_RE = re.compile(rf"(?<![\w.'-])(Drs|Dr|Pr)\b\.?(?=\s+{_NAME})")
_DR_PR_FULL = {"Dr": "docteur", "Drs": "docteurs", "Pr": "professeur"}
# Casing of the spelled-out titles before a name.
_UPPER_TITLE_RE = re.compile(rf"\b(monsieur|madame|mademoiselle)(?=\s+{_NAME})")
_LOWER_TITLE_RE = re.compile(rf"\b(Docteur|Professeur)(?=\s+{_NAME})")


def _is_name(following: str) -> bool:
    """Whether the text after a title starts with a name. A sentence opener counts
    only when a capitalized word follows it (``M. Le Corre``, not ``M. Le patient``)."""
    words = following.split(maxsplit=2)
    if not words:
        return False
    if words[0].strip(".,;:") not in _OPENERS:
        return True
    return len(words) > 1 and words[1][:1].isupper()


def expand_titles(text: str) -> str:
    """``M. Dupont`` -> ``Monsieur Dupont``, ``Dr Martin`` -> ``docteur Martin``, etc. Idempotent."""
    text = _M_AND_MME_RE.sub("Monsieur et Madame", text)

    def m_dot(m: re.Match) -> str:
        if not _is_name(text[m.end():]) or _TITLE_BEFORE.search(text[:m.start()]):
            return m.group(0)
        return "Monsieur"
    text = _M_DOT_RE.sub(m_dot, text)
    text = _ABBREV_RE.sub(lambda m: _ABBREV_FULL[m.group(1)], text)

    def dr_pr(m: re.Match) -> str:
        full = _DR_PR_FULL[m.group(1)]
        return _cap(full) if at_sentence_start(text, m.start()) else full
    text = _DR_PR_RE.sub(dr_pr, text)

    def upper(m: re.Match) -> str:
        return _cap(m.group(1)) if _is_name(text[m.end():]) else m.group(1)
    text = _UPPER_TITLE_RE.sub(upper, text)
    return _LOWER_TITLE_RE.sub(
        lambda m: m.group(1) if at_sentence_start(text, m.start()) else m.group(1).lower(), text)


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

_UNITS = ["zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix",
          "onze", "douze", "treize", "quatorze", "quinze", "seize"]
_TENS = {20: "vingt", 30: "trente", 40: "quarante", 50: "cinquante", 60: "soixante"}


def _spell(n: int) -> set[str]:
    """Every spelling of 0..99, words joined by single spaces (hyphens folded to spaces)."""
    if n <= 16:
        return {_UNITS[n]}
    if n < 20:
        return {f"dix {_UNITS[n - 10]}"}
    if n < 70:
        tens, unit = n // 10 * 10, n % 10
        if unit == 0:
            return {_TENS[tens]}
        if unit == 1:
            return {f"{_TENS[tens]} et un"}
        return {f"{_TENS[tens]} {u}" for u in _spell(unit)}
    if n < 80:
        rest = n - 60
        if rest == 11:
            return {"soixante et onze", "soixante onze"}
        return {f"soixante {s}" for s in _spell(rest)}
    if n == 80:
        return {"quatre vingts", "quatre vingt"}
    return {f"quatre vingt {s}" for s in _spell(n - 80)}


def _build_years() -> dict[str, int]:
    years: dict[str, int] = {}
    for rest in range(100):
        tails = [""] if rest == 0 else [" " + s for s in _spell(rest)]
        for tail in tails:
            for head in ("mille neuf cent", "mil neuf cent", "dix neuf cent"):
                years[head + tail] = 1900 + rest
            years["deux mille" + tail] = 2000 + rest
    for head in ("mille neuf cents", "dix neuf cents"):
        years[head] = 1900
    return years


_YEARS = _build_years()
_DAYS = {s: n for n in range(2, 32) for s in _spell(n)}
_NUM_WORDS = sorted({w for s in list(_YEARS) + list(_DAYS) for w in s.split()} | {"cents"},
                    key=len, reverse=True)
# A run of number words, hyphens or spaces between them.
_SEQ = rf"(?i:\b(?:{'|'.join(_NUM_WORDS)})(?:[\s-]+(?:{'|'.join(_NUM_WORDS)}))*\b)"
_MONTHS = (r"(?:janvier|f[ée]vrier|mars|avril|mai|juin|juillet|ao[uû]t|septembre|octobre"
           r"|novembre|d[ée]cembre)")
_YEAR_AFTER_MONTH_RE = re.compile(rf"(\b{_MONTHS}\s+)({_SEQ})")
_YEAR_AFTER_EN_RE = re.compile(rf"(\b(?:en|depuis)\s+)({_SEQ})")
_DAY_BEFORE_MONTH_RE = re.compile(rf"({_SEQ})(\s+{_MONTHS}\b)")
# The first day of a range written before a converted one: "du six au 13 mai".
_DAY_BEFORE_RANGE_RE = re.compile(rf"({_SEQ})(\s+au\s+(?:\d{{1,2}}|1er)\s+{_MONTHS}\b)")
_PREMIER_RE = re.compile(rf"\b[Pp]remier(\s+{_MONTHS}\b|\s+au\s+\d{{1,2}}\s+{_MONTHS}\b)")
# What may follow a year after "en" / "depuis" for it to be a year and not a count.
_AFTER_EN_YEAR_RE = re.compile(r"^(?:\s*(?:[.,;:!?)]|$)|\s+(?:au|et|puis)\b)")


def _words(seq: str) -> list[str]:
    return re.split(r"[\s-]+", seq)


def _year_prefix(seq: str) -> tuple[int, int] | None:
    """Longest leading run of ``seq``'s words that spells a year: (year, n_words)."""
    words = _words(seq)
    for n in range(len(words), 1, -1):
        year = _YEARS.get(" ".join(words[:n]).lower())
        if year:
            return year, n
    return None


def _replace_year(m: re.Match, text: str, guard: bool) -> str:
    seq = m.group(2)
    found = _year_prefix(seq)
    if not found:
        return m.group(0)
    year, n = found
    # Keep the separators of the words after the year exactly as written.
    parts = re.split(r"([\s-]+)", seq)
    rest = "".join(parts[2 * n - 1:])
    if guard and not _AFTER_EN_YEAR_RE.match(rest + text[m.end():]):
        return m.group(0)
    return f"{m.group(1)}{year}{rest}"


def _replace_day(m: re.Match) -> str:
    seq = m.group(1)
    words = _words(seq)
    for n in range(len(words), 0, -1):
        day = _DAYS.get(" ".join(words[-n:]).lower())
        if day:
            parts = re.split(r"([\s-]+)", seq)
            head = "".join(parts[:len(parts) - (2 * n - 1)])
            return f"{head}{day}{m.group(2)}"
    return m.group(0)


def normalize_dates(text: str) -> str:
    """``le quinze mars deux mille vingt`` -> ``le 15 mars 2020``, ``premier mai`` -> ``1er mai``. Idempotent."""
    text = _YEAR_AFTER_MONTH_RE.sub(lambda m: _replace_year(m, text, guard=False), text)
    text = _YEAR_AFTER_EN_RE.sub(lambda m: _replace_year(m, text, guard=True), text)
    text = _DAY_BEFORE_MONTH_RE.sub(_replace_day, text)
    text = _DAY_BEFORE_RANGE_RE.sub(_replace_day, text)
    return _PREMIER_RE.sub(r"1er\1", text)


# ---------------------------------------------------------------------------
# Clock times
# ---------------------------------------------------------------------------

# "14h30", "14 h 30", "20h", "20 h 00"; an uppercase H only attached and with
# minutes ("14H30"), since "les 4H et 4T" is the cardiac-arrest mnemonic. A number
# followed by a comma and another number is an enumeration of labels ("blocs 1a à
# 1h, 2, 3a"), not a time.
_CLOCK_RE = re.compile(r"(?<![\w,.])(\d{1,2})(?:\s?h(?:\s?(\d{2}))?|H(\d{2}))(?![\w])(?!,\s*\d)")


def normalize_clock(text: str) -> str:
    """``14h30`` -> ``14 heures 30``, ``09h00`` -> ``9 heures``, ``1h`` -> ``1 heure``. Idempotent."""
    def repl(m: re.Match) -> str:
        hours, minutes = int(m.group(1)), m.group(2) or m.group(3)
        if minutes is not None and int(minutes) >= 60:
            return m.group(0)
        out = f"{hours} {'heure' if hours <= 1 else 'heures'}"
        return out if minutes in (None, "00") else f"{out} {minutes}"
    return _CLOCK_RE.sub(repl, text)


# ---------------------------------------------------------------------------
# Drug names
# ---------------------------------------------------------------------------

DRUG_CASING_PATH = Path(__file__).resolve().parent / "drug_casing.json"


class DrugCaser:
    """Rewrite each drug word to its canonical spelling (see ``02_drugs/02_build_drug_casing.py``).

    Parameters
    ----------
    caps : dict[str, str]
        Accent-free ALLCAPS key -> canonical form, for ALLCAPS spellings.
    variants : dict[str, str]
        Lowercased non-ALLCAPS spelling -> canonical form.
    """

    def __init__(self, caps: dict[str, str], variants: dict[str, str]) -> None:
        self.caps = caps
        self.variants = variants

    @classmethod
    def from_json(cls, path: Path = DRUG_CASING_PATH) -> "DrugCaser":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(caps=data["caps"], variants=data["variants"])

    def __call__(self, text: str) -> str:
        """Idempotent: every canonical form maps to itself."""
        def repl(m: re.Match) -> str:
            w = m.group(0)
            if w.isupper() and len(w) > 1:
                canon = self.caps.get(fold_key(w))
            else:
                canon = self.variants.get(w.lower())
                # A lowercase canonical keeps the capital the spelling had (a
                # list item, a heading): only the accents are harmonized.
                if canon and w[0].isupper():
                    canon = _cap(canon)
            if canon is None:
                return w
            return _cap(canon) if at_sentence_start(text, m.start()) else canon
        return WORD_RE.sub(repl, text)


@lru_cache(maxsize=1)
def default_drug_caser() -> DrugCaser:
    return DrugCaser.from_json()


def apply_label_conventions(text: str) -> str:
    """All four rewrites, titles first so ``M.`` is gone before sentence starts are judged."""
    return default_drug_caser()(normalize_clock(normalize_dates(expand_titles(text))))
