"""One written form for titles, dates, clock times, quantities, sutures, compounds, spelling variants and drug names in the ASR label.

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
- **Quantities** (``normalize_quantities``). A spelled number before a unit of measure
  or a duration becomes digits (``quatre milligrammes`` -> ``4 milligrammes``, 62k
  against 1.9k; ``zéro virgule vingt-cinq microgrammes`` -> ``0,25 microgrammes``;
  ``pendant dix-huit mois`` -> ``pendant 18 mois``), and so do ``fois`` / ``séances``
  counts, a range head and a score (``deux fois`` -> ``2 fois``, ``un à deux jours`` ->
  ``1 à 2 jours``, ``deux sur dix`` -> ``2 sur 10``, 2026-09-28). Other counts of things
  stay spelled, as the corpus spells them (``deux comprimés``), and so does a single
  ``un``/``une`` (``un an``, ``une heure``). Added 2026-09-28 because the hand-made
  ``drug_sentence`` sets spelled every dose while UltiMed writes digits: the model
  wrote ``4 milligrammes`` on the synthetic voice and 63% of that set's WER was
  number formatting, not recognition. The radiology units (``grays``, ``hertz``,
  ``teslas``) count as units, and a decimal written half in digits becomes a decimal
  (``3 virgule 5 mégahertz`` -> ``3,5 mégahertz``).
- **Suture gauges** (``normalize_sutures``, label only). ``Vicryl trois zéro`` /
  ``Vicryl 3 zéros`` / ``Vicryl 3/0`` -> ``Vicryl 3-0``, only after a suture material.
  Skipped for a TTS source (``tts_source=True``) because voxtral does not read ``3-0``
  reliably as what the source said.
- **Staging numbers** (``normalize_staging``). After ``stade``, ``grade``, ``type``,
  ``classe``, ``palier``, ``niveau`` or ``NYHA``, a Roman or spelled number becomes digits
  (``stade IIIb`` -> ``stade 3b``, ``palier deux`` -> ``palier 2``), the author's choice
  (2026-09-28) over a corpus that mixed all three. Roman numerals elsewhere
  (``angiotensine II``) stay.
- **Compounds** (``normalize_compounds``). ``petit déjeuner`` -> ``petit-déjeuner``
  (213 against 35, and the dictionary spelling of the noun), and a split prefix is glued
  (``extra hépatiques`` -> ``extra-hépatiques``, ``multi lithiasique`` ->
  ``multilithiasique``).
- **Spelling variants** (``normalize_spelling``). ``œ`` -> ``oe`` everywhere (``cœur`` ->
  ``coeur``), ``aigüe`` -> ``aiguë``, ``compte-rendu`` -> ``compte rendu``,
  ``bêta-bloquant`` / ``bêtabloquant`` -> ``bétabloquant``, ``anévrysme`` -> ``anévrisme``,
  ``urèthre`` / ``uréthral`` -> ``urètre`` / ``urétral``, and a sentence-initial ``A``
  before an infinitive -> ``À`` (``A surveiller`` -> ``À surveiller``). Added 2026-09-28 from an
  inference sweep of val/test, where these pairs were among the most frequent diffs.
- **Drug names** (``DrugCaser``). One spelling per drug word, from the lexicon
  ``drug_casing.json`` built by ``02_drugs/02_build_drug_casing.py`` (see its
  docstring for how the canonical form is chosen): ``PRIMPERAN`` / ``Primperan`` ->
  ``Primpéran``, ``PARACETAMOL`` -> ``paracétamol``, ``UVEDOSE`` -> ``Uvedose``, with
  ALLCAPS acronyms (``LP``, ``BCG``) protected by the lexicon.

``apply_label_conventions`` runs them all. The text generators apply it in
``_pipeline_shared.parse_asr_training_target``, so every freshly generated label is
canonical (and so is the TTS source ``voxtral_normalize`` derives from it);
``99_hf_release/05_normalize_text.py`` applies it to the v1 manifests, to both the
label and the TTS source: every rewrite here is spoken identically before and after
(``M.`` was read ``monsieur``, ``2019`` and ``deux mille dix-neuf`` are the same
words, ``4 milligrammes`` is read ``quatre milligrammes``, a hyphen and casing are
silent), so the audio still says the rewritten text.

Stdlib only, so both the LLM stack and the light release scripts can import it.
Tested by ``tests/test_label_conventions.py``.

This file was written by Claude Code.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

# voxtral_normalize is a sibling module (stdlib only too), whatever the caller's sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from voxtral_normalize import roman_to_int  # noqa: E402

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
# Quantities
# ---------------------------------------------------------------------------

# Units of measure and durations: the corpus writes the number before them in digits
# (``milligrammes`` 62k digits against 1.9k words, ``ans`` 241k / 2.9k, ``mois``
# 64k / 11k, ``heures`` 38k / 3.3k). ``fois`` and ``séances`` were split (8.9k digits
# against 31.7k words, 1.8k against 2.2k) and the author chose digits (2026-09-28,
# "2 fois par jour"), although words were the majority. Other counts of things are NOT
# here, the corpus spells them (``comprimés`` 263 digits / 406 words, ``bouffées``,
# ``gélules``, ``prises``, ``doses``): "deux comprimés de 4 milligrammes". The singular
# forms only follow a decimal (``zéro virgule cinq milligramme``), since ``un``/``une``
# is never converted (``une fois par jour``: 4.8k against 64 ``1 fois``).
_QTY_UNITS = (r"(?:(?:milli|micro|nano|kilo)?grammes?|kilos?|(?:milli|micro|centi)?litres?"
              r"|(?:milli|centi|kilo)?mètres?|(?:milli|micro)moles?|unités?|degrés?"
              r"|heures?|minutes?|secondes?|jours?|semaines?|mois|ans?|fois|séances?"
              # Radiology units: 588 labels write "N grays", 11 spelled them; hertz
              # 356 / 3, teslas 114 / 3.
              r"|(?:milli|centi)?grays?|(?:kilo|méga)?hertz|teslas?)")
# Every word a French number up to 999,999 is spelled with ("et" only inside one).
_NUMBER_VALUES = {w: i for i, w in enumerate(_UNITS)} | {w: n for n, w in _TENS.items()}
_NUMBER_VALUES |= {"une": 1, "vingts": 20}
_QTY_WORD = rf"(?:{'|'.join(sorted(_NUMBER_VALUES, key=len, reverse=True))}|cents?|mille|mil|et)"
_QTY_SEQ = rf"(?i:(?<![\w-]){_QTY_WORD}(?:[\s-]+{_QTY_WORD})*)"
# A ratio's second number ("cinq jours sur sept") follows the first, else the label
# mixes forms ("5 jours sur sept") the corpus never uses (362 all-words, 280 all-digits).
# Not right after "pour": "0,10 gramme pour cent une fois par jour" is a percentage
# followed by a count, not "101 fois" (the match then starts at "une", which stays).
_QUANTITY_RE = re.compile(rf"(?<!\bpour )({_QTY_SEQ})(?:(\s+virgule\s+)({_QTY_SEQ}))?(\s+{_QTY_UNITS})\b"
                          rf"(?:(\s+sur\s+)({_QTY_SEQ})\b)?")
# A decimal the LLM wrote half in digits (94 labels, "3 virgule 5 mégahertz"): the
# corpus writes "3,5" everywhere else.
_DIGIT_VIRGULE_RE = re.compile(r"(?<![\w,])(\d+) virgule (\d+)\b")
# The first number of a range whose second one is already digits ("deux à 3 jours",
# ~200 labels, and what ``_QUANTITY_RE`` alone makes of "deux à trois jours", since only
# the number next to the unit matches it). "un à 2 jours" -> "1 à 2 jours" too.
_RANGE_HEAD_RE = re.compile(rf"({_QTY_SEQ})(\s+(?:à|ou)\s+\d+(?:,\d+)?\s+{_QTY_UNITS})\b")
# A score out of ten, twenty or a hundred (pain scale, "deux sur dix"): 6k labels write
# "2 sur 10", 146 spell it.
_SCORE_BASE = {"dix": 10, "vingt": 20, "cent": 100}
_SCORE_RE = re.compile(rf"({_QTY_SEQ})(\s+sur\s+)(dix|vingt|cent)\b")
# Suture gauges: "Vicryl trois zéro" (~250), "Vicryl 3 zéro" (~255), "Vicryl 3-0" (~300,
# also what the model writes). Only right after a suture material, optionally followed
# by "rapide" / "résorbable" / "fast", because "N zéro" elsewhere is not a gauge
# ("schéma zéro un zéro", a score). "Vicryl zéro" (a gauge of plain 0) stays.
_SUTURE_MATERIAL = (r"vicryl|monocryl|prol[èe]ne|pds|surgil[èe]ne|ticron|monofil(?:ament)?|fils?"
                    r"|[dl]'[eé]th[iy]lon|[eé]th[iy]lon|polysorb|fil ?[àa] ?peau|polydioxanone"
                    r"|monosyn|v-?lo[c]?k|velock|flexocrin|nylon|soie|caprosyn|dafilon"
                    r"|polypropyl[èe]ne|r[ée]sorbables?")
_GAUGE_NUM = r"un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|\d{1,2}"
_SUTURE_RE = re.compile(
    rf"(?i:\b((?:{_SUTURE_MATERIAL})(?:\s+(?:rapide|r[ée]sorbable|fast))?\s+)({_GAUGE_NUM})"
    rf"(?:[\s-]+zéros?\b|/0\b))")


def parse_french_number(words: list[str]) -> int | None:
    """Value of spelled-out French number words, or None when they do not spell one.

    ``["deux", "cent", "cinquante"]`` -> 250, ``["quatre", "vingt", "dix", "sept"]``
    -> 97, ``["trois", "mille", "cinq", "cents"]`` -> 3500. Hyphens must already be
    split off. Lenient on spelling variants (``cent``/``cents``, ``vingt``/``vingts``),
    strict on structure (``_can_follow``, one ``mille``, one ``cent`` per thousand).
    """
    total, current = 0, 0
    for i, raw in enumerate(words):
        w = raw.lower()
        if w == "et":
            if i == 0 or i == len(words) - 1:
                return None
            continue
        if w in ("cent", "cents"):
            if current >= 100:
                return None
            current = (current or 1) * 100
        elif w in ("mille", "mil"):
            if total:
                return None
            total, current = (current or 1) * 1000, 0
        elif w in ("vingt", "vingts") and current % 100 == 4:
            current += 76  # "quatre vingt" is 80
        elif w in _NUMBER_VALUES:
            if not _can_follow(current % 100, _NUMBER_VALUES[w]):
                return None
            current += _NUMBER_VALUES[w]
        else:
            return None
    return total + current


def _can_follow(r: int, v: int) -> bool:
    """Whether a unit/tens word worth ``v`` may follow a number ending in ``r`` (its last
    two digits). Rejects juxtapositions that are not one number: ``deux trois jours``
    (two or three days) is not 5, ``vingt trente`` is not 50."""
    if r == 0:
        return True
    if r % 10 == 0 and r >= 20:
        return v < 20 if r in (60, 80) else v < 10
    return r in (10, 70, 90) and v in (7, 8, 9)


def _decimal_digits(words: list[str]) -> str | None:
    """Digits after "virgule" as said: ``zéro cinq`` -> ``05``, ``cinquante`` -> ``50``."""
    zeros = 0
    while zeros < len(words) and words[zeros].lower() == "zéro":
        zeros += 1
    rest = parse_french_number(words[zeros:]) if words[zeros:] else None
    if words[zeros:] and rest is None:
        return None
    return "0" * zeros + ("" if rest is None else str(rest))


def _replace_quantity(m: re.Match) -> str:
    whole = _words(m.group(1))
    # A leading "et" belongs to the sentence ("et deux milligrammes"), not the number.
    head = ""
    while whole and whole[0].lower() == "et":
        head += whole.pop(0) + " "
    value = parse_french_number(whole) if whole else None
    if value is None:
        return m.group(0)
    ratio = ""
    if m.group(6) is not None:
        per = parse_french_number(_words(m.group(6)))
        # Not a ratio number ("sur deux trois"): leave the tail as written.
        ratio = f"{m.group(5)}{per}" if per is not None else f"{m.group(5)}{m.group(6)}"
    if m.group(3) is not None:
        decimals = _decimal_digits(_words(m.group(3)))
        if decimals is None:
            return m.group(0)
        return f"{head}{value},{decimals}{m.group(4)}{ratio}"
    if value == 1:
        # "un milligramme", "une heure", "un an": the corpus spells a single one.
        return m.group(0)
    return f"{head}{value}{m.group(4)}{ratio}"


def normalize_quantities(text: str) -> str:
    """``quatre milligrammes`` -> ``4 milligrammes``, ``zéro virgule vingt-cinq microgrammes``
    -> ``0,25 microgrammes``, ``toutes les huit heures`` -> ``toutes les 8 heures``.

    Only before a unit of measure, a duration, ``fois`` or ``séances`` (``_QTY_UNITS``);
    other counts (``deux comprimés``) and a single ``un``/``une`` stay spelled, except as
    the head of a range (``un à deux jours`` -> ``1 à 2 jours``). A score becomes digits
    too (``deux sur dix`` -> ``2 sur 10``). Minutes after
    a spelled hour (``huit heures trente``) are left alone: telling them from a count
    that follows (``deux heures deux fois par jour``) needs more than a regex. Idempotent.
    """
    text = _QUANTITY_RE.sub(_replace_quantity, text)
    text = _RANGE_HEAD_RE.sub(lambda m: _digits_or_keep(m, 1) + m.group(2), text)
    text = _SCORE_RE.sub(lambda m: f"{_digits_or_keep(m, 1)}{m.group(2)}{_SCORE_BASE[m.group(3)]}"
                         if parse_french_number(_words(m.group(1))) is not None else m.group(0), text)
    return _DIGIT_VIRGULE_RE.sub(r"\1,\2", text)


def _digits_or_keep(m: re.Match, group: int) -> str:
    """Group ``group`` of ``m`` (spelled number words, maybe led by a sentence "et") as
    digits, or unchanged when the words do not spell one number."""
    whole = _words(m.group(group))
    head = ""
    while whole and whole[0].lower() == "et":
        head += whole.pop(0) + " "
    value = parse_french_number(whole) if whole else None
    return m.group(group) if value is None else f"{head}{value}"


def normalize_sutures(text: str) -> str:
    """Suture gauges to ``N-0``: ``Vicryl trois zéro`` -> ``Vicryl 3-0``.

    Label only (``apply_label_conventions(..., tts_source=True)`` skips it): voxtral
    does not read ``3-0`` reliably as "trois zéro" (Whisper heard ``3-0``, ``3.0`` and
    ``2 à 0`` on such clips), so a TTS source that said ``trois zéro`` keeps saying it.
    """
    return _SUTURE_RE.sub(_replace_suture, text)


def _replace_suture(m: re.Match) -> str:
    """``Vicryl trois zéro`` / ``Vicryl 3 zéros`` / ``Vicryl 3/0`` -> ``Vicryl 3-0``."""
    num = m.group(2)
    value = num if num.isdigit() else parse_french_number([num.lower()])
    return f"{m.group(1)}{value}-0"


# ---------------------------------------------------------------------------
# Staging numbers
# ---------------------------------------------------------------------------

# After a staging / classification word the corpus mixed Roman numerals, digits and
# words (stade 5.4k Roman / 0.4k digits, grade 2.9k / 2.4k, type 6.2k / 3.4k, classe
# 0.6k / 0.7k, palier 0.8k / 5.0k / 0.5k "palier deux"), the same sound written three
# ways. The author chose digits for all of them (2026-09-28): "stade 3", "grade 2b",
# "type 1 et 2". Only these words: a Roman numeral elsewhere is a name ("angiotensine
# II", "APACHE II", "métaphase II", "Henri IV") and stays. Voxtral read every form the
# same, so the TTS source gets the rewrite too.
_STAGING_WORD = r"\b(?:[Ss]tades?|[Gg]rades?|[Tt]ypes?|[Cc]lasses?|[Pp]aliers?|[Nn]iveaux?|NYHA)"
# A Roman numeral (uppercase only: "civil" is valid Roman letters) with an optional
# sub-stage letter ("IIIb", "IVB"), a spelled number, or digits. "un" not before "peu"
# ("un type un peu particulier").
_STAGE_NUM = (r"(?:[IVX]+[A-Da-d]?|(?:un(?!\s+peu\b)|deux|trois|quatre|cinq|six|sept|huit|neuf|dix)"
              r"|\d+[A-Da-d]?)")
_STAGING_RE = re.compile(rf"({_STAGING_WORD}\s+)({_STAGE_NUM}(?:\s*(?:-|/|à|et|ou|,)\s*{_STAGE_NUM})*)\b")
_STAGE_TOKEN_RE = re.compile(r"\b([IVX]+)([A-Da-d]?)\b|\b(un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix)\b")


def _stage_token(m: re.Match) -> str:
    if m.group(3):
        return str(_NUMBER_VALUES[m.group(3)])
    n = roman_to_int(m.group(1))
    # A letter that happens to be Roman ("type C" is 100, "classe D" is 500) is not a
    # stage, nor is a karyotype ("type XX" is not 20): stages stop at 12.
    return m.group(0) if n is None or not 1 <= n <= 12 else f"{n}{m.group(2)}"


def normalize_staging(text: str) -> str:
    """``stade IIIb`` -> ``stade 3b``, ``palier deux`` -> ``palier 2``, ``grade I à II``
    -> ``grade 1 à 2``, ``classe III NYHA`` -> ``classe 3 NYHA``. Idempotent."""
    return _STAGING_RE.sub(lambda m: m.group(1) + _STAGE_TOKEN_RE.sub(_stage_token, m.group(2)), text)


# ---------------------------------------------------------------------------
# Hyphenated compounds
# ---------------------------------------------------------------------------

# ``petit-déjeuner`` (213 labels) against ``petit déjeuner`` (35); also the
# dictionary spelling of the noun. The WER normaliser deletes hyphens, so the two
# spellings are two different tokens (``petitdéjeuner`` / ``petit déjeuner``) and
# cost a word every time they disagree.
_PETIT_DEJEUNER_RE = re.compile(r"\b([Pp]etits?) (déjeuners?)\b")


# Prefixes that are never a word on their own, written split before an adjective in
# ~560 labels ("extra hépatiques", "multi lithiasique", "péri ombilicale") against 13.6k
# hyphenated ("intra-utérin") and many glued ("intracrânienne"). The normaliser deletes
# hyphens, so hyphenated and glued are the same token and only the split form costs a
# word. Glued, or hyphenated when the word starts with a vowel ("intra-abdominal", not
# "intraabdominal"), the author's choice (2026-09-28). Not before a conjunction or a
# preposition: "intra et extra-hépatiques" keeps its "intra".
_PREFIX_RE = re.compile(r"\b((?i:multi|extra|intra|supra|infra|péri)) "
                        r"(?!(?:et|ou|ni|sur|de|du|des|à|en)\b)([a-zà-ÿ][^\W\d_]{3,})")


def _glue_prefix(m: re.Match) -> str:
    sep = "-" if m.group(2)[0] in "aeiouyàâéèêëîïôöûüh" else ""
    return f"{m.group(1)}{sep}{m.group(2)}"


def normalize_compounds(text: str) -> str:
    """``petit déjeuner`` -> ``petit-déjeuner``, ``extra hépatiques`` -> ``extra-hépatiques``,
    ``multi lithiasique`` -> ``multilithiasique``. Idempotent."""
    return _PREFIX_RE.sub(_glue_prefix, _PETIT_DEJEUNER_RE.sub(r"\1-\2", text))


# ---------------------------------------------------------------------------
# Spelling variants
# ---------------------------------------------------------------------------

# Pairs spelled two ways with no difference in sound. The WER normaliser keeps both
# spellings apart, so every disagreement costs a word even though the audio cannot
# tell them apart. The choices below are the author's (2026-09-28), mostly the majority:
# - ``oe`` for the ``œ`` ligature, everywhere (``oeil`` 10.7k against ``œil`` 1.5k,
#   ``oedème`` 4.4k against 2.7k; ``cœur`` and ``œsophage`` were the majority but one
#   rule for every word beats a per-word choice nobody can remember).
# - ``aiguë`` (28.4k) over ``aigüe`` (1.4k), same for ``subaiguë``, ``contiguë``,
#   ``ambiguïté``... (the pre-1990 spelling, the corpus majority). ``Argüelles`` (a name)
#   is left alone because ``güe`` must end the word.
# - ``compte rendu`` (81k) over ``compte-rendu`` (2k).
# - ``bétabloquant`` (1.2k) over ``bêtabloquant`` (1.1k) / ``bêta-bloquant`` (0.1k).
# - ``anévrisme`` (2.7k) over ``anévrysme`` (1.0k).
# - ``urètre`` / ``urétral`` without the h, everywhere (2.8k against 2.3k over the
#   noun and its derivatives together; the noun alone preferred ``urèthre``, 741 / 274,
#   but one rule for the family beats a noun and adjective spelled differently).
_SPELLING_RULES = (
    (re.compile("œ"), "oe"),
    (re.compile("Œ"), "Oe"),
    (re.compile(r"güe(s?)\b"), r"guë\1"),
    (re.compile(r"güité"), "guïté"),
    (re.compile(r"\b([Cc]omptes?)-(rendus?)\b"), r"\1 \2"),
    (re.compile(r"\b([Bb])[êée]ta[- ]?(bloqu(?:ant|eur)\w*)"), r"\1éta\2"),
    (re.compile(r"([Aa])névrysm"), r"\1névrism"),
    (re.compile(r"([Uu])réthr"), r"\1rétr"),
    (re.compile(r"([Uu])rèthr"), r"\1rètr"),
)


# "A surveiller", "A jeun", "A l'examen" at a sentence start (~720 labels, against 8.9k
# "À"): the preposition lost its accent. Only before an infinitive, "jeun", "distance"
# or an article: before a participle it is the verb ("A présenté", "A bien toléré",
# telegraphic notes drop the subject), and "A encore" is ambiguous.
_A_GRAVE_RE = re.compile(r"\bA (?=(?:l'|la\b|jeun\b|distance\b|(?!encore\b|\w+oire\b)[^\W\d_]+(?:er|ir|oir|re)\b))")


def _a_grave(text: str) -> str:
    return _A_GRAVE_RE.sub(lambda m: "À " if at_sentence_start(text, m.start()) else m.group(0), text)


def normalize_spelling(text: str) -> str:
    """``cœur`` -> ``coeur``, ``aigüe`` -> ``aiguë``, ``compte-rendu`` -> ``compte rendu``,
    ``bêta-bloquant`` -> ``bétabloquant``, ``anévrysme`` -> ``anévrisme``, ``urèthre`` ->
    ``urètre``, ``A surveiller.`` -> ``À surveiller.`` Idempotent."""
    for pattern, repl in _SPELLING_RULES:
        text = pattern.sub(repl, text)
    text = _a_grave(text)
    return text


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


def apply_label_conventions(text: str, tts_source: bool = False) -> str:
    """Every rewrite, titles first so ``M.`` is gone before sentence starts are judged,
    and dates before quantities so a spelled year is read as a year, not as a count.

    ``tts_source=True`` is for a text the TTS already read (05 on the v1 sources): it
    skips ``normalize_sutures``, the one rewrite voxtral does not speak identically.
    """
    text = normalize_staging(normalize_quantities(normalize_clock(normalize_dates(expand_titles(text)))))
    if not tts_source:
        text = normalize_sutures(text)
    return default_drug_caser()(normalize_spelling(normalize_compounds(text)))
