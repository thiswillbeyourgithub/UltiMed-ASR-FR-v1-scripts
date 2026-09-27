#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru"]
# ///
"""Build ``utils/drug_casing.json``, the lexicon that gives each drug word ONE written form.

UltiMed v1 labels spell the same drug several ways, because the LLM copied the
drug database's ALLCAPS spelling in some sentences and wrote normal French in
others: ``PRIMPERAN`` / ``Primperan`` / ``Primpéran``, ``PARACETAMOL`` / ``paracétamol``,
``SKENAN`` / ``Skénan``. A model trained on that learns no consistent form. The
lexicon this script writes is applied by ``utils/label_conventions.DrugCaser`` (at
generator parse time and by ``99_hf_release/05_normalize_text.py`` on the v1
manifests).

Run from the repo root, after the text stages produced their ``generated_dataset.jsonl``:

    uv run 02_drugs/02_build_drug_casing.py

How the canonical form of a word is chosen (a "word" is one alphabetic token of a
drug term, compared accent- and case-insensitively, so ``PRIMPERAN``, ``Primperan``
and ``Primpéran`` share the key ``PRIMPERAN``):

1. Count every spelling of the key in the labels of the text stages, ignoring
   spellings at a sentence start (their capital says nothing about the word).
2. The most frequent spelling that is not ALLCAPS wins. Substances therefore stay
   lowercase (``paracétamol``, ``bisoprolol``), brands keep their capital
   (``Eliquis``, ``Kardégic``) and presentation words go lowercase (``FORTE`` ->
   ``forte``, ``ADULTES`` -> ``adultes``), all as the corpus itself mostly writes them.
   Two exceptions for a lowercase winner: an ANSM substance takes the ANSM accents,
   and when the accent-free spelling is common too (``MINIMAL_PAIR_SHARE``), the key
   folds two different words (``sucre`` / ``sucré``) and the ALLCAPS letters are kept
   as written, just lowercased.
3. A key never written other than ALLCAPS becomes Title case (``UVEDOSE`` ->
   ``Uvedose``), with the accents of the ANSM substance list
   (``drugs_dosages.jsonl``) when it has the word, none otherwise: there is no source
   to guess a brand's accents from.

ALLCAPS spellings that ARE the right form are protected (``protected`` in the JSON,
for review): words shorter than 4 letters (``LP``, ``BCG``, ``II``), Roman numerals
(``VIII``), words listed in the acronym stage's CSV or written ALLCAPS as a
dictionary term (``AINS``), and 4-letter words with fewer than two vowels
(``DMSA``, ``LHRH``, ``JEXT``: nothing sayable as one word). Longer
vowel-poor words are recased anyway (``TALTZ``, ``VPRIV``, ``VFEND`` are brands).

The JSON holds two maps:

- ``caps``: key -> canonical, applied to every ALLCAPS spelling of the key.
- ``variants``: lowercased non-ALLCAPS spelling -> canonical, the observed
  spellings that differ from the canonical one. Restricted to keys whose canonical
  form is capitalized (a brand: ``primperan`` -> ``Primpéran``, ``skénan`` ->
  ``Skenan``, a lowercase ``eliquis`` -> ``Eliquis``) or that are ANSM substances
  (``paracetamol`` -> ``paracétamol``), so common French words and minimal pairs
  (``des`` / ``dès``) are never touched. A lowercase spelling is only capitalized
  when the word starts some drug term (where brand names sit): ``menthe`` or
  ``fraîcheur`` also end brand names (``... ORAL MENTHE``) but stay lowercase in
  running text. Spellings shorter than 4 letters and those whose last letter differs
  (``codéiné``, ``bétadiné``: adjectives, not misspellings) are left alone.

The output is committed; the stage outputs it is built from are not (they are
large and local), so rerun this only after regenerating texts, and review the diff.

This file was written by Claude Code.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import click
from loguru import logger

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "utils"))
from label_conventions import WORD_RE, at_sentence_start, fold_key  # noqa: E402

STAGES = ("01_dictionnary", "02_drugs", "03_PARHAF", "04_PARROT", "07_acronyms")
ROMAN_RE = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
VOWELS = set("AEIOUY")
# How common the accent-free spelling of a lowercase word must be, relative to its
# most common spelling, to count as a different word rather than a missing accent
# (sucre 139 against sucré 153; the missing accents top out near 0.2, e.g.
# irbesartan 29 against irbésartan 132).
MINIMAL_PAIR_SHARE = 0.3


def title(word: str) -> str:
    """``UVEDOSE`` -> ``Uvedose``."""
    return word[:1].upper() + word[1:].lower()


def protect_reason(key: str, acronyms: set[str]) -> str | None:
    """Why the ALLCAPS spelling of ``key`` must stay ALLCAPS, or None."""
    if len(key) < 4:
        return "len<4"
    if ROMAN_RE.match(key):
        return "roman"
    if key in acronyms:
        return "acronym"
    if len(key) == 4 and sum(c in VOWELS for c in key) < 2:
        return "4 letters, vowels<2"
    return None


@click.command()
@click.option("--out", default=str(_REPO / "utils" / "drug_casing.json"), show_default="utils/drug_casing.json")
def main(out: str) -> None:
    """Build the drug casing / accent lexicon from the stage outputs."""
    # Keys: every word of every drug term.
    # Keys: every word of every drug term. The first word of a term is where the
    # brand name sits in a brand row (DEPAKINE CHRONO; ORAL MENTHE comes after the brand), which
    # decides below whether a lowercase spelling may be capitalized.
    keys: set[str] = set()
    first_words: set[str] = set()
    for row in map(json.loads, open(_REPO / "02_drugs" / "generated_dataset.jsonl", encoding="utf-8")):
        words = [fold_key(w) for w in WORD_RE.findall(row["term"])]
        keys.update(words)
        # A hyphenated head (CHRONO-INDOCID) does not make CHRONO a brand.
        head = WORD_RE.findall(row["term"].split()[0]) if row["term"].split() else []
        # Substance rows start with class words (DIVERS MEDICAMENTS ...), not brands.
        if len(head) == 1 and row.get("type") == "brand":
            first_words.add(fold_key(head[0]))

    # ALLCAPS spellings that are acronyms: the acronym stage's list, plus the
    # dictionary terms written ALLCAPS (AINS, SIDA, ...).
    acronyms = {fold_key(r["TERM"]) for r in csv.DictReader(
        open(_REPO / "07_acronyms" / "wikipedia_acronyms.filtered.authorfiltered.csv", encoding="utf-8"))}
    for row in map(json.loads, open(_REPO / "01_dictionnary" / "generated_dataset.jsonl", encoding="utf-8")):
        term = row["term"]
        if term.isupper() and WORD_RE.fullmatch(term):
            acronyms.add(fold_key(term))

    # ANSM accented substance words: the only accent source for words the corpus
    # never writes in normal case.
    ansm: dict[str, str] = {}
    for row in map(json.loads, open(_REPO / "02_drugs" / "drugs_dosages.jsonl", encoding="utf-8")):
        for w in WORD_RE.findall(row["substance"]):
            ansm.setdefault(fold_key(w), w.lower())

    forms: dict[str, Counter] = defaultdict(Counter)
    caps_count: Counter = Counter()
    for stage in STAGES:
        path = _REPO / stage / "generated_dataset.jsonl"
        if not path.exists():
            logger.warning(f"missing {path}, skipped")
            continue
        for row in map(json.loads, open(path, encoding="utf-8")):
            text = row["asr_training_target"]
            for m in WORD_RE.finditer(text):
                w = m.group(0)
                k = fold_key(w)
                if k not in keys:
                    continue
                if w.isupper() and len(w) > 1:
                    caps_count[k] += 1
                elif not at_sentence_start(text, m.start()):
                    forms[k][w] += 1

    caps: dict[str, str] = {}
    protected: dict[str, str] = {}
    variants: dict[str, str] = {}
    for k in sorted(keys):
        seen = forms.get(k, Counter())
        canonical = seen.most_common(1)[0][0] if seen else None
        if canonical and canonical[0].islower():
            if k in ansm:
                # A substance: the ANSM spelling is the reference for its accents.
                canonical = ansm[k]
            elif seen[k.lower()] >= MINIMAL_PAIR_SHARE * seen[canonical]:
                # The accent-free spelling is common in its own right: the key
                # folds two words together (SANS SUCRE is "sucre", not "sucré"),
                # so the ALLCAPS letters are taken as written.
                canonical = k.lower()
        if k in caps_count:
            why = protect_reason(k, acronyms)
            if why:
                protected[k] = why
            else:
                caps[k] = canonical or title(ansm.get(k, k))
        if canonical is None:
            continue
        if not (canonical[0].isupper() or k in ansm):
            continue
        # Keyed by the lowercased spelling, which DrugCaser looks up. For a
        # lowercase canonical form, a spelling that differs only by its first
        # capital lowercases to the canonical itself and needs no entry.
        # Words shorter than 4 letters are skipped (a lone "f" is not a brand), and
        # so is a spelling whose last letter differs: "codéiné" / "bétadiné" are
        # adjectives in their own right, not misspellings of codéine / Bétadine.
        # A key that never starts a term (CHRONO, MENTHE, FRAICHEUR, DIVERS) is a
        # common word that brand names append, so its lowercase spellings are the
        # common word and must keep their case.
        # DrugCaser looks spellings up lowercased, so any entry of such a key would
        # capitalize the lowercase word too: the key gets no variants at all.
        if k not in first_words and canonical[0].isupper():
            continue
        for w in seen:
            if len(w) >= 4 and w.lower() != canonical and w[-1] == canonical[-1]:
                variants[w.lower()] = canonical

    payload = {
        "_doc": "Built by 02_drugs/02_build_drug_casing.py, applied by utils/label_conventions.DrugCaser. "
                "caps: accent-free ALLCAPS key -> canonical form. variants: lowercased spelling -> canonical. "
                "protected: ALLCAPS keys left ALLCAPS, with the reason.",
        "caps": caps,
        "variants": dict(sorted(variants.items())),
        "protected": protected,
    }
    Path(out).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    logger.info(f"{len(caps)} ALLCAPS keys recased ({sum(caps_count[k] for k in caps)} occurrences), "
                f"{len(protected)} protected, {len(variants)} variant spellings harmonized -> {out}")


if __name__ == "__main__":
    main()
