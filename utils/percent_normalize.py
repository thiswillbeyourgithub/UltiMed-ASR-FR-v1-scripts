"""One written form for a percentage: ``%`` in the ASR label.

UltiMed v1 labels mixed three spellings of the same spoken word (16.6k ``pour cent``,
0.4k ``pourcent``, 11.6k ``%``), so the model learnt no consistent form. The LLM
chooses the spelling (the rewrite prompts' own worked example writes ``35 pour
cent``), and the prompts are left as they are because they are the exact v1
provenance. The label is instead canonicalized AFTER the LLM:

- ``percent_to_symbol`` gives the label form, ``95 pour cent`` / ``95 pourcent`` ->
  ``95 %`` (the space before ``%`` is French typography, and what the v1 rows already
  written with ``%`` do). The text generators apply it in
  ``_pipeline_shared.parse_asr_training_target``, so every freshly generated
  ``asr_training_target`` uses ``%``; ``99_hf_release/05_normalize_text.py`` applies
  it to the v1 manifests, generated before that.
- The TTS source (``asr_training_source``) says ``pourcent``, one word, whatever the
  label wrote. ``percent_sign_to_one_word`` (``95 %`` -> ``95 pourcent``) is the rule
  ``voxtral_normalize`` applies when it derives a fresh source from the canonical
  label. Voxtral reads ``%`` correctly too (``01_dictionnary/VOXTRAL_QUIRKS.md``, row
  ``95 %``), so this is a consistency convention, not an audio fix: every source spells
  a percentage the same way, and that way matches the majority of the v1 audio, which
  was read from ``pour cent``. ``percent_to_one_word`` (``95 pour cent`` ->
  ``95 pourcent``) covers the spelled-out forms; ``99_hf_release/05_normalize_text.py``
  applies both to the v1 sources, whose clips were read from ``pour cent`` or ``%``,
  all spoken exactly like ``pourcent``, so the rewrite only makes the shipped text
  consistent.

"pour cent" is only a percentage after a quantity. The rewrite skips it when the
preceding word is not a number (``106 garçons pour cent filles``, ``2 grammes pour cent
grammes``) and when it is followed by what "cent" is counting (``pour cent
millilitres``, ``3 cas pour cent mille habitants``, ``1,70 m pour cent vingt
kilogrammes``). ``6 volumes pour cent`` and ``0,10 gramme pour cent`` are also left
alone: they are the vol% and g% concentration units, and ``volumes %`` would be no
better a label. In the v1 manifests 26 labels keep such a "pour cent".

Stdlib only, so both the LLM stack and the light release scripts can import it.
Tested by ``tests/test_percent_normalize.py``.

This file was written by Claude Code.
"""
from __future__ import annotations

import re

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
# Any spelled-out "pour cent" still present after the rewrite (a "per hundred X"),
# for reporting.
LEFTOVER_RE = re.compile(r"\bpour\s+cent\b|\bpourcent\b", re.IGNORECASE)


def percent_to_symbol(text: str) -> str:
    """ASR label form: ``95 pour cent`` / ``95 pourcent`` -> ``95 %``. Idempotent."""
    return _PERCENT_RE.sub(r"\1\2%", text)


def percent_to_one_word(text: str) -> str:
    """TTS source form: ``95 pour cent`` -> ``95 pourcent``. Idempotent."""
    return _PERCENT_RE.sub(r"\1\2pourcent", text)


# A written percent sign, with any space before it: "95 %", "95%" and "95 %" all
# become "95 pourcent". No guard needed, unlike "pour cent": "%" is never anything
# but "percent".
_PERCENT_SIGN_RE = re.compile(r"\s*%")


def percent_sign_to_one_word(text: str) -> str:
    """TTS source form of a written sign: ``95 %`` / ``95%`` -> ``95 pourcent``. Idempotent."""
    return _PERCENT_SIGN_RE.sub(" pourcent", text)
