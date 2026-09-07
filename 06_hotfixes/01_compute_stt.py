#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.31",
#   "jiwer>=3.0",
#   "tqdm>=4.66",
#   "loguru>=0.7",
#   "click>=8.1",
# ]
# ///
"""Transcribe every audio clip referenced by a dataset jsonl and score it.

The goal of this stage is quality control: run a reference STT model over the
whole synthesized dataset so we can find the clips where the local TTS engine
totally derailed. For each input line we locate the audio, POST it to an
OpenAI-compatible transcription endpoint, then compute WER and CER between the
STT transcript and the reference text (what the TTS was asked to say). A high
WER/CER means the audio probably does not match its label and should be
inspected or regenerated.

Usage (single manifest):

    uv run 01_compute_stt.py \
        --endpoint http://localhost:8000/v1/audio/transcriptions \
        --api-token "$STT_API_TOKEN" \
        --model whisper-large-v3 \
        --input ../99_hf_release/data/NeMO_files/full.jsonl \
        --output ./stt_out

Usage (a directory, recursively finds every *.jsonl that is not a *.stt.jsonl):

    uv run 01_compute_stt.py --endpoint ... --api-token ... --model ... \
        --input ../99_hf_release/data/NeMO_files --output ./stt_out

For each input jsonl the script writes a sibling `<name>.stt.jsonl` under
--output, mirroring the input folder hierarchy. Every output line keeps all the
original keys and adds a `transcriptions` object keyed by model name:

    "transcriptions": {
        "whisper-large-v3": {"text": "...", "wer": 0.12, "cer": 0.04, ...}
    }

Keying by model means you can rerun with a different --model against the same
--output and the new result is merged into the existing lines. Runs resume:
a line is skipped only if the current model already has a successful entry
(errored attempts are retried). Output is written in input order, flushed to
disk regularly, and every write is atomic (temp file + os.replace).

A row resumes on its audio path AND its text. If the manifest was rebuilt and a
clip filename now carries different text (re-chunked source, regenerated audio),
the stored transcription describes audio that no longer exists, so it is dropped
and the row starts fresh. Rows the new input no longer lists are not carried over
at all, since the output is built from the input file.

Pass --n-parallel N to transcribe N clips concurrently (a thread pool), which
keeps a batching STT server busy; the default of 1 is sequential. Output order,
resume and atomic flushing are unaffected by the concurrency.

Pass --shuffle to randomize the processing order (the written output stays in
input order regardless), so the running rate settles on a representative mix of
clip lengths early. A second progress bar tracks audio-seconds processed out of
the summed clip duration (from each line's `duration`), giving an ETA weighted
by audio length rather than clip count.

This file was written with Claude Code.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
import jiwer
import requests
from loguru import logger
from tqdm import tqdm

# The scorer folds the same symbol / spoken pairs the TTS normalizer spells out, so it
# reads that table instead of keeping a second copy of it (see _UNIT_RULES below).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from voxtral_normalize import (  # noqa: E402
    FR_CARDINAL, STAGING_ROMAN_RE, UNIT_SPOKEN, compile_unit_rules, roman_to_int,
)

# Reference-text fields tried in order when --reference-key is not given. The
# audio was synthesized from asr_training_source, so that is the fairest thing
# to compare the STT transcript against; text / asr_training_target are the
# canonical label fallbacks.
_REFERENCE_FALLBACKS = ("asr_training_source", "text", "asr_training_target")

# WER/CER normalization. The point is to score the WHISPER transcript against a
# Parakeet-style reference label in ONE common character space, so differences that
# are purely how the two systems render text (not real recognition errors) do not
# inflate CER. The steps: lowercase, fold ligatures (see _LIGATURES), strip every
# accent/diacritic, drop punctuation, collapse whitespace. Applied identically to
# reference and hypothesis by compute_metrics().
#
# Relationship to the pipeline's glyph gate (01_dictionnary/02_token_check.py
# NORMALIZATION_RULES): that gate deliberately KEEPS oe/ae ligatures and accents
# because Parakeet's vocab represents them, so the reference labels carry them. This
# scorer deliberately does the OPPOSITE and folds them away, because Whisper renders
# them inconsistently (oe vs oe-ligature, dropped accents) and we do not want that
# style gap counted as error. It is therefore NOT a copy of those rules: the
# typographic rewrites there (dashes, quotes, guillemets, (R), +/*) are all
# non-word characters that this scorer's punctuation strip removes anyway, and the
# letter rewrites there (macron O, g-breve, s-cedilla) are subsumed by the accent
# fold below. Nothing to keep in sync; the two transforms are independent by design.
#
# Numbers are settled (they used to be an open TODO here): both sides are read into
# digits, "stade quatre" and "stade 4" alike, by the number rules further down.
#
# oe/ae ligatures are NOT decomposed by Unicode NFKD, so map them explicitly. Done
# after lowercasing, so the uppercase forms are already folded to these.
_LIGATURES = str.maketrans({"œ": "oe", "æ": "ae"})
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


class EmptyTranscriptError(RuntimeError):
    """The STT endpoint returned empty / blank text on every attempt. Raised so
    callers record it as an error (and --resume retries it) instead of ever storing
    a blank transcript as if it were a valid result. Per-clip, not fatal."""


class TranscriptionError(RuntimeError):
    """The STT request kept failing (network / HTTP) after every retry. A caller
    that wants fail-fast behavior (the improvement pipeline) lets this propagate to
    abort the run; the plain QC pass records it and moves on."""


def strip_accents(text: str) -> str:
    """Remove every diacritic by NFKD-decomposing and dropping the combining marks:
    e-acute -> e, c-cedilla -> c, u-diaeresis -> u, and also non-French ones (macron O,
    g-breve, s-cedilla), so accent-rendering differences never count as CER errors."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


# --- one written form per spoken thing ----------------------------------------------
#
# The audio is the only thing being judged here, but the two sides of the comparison
# write it down differently. The label spells a unit out ("0,8 centimetre", because the
# generator wrote dictation-style French) and Whisper abbreviates whatever it hears
# ("0,8 cm"), so CER pays for 9 characters that no listener would call an error. Measured
# over 60k stored transcripts, the top of the substitution list is almost entirely this:
# monsieur/m (7,662), milligrammes/mg (5,756), madame/mme (4,860), millimetres/mm
# (3,144), pour cent/% (1,195, and worse: the punctuation strip DELETED the % so the
# whole word counted as missing), plus digits against number words both ways (~4,000).
#
# So each of those gets one canonical written form, applied to BOTH sides. The canonical
# is the SPOKEN one (units expand rather than contract) for two reasons: it is what the
# audio actually contains, and it leaves the reference the length it already had, so the
# CER thresholds keep meaning what they meant.
#
# What is deliberately NOT folded: French singular/plural and gender endings
# ("antalgiques"/"antalgique", "suivie"/"suivi"). They are inaudible in French and do show
# up in the diffs, but folding them would also hide a TTS that dropped a word, and at 1-2
# characters they cost far less than the classes above.
def _fold(text: str) -> str:
    """lower + ligatures + accents, the part of the normalization the tables must have
    already been through so their entries match the text the rules run against."""
    return strip_accents((text or "").lower().translate(_LIGATURES))


# Units Voxtral reads correctly and so are absent from its table, but that Whisper still
# abbreviates. Same (symbol, spoken) shape and the same number-anchored compiler, so the
# two tables cannot disagree about what a symbol says. The number anchor is what makes
# the short ones safe: bare "l" is the elided article in "l'artere", bare "g" and "m" are
# not units either unless a number precedes them.
_SCORING_ONLY_UNITS: list[tuple[str, str]] = [
    ("mg", "milligrammes"), ("ng", "nanogrammes"), ("mcg", "microgrammes"),
    ("kg", "kilogrammes"), ("g", "grammes"),
    ("mL", "millilitres"), ("cL", "centilitres"), ("dL", "decilitres"), ("L", "litres"),
    ("mm", "millimetres"), ("cm", "centimetres"), ("m", "metres"),
    ("mmHg", "millimetres de mercure"),
    ("mmol", "millimoles"), ("mol", "moles"), ("mEq", "milliequivalents"),
    ("mUI", "milli-unites internationales"), ("UI", "unites internationales"),
    ("min", "minutes"), ("h", "heures"), ("j", "jours"),
    # Squared / cubed. One transcript token ("750 cm3") against three label words, so
    # these were the second most expensive fake error in the corpus (6,585 characters
    # over 60k clips for cm3 alone). They must precede the bare "cm" / "m" entries,
    # which the length sort below does.
    ("cm2", "centimetres carres"), ("cm3", "centimetres cubes"),
    ("mm2", "millimetres carres"), ("mm3", "millimetres cubes"),
    ("m2", "metres carres"), ("m3", "metres cubes"),
    # Doses per weight or surface the shared table does not carry (it only needs the
    # ones voxtral mispronounces; Whisper abbreviates all of them).
    ("UI/kg", "unites internationales par kilogramme"),
    ("mUI/kg", "milli-unites internationales par kilogramme"),
    ("UI/j", "unites internationales par jour"),
    ("kg/m2", "kilogrammes par metre carre"),
    # Nuclear medicine (dictionary stage: iodine and technetium activities).
    ("GBq", "gigabecquerels"), ("MBq", "megabecquerels"), ("kBq", "kilobecquerels"),
    # Laser fluence (dermatology). Only the milli- form: bare "J" cannot be added
    # because the fold lowercases it into "j", which this table already reads as
    # "jours", and days outnumber joules here by four orders of magnitude.
    ("mJ", "millijoules"),
    # Imaging and physiology, the units the clips no redraw could fix were still being
    # charged for. Counted over the corpus as clips whose label spells the unit out while
    # the transcript abbreviates it: ms 259, kPa 146, MHz 126, kV 50, mA 48. mGy rides
    # along with them (radiology doses, none measured yet, same family and no collision).
    ("ms", "millisecondes"), ("kPa", "kilopascals"), ("MHz", "megahertz"),
    ("kV", "kilovolts"), ("mA", "milliamperes"), ("mGy", "milligrays"),
]

# Case-sensitive, applied to the RAW text before the fold throws case away. In a blood
# count "G/L" is giga (10^9) per litre and "g/L" is grammes per litre: the same three
# characters once lowercased, and the lowercase reading won, so every platelet count
# scored as an error (3,718 characters over 60k clips). Case is the only discriminator,
# so this pair has to be settled while it is still there.
_CASE_UNITS: list[tuple[str, str]] = [
    ("G/L", "giga par litre"), ("T/L", "tera par litre"), ("M/L", "mega par litre"),
]
def _letter_guarded(
    rules: list[tuple[re.Pattern[str], str]],
) -> list[tuple[re.Pattern[str], str]]:
    """Same rules, minus the numbers that hang off a letter.

    Scoring-only, hence not in `compile_unit_rules`: clinical text is full of
    letter+digit codes ("M1M2" for the metatarsals, "T1T2" for the vertebrae) whose
    tail reads as a unit. "M1M2" became "m1 metres carres" while a hyphenated "M1-M2"
    on the other side stayed put, so the code cost more CER than the unit ever saved.
    A real measurement always has a separator or a line start before its number.
    """
    return [(re.compile(r"(?<![A-Za-z])" + pattern.pattern), repl)
            for pattern, repl in rules]


_CASE_UNIT_RULES = _letter_guarded(compile_unit_rules(_CASE_UNITS))

# Symbol -> spoken, number-anchored. Longest symbols first (compile_unit_rules keeps the
# given order) so "mg/L" wins over "mg" and "mmHg" over "mm".
_UNIT_RULES = _letter_guarded(compile_unit_rules(
    sorted([(_fold(sym), _fold(spoken)) for sym, spoken in
            UNIT_SPOKEN + _SCORING_ONLY_UNITS + _CASE_UNITS],
           key=lambda pair: -len(pair[0]))
))


# A unit hanging off "par" instead of off a number. The rules above are number-anchored,
# which covers "42 mg/L" and "42 mg" but not the tail of a rate the two sides split
# differently: the STT writes "15 milligrammes par kg" where the label writes "15
# milligrammes par kilogramme", and only the head carries the number. Left unfolded, "kg"
# alone was the most common surviving diff in the clips no redraw could fix (274
# occurrences over 173 of them) and it appears somewhere in 8,015 clips of the corpus.
#
# Single-letter symbols are EXCLUDED, which is why this is not simply the same table: the
# fold turns "par l'artere" into "par l artere", so a bare "l" after "par" is far more
# often the elided article than a litre, and "g" / "m" / "j" / "h" are no better. Compound
# symbols are excluded too ("par mg/L" is not a thing). What is left is unambiguous:
# nothing but a unit is spelled "kg" or "mmol" after "par".
#
# "kilo" rides here and ONLY here. It is how the generators write a dose ("un milligramme
# par kilo") and how the TTS says it, so the rate tail needs it, but it cannot join the
# table above: number-anchored, it eats the head of `87,5 kilo-unites par litre` (the IgE
# assay unit, kU/L) and turns Whisper's mishearing of "3 culots globulaires" as "3 kilos"
# into a longer error. Both were measured, over the full corpus, not imagined.
_RATE_TAIL_ONLY = [("kilo", "kilogrammes"), ("kilos", "kilogrammes")]
# And bare "mm" is barred. A rate tail is a dose denominator, not a length, so "par mm"
# earns almost nothing (4 clips in the whole corpus, and the compound "par mm3" is covered
# by its own entry), while it does collide: Whisper writes the vaccine M-M-RVAXPRO as
# "MM-VAX-PRO", and "par MM" then becomes "par millimetres".
# "ma" is barred for a blunter reason: after the fold it is the French possessive, so
# "transmis par ma consoeur" would read "par milliamperes consoeur". Nothing is measured
# in milliamperes per anything anyway.
_RATE_TAIL_BARRED = {"mm", "ma"}
_RATE_TAIL_RULES = [
    (re.compile(r"(?<![A-Za-z])par\s+" + re.escape(symbol) + r"(?![/\w²³°µ])"),
     "par " + spoken)
    for symbol, spoken in sorted(
        {(_fold(sym), _fold(sp)) for sym, sp in
         UNIT_SPOKEN + _SCORING_ONLY_UNITS + _CASE_UNITS + _RATE_TAIL_ONLY
         if len(sym) > 1 and "/" not in sym and _fold(sym) not in _RATE_TAIL_BARRED},
        key=lambda pair: -len(pair[0]))
]


def _plural_optional(spoken: str) -> str:
    """Pattern matching a spoken unit however either side inflects it: EVERY word may or
    may not carry its plural "s". Relaxing only the "s" the table itself writes was not
    enough, because the two sides disagree word by word: the table says "milligrammes par
    kilogramme", the label writes "milligrammes par kilogrammes" and the symbol rule emits
    the table's form, so the plural on the last word alone cost 7,128 characters over 60k
    clips, the single most expensive fake error in the corpus."""
    return " ".join(re.escape(w.rstrip("s")) + "s?" for w in spoken.split())


# Spoken -> spoken, to settle singular against plural on either side ("0,8 centimetre"
# in the label, "centimetres" out of the rule above). Built from the same table, so it
# never needs its own list of words.
_SPOKEN_RULES = [
    (re.compile(r"\b" + _plural_optional(spoken) + r"\b"), spoken)
    for spoken in sorted({_fold(sp) for _, sp in
                          UNIT_SPOKEN + _SCORING_ONLY_UNITS + _CASE_UNITS},
                         key=len, reverse=True)
]

# Civility titles. No number anchor (a title never follows one), and none of the
# multi-letter ones is a French word. Bare "m" is the exception and needs context: it is
# also a spelled-out initial, and folding it blindly turned a transcript's "gene L.M.B.R.1"
# into "gene l monsieur b r 1". So it only counts as a title in front of a real word,
# which also rules out the elision "m'a" (no space after the m).
_TITLE_RULES = [
    (re.compile(r"\b(?:" + "|".join(forms) + r")\b(?![’'])"), spoken)
    for forms, spoken in [
        (["mme"], "madame"), (["mlle"], "mademoiselle"), (["mr"], "monsieur"),
        (["dr"], "docteur"), (["pr"], "professeur"),
    ]
] + [(re.compile(r"\bm\.?(?=\s+[^\W\d_]{2,})"), "monsieur")]

# Percent, before the punctuation strip deletes the symbol and leaves the label's
# "pour cent" scoring as three missing words. The canonical is one word on purpose, so
# that its "cent" is not read as the number 100 by the rule below.
_PERCENT_RE = re.compile(r"\s*%|\bpour[-\s]?cents?\b")

# Number words -> digits. Digits are the canonical because Whisper writes them and
# because going the other way would need a French number speller; this direction only
# needs a reader. FR_CARDINAL (0..20) is the same table Voxtral spells staging numerals
# with, plus the tens and scales French composes with.
_NUM_WORD_VALUE: dict[str, int] = {
    # _fold on the way in: FR_CARDINAL is written for the TTS, so its first entry is the
    # accented "zero", which could never match text this module has already accent-folded.
    # That one missing key was 3,832 characters of fake CER over 60k clips.
    **{_fold(w): i for i, w in enumerate(FR_CARDINAL)}, "une": 1,
    "trente": 30, "quarante": 40, "cinquante": 50, "soixante": 60,
    "septante": 70, "octante": 80, "huitante": 80, "nonante": 90,  # Belgian / Swiss
    "vingts": 20, "cent": 100, "cents": 100, "mille": 1000, "milles": 1000,
}
# A run is number words joined by spaces or hyphens, optionally with "et" ("vingt et
# un"). Longest alternatives first so "dix-sept" is not eaten by "dix". The run stops at
# any other word or punctuation, so two numbers in one sentence stay two numbers.
_NUM_WORD_ALT = "|".join(
    re.escape(w) for w in sorted(_NUM_WORD_VALUE, key=len, reverse=True))
_NUMBER_RUN_RE = re.compile(
    rf"\b(?:{_NUM_WORD_ALT})(?:[-\s]+(?:et[-\s]+)?(?:{_NUM_WORD_ALT}))*\b")
_NUM_SPLIT_RE = re.compile(r"[-\s]+")


def _read_fr_number(match: re.Match[str]) -> str:
    """Value of a run of French number words. Handles the composed forms: "quatre-vingt"
    is 4x20 and not 4+20, "soixante-dix" is 60+10, and "cent"/"mille" multiply what came
    before them ("deux cents" 200, "mille neuf cent quarante-neuf" 1949)."""
    total = current = 0
    previous = None
    for word in _NUM_SPLIT_RE.split(match.group(0)):
        if word == "et":
            continue
        value = _NUM_WORD_VALUE[word]
        if value == 20 and previous == 4:      # quatre-vingt: multiplicative, not additive
            current += previous * (value - 1)  # the 4 is already in current: 4 -> 4*20
        elif value == 100:
            current = (current or 1) * 100
        elif value == 1000:
            total += (current or 1) * 1000
            current = 0
        else:
            current += value
        previous = value
    return str(total + current)


def _fold_staging_roman(match: re.Match[str]) -> str:
    """"stade IV" -> "stade 4", the same trigger words and the same 0..20 bound Voxtral
    uses to decide a Roman numeral is a staging number (it speaks "quatre", Whisper
    writes "4", the label kept "IV": three spellings, one number)."""
    word, roman = match.group(1), match.group(2)
    n = roman_to_int(roman.upper())
    return f"{word} {n}" if n is not None and 0 <= n < len(FR_CARDINAL) else match.group(0)


# Case-insensitive twin of Voxtral's staging regex, because scoring lowercases first.
_STAGING_ROMAN_LOWER = re.compile(STAGING_ROMAN_RE.pattern, re.IGNORECASE)


# --- notation the two sides write differently ---------------------------------------
#
# Everything below is the same idea as the unit table, applied to the notation around
# numbers: the label spells a symbol out (the generators speak "+", "/" and "%" rather
# than emit a character Parakeet cannot tokenize) while Whisper writes the symbol, and
# the punctuation strip then DELETES it, so the label's word scores as a pure deletion.
# Each class below was picked off the measured cost ranking, not guessed.

# Ordinals: "le premier mars" against "le 1er mars", and "3eme" / "3eme" / "3e" for one
# spoken word. Canonical is digits + "e", the direction the cardinal reader already goes.
# "second" / "seconde" are deliberately absent: "seconde" is also the time unit, and a
# label that says "deuxieme" against audio that says "second" IS a difference.
_ORDINAL_WORDS = {
    "premier": 1, "premiere": 1, "premiers": 1, "premieres": 1, "unieme": 1,
    "deuxieme": 2, "troisieme": 3, "quatrieme": 4, "cinquieme": 5, "sixieme": 6,
    "septieme": 7, "huitieme": 8, "neuvieme": 9, "dixieme": 10, "onzieme": 11,
    "douzieme": 12, "treizieme": 13, "quatorzieme": 14, "quinzieme": 15,
    "seizieme": 16, "dix-septieme": 17, "dix-huitieme": 18, "dix-neuvieme": 19,
    "vingtieme": 20,
}
# The trailing "s" is optional and dropped before the lookup: "les quatriemes et
# cinquiemes cotes" is written "4e et 5e" by Whisper, which the singular-only rule left as
# a two-word diff on 84 clips.
_ORDINAL_WORD_RE = re.compile(
    r"\b(?:" + "|".join(sorted(_ORDINAL_WORDS, key=len, reverse=True)) + r")s?\b")
# Digit forms, every suffix French and Whisper use: 1er, 1re, 1ere, 2nd is left out with
# "second", 3e, 3eme, 40emes. No \b before the digits is needed, but the digit run must
# start on a boundary so a code like "L2e" is untouched.
_ORDINAL_DIGIT_RE = re.compile(r"\b(\d+)\s*(?:ers?|eres?|res?|emes?|es?)\b")

# Decimal separator, in THREE spellings for one sound: Whisper writes "0,5", the
# generators often spell it ("zero virgule cinq"), and a suture gauge is a fourth case
# again, where the label writes "3 0" for what Whisper writes "3,0". Canonical is the
# separator REMOVED, the only form all of them reach ("0 5", "3 0"). Spelling it out as
# " virgule " instead was measured over 30k clips and made things worse: 3/0 and 4/0
# sutures are in every surgical report, and each one cost 8 characters.
# TIGHT on purpose: no whitespace allowed around the symbol, because a spaced comma
# between two numbers is an enumeration ("deux, trois patients"), not a decimal.
_DECIMAL_RE = re.compile(r"(?<=\d)[.,](?=\d)")
_DECIMAL_WORD_RE = re.compile(r"(?<=\d)\s+virgules?\s+(?=\d)")
# Thousands separator: "11 500" (Whisper, French typography) against the "11500" the
# cardinal reader emits from "onze mille cinq cents". Exactly one group of three digits,
# so it cannot eat two unrelated numbers.
_THOUSANDS_RE = re.compile(r"(?<=\d)[ .](?=\d{3}\b)")
# Leading zeros carry no sound: "05" and "5" are one spoken number. Blocked after a
# decimal separator, where the zero DOES sound ("3,05" must not become "3,5"), which is
# why this runs before the separator is dropped.
_LEADING_ZERO_RE = re.compile(r"(?<![\d.,])\b0+(\d)")

# "+" and "*" are spoken, not dropped, by the generators ("CD4+" -> "CD4 plus", "T2*" ->
# "T2 etoile", see 01_dictionnary/02_token_check.py), so the label carries the word.
_DIGIT_PLUS_RE = re.compile(r"(?<=\d)\s*\+")
_STAR_RE = re.compile(r"(?<=\w)\s*\*")
# A slash between two numbers is read aloud, and which word depends on what is being
# read: "3 barre 0" for a suture gauge, "120 sur 80" for a blood pressure, and sometimes
# nothing at all ("trois zero"). Dropped like the decimal separator, for the same reason:
# deletion is the one form every spelling reaches, and the numbers themselves still have
# to match.
_RATIO_RE = re.compile(r"(?<=\d)\s*(?:/|\bbarres?\b|\bsurs?\b)\s*(?=\d)")
# Day offsets: "J+2" is spoken "J plus 2" and the generators write the words; Whisper
# writes "J2" or "J+2". Restricted to "j" (an "h"/"d" twin would collide with times).
# Canonical is the SHORT form, as for the other separators: Whisper mishears the letter
# often enough ("J1" heard as "G1") that the long form would charge 7 characters for a
# 1-character error.
_DAY_OFFSET_RE = re.compile(r"\bj\s*(?:\+|plus)\s*(\d+)\b|\bj\s*(\d+)\b")
# Compact times: "2h05" is one token to Whisper, four words to the label. Runs before the
# unit rules, which cannot see it ("h" there must not be followed by a digit).
_COMPACT_TIME_RE = re.compile(r"\b(\d{1,2})\s*h\s*(\d{2})\b")


# ...and the word the hour does NOT carry. The compact form is almost entirely Whisper's
# (596 clips against 9 for the label), so an expansion it alone reaches is an expansion
# that charges: measured over the clips whose transcript writes "NhMM", the label stops at
# the number 476 times and says "minutes" 21 times. Rather than pick the majority, the
# word is folded away on BOTH sides, so all four spellings ("14h30", "14 heures 30",
# "14 heures 30 min", "14 heures 30 minutes") land on one string.
_TIME_MINUTES_RE = re.compile(r"\b(\d{1,2} heures \d{1,2}) minutes\b")


def _read_compact_time(match: re.Match[str]) -> str:
    """"2h05" -> "2 heures 5", and "2h00" -> "2 heures", which is how both sides say a
    round hour."""
    hours, minutes = int(match.group(1)), int(match.group(2))
    return f"{hours} heures" if minutes == 0 else f"{hours} heures {minutes}"


# Letter-spelled acronyms: the label writes the sigle joined ("GGT", "ESAT") while
# Whisper sometimes writes the spelling it heard, letter by letter ("G-G-T", "E.S.A.T."),
# and the acronyms stage's TTS sources make that spelling explicit, so neither form is an
# error. Fold a chain of >= 3 single letters joined by "-" or "." into the joined form.
# The chain may not touch surrounding word characters or further separators, and the
# >= 3 minimum keeps two-letter accidents out, so the French euphonic t ("a-t-il",
# "y a-t-il"), digit-bearing codes ("5-F-U", "L.M.B.R.1"), hyphen compounds
# ("post-AVC", "S-T-plus": partial folds are excluded on purpose) and sentence
# boundaries ("vitamine B. Il...") all stay untouched. Runs right after _fold, so it
# sees lowercase text and its output feeds every later number/unit rule unchanged.
_LETTER_CHAIN_RE = re.compile(r"(?<![\w-])[a-z](?:[-.][a-z]){2,}(?![-.]?\w)")
# A spaced variant (folding "o h u v u" the same way, for the rare label that writes
# the spelled letters space-separated) was tried and REJECTED by the replay: it runs
# before the unit rules, so the stt-side unit letter in "2 g à l'induction" formed a
# "g a l" run and folded into "gal'induction", destroying the unit token the label
# spells as "grammes" (52 -> 273 worsened clips). Do not re-add it in this position.


def normalize_for_scoring(text: str) -> str:
    # Case-sensitive first, on the raw text: "G/L" (giga) and "g/L" (grammes) are the
    # same string once folded, and only the label knows which one it meant.
    for pattern, repl in _CASE_UNIT_RULES:
        text = pattern.sub(repl, text or "")
    text = _fold(text)
    text = _LETTER_CHAIN_RE.sub(lambda m: re.sub(r"[-.]", "", m.group(0)), text)
    text = _PERCENT_RE.sub(" pourcent", text)
    # Numbers before units: the unit rules are number-anchored, so "deux mg" only
    # becomes "2 milligrammes" once "deux" is a digit. Ordinals go first so the cardinal
    # reader never sees "cinquieme" as a stray "cinq" (it cannot: no word boundary), and
    # so both spellings are already one token when the run regex scans.
    text = _ORDINAL_WORD_RE.sub(
        lambda m: f"{_ORDINAL_WORDS[m.group(0).rstrip('s') if m.group(0) not in _ORDINAL_WORDS else m.group(0)]}e",
        text)
    text = _ORDINAL_DIGIT_RE.sub(r"\1e", text)
    text = _STAGING_ROMAN_LOWER.sub(_fold_staging_roman, text)
    text = _NUMBER_RUN_RE.sub(_read_fr_number, text)
    text = _COMPACT_TIME_RE.sub(_read_compact_time, text)  # before the "h" unit rule
    for pattern, repl in _UNIT_RULES:  # repl re-emits the captured number: r"\1 <spoken>"
        text = pattern.sub(repl, text)
    # After them: what is left of a rate whose head the rules above just expanded
    # ("15 milligrammes par kg"). Before _SPOKEN_RULES, which settles the plural the
    # replacement introduces against whatever the other side wrote.
    for pattern, repl in _RATE_TAIL_RULES:
        text = pattern.sub(repl, text)
    for pattern, spoken in _SPOKEN_RULES:
        text = pattern.sub(spoken, text)
    # After them, so "2 heures 5 min" has already become "2 heures 5 minutes" and every
    # spelling of a clock time is one string.
    text = _TIME_MINUTES_RE.sub(r"\1", text)
    for pattern, spoken in _TITLE_RULES:
        text = pattern.sub(spoken, text)
    # Notation last, and after the unit rules on purpose: those capture the number with
    # its separators ("4,2 mmol/L"), so the decimal must still be a comma at that point.
    text = _DAY_OFFSET_RE.sub(lambda m: f"j{m.group(1) or m.group(2)}", text)
    text = _DIGIT_PLUS_RE.sub(" plus", text)
    text = _STAR_RE.sub(" etoile", text)
    text = _RATIO_RE.sub(" ", text)
    text = _LEADING_ZERO_RE.sub(r"\1", text)  # while the decimal separator is still there
    text = _DECIMAL_RE.sub(" ", text)
    text = _DECIMAL_WORD_RE.sub(" ", text)
    text = _THOUSANDS_RE.sub("", text)
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def compute_metrics(reference: str, hypothesis: str) -> tuple[float | None, float | None]:
    """Return (wer, cer) on normalized text, or None when reference is empty."""
    ref = normalize_for_scoring(reference or "")
    hyp = normalize_for_scoring(hypothesis or "")
    wer = jiwer.wer(ref, hyp) if ref.split() else None
    cer = jiwer.cer(ref, hyp) if ref else None
    return wer, cer


# --- tail scoring -------------------------------------------------------------------
# A whole-clip CER averages over the whole reference, so a defect confined to the END of
# a long clip is diluted by all the text that came out right: a clip truncated at the TTS
# output cap keeps ~85% of its text and lands at CER ~0.13, under any sane gate. The tail
# CER scores just the last TAIL_SECONDS of the clip, where those failures live.
#
# There is no time index on the text, so the tail is approximated by character count:
# TAIL_SECONDS at the corpus median pace (02_statistics.py, "Audio pace": 0.0603 s per
# unnormalized character over 563,910 clips, and the P10-P90 spread is only +/-10%, so a
# single constant is accurate enough to slice a window). Keep this in sync with that
# report if the voice or its speed changes.
TAIL_SECONDS = 30.0
TAIL_SECONDS_PER_CHAR = 0.0603
TAIL_CHARS = round(TAIL_SECONDS / TAIL_SECONDS_PER_CHAR)  # 498


def tail_cer(reference: str, hypothesis: str, duration: float) -> float | None:
    """CER of the last ``TAIL_CHARS`` normalized characters of each side, i.e. roughly the
    last ``TAIL_SECONDS`` of audio. None when the clip is at most ``TAIL_SECONDS`` long
    (nothing to isolate: the whole-clip CER already is the tail) or nothing is scorable.

    Both sides are cut to the same window and anchored at their own end, so the metric
    answers "did the clip END the way it was supposed to": a derailed or looped ending
    scores badly, and so does a truncated one (its transcript ends mid-document, against a
    reference tail that was never spoken). The window is ~500 characters, i.e. the size of
    a whole PARHAF chunk rather than a fifth of one, so it is only mildly noisier than a
    whole-clip score (its boundary cuts mid-sentence): measured over 23.6k long clips, tail
    P95 is 0.076 and P99 0.193, against 0.037 and 0.061 whole-clip. That is why the gate on
    it is looser than the whole-clip one, but not by the same factor (0.12 against 0.08 at
    the current sweep depth): raising it tracks the metric's noise, not its signal, see the
    README."""
    if duration is None or duration <= TAIL_SECONDS:
        return None
    ref = normalize_for_scoring(reference or "")[-TAIL_CHARS:]
    hyp = normalize_for_scoring(hypothesis or "")[-TAIL_CHARS:]
    if not ref:
        return None
    return jiwer.cer(ref, hyp)


def duration_of(rec: dict) -> float:
    """Clip length in seconds from the manifest's `duration` field, 0 if absent
    or unparsable. Used only to drive the audio-time progress bar / ETA."""
    try:
        return float(rec.get("duration") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def get_reference(rec: dict, reference_key: str) -> str | None:
    if reference_key:
        return rec.get(reference_key)
    for key in _REFERENCE_FALLBACKS:
        value = rec.get(key)
        if value:
            return value
    return None


def resolve_audio(
    rec: dict, jsonl_path: Path, audio_key: str, audio_root: Path | None
) -> Path | None:
    """Find the clip on disk. Paths in NeMo manifests are relative to the
    manifest's own directory; we also try --audio-root, absolute, and CWD."""
    raw = rec.get(audio_key)
    if not raw:
        return None
    raw_path = Path(raw)
    candidates = []
    if audio_root is not None:
        candidates.append(audio_root / raw)
    candidates.append((jsonl_path.parent / raw_path))
    candidates.append(raw_path)
    candidates.append(Path.cwd() / raw_path)
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except OSError:
            continue
        if resolved.is_file():
            return resolved
    return None


def transcribe(
    audio_path: Path,
    endpoint: str,
    token: str,
    model: str | None,
    temperature: float,
    language: str,
    response_format: str,
    extra_params: dict,
    timeout: float,
    max_retries: int,
) -> str:
    """POST the clip to an OpenAI-compatible transcription endpoint. Retries
    transient network / 5xx / 429 errors (and blank results) with exponential
    backoff. On exhaustion raises EmptyTranscriptError (persistent blank text,
    per-clip) or TranscriptionError (persistent network / HTTP failure)."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    data = {"temperature": str(temperature), "response_format": response_format}
    if model:  # only pin a model when the user gave one; else let the server pick
        data["model"] = model
    if language:
        data["language"] = language
    data.update(extra_params)

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            with audio_path.open("rb") as fh:
                files = {"file": (audio_path.name, fh, "audio/flac")}
                resp = requests.post(
                    endpoint, headers=headers, data=data, files=files, timeout=timeout
                )
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            if response_format == "json" or response_format == "verbose_json":
                text = (resp.json() or {}).get("text", "") or ""
            else:
                text = resp.text.strip()
            if text.strip():
                return text
            # Never return a blank transcript. Treat a blank result as retryable
            # too (it can be a transient server hiccup); keep it as the pending
            # error so we raise EmptyTranscriptError if every attempt stays blank.
            last_exc = EmptyTranscriptError(f"STT returned empty text for {audio_path.name}")
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
        if attempt < max_retries - 1:
            backoff = 2.0 * (2 ** attempt)
            logger.debug(f"transcribe retry {attempt + 1}/{max_retries} after {backoff}s: {last_exc}")
            time.sleep(backoff)
    # Exhausted every attempt. A persistently blank result is a per-clip data
    # issue (EmptyTranscriptError, recorded and resumed); a persistent network /
    # HTTP failure is a server-health issue (TranscriptionError) a fail-fast
    # caller can let propagate to abort the run.
    if isinstance(last_exc, EmptyTranscriptError):
        raise last_exc
    raise TranscriptionError(f"transcription failed after {max_retries} attempts: {last_exc}")


def transcribe_timed(*args, **kwargs) -> tuple[str, float]:
    """transcribe() plus the wall-clock seconds the request took, so callers can
    report a real-time factor (RTF = STT seconds / audio duration). Only the call
    is timed; when wrapped in a concurrency semaphore, time it inside the `with` so
    the queue wait is excluded and the number reflects the request itself."""
    t0 = time.monotonic()
    text = transcribe(*args, **kwargs)
    return text, time.monotonic() - t0


def compute_entry(
    rec: dict,
    input_file: Path,
    *,
    endpoint: str,
    token: str,
    model: str | None,
    temperature: float,
    language: str,
    response_format: str,
    extra_params: dict,
    timeout: float,
    max_retries: int,
    audio_key: str,
    reference_key: str,
    audio_root: Path | None,
    wer_threshold: float,
    cer_threshold: float,
) -> dict:
    """Transcribe and score one record. Pure per-clip work with no shared
    state, so it is safe to run concurrently across threads (the blocking HTTP
    POST releases the GIL). Never raises: failures are recorded on the entry."""
    ident = short_ident(rec, audio_key)
    audio_path = resolve_audio(rec, input_file, audio_key, audio_root)
    reference = get_reference(rec, reference_key)

    entry: dict = {"text": None, "wer": None, "cer": None}
    if audio_path is None:
        entry["error"] = f"audio not found for {rec.get(audio_key)!r}"
        logger.warning(f"{ident}: {entry['error']}")
        return entry
    try:
        transcript, stt_seconds = transcribe_timed(
            audio_path, endpoint, token, model, temperature, language,
            response_format, extra_params, timeout, max_retries,
        )
        entry["text"] = transcript
        # Real-time factor: how long the transcription took relative to the audio
        # length (rtf < 1 = faster than real time). Stored so it lands in the jsonl.
        entry["stt_seconds"] = round(stt_seconds, 3)
        dur = duration_of(rec)
        rtf = stt_seconds / dur if dur > 0 else None
        if rtf is not None:
            entry["stt_rtf"] = round(rtf, 3)
        rtf_s = f" stt={stt_seconds:.2f}s rtf={rtf:.3f}" if rtf is not None else f" stt={stt_seconds:.2f}s"
        if reference is None:
            entry["error"] = "no reference text, cannot score"
            logger.warning(f"{ident}: {entry['error']}")
        else:
            wer, cer = compute_metrics(reference, transcript)
            entry["wer"] = wer
            entry["cer"] = cer
            # Same reading scored again over its last ~30s only, so an ending that
            # derailed or was cut off does not average away (omitted on short clips).
            t_cer = tail_cer(reference, transcript, dur)
            if t_cer is not None:
                entry["cer_tail"] = t_cer
            wer_s = f"{wer:.3f}" if wer is not None else "n/a"
            cer_s = f"{cer:.3f}" if cer is not None else "n/a"
            tail_s = f" cer_tail={t_cer:.3f}" if t_cer is not None else ""
            logger.info(f"{ident}  wer={wer_s} cer={cer_s}{tail_s}{rtf_s}")
            bad_wer = wer is not None and wer >= wer_threshold
            bad_cer = cer is not None and cer >= cer_threshold
            if bad_wer or bad_cer:
                logger.warning(
                    f"LOW QUALITY {ident} wer={wer_s} cer={cer_s}{tail_s}\n"
                    f"  reference : {reference}\n"
                    f"  transcript: {transcript}"
                )
    except Exception as exc:  # keep going, record the failure
        entry["error"] = str(exc)
        logger.warning(f"{ident}: transcription error: {exc}")
    return entry


def record_key(rec: dict, audio_key: str, index: int) -> str:
    """Stable identity for resume/merge: the audio path, else the line index."""
    value = rec.get(audio_key)
    return value if value else f"__idx_{index}"


def short_ident(rec: dict, audio_key: str) -> str:
    term = rec.get("term")
    if term:
        return str(term)
    raw = rec.get(audio_key)
    return Path(raw).name if raw else "?"


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes so a reader never sees a half-written file: stage to a unique
    temp sibling, fsync, then os.replace (atomic on the same filesystem). Used for
    replacing / backing up audio clips so an interrupted run cannot corrupt them.
    The temp name includes the pid so concurrent writers never collide."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        # If os.replace never ran (write failed), don't leave the temp behind.
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def load_existing(path: Path, audio_key: str) -> dict[str, dict]:
    """Preload a prior output file so we can resume/merge. Tolerates a torn
    last line from a crash mid-flush.

    Bars its own progress: on a 600k-row corpus this is a ~700 MB json parse, which is
    a minute of apparent silence before anything else happens, and a silent minute is
    indistinguishable from a hang. Read as bytes so the bar is exact (json.loads takes
    bytes directly, which also skips a decode)."""
    existing: dict[str, dict] = {}
    if not path.is_file():
        return existing
    t0 = time.monotonic()
    with path.open("rb") as fh, tqdm(
            total=path.stat().st_size, desc=f"resuming {path.name}", unit="B",
            unit_scale=True, unit_divisor=1024, mininterval=0.5, leave=False) as bar:
        for i, raw in enumerate(fh):
            bar.update(len(raw))
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # A crash mid-flush can also tear a multi-byte character in half.
                logger.warning(f"skipping unreadable line {i} in existing {path.name}")
                continue
            existing[record_key(rec, audio_key, i)] = rec
    if existing:
        logger.info(f"resuming from {path.name}: {len(existing)} lines already present "
                    f"({time.monotonic() - t0:.0f}s to read)")
    return existing


# Share of wall-clock the pass may spend rewriting its output. Every flush writes the
# whole file, so on a big manifest the only way to keep --flush-every from eating the run
# is to make the cadence proportional to what a rewrite actually costs.
FLUSH_DUTY_CYCLE = 0.05


def flush_backoff_seconds(last_flush_cost: float,
                          duty_cycle: float = FLUSH_DUTY_CYCLE) -> float:
    """How long a flush that cost ``last_flush_cost`` seconds buys before the next one
    is worth doing. 10 s to rewrite at a 5% duty cycle means one flush per 200 s, so a
    crash loses at most that much work. 0 (nothing written yet, or an instant write)
    means no hold-off, which keeps small files behaving exactly as before."""
    if last_flush_cost <= 0 or duty_cycle <= 0:
        return 0.0
    return last_flush_cost / duty_cycle


def discover_inputs(input_path: Path) -> tuple[Path, list[Path]]:
    """Return (input_root, [jsonl files]). Skips already-produced .stt.jsonl."""
    if input_path.is_file():
        return input_path.parent, [input_path]
    if input_path.is_dir():
        files = sorted(
            p for p in input_path.rglob("*.jsonl") if not p.name.endswith(".stt.jsonl")
        )
        return input_path, files
    raise click.BadParameter(f"--input {input_path} is neither a file nor a directory")


def output_path_for(input_file: Path, input_root: Path, output_root: Path) -> Path:
    rel = input_file.relative_to(input_root)
    return output_root / rel.parent / (rel.stem + ".stt.jsonl")


def process_file(
    input_file: Path,
    out_file: Path,
    *,
    endpoint: str,
    token: str,
    model: str | None,
    temperature: float,
    language: str,
    response_format: str,
    extra_params: dict,
    timeout: float,
    max_retries: int,
    audio_key: str,
    reference_key: str,
    audio_root: Path | None,
    wer_threshold: float,
    cer_threshold: float,
    flush_every: int,
    flush_interval: float,
    limit: int,
    n_parallel: int,
    shuffle: bool,
    worker=None,
    apply_result=None,
    needs_work=None,
    on_flush=None,
    postfix_fn=None,
) -> None:
    # Key used under `transcriptions`; when no model is pinned we still need a
    # stable slot so resume/merge works, so fall back to "default".
    model_key = model or "default"

    # Per-clip hooks let another stage (01_recursive_improvement.py) reuse this
    # whole orchestration (dual progress bars, resume, atomic flushing) with an
    # augmented worker instead of duplicating it. The defaults below reproduce the
    # plain transcribe-and-score behavior exactly.
    #   worker(pos, rec) -> result           runs in a pool thread
    #   apply_result(pos, result, results) -> entry   runs in the main thread,
    #       stores the result and returns the {wer,cer,...} entry for the bar
    #   needs_work(out_rec) -> bool          whether a row goes in the todo list
    #   on_flush()                           extra work on each atomic flush
    if worker is None:
        def worker(pos, rec):
            return compute_entry(
                rec, input_file,
                endpoint=endpoint, token=token, model=model, temperature=temperature,
                language=language, response_format=response_format,
                extra_params=extra_params, timeout=timeout, max_retries=max_retries,
                audio_key=audio_key, reference_key=reference_key, audio_root=audio_root,
                wer_threshold=wer_threshold, cer_threshold=cer_threshold,
            )
    if apply_result is None:
        def apply_result(pos, result, results):
            results[pos]["transcriptions"][model_key] = result
            return result
    if needs_work is None:
        def needs_work(out_rec):
            prior = out_rec["transcriptions"].get(model_key)
            if prior is None or prior.get("error"):
                return True
            text = prior.get("text")
            return not (text and text.strip())  # re-do blank transcripts too

    existing = load_existing(out_file, audio_key)

    # Read every record up front, in input order. `results` stays index-aligned
    # so the output file is always written in input order regardless of the
    # order clips finish transcribing; `todo` holds only the ones needing work.
    #
    # Barred, because this scan is NOT just a parse: it calls needs_work on every row,
    # and on a rescore that means re-deriving the CER of every stored transcript
    # (normalize + jiwer, hundreds of thousands of times). It is minutes of real work
    # before a single clip is transcribed, so it gets to say so. Driven by bytes read
    # rather than rows, which needs no line count pass to know its total.
    scan_t0 = time.monotonic()
    results: list[dict] = []
    todo: list[tuple[int, dict]] = []  # (position in results, the out_rec to work on)
    stale_rows = 0
    with input_file.open("rb") as fh, tqdm(
            total=input_file.stat().st_size, desc=f"{input_file.name} scan", unit="B",
            unit_scale=True, unit_divisor=1024, mininterval=0.5, leave=False) as scan_bar:
        for i, raw in enumerate(fh):
            scan_bar.update(len(raw))
            if limit and i >= limit:
                break
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            key = record_key(rec, audio_key, i)

            # Start from any previously stored record so we keep other
            # models' transcriptions when merging.
            prior = existing.get(key)
            # ...unless the manifest was rebuilt under the same clip filename (re-chunked
            # source, regenerated audio). A row's identity is its audio path AND its
            # label: same path with a different label means the stored transcription,
            # CER and improvement state describe OTHER audio. Keeping them would score
            # the new clip against the old label and could regenerate audio for a
            # sentence the manifest no longer holds, so the stored work is dropped and
            # the row starts from the input again.
            if prior is not None and prior.get("text") != rec.get("text"):
                prior = None
                stale_rows += 1
            out_rec = prior if prior is not None else dict(rec)
            out_rec.setdefault("transcriptions", {})
            pos = len(results)
            results.append(out_rec)
            if needs_work(out_rec):
                # Pass out_rec (not the raw line) so an injected worker can reuse a
                # transcript stored on a prior run instead of re-transcribing.
                todo.append((pos, out_rec))

    total = len(results)
    logger.info(f"{input_file.name}: scanned {total} row(s) in "
                f"{time.monotonic() - scan_t0:.0f}s, {len(todo)} need work this pass")

    if stale_rows:
        logger.warning(
            f"{input_file.name}: {stale_rows} row(s) kept the same clip filename but "
            "changed their text since the stored run (manifest rebuilt?); their stored "
            "transcription and improvement state describe the old audio and were dropped")

    # Randomize the processing order so the running rate (and thus the ETA) is
    # estimated from a representative sample of clip lengths early on. This only
    # reorders how clips are submitted; `results` stays index-aligned to the
    # input, so the written output order is unchanged.
    if shuffle:
        random.shuffle(todo)

    dirty = False
    processed_since_flush = 0
    last_flush_t = time.monotonic()
    flush_cost = 0.0      # seconds the last full rewrite took, 0 until one has happened
    flush_warned = False  # log the derived cadence once, not on every held-off flush
    # Rolling real-time factor for the bar postfix: total STT seconds / total audio
    # seconds transcribed this run (duration-weighted). Reflects per-request cost;
    # with concurrency the wall-clock throughput is faster (see the audio-s/s bar).
    rtf_stt_sum = 0.0
    rtf_audio_sum = 0.0

    def flush(force: bool = False) -> None:
        nonlocal dirty, processed_since_flush, last_flush_t, flush_cost, flush_warned
        if not dirty:
            return
        # A flush rewrites the WHOLE output, so it costs a rewrite of every row, not of
        # the ones just processed. On a 600k-row manifest that is ~10 s of json.dumps,
        # and honouring --flush-every 50 literally would spend all day rewriting: a pass
        # with nothing to wait for (a rescore, a cache-warm resume) finishes rows far
        # faster than the file can be written. So the caps below are treated as "ask to
        # flush", and a flush that turned out expensive holds the next one off until it
        # has been earned, keeping the rewrite overhead near FLUSH_DUTY_CYCLE.
        now = time.monotonic()
        if not force and now - last_flush_t < flush_backoff_seconds(flush_cost):
            if not flush_warned:
                logger.info(
                    f"{out_file.name}: {len(results)} rows take {flush_cost:.1f}s to write, "
                    f"so flushing at most every {flush_backoff_seconds(flush_cost):.0f}s "
                    f"(a crash costs that much work, and nothing paid for)")
                flush_warned = True
            return
        t0 = time.monotonic()
        atomic_write_jsonl(out_file, results)
        flush_cost = time.monotonic() - t0
        if on_flush is not None:
            on_flush()
        dirty = False
        processed_since_flush = 0
        last_flush_t = time.monotonic()

    # Audio-time bar: total is the summed clip duration (done + todo), advanced
    # by each clip's own duration as it finishes. Its ETA is driven by
    # audio-seconds per wall-second, a truer estimate than a per-clip count when
    # clip lengths vary. Only shown when the manifest carries durations.
    total_audio = sum(duration_of(r) for r in results)
    todo_audio = sum(duration_of(results[pos]) for pos, _ in todo)
    done_audio = total_audio - todo_audio

    # Pass the already-done count as `initial=` (like the audio bar below) rather
    # than update()-ing after construction: tqdm excludes `initial` from its rate
    # so the ETA is timed from the first real clip, not skewed super-fast by the
    # instantaneous jump over the resumed lines.
    bar = tqdm(total=total, initial=len(results) - len(todo),
               desc=input_file.name, unit="clip", position=0)
    audio_bar = None
    if total_audio > 0:
        audio_bar = tqdm(
            total=round(total_audio), initial=round(done_audio),
            desc=f"{input_file.name} audio", unit="s", unit_scale=True, position=1,
        )
    try:
        with ThreadPoolExecutor(max_workers=max(1, n_parallel)) as pool:
            futures = {pool.submit(worker, pos, rec): pos for pos, rec in todo}
            for fut in as_completed(futures):
                pos = futures[fut]
                result = fut.result()  # default worker never raises
                entry = apply_result(pos, result, results) or {}
                dirty = True
                processed_since_flush += 1
                # Accumulate the rolling RTF from whatever STT time this entry cost.
                stt_seconds = entry.get("stt_seconds")
                if stt_seconds is not None:
                    dur = duration_of(results[pos])
                    if dur > 0:
                        rtf_stt_sum += stt_seconds
                        rtf_audio_sum += dur
                if postfix_fn is not None:
                    base = postfix_fn(entry)
                else:
                    wer_s = f"{entry['wer']:.3f}" if entry.get("wer") is not None else "n/a"
                    cer_s = f"{entry['cer']:.3f}" if entry.get("cer") is not None else "n/a"
                    base = f"wer={wer_s} cer={cer_s}"
                if rtf_audio_sum > 0:
                    base += f" rtf={rtf_stt_sum / rtf_audio_sum:.3f}"
                bar.set_postfix_str(base)
                bar.update(1)
                if audio_bar is not None:
                    audio_bar.update(duration_of(results[pos]))
                # Ask to flush on whichever cap trips first: N processed clips, or T
                # seconds since the last flush. Either bound can be disabled with 0.
                # flush() may still hold off if the last rewrite was expensive.
                if (flush_every > 0 and processed_since_flush >= flush_every) or (
                    flush_interval > 0 and time.monotonic() - last_flush_t >= flush_interval
                ):
                    flush()
    finally:
        # Forced: the hold-off only makes sense while more work is coming, and this is
        # also the flush that persists an interrupted run (Ctrl-C included).
        flush(force=True)
        bar.close()
        if audio_bar is not None:
            audio_bar.close()
    logger.success(f"wrote {out_file} ({len(results)} lines)")


def _parse_params(pairs: tuple[str, ...]) -> dict:
    out: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise click.BadParameter(f"--param must be KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        out[key.strip()] = value
    return out


@click.command(context_settings={"show_default": True})
@click.option("--endpoint", required=True, help="Full transcription URL (OpenAI-compatible /v1/audio/transcriptions).")
@click.option("--api-token", default="", envvar="STT_API_TOKEN", help="Bearer token; sent only if non-empty. Falls back to $STT_API_TOKEN.")
@click.option("--model", default=None, help="STT model name; also the key under `transcriptions`. Omitted from the request (and keyed as 'default') when not given, letting the server pick.")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path), help="A jsonl file or a directory searched recursively for *.jsonl.")
@click.option("--output", "output_root", required=True, type=click.Path(path_type=Path), help="Output root; mirrors input hierarchy, files get the .stt.jsonl extension.")
@click.option("--temperature", default=0.0, type=float, help="Transcription temperature.")
@click.option("--language", default="fr", help="Transcription language hint (empty to omit).")
@click.option("--response-format", default="json", help="Endpoint response_format (json / verbose_json / text).")
@click.option("--param", "params", multiple=True, help="Extra transcription form field KEY=VALUE (repeatable).")
@click.option("--audio-key", default="audio_filepath", help="JSONL key holding the audio path.")
@click.option("--reference-key", default="", help="JSONL key for the reference text; empty auto-tries asr_training_source, text, asr_training_target.")
@click.option("--audio-root", default=None, type=click.Path(path_type=Path), help="Optional base dir to resolve audio paths against.")
@click.option("--wer-threshold", default=0.5, type=float, help="Warn (with texts) when WER >= this.")
@click.option("--cer-threshold", default=0.15, type=float, help="Warn (with texts) when CER >= this.")
@click.option("--n-parallel", "n_parallel", default=1, type=int, help="Number of clips to transcribe concurrently (thread pool). Raise to keep a batching STT server busy; 1 = sequential.")
@click.option("--shuffle/--no-shuffle", default=False, help="Randomize the processing order (not the output order) so the running rate / ETA reflects a representative mix of clip lengths early on.")
@click.option("--flush-every", default=50, type=int, help="Atomically write the output every N newly processed clips (0 = disable this cap).")
@click.option("--flush-interval", default=0.0, type=float, help="Also flush every this many seconds since the last write (e.g. 300 = every 5 min); 0 = disabled. Whichever of --flush-every / --flush-interval trips first wins.")
@click.option("--timeout", default=120.0, type=float, help="Per-request timeout (seconds).")
@click.option("--max-retries", default=4, type=int, help="Retries on transient network/5xx/429 errors.")
@click.option("--limit", default=0, type=int, help="Process at most this many lines per file (0 = all), for testing.")
def main(
    endpoint, api_token, model, input_path, output_root, temperature, language,
    response_format, params, audio_key, reference_key, audio_root, wer_threshold,
    cer_threshold, n_parallel, shuffle, flush_every, flush_interval, timeout, max_retries, limit,
):
    """Transcribe dataset audio and score WER/CER to find derailed TTS clips."""
    logger.remove()
    logger.add(lambda m: tqdm.write(m, end=""), colorize=True,
               format="<level>{level: <8}</level> | {message}", level="INFO")

    extra_params = _parse_params(params)
    input_root, input_files = discover_inputs(input_path)
    if not input_files:
        logger.warning(f"no *.jsonl files found under {input_path}")
        return
    logger.info(f"{len(input_files)} input file(s); model={model or '(server default)'} endpoint={endpoint}")

    if shuffle:
        random.shuffle(input_files)

    for input_file in input_files:
        out_file = output_path_for(input_file, input_root, output_root)
        process_file(
            input_file, out_file,
            endpoint=endpoint, token=api_token, model=model, temperature=temperature,
            language=language, response_format=response_format, extra_params=extra_params,
            timeout=timeout, max_retries=max_retries, audio_key=audio_key,
            reference_key=reference_key, audio_root=audio_root,
            wer_threshold=wer_threshold, cer_threshold=cer_threshold,
            flush_every=flush_every, flush_interval=flush_interval,
            limit=limit, n_parallel=n_parallel,
            shuffle=shuffle,
        )


if __name__ == "__main__":
    main()
