"""Deterministic text normalizer for local voxtral-tts.

Turns an ``asr_training_target`` (written clinical French, the ASR label) into
the ``asr_training_source`` (the text fed to voxtral-tts) by applying ONLY the
transforms voxtral demonstrably needs. Everything else passes through
unchanged: the sweep in ``01_dictionnary/VOXTRAL_QUIRKS.md`` showed voxtral
already reads written acronyms, numbers, dates, ``%`` and bare units correctly,
and that respelling those (spacing letters, spelling digits) makes it *worse*.

The FIX set the sweep left is small and fully deterministic, so this is a
regex/lookup module, NOT an LLM pass:

  1. units written with ``/``, ``µ`` or ``°``      -> spelled out in French
  2. Roman numeral after a staging/anatomy word    -> French cardinal word
  3. ``ARNm``                                       -> ``ARN-m``
  4. ``RAS``                                        -> TODO (see quirks file)

This module is the executable form of the FIX list in ``VOXTRAL_QUIRKS.md``;
keep the two in sync. The unit table does NOT need to be exhaustive:
``find_residual_units`` flags any ``/ µ °`` that survives so an uncovered unit
goes to a review queue, never to bad audio.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

__all__ = [
    "normalize_for_voxtral",
    "find_residual_units",
    "normalize_and_flag",
    "record_review",
    "append_jsonl_row",
    # Shared with the CER scorer (06_hotfixes/01_compute_stt.py), which has to fold the
    # same symbol / spoken pairs the other way: the TTS spells a unit out, then Whisper
    # abbreviates it back, and neither difference is an error. Public so the scorer reads
    # this table instead of growing a second copy of it.
    "UNIT_SPOKEN",
    "compile_unit_rules",
    "STAGING_ROMAN_RE",
    "roman_to_int",
    "FR_CARDINAL",
]


# ---------------------------------------------------------------------------
# 1. Units containing '/', 'µ' or '°'
# ---------------------------------------------------------------------------
# (symbol, spoken French). Order matters: more specific symbols first so that
# e.g. `mUI/L` is tried before `UI/L`. The required leading number (added when
# compiling) both avoids false hits on ordinary letters and lets the residual
# detector treat any leftover `/ µ °` as an uncovered case. Bare units voxtral
# already reads (`mmHg`, bare `mg`, `kg`, `cm`) are deliberately absent.
UNIT_SPOKEN: list[tuple[str, str]] = [
    # --- concentration / lab values (per litre) ---
    ("µmol/L", "micromoles par litre"),
    ("nmol/L", "nanomoles par litre"),
    ("mmol/L", "millimoles par litre"),
    ("mol/L", "moles par litre"),
    ("mUI/L", "milli-unités internationales par litre"),
    ("UI/L", "unités internationales par litre"),
    ("U/L", "unités par litre"),
    ("mEq/L", "milliéquivalents par litre"),
    ("mOsm/kg", "milliosmoles par kilogramme"),
    ("mOsm/L", "milliosmoles par litre"),
    ("µg/L", "microgrammes par litre"),
    ("ng/L", "nanogrammes par litre"),
    ("mg/dL", "milligrammes par décilitre"),
    ("g/dL", "grammes par décilitre"),
    ("mg/L", "milligrammes par litre"),
    ("g/L", "grammes par litre"),
    # --- per millilitre ---
    ("µg/mL", "microgrammes par millilitre"),
    ("ng/mL", "nanogrammes par millilitre"),
    ("mUI/mL", "milli-unités internationales par millilitre"),
    ("UI/mL", "unités internationales par millilitre"),
    # --- dosing per weight / surface / time ---
    ("µg/kg/min", "microgrammes par kilogramme par minute"),
    ("mg/kg/j", "milligrammes par kilogramme par jour"),
    ("µg/kg", "microgrammes par kilogramme"),
    ("mg/kg", "milligrammes par kilogramme"),
    ("g/kg", "grammes par kilogramme"),
    ("mg/m²", "milligrammes par mètre carré"),
    ("mg/m2", "milligrammes par mètre carré"),
    ("mg/jour", "milligrammes par jour"),
    ("µg/jour", "microgrammes par jour"),
    ("mg/j", "milligrammes par jour"),
    ("µg/j", "microgrammes par jour"),
    ("g/j", "grammes par jour"),
    # --- flow / rate ---
    ("mL/min", "millilitres par minute"),
    ("L/min", "litres par minute"),
    ("mL/kg", "millilitres par kilogramme"),
    ("mL/h", "millilitres par heure"),
    # --- counts per volume (cell counts, viral load) ---
    ("/mm³", "par millimètre cube"),
    ("/mm3", "par millimètre cube"),
    ("/µL", "par microlitre"),
    ("/mL", "par millilitre"),
    # --- bare micro units (no slash) ---
    ("µg", "microgrammes"),
    ("µmol", "micromoles"),
    ("µL", "microlitres"),
    ("µm", "micromètres"),
    # --- temperature ---
    ("°C", "degrés Celsius"),
]


def compile_unit_rules(
    pairs: list[tuple[str, str]],
) -> list[tuple[re.Pattern[str], str]]:
    """Compile (symbol, spoken) into (number-anchored regex, replacement).

    The number is captured and re-emitted, so `42 mg/L` -> `42 milligrammes
    par litre` while spacing is normalised to a single space. The trailing
    negative lookahead stops a bare unit (`µg`) from eating the head of a
    slash-compound the table happens not to cover (`µg/dose`); the compound
    forms are listed earlier and match first when they *are* covered.
    """
    rules: list[tuple[re.Pattern[str], str]] = []
    for symbol, spoken in pairs:
        pattern = re.compile(
            r"(\d[\d.,]*)\s*" + re.escape(symbol) + r"(?![/\w²³°µ])"
        )
        rules.append((pattern, r"\1 " + spoken))
    return rules


_UNIT_RULES = compile_unit_rules(UNIT_SPOKEN)


# ---------------------------------------------------------------------------
# 2. Roman numeral after a staging / anatomy word
# ---------------------------------------------------------------------------
# Trigger word (either capitalisation) + an UPPERCASE Roman token. The Roman
# group is case-sensitive on purpose: lowercase words like "civil" are valid
# Roman letters (C,I,V,I,L) and would otherwise be mangled. `en IV`,
# `Henri IV`, and any Roman not preceded by a trigger word are left untouched.
STAGING_ROMAN_RE = re.compile(
    r"\b([Ss]tade|[Tt]ype|[Gg]rade|[Pp]hase|[Cc]lasse|[Nn]iveau|[Nn]erf"
    r"|[Ff]acteur|[Gg]roupe)\s+([IVXLCDM]+)\b"
)

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

# Index by value; staging numbers are small, so 0..20 is plenty.
FR_CARDINAL = [
    "zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit",
    "neuf", "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize",
    "dix-sept", "dix-huit", "dix-neuf", "vingt",
]


def roman_to_int(s: str) -> int | None:
    """Return the integer value of a Roman string, or None if malformed."""
    total = 0
    prev = 0
    for ch in reversed(s):
        value = _ROMAN_VALUES.get(ch)
        if value is None:
            return None
        if value < prev:
            total -= value
        else:
            total += value
            prev = value
    return total


def _replace_staging(match: re.Match[str]) -> str:
    word, roman = match.group(1), match.group(2)
    n = roman_to_int(roman)
    if n is None or not 1 <= n <= 20:
        return match.group(0)  # leave unusual values for a human to see
    return f"{word} {FR_CARDINAL[n]}"


# ---------------------------------------------------------------------------
# 3. One-off token fixes
# ---------------------------------------------------------------------------
_ARNM_RE = re.compile(r"\bARNm\b")

# TODO(RAS): voxtral reads "RAS" as the French word "race". The fix (respell to
# read as letters vs expand to "rien à signaler") is an OPEN decision in
# VOXTRAL_QUIRKS.md, so it is deliberately NOT applied here yet. Told the user.
_RAS_FIX: str | None = None


# ---------------------------------------------------------------------------
# 4. Residual detector (the review-queue backstop)
# ---------------------------------------------------------------------------
# After normalisation, any surviving `µ`, `°`, or letter-adjacent `/` means an
# uncovered unit slipped through. Digit-only `/` (dates like 12/07/1998) is NOT
# a residual: dates are KEEP RAW and voxtral reads them correctly.
_RESIDUAL_UNIT_RE = re.compile(r"[°µ]|(?<=[A-Za-z])/|/(?=[A-Za-z])")
_RESIDUAL_CONTEXT = 14


def normalize_for_voxtral(text: str) -> str:
    """Apply the deterministic voxtral FIX transforms to one text.

    Order: unit spell-out, then staging-Roman, then ARNm. Idempotent on already
    normalised text (no rule matches a spelled-out form).
    """
    for pattern, replacement in _UNIT_RULES:
        text = pattern.sub(replacement, text)
    text = STAGING_ROMAN_RE.sub(_replace_staging, text)
    text = _ARNM_RE.sub("ARN-m", text)
    if _RAS_FIX is not None:
        text = re.sub(r"\bRAS\b", _RAS_FIX, text)
    return text


def find_residual_units(text: str) -> list[str]:
    """Return context snippets around any unit symbol that survived.

    A non-empty result means the text should go to the review queue: an
    uncovered unit reached the TTS input. Empty means fully handled.
    """
    out: list[str] = []
    for m in _RESIDUAL_UNIT_RE.finditer(text):
        i = m.start()
        out.append(text[max(0, i - _RESIDUAL_CONTEXT): i + _RESIDUAL_CONTEXT])
    return out


def normalize_and_flag(text: str) -> tuple[str, list[str]]:
    """Convenience: normalise, then detect residuals on the result."""
    normalized = normalize_for_voxtral(text)
    return normalized, find_residual_units(normalized)


def append_jsonl_row(path: str | Path, row: dict) -> None:
    """Append one dict as a JSONL line, creating or extending the file.

    The generic queue-append used by both the voxtral residual review queue
    (record_review) and the generator's term-missing skip queue, so the
    "serialize a dict and append a line" logic lives in one place.
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def record_review(
    path: str | Path,
    *,
    term: str,
    original: str,
    normalized: str,
    residuals: list[str],
) -> None:
    """Append one review-queue entry (JSONL). Call only when residuals exist."""
    append_jsonl_row(
        path,
        {
            "term": term,
            "original": original,
            "normalized": normalized,
            "residuals": residuals,
        },
    )


if __name__ == "__main__":
    demo = [
        "La CRP est à 42 mg/L.",
        "Créatinine à 120 µmol/L, clairance à 60 mL/min.",
        "Cancer diagnostiqué au stade IV, atteinte du nerf X.",
        "Vaccin à ARNm à jour.",
        "Opéré le 12/07/1998, TSH normale, saturation à 95 %.",
        "Débit résiduel de 5 cg/semaine.",  # uncovered unit -> residual
    ]
    for d in demo:
        norm, resid = normalize_and_flag(d)
        print(f"IN : {d}\nOUT: {norm}\nRESIDUAL: {resid}\n")
