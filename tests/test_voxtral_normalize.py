#!/usr/bin/env python3
"""Unit tests for voxtral_normalize (pure stdlib; run: python tests/test_voxtral_normalize.py).

Cases mirror the sweep verdicts recorded in 01_dictionnary/VOXTRAL_QUIRKS.md:
FIX rows must transform, KEEP RAW rows must pass through untouched, and an
uncovered unit must surface via the residual detector.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

from voxtral_normalize import (  # noqa: E402
    find_residual_units,
    normalize_and_flag,
    normalize_for_voxtral,
)

# (input, expected output) for the FIX rows.
FIX_CASES = [
    # units with '/'
    ("La CRP est à 42 mg/L.", "La CRP est à 42 milligrammes par litre."),
    ("Créatinine à 120 µmol/L.", "Créatinine à 120 micromoles par litre."),
    ("Clairance à 60 mL/min.", "Clairance à 60 millilitres par minute."),
    ("TSH à 0,3 mUI/L.", "TSH à 0,3 milli-unités internationales par litre."),
    ("Vancomycine 15 mg/kg.", "Vancomycine 15 milligrammes par kilogramme."),
    ("Protéinurie à 2 g/L.", "Protéinurie à 2 grammes par litre."),
    ("Phosphatases à 80 UI/L.", "Phosphatases à 80 unités internationales par litre."),
    ("Glycémie à 5,5 mmol/L.", "Glycémie à 5,5 millimoles par litre."),
    # micro (with and without slash)
    ("Lévothyroxine 75 µg par jour.", "Lévothyroxine 75 microgrammes par jour."),
    ("Débit de 5 µg/kg/min.", "Débit de 5 microgrammes par kilogramme par minute."),
    # degree
    ("Température à 38 °C.", "Température à 38 degrés Celsius."),
    # staging / anatomy Roman
    ("Cancer au stade IV.", "Cancer au stade quatre."),
    ("Atteinte du nerf X.", "Atteinte du nerf dix."),
    ("Diabète de type II.", "Diabète de type deux."),
    ("Déficit en facteur VIII.", "Déficit en facteur huit."),
    # one-off token
    ("Vaccin à ARNm à jour.", "Vaccin à ARN-m à jour."),
]

# Rows the sweep marked KEEP RAW: normalize must be a no-op.
KEEP_RAW = [
    "Le bilan montre une TSH basse.",
    "L'ECG est sans particularité.",
    "La sérologie VIH est négative.",
    "Le taux d'HbA1c est élevé.",
    "Codé selon la CIM-10, critères du DSM-V remplis.",
    "Tension à 140 mmHg.",              # bare unit voxtral reads fine
    "Dexaméthasone 0,5 mg le matin.",   # bare mg
    "Saturation à 95 %.",               # percent
    "Opéré le 12/07/1998.",             # date slash
    "Contrôle prévu le 2024-04-15.",    # ISO date
    "Admis à 14h30.",                   # time
    "Hôpital Henri IV.",                # Roman, no staging word
    "Antibiotiques administrés en IV.",  # IV = intraveineuse
]

# Uncovered units the table does not list: must reach the review queue.
RESIDUAL_CASES = [
    "Débit résiduel de 5 cg/semaine.",   # cg not in table -> '/' letter-adjacent
    "Osmolalité à 290 mOsm/L bizarre µV.",  # stray µ survives
]


def test_fix_cases() -> None:
    for src, expected in FIX_CASES:
        got = normalize_for_voxtral(src)
        assert got == expected, f"\n  in:  {src}\n  got: {got}\n  exp: {expected}"


def test_keep_raw_unchanged() -> None:
    for src in KEEP_RAW:
        got = normalize_for_voxtral(src)
        assert got == src, f"KEEP RAW mutated:\n  in:  {src}\n  got: {got}"


def test_no_residual_after_fix() -> None:
    for src, _ in FIX_CASES:
        norm, residual = normalize_and_flag(src)
        assert residual == [], f"unexpected residual on {src!r}: {residual}"


def test_keep_raw_has_no_residual() -> None:
    # In particular dates (12/07/1998) must NOT look like a residual unit.
    for src in KEEP_RAW:
        assert find_residual_units(src) == [], f"false residual on {src!r}"


def test_uncovered_unit_flagged() -> None:
    for src in RESIDUAL_CASES:
        _, residual = normalize_and_flag(src)
        assert residual, f"uncovered unit not flagged: {src!r}"


def test_idempotent() -> None:
    for src, _ in FIX_CASES:
        once = normalize_for_voxtral(src)
        twice = normalize_for_voxtral(once)
        assert once == twice, f"not idempotent: {once!r} != {twice!r}"


def _run() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
        else:
            passed += 1
            print(f"ok   {t.__name__}")
    print(f"\n{passed}/{len(tests)} test functions passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(_run())
