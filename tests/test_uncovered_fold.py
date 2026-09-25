#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "litellm",
#     "rapidfuzz",
#     "tiktoken",
#     "tenacity",
#     "loguru",
#     "click",
# ]
# ///
"""Regression test: generated labels must not keep a character Parakeet cannot write.

UltiMed v1 shipped labels such as ``Laser CO₂``, ``antigène Brª``, ``séquence CUBE™``
and a ``ç`` stored as ``c`` + U+0327 combining cedilla, plus 6 labels with a raw
newline. The generators' ``<unk>`` gate checks with NFKC (what SentencePiece does at
training time), and under NFKC all of these are covered, so they passed and were
stored unfolded: the label showed ``CO₂`` while the model learnt ``CO2``. They were
fixed after the fact by ``99_hf_release/05_normalize_text.py``.

``parse_asr_training_target`` now runs ``ParakeetTokenizer.clean_label`` (via
``_normalize_tokenizable_text``), so a new generation stores the folded form directly.
Genuinely uncovered symbols (``≥``) must still reach the gate untouched.

Run:  uv run tests/test_uncovered_fold.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "utils"))

from _pipeline_shared import (  # noqa: E402
    _PARAKEET_TOK,
    _normalize_tokenizable_text,
    parse_asr_training_target,
)

COMBINING_CEDILLA = "̧"
# (raw LLM output, label that must be stored)
CASES = [
    ("Laser CO₂ fractionné.", "Laser CO2 fractionné."),
    ("Antigène Brª positif.", "Antigène Bra positif."),
    ("Séquence CUBE™ en pondération T2.", "Séquence CUBE en pondération T2."),
    (f"Nous espac{COMBINING_CEDILLA}ons les prises.", "Nous espaçons les prises."),
    ("Pas de fièvre.\nPas de toux.", "Pas de fièvre. Pas de toux."),
]


def test_v1_characters_are_folded_and_raw_covered():
    raw_tok = type(_PARAKEET_TOK)(nfkc=False)
    for raw, expected in CASES:
        got = _normalize_tokenizable_text(raw)
        assert got == expected, (raw, got)
        assert not raw_tok.offending_chars(got), (raw, got)
        assert _normalize_tokenizable_text(got) == got  # idempotent


def test_parse_stores_folded_label():
    (variant,) = parse_asr_training_target("<t>Laser CO₂,\n à 10 %.</t>", expected=1)
    assert variant == "Laser CO2, à 10 %.", variant


def test_meaning_bearing_uncovered_char_still_reaches_gate():
    # NFKC leaves ≥ as ≥, which the vocab lacks: no silent rewrite, the gate re-prompts
    # so the LLM writes "supérieur ou égal à".
    got = _normalize_tokenizable_text("Glycémie ≥ 7 mmol.")
    assert "≥" in got and _PARAKEET_TOK.offending_chars(got) == {"≥"}, got


def main() -> int:
    tests = [
        test_v1_characters_are_folded_and_raw_covered,
        test_parse_stores_folded_label,
        test_meaning_bearing_uncovered_char_still_reaches_gate,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {t.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
