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
"""Regression test for the curly-apostrophe fold in `utils/_pipeline_shared`.

Run 20260706T074824-bb21ab aborted mid-run: the model wrote the typographic
apostrophe U+2019 ("l'os" style French elision) in an `asr_training_target`
variant, the forbidden-char validator rejected it on all 5 retries, and the
exhausted ValidationError killed the whole thread pool (~180k rows in).

U+2019 (and its mirror U+2018) is a typography-only glyph with a lossless
straight-apostrophe equivalent, and "'" is the correct written form the vocab
covers. So it is folded deterministically at parse time in
`parse_asr_training_target` (via `_TOKENIZABLE_FOLDS` / `_normalize_tokenizable_text`)
BEFORE validation, exactly like 01_dictionnary/02_token_check.py folds it on
input terms. These tests assert the offending variant now parses + validates
clean instead of aborting.

Run:  python tests/test_apostrophe_fold.py
(or)  uv run tests/test_apostrophe_fold.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "utils"))

from _pipeline_shared import (  # noqa: E402
    _FORBIDDEN_W_CHARS,
    _normalize_tokenizable_text,
    parse_asr_training_target,
    validate_asr_training_target,
)

# The exact glyph that aborted the run, and its opening mirror.
RIGHT_QUOTE = "’"
LEFT_QUOTE = "‘"


def test_forbidden_set_still_bans_the_curly_quote():
    # The fold, not a relaxed blacklist, is what fixes this: the raw glyph must
    # still be considered forbidden so any un-folded path is still caught.
    assert RIGHT_QUOTE in _FORBIDDEN_W_CHARS


def test_normalize_folds_both_curly_single_quotes():
    assert _normalize_tokenizable_text(f"l{RIGHT_QUOTE}os") == "l'os"
    assert _normalize_tokenizable_text(f"{LEFT_QUOTE}bord") == "'bord"
    # No curly quote survives the fold.
    folded = _normalize_tokenizable_text(f"d{RIGHT_QUOTE}h{LEFT_QUOTE}…")
    assert RIGHT_QUOTE not in folded and LEFT_QUOTE not in folded


def test_parse_strips_curly_quote_from_variant():
    raw = f"<t>Consultation d{RIGHT_QUOTE}hematologie pour un patient febrile.</t>"
    (variant,) = parse_asr_training_target(raw, expected=1)
    assert RIGHT_QUOTE not in variant
    assert variant == "Consultation d'hematologie pour un patient febrile."


def test_aborting_variant_now_validates_clean():
    # Reconstruction of term 38983 variant 0: a real French-elision apostrophe
    # rendered as U+2019, which previously failed validation on every retry.
    target = (
        f"Consultation de nephrologie pediatrique pour un nourrisson de quatre "
        f"mois. L{RIGHT_QUOTE}examen retrouve un syndrome nephrotique cortico-"
        f"resistant et le sequencage cible a revele une mutation homozygote du "
        f"gene NPHS1, confirmant l{RIGHT_QUOTE}atteinte."
    )
    parsed = parse_asr_training_target(f"<t>{target}</t>", expected=1)
    # Must not raise: the fold removed the only offending characters.
    validate_asr_training_target(parsed)
    assert all(RIGHT_QUOTE not in v for v in parsed)


def main() -> int:
    tests = [
        test_forbidden_set_still_bans_the_curly_quote,
        test_normalize_folds_both_curly_single_quotes,
        test_parse_strips_curly_quote_from_variant,
        test_aborting_variant_now_validates_clean,
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
