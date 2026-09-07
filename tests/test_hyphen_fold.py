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
"""Regression test for the typographic-hyphen fold in `utils/_pipeline_shared`.

PARHAF rewrite chunk 7191 skipped after 5 retries: an otherwise-clean French
paragraph contained U+2010 HYPHEN ("cortico-resistant" style), which the
Parakeet tokenizer maps to <unk>, so the rewrite `<unk>` gate rejected every
attempt. U+2010/U+2011 (and U+2212 minus) are typography-only variants of the
plain hyphen-minus with a lossless ASCII equivalent, so they are folded
deterministically at parse time in `parse_asr_training_target` (via
`_TOKENIZABLE_FOLDS` / `_normalize_tokenizable_text`) BEFORE validation, exactly
like the curly apostrophe is (see test_apostrophe_fold.py).

The en/em dash (U+2013/U+2014) are intentionally NOT folded: they stay forbidden
so a real dash aside is re-prompted rather than silently rewritten.

Run:  python tests/test_hyphen_fold.py
(or)  uv run tests/test_hyphen_fold.py
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
)

HYPHEN = "‐"      # ‐ U+2010 HYPHEN (the glyph that skipped chunk 7191)
NB_HYPHEN = "‑"   # ‑ U+2011 NON-BREAKING HYPHEN
MINUS = "−"       # − U+2212 MINUS SIGN
EN_DASH = "–"     # – U+2013 EN DASH (must stay forbidden, not folded)
EM_DASH = "—"     # — U+2014 EM DASH (must stay forbidden, not folded)


def test_normalize_folds_hyphen_variants_to_ascii():
    assert _normalize_tokenizable_text(f"cortico{HYPHEN}resistant") == "cortico-resistant"
    assert _normalize_tokenizable_text(f"non{NB_HYPHEN}hodgkinien") == "non-hodgkinien"
    assert _normalize_tokenizable_text(f"moins{MINUS}deux") == "moins-deux"
    folded = _normalize_tokenizable_text(f"a{HYPHEN}b{NB_HYPHEN}c{MINUS}d")
    assert HYPHEN not in folded and NB_HYPHEN not in folded and MINUS not in folded


def test_en_and_em_dash_are_not_folded_and_stay_forbidden():
    # The dash family is meaning-bearing (a range / aside), not a hyphen: it must
    # survive the fold so the forbidden-char gate re-prompts instead of silently
    # turning it into a hyphen.
    assert _normalize_tokenizable_text(f"a{EN_DASH}b") == f"a{EN_DASH}b"
    assert _normalize_tokenizable_text(f"a{EM_DASH}b") == f"a{EM_DASH}b"
    assert EN_DASH in _FORBIDDEN_W_CHARS
    assert EM_DASH in _FORBIDDEN_W_CHARS


def test_parse_strips_hyphen_from_paragraph():
    raw = f"<t>Syndrome nephrotique cortico{HYPHEN}resistant chez le nourrisson.</t>"
    (variant,) = parse_asr_training_target(raw, expected=1)
    assert HYPHEN not in variant
    assert variant == "Syndrome nephrotique cortico-resistant chez le nourrisson."


def main() -> int:
    tests = [
        test_normalize_folds_hyphen_variants_to_ascii,
        test_en_and_em_dash_are_not_folded_and_stay_forbidden,
        test_parse_strips_hyphen_from_paragraph,
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
