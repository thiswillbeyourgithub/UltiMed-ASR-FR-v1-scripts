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
"""Tests for the reference-example leak validator (_check_no_example_leak).

The validator scores each (example, variant) pair by the AVERAGE of partial_ratio
(containment) and ratio (length-aware similarity), so a short example merely
appearing in a long variant does not count as leakage (term 107 is "abdomen" and
lists "abdomen" as an example, which the term-presence check in fact requires),
while a whole reference sentence reproduced does.

Run:  python tests/test_example_leak.py
(or)  uv run tests/test_example_leak.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "utils"))

from _pipeline_shared import (  # noqa: E402
    LLMError,
    _EXAMPLE_LEAK_MAX_RATIO,
    _check_no_example_leak,
)


def _raises(variants, examples) -> bool:
    try:
        _check_no_example_leak(variants, examples)
        return False
    except LLMError:
        return True


def test_single_word_example_not_leak():
    # The term-presence check REQUIRES "abdomen" in every variant, so the
    # one-word "abdomen" example must not simultaneously count as leakage.
    variant = (
        "Consultation pour douleurs abdominales. L'examen revele un abdomen "
        "souple, sensible dans la fosse iliaque droite."
    )
    assert not _raises([variant], ["abdomen"]), "one-word example must not be leakage"


def test_short_phrase_example_not_leak():
    # A 2-3 word medical phrase is terminology, not distinctive phrasing.
    variant = "Le patient presente une fosse iliaque droite sensible a la palpation."
    assert not _raises([variant], ["fosse iliaque droite"])


def test_full_sentence_leak_flagged():
    ex = "La radiographie thoracique montre une cardiomegalie moderee sans foyer."
    variant = ex + " parenchymateux."
    assert _raises([variant], [ex]), "echoing a full reference sentence must be flagged"


def test_long_leak_flagged_even_beside_short_example():
    # The short example is skipped, but a genuine long-sentence leak alongside it
    # is still caught (the gate filters examples, it does not disable the check).
    ex_short = "abdomen"
    ex_long = "La radiographie thoracique montre une cardiomegalie moderee sans foyer."
    variant = ex_long + " Abdomen souple par ailleurs."
    assert _raises([variant], [ex_short, ex_long])


def test_average_metric_not_pure_containment():
    # The one-word example scores partial_ratio 100 (it appears verbatim), which
    # the old pure-partial_ratio check flagged. Averaging with the length-aware
    # ratio must keep it below threshold: containment alone no longer triggers.
    from rapidfuzz import fuzz

    ex = "abdomen"
    variant = (
        "Consultation pour douleurs abdominales. L'examen revele un abdomen "
        "souple, sensible dans la fosse iliaque droite."
    ).lower()
    assert fuzz.partial_ratio(ex, variant) >= _EXAMPLE_LEAK_MAX_RATIO
    score = (fuzz.partial_ratio(ex, variant) + fuzz.ratio(ex, variant)) / 2
    assert score < _EXAMPLE_LEAK_MAX_RATIO, f"averaged score {score:.0f} must be below threshold"


def main() -> int:
    tests = [
        test_single_word_example_not_leak,
        test_short_phrase_example_not_leak,
        test_full_sentence_leak_flagged,
        test_long_leak_flagged_even_beside_short_example,
        test_average_metric_not_pure_containment,
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
