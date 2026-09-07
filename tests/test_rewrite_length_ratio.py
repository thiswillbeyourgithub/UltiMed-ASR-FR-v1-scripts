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
"""Regression test for the rewrite length-ratio upper bound in utils/text_rewrite.

17 of 20 PARHAF skips were `validation_after_retries` with the paragraph running
132%-167% of the source length. The rewrite prompt REQUIRES spelling every unit
and number out in full French words ("40 mg" -> "quarante milligrammes"), which
legitimately inflates a dense medication / lab chunk past the old 1.30 cap, so a
faithful rewrite could never pass. The cap is now 2.0: a ~1.6x spell-out passes,
genuine runaway padding (>2.0x) still fails.

40 PARHAF chunks then failed the 2.0 cap too, at 201%-318%, all of them pure
shorthand (lab panels, biology tables) where the spell-out cannot fit under any
useful cap. Those are re-run with the upper bound OFF (`max_len_ratio=None`, the
stage wrappers' `--no-max-len-ratio`), which must still enforce the lower bound.

Run:  python tests/test_rewrite_length_ratio.py
(or)  uv run tests/test_rewrite_length_ratio.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "utils"))

from _pipeline_shared import LLMError  # noqa: E402
import text_rewrite  # noqa: E402
from text_rewrite import MAX_LEN_RATIO, rewrite_validate  # noqa: E402


def _pad_to(target_len: int) -> str:
    """Build a clean French sentence of ~target_len chars, ending on a period.

    Grows a plain word list so it never overshoots the target, then closes with a
    period, so the caller controls the length-ratio precisely.
    """
    body = "le patient reste stable"
    filler = " et calme"
    while len(body) + len(filler) + 1 <= target_len:
        body += filler
    return body + "."


def test_spell_out_expansion_passes_under_new_cap():
    # A 1.6x expansion (spelling units out) that the old 1.30 cap wrongly rejected.
    src = "PANTOPRAZOLE 40 mg 1-0-1 pendant 8 semaines puis 40 mg par jour."
    para = _pad_to(int(len(src) * 1.6))
    ratio = len(para) / len(src)
    assert 1.30 < ratio < MAX_LEN_RATIO, ratio  # in the newly-allowed band
    rewrite_validate([para], {"text": src})  # must not raise


def test_runaway_padding_still_rejected():
    src = "Pantoprazole quarante milligrammes le matin et le soir."
    para = _pad_to(int(len(src) * 2.3))
    assert len(para) / len(src) > MAX_LEN_RATIO
    try:
        rewrite_validate([para], {"text": src})
    except LLMError as e:
        assert "of the source length" in str(e)
    else:
        raise AssertionError("expected LLMError for a >2.0x rewrite")


def test_disabled_cap_accepts_a_shorthand_spell_out():
    # A real skipped chunk shape: pure lab shorthand, whose faithful spell-out runs
    # past 3x. --no-max-len-ratio must let it through.
    src = "Na+ 142meq/L; K+ 3,6meq/L; uree 2,6mmol/L; creatininemie 72umol/L"
    para = _pad_to(int(len(src) * 3.2))
    assert len(para) / len(src) > MAX_LEN_RATIO
    rewrite_validate([para], {"text": src}, max_len_ratio=None)  # must not raise


def test_disabled_cap_still_enforces_the_lower_bound():
    # Turning the upper bound off must not also stop catching a dropped-content
    # rewrite: the two bounds guard different failures.
    src = _pad_to(2000)
    para = "Le patient va bien."
    assert len(para) / len(src) < text_rewrite._MIN_LEN_RATIO
    try:
        rewrite_validate([para], {"text": src}, max_len_ratio=None)
    except LLMError as e:
        assert "too much was dropped" in str(e)
    else:
        raise AssertionError("expected LLMError for a rewrite under the lower bound")


def test_adapter_threads_the_disabled_cap_through():
    # The engine calls adapter.validate_fn(parsed, entry) with no kwargs, so the
    # setting has to be bound into the adapter, not passed at call time.
    src = "Na+ 142meq/L; K+ 3,6meq/L; uree 2,6mmol/L; creatininemie 72umol/L"
    para = _pad_to(int(len(src) * 3.2))
    off = text_rewrite.build_rewrite_adapter("parhaf", "sys", max_len_ratio=None)
    off.validate_fn([para], {"text": src})  # must not raise
    on = text_rewrite.build_rewrite_adapter("parhaf", "sys")
    try:
        on.validate_fn([para], {"text": src})
    except LLMError:
        pass
    else:
        raise AssertionError("default adapter should still enforce the upper bound")


def test_cap_constant_is_the_relaxed_value():
    # Guard the intentional relaxation so a future edit back to 1.30 fails loudly.
    assert text_rewrite.MAX_LEN_RATIO == 2.0


def main() -> int:
    tests = [
        test_spell_out_expansion_passes_under_new_cap,
        test_runaway_padding_still_rejected,
        test_disabled_cap_accepts_a_shorthand_spell_out,
        test_disabled_cap_still_enforces_the_lower_bound,
        test_adapter_threads_the_disabled_cap_through,
        test_cap_constant_is_the_relaxed_value,
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
