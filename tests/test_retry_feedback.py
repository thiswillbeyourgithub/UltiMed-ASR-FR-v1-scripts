#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "click",
#     "loguru",
#     "tqdm",
#     "tenacity",
#     "litellm",
#     "rapidfuzz",
#     "tiktoken",
# ]
# ///
"""Tests for the retry-feedback refinements in `utils/_pipeline_shared`.

Covers the self-contained changes made to the Layer-2 validation retry so a
62k-term run stays cheap and observable:

  - `_strip_reasoning` removes <think>/<thinking>/<reasoning> blocks (closed
    and unclosed-trailing) so the `<t>` parser never eats a mid-thought example.
  - `_preview_head_tail` bounds how much of a failed generation is echoed back.
  - `_apply_retry_context` keeps system+user as an untouched prefix (cache
    reuse), wraps feedback in short <verr>/<prev> tags, and truncates the
    echoed previous output.
  - PricingTracker.note_transport_retry / note_validation_retry feed the live
    per-call retry rates shown in the progress bar and cost line.

Run:  python tests/test_retry_feedback.py
(or)  uv run tests/test_retry_feedback.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "utils"))

from _pipeline_shared import (  # noqa: E402
    _FEEDBACK_PREVIEW_LIMIT,
    _apply_retry_context,
    _preview_head_tail,
    _strip_reasoning,
)


def test_strip_reasoning_removes_closed_block():
    raw = "<think>let me plan the variants</think><t>Bonjour</t>"
    out = _strip_reasoning(raw)
    assert out == "<t>Bonjour</t>", out


def test_strip_reasoning_is_case_insensitive_and_multitag():
    raw = "<Thinking>a</Thinking><REASONING>b</REASONING><t>ok</t>"
    out = _strip_reasoning(raw)
    assert out == "<t>ok</t>", out


def test_strip_reasoning_drops_unclosed_trailing_block():
    # A reasoning stream cut off at max_tokens: nothing valid can follow.
    raw = "<t>garde ceci</t><think>je reflechis mais je suis coupe"
    out = _strip_reasoning(raw)
    assert out == "<t>garde ceci</t>", out


def test_strip_reasoning_protects_the_t_parser():
    # An example <t> written mid-thought must not survive to the parser.
    raw = "<think>example: <t>NE PAS GARDER</t></think><t>vrai</t>"
    out = _strip_reasoning(raw)
    assert "NE PAS GARDER" not in out, out
    assert out == "<t>vrai</t>", out


def test_strip_reasoning_noop_on_plain_output():
    raw = "<t>rien a stripper</t>"
    assert _strip_reasoning(raw) == raw


def test_preview_head_tail_keeps_both_ends():
    text = "A" * 50 + "MIDDLE" + "Z" * 50
    out = _preview_head_tail(text, 20)
    assert out.startswith("AAAAAAAAAA"), out
    assert out.rstrip().endswith("ZZZZZZZZZZ"), out
    assert "elided" in out
    # MIDDLE sits in the elided section, so it should be gone.
    assert "MIDDLE" not in out, out


def test_preview_head_tail_short_text_untouched():
    text = "court"
    assert _preview_head_tail(text, 100) == text


def test_apply_retry_context_none_is_passthrough():
    assert _apply_retry_context("PROMPT", None) == "PROMPT"
    assert _apply_retry_context("PROMPT", {}) == "PROMPT"


def test_apply_retry_context_preserves_prefix_for_caching():
    # The original user prompt must remain a literal prefix so a prefix-caching
    # provider still hits system+user across retries.
    prompt = "Term: aspirine\nProduce 2 variants."
    ctx = {
        "previous_output": "<t>bad</t>",
        "error_message": "variant 0 uses banned char",
        "extra_focus": False,
    }
    out = _apply_retry_context(prompt, ctx)
    assert out.startswith(prompt), "feedback must be appended after the user prompt"


def test_apply_retry_context_uses_short_xml_tags():
    ctx = {
        "previous_output": "<t>bad</t>",
        "error_message": "variant 0 uses banned char '('",
        "extra_focus": False,
    }
    out = _apply_retry_context("P", ctx)
    assert "<verr>variant 0 uses banned char '('</verr>" in out, out
    assert "<prev>" in out and "</prev>" in out, out


def test_apply_retry_context_truncates_long_previous_output():
    long_prev = "X" * (_FEEDBACK_PREVIEW_LIMIT + 5000)
    ctx = {
        "previous_output": long_prev,
        "error_message": "too long",
        "extra_focus": False,
    }
    out = _apply_retry_context("P", ctx)
    assert "elided" in out, "an over-long previous output must be truncated"
    assert len(out) < len(long_prev), out


def test_apply_retry_context_extra_focus_only_has_no_prev_block():
    ctx = {"previous_output": None, "error_message": None, "extra_focus": True}
    out = _apply_retry_context("P", ctx)
    assert "<prev>" not in out, out
    assert "IMPORTANT" in out, out
    assert out.startswith("P"), out


def test_tracker_retry_counters_average_over_calls():
    # PricingTracker.__init__ hits the network for prices; build a bare instance
    # so the test stays offline and deterministic.
    from _pipeline_shared import PricingTracker

    t = PricingTracker.__new__(PricingTracker)
    import threading

    t._lock = threading.Lock()
    t.calls = 0
    t.transport_retries = 0
    t.validation_retries = 0

    t.note_transport_retry()
    t.note_transport_retry()
    t.note_validation_retry()
    t.calls = 4
    assert t.transport_retries == 2
    assert t.validation_retries == 1
    # 2 transport / 4 calls = 0.50 per call.
    assert abs(t.transport_retries / max(1, t.calls) - 0.5) < 1e-9


def main() -> int:
    tests = [
        test_strip_reasoning_removes_closed_block,
        test_strip_reasoning_is_case_insensitive_and_multitag,
        test_strip_reasoning_drops_unclosed_trailing_block,
        test_strip_reasoning_protects_the_t_parser,
        test_strip_reasoning_noop_on_plain_output,
        test_preview_head_tail_keeps_both_ends,
        test_preview_head_tail_short_text_untouched,
        test_apply_retry_context_none_is_passthrough,
        test_apply_retry_context_preserves_prefix_for_caching,
        test_apply_retry_context_uses_short_xml_tags,
        test_apply_retry_context_truncates_long_previous_output,
        test_apply_retry_context_extra_focus_only_has_no_prev_block,
        test_tracker_retry_counters_average_over_calls,
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
