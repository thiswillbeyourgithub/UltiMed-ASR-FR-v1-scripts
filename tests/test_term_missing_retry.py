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
"""Tests for the term-missing retry-and-skip behavior of the dictionary stage.

Covers the two changes that turn a term-missing from a whole-run abort into a
recoverable (and, if persistent, single-term-skip) event:

  1. TermMissingError is an LLMError subclass, so `_retry_with_validation`
     retries it with feedback and only surfaces it (as a ValidationError whose
     __cause__ is the TermMissingError) once every retry is exhausted.
  2. `generate_variants_for_term` converts that persistent term-missing into a
     logged TermSkipped signal, so the run drops one term instead of aborting.

Run:  python tests/test_term_missing_retry.py
(or)  uv run tests/test_term_missing_retry.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "utils"))

from _pipeline_shared import (  # noqa: E402
    BlockCountError,
    LLMError,
    TermMissingError,
    ValidationError,
    _VALIDATION_MAX_ATTEMPTS,
    _retry_with_validation,
    _strip_accents,
    check_term_in_variants,
    parse_asr_training_target,
    term_present_in_variant,
)

# A term made of uncommon letters so it cannot fuzzily match ordinary French
# text: term-less variants score well under the 75 threshold and raise.
TERM = "zzqwxykjpv"


def _load_generator():
    """Import the digit-prefixed generator module by path."""
    spec = importlib.util.spec_from_file_location(
        "gen03", REPO / "01_dictionnary" / "03_generate_texts.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _check_or_raise(term, variants):
    """Mimic the generator: run the term check, attaching raw_output on failure."""
    try:
        check_term_in_variants(term, variants)
    except LLMError as e:
        e.raw_output = "<raw>"
        raise


def test_termmissing_is_llmerror():
    assert issubclass(TermMissingError, LLMError), (
        "TermMissingError must subclass LLMError so _retry_with_validation "
        "retries it"
    )


def test_termless_variants_actually_raise():
    # Guards the test fixtures themselves: the chosen TERM must be absent enough
    # to trip the check, or the other tests would pass vacuously.
    raised = False
    try:
        check_term_in_variants(TERM, ["Le patient se repose bien ce matin."])
    except TermMissingError:
        raised = True
    assert raised, "fixture term unexpectedly fuzzy-matched a term-less variant"


def test_retry_recovers_when_term_reappears():
    calls = {"n": 0}
    good = [f"Le compte rendu numero {i} cite {TERM} clairement." for i in range(2)]

    def produce_fn(retry_context=None):
        calls["n"] += 1
        if calls["n"] < 3:
            variants = [f"Phrase numero {i} sans le mot vise." for i in range(2)]
        else:
            variants = good
        _check_or_raise(TERM, variants)
        return variants

    out = _retry_with_validation(produce_fn, label="test-recover")
    assert out == good
    assert calls["n"] == 3, f"expected recovery on attempt 3, took {calls['n']}"


def test_retry_exhausts_to_validationerror_with_termmissing_cause():
    calls = {"n": 0}

    def produce_fn(retry_context=None):
        calls["n"] += 1
        _check_or_raise(TERM, ["Aucune occurrence ici.", "Toujours rien la non plus."])
        return []  # unreachable

    raised = None
    try:
        _retry_with_validation(produce_fn, label="test-exhaust")
    except ValidationError as e:
        raised = e
    assert raised is not None, "expected ValidationError once retries exhaust"
    assert isinstance(raised.__cause__, TermMissingError), (
        f"exhaustion must preserve the term-missing cause, got {raised.__cause__!r}"
    )
    assert calls["n"] == _VALIDATION_MAX_ATTEMPTS, (
        f"expected {_VALIDATION_MAX_ATTEMPTS} attempts, made {calls['n']}"
    )


def test_generator_skips_persistent_term_missing():
    gen = _load_generator()
    n = 2

    # Fake LLM: N clean French blocks that never contain the term.
    def fake_call_llm(**kwargs):
        return "".join(
            f"<t>Observation numero {i} parfaitement propre ici.</t>"
            for i in range(n)
        )

    gen.call_llm = fake_call_llm
    # Bypass the heavy soft-validator suite: this test targets the term-missing
    # control flow, not the validators (which have their own coverage).
    gen.validate_asr_training_target = lambda variants, examples=None: None

    entry = {"index": 42, "term": TERM, "definition": "", "examples": [], "page": 1}
    with tempfile.TemporaryDirectory() as td:
        skip_file = Path(td) / "skips.jsonl"
        raised = None
        try:
            gen.generate_variants_for_term(
                entry,
                asr_training_target_system_prompt="sys",
                n_variants=n,
                seen_target_hashes=None,
                seen_source_hashes=None,
                seen_lock=None,
                tracker=None,
                review_path=Path(td) / "review.jsonl",
                skip_path=skip_file,
            )
        except gen.TermSkipped as e:
            raised = e
        assert raised is not None, "expected TermSkipped, not a fatal error"
        assert raised.term_index == 42
        rows = [json.loads(x) for x in skip_file.read_text().splitlines() if x.strip()]
        assert len(rows) == 1, f"expected one skip-queue entry, got {len(rows)}"
        assert rows[0]["reason"] == "term_missing_after_retries"
        assert rows[0]["term_index"] == 42


def test_generator_succeeds_when_term_present():
    gen = _load_generator()
    n = 2

    def fake_call_llm(**kwargs):
        return "".join(
            f"<t>Observation numero {i} citant {TERM} sans ambiguite.</t>"
            for i in range(n)
        )

    gen.call_llm = fake_call_llm
    gen.validate_asr_training_target = lambda variants, examples=None: None

    entry = {"index": 7, "term": TERM, "definition": "", "examples": [], "page": 1}
    with tempfile.TemporaryDirectory() as td:
        rows = gen.generate_variants_for_term(
            entry,
            asr_training_target_system_prompt="sys",
            n_variants=n,
            seen_target_hashes=None,
            seen_source_hashes=None,
            seen_lock=None,
            tracker=None,
            review_path=Path(td) / "review.jsonl",
            skip_path=Path(td) / "skips.jsonl",
        )
        assert len(rows) == n, f"expected {n} rows, got {len(rows)}"
        assert all(TERM in r["asr_training_target"] for r in rows)


def test_generator_propagates_category():
    # The `category` field on the input row must flow through to every output
    # row (not be dropped by the fixed row schema), so a merged multi-source
    # dataset can tell dictionary pairs from drugs / PARHAF / PARROT pairs.
    gen = _load_generator()
    n = 2

    def fake_call_llm(**kwargs):
        return "".join(
            f"<t>Observation numero {i} citant {TERM} sans ambiguite.</t>"
            for i in range(n)
        )

    gen.call_llm = fake_call_llm
    gen.validate_asr_training_target = lambda variants, examples=None: None

    entry = {
        "index": 7, "term": TERM, "definition": "", "examples": [],
        "page": 1, "category": "dictionary",
    }
    with tempfile.TemporaryDirectory() as td:
        rows = gen.generate_variants_for_term(
            entry,
            asr_training_target_system_prompt="sys",
            n_variants=n,
            seen_target_hashes=None,
            seen_source_hashes=None,
            seen_lock=None,
            tracker=None,
            review_path=Path(td) / "review.jsonl",
            skip_path=Path(td) / "skips.jsonl",
        )
        assert rows, "expected rows"
        assert all(r["category"] == "dictionary" for r in rows), (
            "category must propagate from the input row to every output row"
        )


def test_blockcount_is_llmerror_and_parser_raises_it():
    # BlockCountError must be an LLMError so _retry_with_validation retries it
    # (rather than the old CountMismatchError, which bubbled up and aborted the
    # whole run). The parser raises it on a wrong <t> count.
    assert issubclass(BlockCountError, LLMError), (
        "BlockCountError must subclass LLMError so it is retried, not fatal"
    )
    raised = None
    try:
        parse_asr_training_target("<t>only one block</t>", expected=2)
    except BlockCountError as e:
        raised = e
    assert raised is not None, "parser must raise BlockCountError on a wrong count"


def test_generator_skips_persistent_wrong_block_count():
    gen = _load_generator()
    n = 3

    # Fake LLM that always emits ONE TOO MANY <t> blocks: parse_asr_training_target
    # raises BlockCountError every attempt, so after retries are exhausted the one
    # term is skipped (previously a wrong count aborted the entire run).
    def fake_call_llm(**kwargs):
        return "".join(
            f"<t>Observation numero {i} citant {TERM} correctement.</t>"
            for i in range(n + 1)
        )

    gen.call_llm = fake_call_llm
    gen.validate_asr_training_target = lambda variants, examples=None: None

    entry = {"index": 99, "term": TERM, "definition": "", "examples": [], "page": 1}
    with tempfile.TemporaryDirectory() as td:
        skip_file = Path(td) / "skips.jsonl"
        raised = None
        try:
            gen.generate_variants_for_term(
                entry,
                asr_training_target_system_prompt="sys",
                n_variants=n,
                seen_target_hashes=None,
                seen_source_hashes=None,
                seen_lock=None,
                tracker=None,
                review_path=Path(td) / "review.jsonl",
                skip_path=skip_file,
            )
        except gen.TermSkipped as e:
            raised = e
        assert raised is not None, "expected TermSkipped, not a fatal count-mismatch abort"
        assert raised.term_index == 99
        rows = [json.loads(x) for x in skip_file.read_text().splitlines() if x.strip()]
        assert len(rows) == 1, f"expected one skip-queue entry, got {len(rows)}"
        assert rows[0]["reason"] == "block_count_after_retries"
        assert rows[0]["term_index"] == 99


def test_generator_recovers_from_transient_wrong_block_count():
    gen = _load_generator()
    n = 2
    state = {"n": 0}

    # Attempt 1 emits N+1 blocks (BlockCountError -> retry); attempt 2 emits
    # exactly N valid blocks, so the term succeeds without a skip or an abort.
    def fake_call_llm(**kwargs):
        state["n"] += 1
        count = n + 1 if state["n"] == 1 else n
        return "".join(
            f"<t>Observation numero {i} citant {TERM} correctement.</t>"
            for i in range(count)
        )

    gen.call_llm = fake_call_llm
    gen.validate_asr_training_target = lambda variants, examples=None: None

    entry = {"index": 8, "term": TERM, "definition": "", "examples": [], "page": 1}
    with tempfile.TemporaryDirectory() as td:
        rows = gen.generate_variants_for_term(
            entry,
            asr_training_target_system_prompt="sys",
            n_variants=n,
            seen_target_hashes=None,
            seen_source_hashes=None,
            seen_lock=None,
            tracker=None,
            review_path=Path(td) / "review.jsonl",
            skip_path=Path(td) / "skips.jsonl",
        )
    assert state["n"] == 2, f"expected recovery on attempt 2, took {state['n']}"
    assert len(rows) == n, f"expected {n} rows, got {len(rows)}"
    assert all(TERM in r["asr_training_target"] for r in rows)


def test_generator_skips_persistent_forbidden_char():
    # Regression for run 20260706T195831-c96c14: term 38706 aborted the whole run
    # because its final retry failed the forbidden-char validator on a genetics
    # notation "c.1964G>A" the model would not verbalise. A forbidden char is NOT
    # a TermMissingError/BlockCountError, so it used to bubble up as a fatal
    # ValidationError. It must now be skipped like any other soft-validation
    # exhaustion. This drives the REAL validator (not a stub) so a literal ">"
    # exercises the same path the run hit.
    gen = _load_generator()
    n = 2

    # N distinct, otherwise-valid French variants that each contain the term AND a
    # ">" (HGVS substitution). Every attempt fails identically on the forbidden
    # char, so retries exhaust and the one term is skipped instead of aborting.
    def fake_call_llm(**kwargs):
        return (
            f"<t>Le patient {TERM} presente une mutation faux-sens notee "
            f"c.100G>A a l'analyse moleculaire.</t>"
            f"<t>Chez {TERM}, le laboratoire rapporte un variant c.250C>T "
            f"juge pathogene selon les criteres retenus.</t>"
        )

    gen.call_llm = fake_call_llm  # real validate_asr_training_target left in place

    entry = {"index": 38706, "term": TERM, "definition": "", "examples": [], "page": 1}
    with tempfile.TemporaryDirectory() as td:
        skip_file = Path(td) / "skips.jsonl"
        raised = None
        try:
            gen.generate_variants_for_term(
                entry,
                asr_training_target_system_prompt="sys",
                n_variants=n,
                seen_target_hashes=None,
                seen_source_hashes=None,
                seen_lock=None,
                tracker=None,
                review_path=Path(td) / "review.jsonl",
                skip_path=skip_file,
            )
        except gen.TermSkipped as e:
            raised = e
        assert raised is not None, "expected TermSkipped, not a fatal forbidden-char abort"
        assert raised.term_index == 38706
        rows = [json.loads(x) for x in skip_file.read_text().splitlines() if x.strip()]
        assert len(rows) == 1, f"expected one skip-queue entry, got {len(rows)}"
        assert rows[0]["reason"] == "validation_after_retries", rows[0]["reason"]
        assert rows[0]["term_index"] == 38706
        assert "forbidden char" in rows[0]["detail"] and ">" in rows[0]["detail"], (
            f"skip detail should record the forbidden char: {rows[0]['detail']!r}"
        )


def test_accent_insensitive_term_presence():
    # The model writes the accented French form ("hérédité") while the dictionary
    # term may be unaccented ("heredite"). Without accent normalization the raw
    # partial_ratio falls below the 75 threshold and the term is reported missing;
    # the check now strips accents on both sides so it passes.
    from rapidfuzz import fuzz

    assert _strip_accents("gène ABCG5") == "gene ABCG5"
    assert _strip_accents("hérédité") == "heredite"

    term = "heredite"
    variant = "Un antecedent d'hérédité familiale au premier degré est retrouve."
    # Guard: the case is only meaningful because the un-normalized score fails.
    assert fuzz.partial_ratio(term.lower(), variant.lower()) < 75
    check_term_in_variants(term, [variant], term_index=1)  # must not raise


def test_slash_acronym_presence():
    # "AC/A" is written "AC par A" in the label (the model and voxtral_normalize
    # both spell the slash). The bare needle strips "/" to a run-on "ACA" that
    # scores low; the " par " candidate must let the spoken form match.
    from rapidfuzz import fuzz

    term = "AC/A"
    variant = "Le rapport AC par A est mesure a 5 dioptries prismatiques."
    assert fuzz.partial_ratio("aca", _strip_accents(variant.lower())) < 75
    check_term_in_variants(term, [variant], term_index=1)  # must not raise


def test_trailing_hyphen_prefix_presence():
    # "acantho-" is a combining form; the text completes it ("acanthose").
    # Matching the raw morpheme scores 88 (a warn), matching the stem scores
    # 100 (silent). Either way it must not raise; assert it does not.
    term = "acantho-"
    variant = "La biopsie cutanee montre une acanthose reguliere avec hyperkeratose."
    check_term_in_variants(term, [variant], term_index=1)  # must not raise


def test_scattered_multiword_term_presence():
    # A long descriptive term is woven into the sentence, its words spread apart,
    # so the whole-span partial_ratio falls below 75, but every distinctive word
    # is individually present. The scattered-match fallback must accept it.
    from rapidfuzz import fuzz

    term = "polydactylie postaxiale et duplication du gros orteil"
    variant = (
        "La radiographie des pieds retrouve une polydactylie postaxiale bilaterale, "
        "avec une duplication du gros orteil droit et un metatarsien surnumeraire."
    )
    # Guard: the case only matters because the single-span score fails.
    assert fuzz.partial_ratio(_strip_accents(term.lower()), _strip_accents(variant.lower())) < 75
    check_term_in_variants(term, [variant], term_index=1)  # must not raise


def test_scattered_fallback_still_raises_when_a_word_is_absent():
    # The fallback requires EVERY distinctive word: a variant missing one of them
    # ("duplication"/"orteil" absent here) must still be reported missing so a
    # dropped term is not silently accepted.
    term = "polydactylie postaxiale et duplication du gros orteil"
    variant = "On note une polydactylie postaxiale isolee, sans autre anomalie du pied."
    try:
        check_term_in_variants(term, [variant], term_index=1)
    except TermMissingError:
        pass
    else:
        raise AssertionError("a term with a missing distinctive word must raise")


def test_combination_drug_order_independent_presence():
    # A "+"-joined combination drug ("codeine + paracetamol") is spoken with the
    # two names in any order and far apart in the sentence. Both orders, and a
    # reversed one, must pass (order- and position-independent per component).
    term = "codeine + paracetamol"
    forward = "Je donne au patient de la codeine et ensuite un peu de paracetamol."
    reversed_ = "Le patient prend du paracetamol puis, plus tard, de la codeine."
    check_term_in_variants(term, [forward], term_index=1)  # must not raise
    check_term_in_variants(term, [reversed_], term_index=1)  # must not raise


def test_combination_drug_multiword_component_presence():
    # A combination whose component is itself multi-word ("acide clavulanique")
    # must match that component as a unit, in any position.
    term = "amoxicilline + acide clavulanique"
    variant = "On associe de l'amoxicilline a de l'acide clavulanique par voie orale."
    check_term_in_variants(term, [variant], term_index=1)  # must not raise


def test_combination_drug_raises_when_a_component_is_absent():
    # Every component of a combination must be present: naming only one of the two
    # drugs must still be reported missing (a dropped component is invalid).
    term = "codeine + paracetamol"
    variant = "Je prescris du paracetamol seul, un comprime trois fois par jour."
    try:
        check_term_in_variants(term, [variant], term_index=1)
    except TermMissingError:
        pass
    else:
        raise AssertionError("a combination missing one component must raise")


def test_term_present_in_variant_boolean_core():
    # The boolean core mirrors check_term_in_variants' pass/fail decision without
    # raising: exact, accent-insensitive, scattered multi-word, and combination.
    assert term_present_in_variant("paracetamol", "on donne du paracetamol")
    assert term_present_in_variant("stenose", "une sténose serrée")  # accents
    assert term_present_in_variant(
        "codeine + paracetamol",
        "de la codeine puis un peu de paracetamol",
    )
    # A missing anchor (or a combination missing one component) is False, not raise.
    assert not term_present_in_variant("ibuprofene", "on donne du paracetamol")
    assert not term_present_in_variant(
        "codeine + paracetamol", "du paracetamol seul"
    )
    # A degenerate / empty needle is not an anchor: False, never "matches all".
    assert not term_present_in_variant("", "n'importe quel texte")


def test_reasoning_escalates_on_validation_retry():
    gen = _load_generator()
    seen_reasoning = []
    state = {"n": 0}

    def fake_call_llm(**kwargs):
        # Attempt 1 drops the term (fails check_term_in_variants -> retry);
        # attempt 2 includes it and succeeds. Record the reasoning level each
        # call so we can assert non-think -> omitted (provider default).
        seen_reasoning.append(kwargs.get("reasoning"))
        state["n"] += 1
        if state["n"] == 1:
            return "".join(
                f"<t>Phrase numero {i} sans le mot vise.</t>" for i in range(2)
            )
        return "".join(
            f"<t>Observation numero {i} citant {TERM} sans ambiguite.</t>"
            for i in range(2)
        )

    gen.call_llm = fake_call_llm
    gen.validate_asr_training_target = lambda variants, examples=None: None

    entry = {"index": 3, "term": TERM, "definition": "", "examples": [], "page": 1}
    with tempfile.TemporaryDirectory() as td:
        gen.generate_variants_for_term(
            entry,
            asr_training_target_system_prompt="sys",
            n_variants=2,
            seen_target_hashes=None,
            seen_source_hashes=None,
            seen_lock=None,
            tracker=None,
            review_path=Path(td) / "review.jsonl",
            skip_path=Path(td) / "skips.jsonl",
        )
    assert len(seen_reasoning) >= 2, seen_reasoning
    assert seen_reasoning[0] == gen.DEFAULT_REASONING == {"enabled": False}, (
        f"first attempt must be non-think, got {seen_reasoning[0]!r}"
    )
    # Retry omits reasoning (RETRY_REASONING is None) so the provider default
    # thinking level applies; call_llm then sends no `reasoning` at all.
    assert seen_reasoning[1] == gen.RETRY_REASONING is None, (
        f"validation retry must omit reasoning (got {seen_reasoning[1]!r})"
    )


def main() -> int:
    tests = [
        test_termmissing_is_llmerror,
        test_termless_variants_actually_raise,
        test_retry_recovers_when_term_reappears,
        test_retry_exhausts_to_validationerror_with_termmissing_cause,
        test_generator_skips_persistent_term_missing,
        test_generator_succeeds_when_term_present,
        test_generator_propagates_category,
        test_blockcount_is_llmerror_and_parser_raises_it,
        test_generator_skips_persistent_wrong_block_count,
        test_generator_recovers_from_transient_wrong_block_count,
        test_generator_skips_persistent_forbidden_char,
        test_accent_insensitive_term_presence,
        test_slash_acronym_presence,
        test_trailing_hyphen_prefix_presence,
        test_scattered_multiword_term_presence,
        test_scattered_fallback_still_raises_when_a_word_is_absent,
        test_combination_drug_order_independent_presence,
        test_combination_drug_multiword_component_presence,
        test_combination_drug_raises_when_a_component_is_absent,
        test_term_present_in_variant_boolean_core,
        test_reasoning_escalates_on_validation_retry,
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
