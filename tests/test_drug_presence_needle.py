#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "click",
#     "loguru",
#     "litellm",
#     "tiktoken",
#     "tenacity",
#     "rapidfuzz",
# ]
# ///
"""Tests for the drugs stage's ATC combination-label presence handling.

`presence_needle` strips the generic therapeutic-class filler from an ATC
combination label so the term-presence check only requires the concrete drug
anchor(s). These pin: filler is dropped, real "+"/"ET" combinations require both
active drugs (order-independent), a fully generic label skips the check, and a
plain single drug is unaffected.

Run: `uv run tests/test_drug_presence_needle.py`
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "utils"))

from _pipeline_shared import TermMissingError  # noqa: E402


def _load_drug_module():
    """Import the digit-prefixed drugs generator module by path."""
    spec = importlib.util.spec_from_file_location(
        "gen_drugs", REPO / "02_drugs" / "01_generate_drug_texts.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DRUGS = _load_drug_module()


def test_needle_strips_en_association_tail():
    assert DRUGS.presence_needle("LIDOCAINE EN ASSOCIATION") == "lidocaine"
    assert (
        DRUGS.presence_needle("PARACETAMOL EN ASSOCIATION AVEC DES PSYCHOLEPTIQUES")
        == "paracetamol"
    )


def test_needle_strips_generic_class_after_et():
    assert DRUGS.presence_needle("IRBESARTAN ET DIURETIQUES") == "irbesartan"
    assert DRUGS.presence_needle("AMOXICILLINE ET INHIBITEUR D'ENZYME") == "amoxicilline"
    assert DRUGS.presence_needle("ATENOLOL ET AUTRES ANTIHYPERTENSEURS") == "atenolol"


def test_needle_keeps_real_combination_components():
    # Both active drugs are real -> both required, rejoined as a "+" combination.
    assert DRUGS.presence_needle("SALMETEROL ET FLUTICASONE") == "salmeterol + fluticasone"
    assert DRUGS.presence_needle("CODEINE + PARACETAMOL") == "codeine + paracetamol"


def test_needle_empty_for_pure_generic_label():
    assert DRUGS.presence_needle("ASSOCIATIONS") == ""
    assert DRUGS.presence_needle("SELS MINERAUX EN ASSOCIATION") == ""
    assert DRUGS.presence_needle("ASSOCIATIONS DE VITAMINES") == ""


def test_needle_passes_plain_single_drug_through():
    assert DRUGS.presence_needle("PARACETAMOL") == "paracetamol"


def test_needle_keeps_french_decimal_comma_in_dosage():
    # A comma flanked by digits is a decimal point inside a dosage, NOT a
    # combination separator: it must not fabricate a spurious "+" component.
    needle = DRUGS.presence_needle("TREPROSTIN.TLO2,5MG/ML")
    assert "+" not in needle
    assert needle == "treprostin tlo2 5mg"


def test_needle_still_splits_real_comma_combination():
    # A comma with non-digit neighbours is a genuine list separator: still split.
    assert (
        DRUGS.presence_needle("CODEINE, PARACETAMOL") == "codeine + paracetamol"
    )


def test_validate_ignores_filler_when_anchor_present():
    # A natural sentence names irbesartan but never "diuretiques": must pass.
    entry = {"term": "IRBESARTAN ET DIURETIQUES", "index": 0}
    variant = "On introduit de l'irbesartan associe a un diuretique thiazidique le matin."
    DRUGS.drug_validate([variant], entry)  # must not raise


def test_validate_requires_the_real_anchor():
    # The concrete drug must still appear: a sentence naming neither raises.
    entry = {"term": "IRBESARTAN ET DIURETIQUES", "index": 0}
    variant = "Le patient prend un comprime chaque matin depuis une semaine."
    try:
        DRUGS.drug_validate([variant], entry)
    except TermMissingError:
        pass
    else:
        raise AssertionError("a missing real anchor must raise")


def test_validate_skips_check_for_pure_generic_label():
    # No specific name to require: any (validator-clean) sentence passes.
    entry = {"term": "ASSOCIATIONS", "index": 0}
    variant = "Le traitement associe plusieurs molecules pour un meilleur controle."
    DRUGS.drug_validate([variant], entry)  # must not raise


def test_validate_combination_requires_both_active_drugs():
    entry = {"term": "SALMETEROL ET FLUTICASONE", "index": 0}
    both = "On prescrit du salmeterol et de la fluticasone en inhalation deux fois par jour."
    DRUGS.drug_validate([both], entry)  # must not raise
    only_one = "On prescrit du salmeterol en inhalation deux fois par jour."
    try:
        DRUGS.drug_validate([only_one], entry)
    except TermMissingError:
        pass
    else:
        raise AssertionError("a combination missing one active drug must raise")


def test_acceptable_needles_include_term_and_substances():
    # A truncated brand code plus its real active ingredients: both are anchors.
    entry = {
        "term": "EMTRICIT/TENOF.MYL200/245",
        "substances": ["TENOFOVIR DISOPROXIL ET EMTRICITABINE"],
    }
    needles = DRUGS._acceptable_needles(entry)
    assert "tenofovir disoproxil + emtricitabine" in needles
    # The reduced raw code is kept too (a variant may name the brand verbatim).
    assert any("emtricit" in n for n in needles)


def test_validate_accepts_substances_when_brand_code_absent():
    # The model can only speak the molecules, not the truncated code: must pass
    # off the substances anchor.
    entry = {
        "term": "EMTRICIT/TENOF.MYL200/245",
        "substances": ["TENOFOVIR DISOPROXIL ET EMTRICITABINE"],
        "index": 0,
    }
    variant = (
        "Prescription d'emtricitabine 200 milligrammes et tenofovir disoproxil "
        "245 milligrammes, un comprime par jour."
    )
    DRUGS.drug_validate([variant], entry)  # must not raise


def test_validate_substances_combination_requires_every_molecule():
    # A "X ET Y" substances string still requires both molecules present.
    entry = {
        "term": "EMTRICIT/TENOF.MYL200/245",
        "substances": ["TENOFOVIR DISOPROXIL ET EMTRICITABINE"],
        "index": 0,
    }
    only_one = "Prescription de tenofovir disoproxil 245 milligrammes par jour."
    try:
        DRUGS.drug_validate([only_one], entry)
    except TermMissingError:
        pass
    else:
        raise AssertionError("naming only one of two active molecules must raise")


def test_validate_raises_when_neither_term_nor_substances_present():
    entry = {
        "term": "PARACET.KBI10MG/ML",
        "substances": ["PARACETAMOL"],
        "index": 0,
    }
    variant = "Le patient prend un comprime chaque matin depuis une semaine."
    try:
        DRUGS.drug_validate([variant], entry)
    except TermMissingError:
        pass
    else:
        raise AssertionError("no acceptable anchor present must raise")


def test_validate_single_brand_accepts_brand_name():
    # A normal brand (real substances) still passes when only the brand is named.
    entry = {"term": "DOLIPRANE", "substances": ["PARACETAMOL"], "index": 0}
    variant = "On prescrit du Doliprane un gramme trois fois par jour."
    DRUGS.drug_validate([variant], entry)  # must not raise
    # ... and equally when only the molecule is named.
    DRUGS.drug_validate(["On donne un peu de paracetamol au patient."], entry)


def main() -> int:
    tests = [
        test_needle_strips_en_association_tail,
        test_needle_strips_generic_class_after_et,
        test_needle_keeps_real_combination_components,
        test_needle_empty_for_pure_generic_label,
        test_needle_passes_plain_single_drug_through,
        test_needle_keeps_french_decimal_comma_in_dosage,
        test_needle_still_splits_real_comma_combination,
        test_validate_ignores_filler_when_anchor_present,
        test_validate_requires_the_real_anchor,
        test_validate_skips_check_for_pure_generic_label,
        test_validate_combination_requires_both_active_drugs,
        test_acceptable_needles_include_term_and_substances,
        test_validate_accepts_substances_when_brand_code_absent,
        test_validate_substances_combination_requires_every_molecule,
        test_validate_raises_when_neither_term_nor_substances_present,
        test_validate_single_brand_accepts_brand_name,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
