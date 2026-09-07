#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "click",
# ]
# ///
"""Tests for the dosage de-duplication in create_drug_db.py.

BDPM lists one strength several ways for a single presentation ("500 mg",
"500,0 mg", "500,00 mg"). Unfixed, all three reach the LLM's presentation hint
and read as three distinct doses of the same drug. 65 of the 482 presentation
lists in drugs_dosages.jsonl were affected.

Run: uv run tests/test_drug_dosage_dedup.py
"""
import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "02_drugs/sources/base_de_donnee_medicament/create_drug_db.py"
)
_spec = importlib.util.spec_from_file_location("create_drug_db", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["create_drug_db"] = _mod
_spec.loader.exec_module(_mod)

canonical_dosage = _mod.canonical_dosage
dedupe_dosages = _mod.dedupe_dosages
dedupe_forms = _mod.dedupe_forms
dosage_key = _mod.dosage_key


def test_key_folds_trailing_zeros():
    assert dosage_key("500 mg") == dosage_key("500,0 mg") == dosage_key("500,00 mg")
    assert dosage_key("500 mg") == (Decimal("500"), "mg")


def test_key_accepts_dot_decimal_separator():
    assert dosage_key("0.5 mg") == dosage_key("0,50 mg")


def test_key_handles_french_thousands_separators():
    assert dosage_key("1 000 mg") == (Decimal("1000"), "mg")
    assert dosage_key("2 000 000 UI") == (Decimal("2000000"), "ui")


def test_key_does_not_merge_different_units():
    assert dosage_key("1 g") != dosage_key("1 mg")
    # 0,5 g and 500 mg are the same quantity but not the same spoken dose, and
    # the unit is part of the key, so they stay separate.
    assert dosage_key("0,5 g") != dosage_key("500 mg")


def test_key_keeps_punctuated_unit_variants_distinct():
    # Deliberately NOT merged: guessing that "M UI" and "M.U.I." are one unit is
    # a different change from folding trailing zeros.
    assert dosage_key("1 M UI") != dosage_key("1 M.U.I.")


def test_key_rejects_non_numeric():
    assert dosage_key("un comprimé") is None
    assert dosage_key("") is None


def test_canonical_strips_zero_padding():
    assert canonical_dosage("25,00 mg") == "25 mg"
    assert canonical_dosage("12,500 mg") == "12,5 mg"
    assert canonical_dosage("2,00 mg") == "2 mg"
    assert canonical_dosage("0,50 mg") == "0,5 mg"


def test_canonical_collapses_whitespace():
    assert canonical_dosage("25,0000  mg") == "25 mg"


def test_canonical_preserves_thousands_grouping_and_unit():
    # Only the fraction is touched; the integer part keeps its spacing.
    assert canonical_dosage("1 000 mg") == "1 000 mg"
    assert canonical_dosage("2 000 000 UI") == "2 000 000 UI"


def test_canonical_leaves_clean_values_alone():
    for value in ("500 mg", "0,25 mg", "12,5 mg", "65 mg"):
        assert canonical_dosage(value) == value


def test_canonical_passes_unparseable_through():
    assert canonical_dosage("dose  adaptée") == "dose adaptée"


def test_dedupe_keeps_canonical_spelling():
    out = dedupe_dosages(["un comprimé", "200 mg", "200,00 mg"])
    assert out == ["un comprimé", "200 mg"]


def test_dedupe_fixes_a_lone_padded_dosage():
    # 88 presentations had a padded value with no shorter twin, so plain
    # de-duplication would have left them untouched.
    assert dedupe_dosages(["un comprimé", "25,00 mg"]) == ["un comprimé", "25 mg"]


def test_dedupe_keeps_head_untouched():
    out = dedupe_dosages(["une gélule", "1 mg", "1,0 mg", "1,00 mg"])
    assert out[0] == "une gélule"
    assert out == ["une gélule", "1 mg"]


def test_dedupe_preserves_source_order():
    # The real ALPRAZOLAM row: 0,50 must fold into 0,5 without reordering.
    out = dedupe_dosages(["un comprimé", "0,25 mg", "0,5 mg", "0,50 mg", "1 mg"])
    assert out == ["un comprimé", "0,25 mg", "0,5 mg", "1 mg"]


def test_dedupe_is_idempotent():
    once = dedupe_dosages(["un comprimé", "100 mg", "100,00 mg", "200 mg"])
    assert dedupe_dosages(once) == once


def test_dedupe_keeps_distinct_doses():
    out = dedupe_dosages(["un comprimé", "10 mg", "20 mg", "30 mg", "40 mg"])
    assert out == ["un comprimé", "10 mg", "20 mg", "30 mg", "40 mg"]


def test_dedupe_drops_empty_entries():
    assert dedupe_dosages(["un comprimé", "", "  ", "5 mg"]) == ["un comprimé", "5 mg"]


def test_dedupe_keeps_unparseable_but_dedupes_identical():
    out = dedupe_dosages(["un comprimé", "dose adaptée", "dose adaptée", "5 mg"])
    assert out == ["un comprimé", "dose adaptée", "5 mg"]


def test_dedupe_forms_counts_removals():
    forms = {
        "comprimé": [["un comprimé", "50 mg", "50,0 mg", "50,00 mg"]],
        "gélule": [["une gélule", "25 mg"]],
    }
    cleaned, removed = dedupe_forms(forms)
    assert removed == 2
    assert cleaned["comprimé"] == [["un comprimé", "50 mg"]]
    assert cleaned["gélule"] == [["une gélule", "25 mg"]]


def test_dedupe_forms_collapses_presentations_made_identical():
    forms = {"comprimé": [["un comprimé", "1 mg"], ["un comprimé", "1,00 mg"]]}
    cleaned, _ = dedupe_forms(forms)
    assert cleaned["comprimé"] == [["un comprimé", "1 mg"]]


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run()
