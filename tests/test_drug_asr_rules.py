"""Tests for 08_drug_asr_rules: the doctests of 01_extract_drug_errors.py (tokenizing,
drug spotting, the character-level mapping inside a multi-word replace, glued elisions)
and of 02_build_fix_rules.py (pattern bounds, rejection reasons, ordering, apply_rules),
03_merge_rules.py, 04_drop_unk_variants.py and 05_build_manual_rules.py.

    uv run --with click --with loguru --with litellm --with tiktoken --with tqdm \
        --with tenacity --with rapidfuzz --with wordfreq tests/test_drug_asr_rules.py

This file was written by Claude Code.
"""
import doctest
import importlib.util
from pathlib import Path

STAGE = Path(__file__).resolve().parent.parent / "08_drug_asr_rules"


def _run_doctests(name: str) -> None:
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), STAGE / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    failed, attempted = doctest.testmod(mod)
    assert attempted and not failed, (name, failed, attempted)
    print(f"{name}: OK ({attempted} cases)")


def test_extract_drug_errors() -> None:
    _run_doctests("01_extract_drug_errors.py")


def test_build_fix_rules() -> None:
    _run_doctests("02_build_fix_rules.py")


def test_merge_rules() -> None:
    _run_doctests("03_merge_rules.py")


def test_drop_unk_variants() -> None:
    _run_doctests("04_drop_unk_variants.py")


def test_build_manual_rules() -> None:
    _run_doctests("05_build_manual_rules.py")


if __name__ == "__main__":
    test_extract_drug_errors()
    test_build_fix_rules()
    test_merge_rules()
    test_drop_unk_variants()
    test_build_manual_rules()
