"""Tests for 06_hotfixes/04_flag_asr_defects.py: its ``find_defect`` doctests hold the
cases the rules were tuned on (a TTS preamble, a skip both recognizers confirm, and the
false positives a glued code / compound / reworded stretch produced).

    uv run --with click --with loguru tests/test_flag_asr_defects.py

This file was written by Claude Code.
"""
import doctest
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "06_hotfixes" / "04_flag_asr_defects.py"


def test_find_defect() -> None:
    spec = importlib.util.spec_from_file_location("flag_asr_defects", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    failed, attempted = doctest.testmod(mod)
    assert attempted and not failed, (failed, attempted)
    print(f"test_find_defect: OK ({attempted} cases)")


if __name__ == "__main__":
    test_find_defect()
