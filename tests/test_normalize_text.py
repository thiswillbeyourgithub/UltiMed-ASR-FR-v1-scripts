"""Stdlib test for 99_hf_release/05_normalize_text.py.

Checks that a v1 manifest row gets both of its texts normalized: the label through
``ParakeetTokenizer.clean_label`` + ``%``, the TTS source through ``pourcent``. The percent
rewrite itself is tested in ``tests/test_percent_normalize.py``.

Run: python tests/test_normalize_text.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_sync_pass_guard import load_sync_module  # noqa: E402

SCRIPT = "05_normalize_text.py"

def test_normalize_row() -> None:
    mod = load_sync_module(script=SCRIPT)
    tok = mod.ParakeetTokenizer(nfkc=False)
    row = {
        "audio_filepath": "a.flac",
        "text": "Laser CO₂ et séquence CUBE™,\nantigène Brª  espaçons 10 pour cent.",
        "asr_training_source": "Laser CO2,\nà 10 pour cent, puis 5%.",
    }
    out = mod.normalize_row(row, tok)
    assert out["text"] == "Laser CO2 et séquence CUBE, antigène Bra espaçons 10 %.", out["text"]
    assert out["asr_training_source"] == "Laser CO2, à 10 pourcent, puis 5 pourcent.", out["asr_training_source"]
    assert not tok.offending_chars(out["text"])
    assert mod.normalize_row(out, tok) == out  # idempotent
    assert row["text"].startswith("Laser CO₂")  # input not mutated
    print("test_normalize_row: OK")


if __name__ == "__main__":
    test_normalize_row()
