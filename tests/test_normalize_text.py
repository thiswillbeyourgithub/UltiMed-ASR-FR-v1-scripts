"""Stdlib test for 99_hf_release/05_normalize_text.py.

Checks that a v1 manifest row gets both of its texts normalized: the label through
``ParakeetTokenizer.clean_label`` + ``%``, the TTS source through ``pourcent``. The percent
rewrite itself is tested in ``tests/test_percent_normalize.py``, the label conventions
(titles, dates, clock, drug casing, applied to both texts) in ``tests/test_label_conventions.py``.

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


def test_label_conventions_both_texts() -> None:
    mod = load_sync_module(script=SCRIPT)
    tok = mod.ParakeetTokenizer(nfkc=False)
    row = {"audio_filepath": "a.flac",
           "text": "M. Petit prend du KARDEGIC depuis le quinze mars à 14h30.",
           "asr_training_source": "M. Petit prend du KARDEGIC depuis le quinze mars à 14h30."}
    out = mod.normalize_row(row, tok)
    want = "Monsieur Petit prend du Kardégic depuis le 15 mars à 14 heures 30."
    assert out["text"] == want, out["text"]
    assert out["asr_training_source"] == want, out["asr_training_source"]
    assert mod.normalize_row(out, tok) == out
    print("test_label_conventions_both_texts: OK")


def test_cp1252_mojibake() -> None:
    # The oli_drug_sentence manifest stored "œil" as U+009C + "il" (the cp1252 byte of
    # "œ" read as Latin-1), so the reference could never match a correct transcript.
    mod = load_sync_module(script=SCRIPT)
    tok = mod.ParakeetTokenizer(nfkc=False)
    row = {"audio_filepath": "a.flac", "text": "Une goutte dans chaque \u009cil, c\u009cur \u0081."}
    out = mod.normalize_row(row, tok)
    # U+0081 is undefined in cp1252: left for the <unk> gate, never guessed. The repaired
    # "œ" then follows the label convention like any other, hence "oe".
    assert out["text"] == "Une goutte dans chaque oeil, coeur \u0081.", out["text"]
    assert mod.normalize_row(out, tok) == out
    print("test_cp1252_mojibake: OK")


if __name__ == "__main__":
    test_normalize_row()
    test_label_conventions_both_texts()
    test_cp1252_mojibake()
