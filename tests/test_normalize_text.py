"""Stdlib test for 99_hf_release/05_normalize_text.py.

The percent cases are real UltiMed v1 sentences (shortened): the rewrite must turn
every percentage into ``%`` in the label and ``pourcent`` in the TTS source, and
leave "pour cent" alone where it means "per hundred X".

Run: python tests/test_normalize_text.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_sync_pass_guard import load_sync_module  # noqa: E402

SCRIPT = "05_normalize_text.py"

# (input, expected label, expected TTS source)
PERCENT_CASES = [
    ("La saturation est à 95 pour cent en air ambiant.",
     "La saturation est à 95 % en air ambiant.",
     "La saturation est à 95 pourcent en air ambiant."),
    ("Ropivacaïne 0,5 pourcent 30 millilitres.",
     "Ropivacaïne 0,5 % 30 millilitres.",
     "Ropivacaïne 0,5 pourcent 30 millilitres."),
    ("Chlorure de sodium 0,9 pour cent 250 millilitres.",
     "Chlorure de sodium 0,9 % 250 millilitres.",
     "Chlorure de sodium 0,9 pourcent 250 millilitres."),
    ("Ki67 à quatre-vingt-dix-sept pour cent.",
     "Ki67 à quatre-vingt-dix-sept %.",
     "Ki67 à quatre-vingt-dix-sept pourcent."),
    ("Soixante pour cent du matériel, crème à 5 pour cent cinq fois par semaine.",
     "Soixante % du matériel, crème à 5 % cinq fois par semaine.",
     "Soixante pourcent du matériel, crème à 5 pourcent cinq fois par semaine."),
    # A spelled-out dose after the percentage is not what "cent" counts.
    ("Glucosé à cinq pour cent un litre, lidocaïne zéro virgule un pour cent vingt centimètres cubes.",
     "Glucosé à cinq % un litre, lidocaïne zéro virgule un % vingt centimètres cubes.",
     "Glucosé à cinq pourcent un litre, lidocaïne zéro virgule un pourcent vingt centimètres cubes."),
    ("Fungizone 10 pour cent quarante millilitres.",
     "Fungizone 10 % quarante millilitres.",
     "Fungizone 10 pourcent quarante millilitres."),
    ("Hémoglobine glyquée à sept et demi pour cent.",
     "Hémoglobine glyquée à sept et demi %.",
     "Hémoglobine glyquée à sept et demi pourcent."),
]

# "pour cent" meaning "per hundred X": untouched on both sides.
NOT_PERCENT = [
    "Il y a cent six garçons pour cent filles.",
    "Acide ascorbique 2 grammes pour cent grammes de préparation.",
    "Elle mesure un mètre soixante-dix pour cent vingt kilogrammes.",
    "Trois cas pour cent mille habitants.",
    "Cyanocobalamine collyre 0,05 grammes pour cent millilitres.",
    "La MAC du desflurane est de 6 volumes pour cent chez l'adulte.",
    "Il est parti pour centraliser les dossiers.",
]


def test_percent() -> None:
    mod = load_sync_module(script=SCRIPT)
    for raw, label, source in PERCENT_CASES:
        assert mod.percent_to_symbol(raw) == label, (raw, mod.percent_to_symbol(raw))
        assert mod.percent_to_one_word(raw) == source, (raw, mod.percent_to_one_word(raw))
        # Idempotent.
        assert mod.percent_to_symbol(label) == label
        assert mod.percent_to_one_word(source) == source
    for raw in NOT_PERCENT:
        assert mod.percent_to_symbol(raw) == raw, (raw, mod.percent_to_symbol(raw))
        assert mod.percent_to_one_word(raw) == raw, raw
    print("test_percent: OK")


def test_normalize_row() -> None:
    mod = load_sync_module(script=SCRIPT)
    tok = mod.ParakeetTokenizer(nfkc=False)
    row = {
        "audio_filepath": "a.flac",
        "text": "Laser CO₂ et séquence CUBE™,\nantigène Brª  espaçons 10 pour cent.",
        "asr_training_source": "Laser CO2,\nà 10 pour cent.",
    }
    out = mod.normalize_row(row, tok)
    assert out["text"] == "Laser CO2 et séquence CUBE, antigène Bra espaçons 10 %.", out["text"]
    assert out["asr_training_source"] == "Laser CO2, à 10 pourcent.", out["asr_training_source"]
    assert not tok.offending_chars(out["text"])
    assert mod.normalize_row(out, tok) == out  # idempotent
    assert row["text"].startswith("Laser CO₂")  # input not mutated
    print("test_normalize_row: OK")


if __name__ == "__main__":
    test_percent()
    test_normalize_row()
