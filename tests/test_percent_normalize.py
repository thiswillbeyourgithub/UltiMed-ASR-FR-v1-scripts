"""Stdlib test for utils/percent_normalize.py and its use by the text generators.

The cases are real UltiMed v1 sentences (shortened): the rewrite must turn every
percentage into ``%`` in the label and ``pourcent`` in the TTS source, and leave
"pour cent" alone where it means "per hundred X". ``test_generator_parse`` checks that
``parse_asr_training_target`` (every generator's parse step) applies the label form,
so a fresh generation writes ``%`` whatever spelling the LLM chose; it needs the LLM
stack and is skipped when that is not installed.

Run: uv run --with litellm --with rapidfuzz --with tiktoken --with tenacity --with loguru \
         --with click tests/test_percent_normalize.py
(or) python tests/test_percent_normalize.py   # generator part skipped
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from percent_normalize import (  # noqa: E402
    LEFTOVER_RE, percent_sign_to_one_word, percent_to_one_word, percent_to_symbol,
)

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
    for raw, label, source in PERCENT_CASES:
        assert percent_to_symbol(raw) == label, (raw, percent_to_symbol(raw))
        assert percent_to_one_word(raw) == source, (raw, percent_to_one_word(raw))
        # Idempotent.
        assert percent_to_symbol(label) == label
        assert percent_to_one_word(source) == source
    for raw in NOT_PERCENT:
        assert percent_to_symbol(raw) == raw, (raw, percent_to_symbol(raw))
        assert percent_to_one_word(raw) == raw, raw
    assert LEFTOVER_RE.search(NOT_PERCENT[0]) and not LEFTOVER_RE.search(PERCENT_CASES[0][1])
    print("test_percent: OK")


def test_percent_sign() -> None:
    # The fresh-source rule voxtral_normalize applies to the canonical label.
    for label, source in [("à 95 %.", "à 95 pourcent."), ("Ki67 à 30%,", "Ki67 à 30 pourcent,"),
                          ("à 95\u202f%", "à 95 pourcent")]:
        assert percent_sign_to_one_word(label) == source, (label, percent_sign_to_one_word(label))
        assert percent_sign_to_one_word(source) == source
    # Every label form of PERCENT_CASES reaches the same source as the v1 rewrite.
    for _, label, source in PERCENT_CASES:
        assert percent_sign_to_one_word(label) == source, label
    print("test_percent_sign: OK")


def test_generator_parse() -> None:
    try:
        from _pipeline_shared import parse_asr_training_target
    except ImportError as exc:
        print(f"test_generator_parse: SKIPPED ({exc})")
        return
    raw = "".join(f"<t>{r}</t>" for r, _, _ in PERCENT_CASES)
    raw += "".join(f"<t>{r}</t>" for r in NOT_PERCENT)
    parsed = parse_asr_training_target(raw, expected=len(PERCENT_CASES) + len(NOT_PERCENT))
    assert parsed == [lab for _, lab, _ in PERCENT_CASES] + NOT_PERCENT, parsed
    print("test_generator_parse: OK")


if __name__ == "__main__":
    test_percent()
    test_percent_sign()
    test_generator_parse()
