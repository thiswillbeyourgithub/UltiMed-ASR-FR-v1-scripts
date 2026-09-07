#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests>=2.31","jiwer>=3.0","tqdm>=4.66","loguru>=0.7","click>=8.1"]
# ///
"""Tests for 01_compute_stt.py's scoring normalization (normalize_for_scoring /
compute_metrics). Run: `uv run test_normalize_for_scoring.py` (needs the module's deps
to import it, hence the uv header). Written with Claude Code.

Locks the "score Whisper and a Parakeet-style label in one common character space"
behavior: fold oe/ae ligatures, strip accents, drop punctuation, lowercase, collapse
whitespace, so representational differences do not count as CER errors."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("compute_stt_mod", Path(__file__).with_name("01_compute_stt.py"))
_cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cs)
norm = _cs.normalize_for_scoring
cer = lambda ref, hyp: _cs.compute_metrics(ref, hyp)[1]

_failures = []


def eq(label, got, want):
    if got != want:
        _failures.append(f"{label}: got {got!r} want {want!r}")


def approx0(label, val):
    if not (val is not None and abs(val) < 1e-9):
        _failures.append(f"{label}: expected ~0.0, got {val!r}")


# --- normalize_for_scoring ---
eq("oe ligature", norm("cœur"), "coeur")
eq("oe ligature word", norm("œsophage"), "oesophage")
eq("ae ligature", norm("ex æquo"), "ex aequo")
eq("accents + case + punct", norm("Étalé, à l'Hôpital"), "etale a l hopital")
eq("diaeresis folded", norm("aiguë"), "aigue")
eq("cedilla folded", norm("ça"), "ca")
eq("non-french diacritics", norm("Sağiroğlu Ō"), "sagiroglu o")  # g-breve, s-cedilla, macron
eq("plus is spoken, not stripped", norm("CD4+"), "cd4 plus")
eq("registered mark stripped", norm("Doliprane®"), "doliprane")
eq("whitespace collapse", norm("a\t b\n\nc"), "a b c")
eq("none safe", norm(None), "")
eq("empty safe", norm(""), "")

# --- one written form per spoken thing ---
# The TTS spells a unit out and Whisper abbreviates it back; neither side is wrong, so
# both fold to the spoken form (measured as the single biggest source of fake CER).
eq("bare unit expands", norm("500 mg"), "500 milligrammes")
eq("compound unit expands", norm("4,2 mmol/L"), "4 2 millimoles par litres")
eq("spelled unit is the canonical", norm("500 milligrammes"), "500 milligrammes")
eq("singular meets plural", norm("0,8 centimetre"), "0 8 centimetres")
eq("mmHg", norm("130 mmHg"), "130 millimetres de mercure")
approx0("unit spelling is free", cer("Le patient prend 500 mg.", "Le patient prend 500 milligrammes."))
approx0("compound unit is free", cer("kaliémie à 4,2 mmol/L", "kaliémie à 4,2 millimoles par litre"))

# The tail of a rate the two sides split differently: only the head carries the number,
# so the rules above expand "15 mg" and leave "par kg" as the whole diff.
eq("rate tail expands", norm("15 mg par kg"), "15 milligrammes par kilogrammes")
eq("rate tail already spelled", norm("15 milligrammes par kilogramme"),
   "15 milligrammes par kilogrammes")
# Single-letter symbols are excluded on purpose: in French "par l" is nearly always the
# elided article, not litres, and the punctuation strip turns "l'aorte" into "l aorte".
eq("elision is not a litre", norm("perfusion par l'aorte"), "perfusion par l aorte")
eq("ordinary word after par", norm("par voie orale"), "par voie orale")
# "kilo" folds in the rate tail only. After a number it is a trap: the IgE assay unit
# spells out as "kilo-unites par litre", and Whisper mishears "culots globulaires" as
# "kilos globulaires" often enough to matter.
eq("kilo folds after par", norm("15 milligrammes par kilo"), "15 milligrammes par kilogrammes")
eq("kilo-unites is left alone", norm("87,5 kilo-unités par litre"),
   "87 5 kilo unites par litres")  # "litre" -> "litres" is the older inflection fold
eq("a miscounted kilo is left alone", norm("3 kilos globulaires"), "3 kilos globulaires")
# Bare "mm" is barred from the rate tail: it earns almost nothing and it collides with
# how Whisper writes the M-M-RVAXPRO vaccine.
eq("bare mm after par is left alone", norm("Vaccination par MM-VAX-PRO"),
   "vaccination par mm vax pro")
eq("the mm3 compound still folds", norm("650 par mm3"), "650 par millimetres cubes")
# "ma" is barred too: after the fold it is the possessive, not milliamperes.
eq("par ma is the possessive", norm("adressé par ma consoeur"), "adresse par ma consoeur")

# Imaging / physiology units: the label spells them, Whisper abbreviates them.
eq("ms expands", norm("un temps de 250 ms"), "1 temps de 250 millisecondes")
eq("kPa expands", norm("pression à 4,5 kPa"), "pression a 4 5 kilopascals")
eq("MHz expands", norm("sonde de 7,5 MHz"), "sonde de 7 5 megahertz")
eq("kV and mA expand", norm("80 kV et 200 mA"), "80 kilovolts et 200 milliamperes")

# A clock time: the compact form is Whisper's, the spelled one is the label's, and only
# one of them used to end in "minutes".
eq("compact time drops minutes", norm("le 29 juin à 14h30"), "le 29 juin a 14 heures 30")
eq("spelled time matches it", norm("le 29 juin à 14 heures 30 minutes"),
   "le 29 juin a 14 heures 30")
eq("abbreviated minutes too", norm("le 29 juin à 14 heures 30 min"),
   "le 29 juin a 14 heures 30")
eq("round hour is unchanged", norm("à 14h00"), "a 14 heures")

# Plural ordinals, which the singular-only rule left as a diff.
eq("plural ordinal words", norm("les quatrièmes et cinquièmes côtes"), "les 4e et 5e cotes")
eq("plural ordinal digits", norm("les 4e et 5e côtes"), "les 4e et 5e cotes")
approx0("mm3 spelling is free", cer("CD4 à 650 par millimètre cube", "CD4 à 650 par mm3"))

# Titles: Whisper abbreviates them too.
eq("title expands", norm("M. Dupont"), "monsieur dupont")
eq("mme expands", norm("Mme Dupont"), "madame dupont")
eq("dr expands", norm("Dr Martin"), "docteur martin")
approx0("title spelling is free", cer("Monsieur Dupont", "M. Dupont"))

# Percent: the punctuation strip used to DELETE the %, so the label's "pour cent"
# scored as a pure deletion. Both spellings now land on one word.
eq("percent symbol", norm("96 %"), "96 pourcent")
eq("percent spelled", norm("96 pour cent"), "96 pourcent")
approx0("percent spelling is free", cer("saturation à 96 %", "saturation à quatre-vingt-seize pour cent"))

# Numbers, both directions, including the composed French forms.
eq("number word to digit", norm("trois seances"), "3 seances")
eq("quatre-vingt is multiplicative", norm("quatre-vingt-seize"), "96")
eq("soixante-dix", norm("soixante-dix ans"), "70 ans")
eq("vingt et un", norm("vingt et un"), "21")
eq("scales compose", norm("mille neuf cent quarante-neuf"), "1949")
eq("cents multiplies", norm("deux cents milligrammes"), "200 milligrammes")
eq("staging roman", norm("stade IV"), "stade 4")
approx0("roman vs digit vs word is free", cer("stade IV, grade III", "stade quatre, grade 3"))
eq("runs stop at punctuation", norm("deux, trois patients"), "2 3 patients")

# --- the guards: what must NOT be folded ---
# Bare unit letters are also ordinary French, so unit rules need a number in front.
eq("elided article is not a litre", norm("l'artère"), "l artere")
eq("de is not a gram", norm("un peu de sel"), "1 peu de sel")
# Bare "m" is also a spelled-out initial: only a following word makes it a title.
eq("initial is not monsieur", norm("le gène L.M.B.R.1"), "le gene l m b r 1")
eq("elision is not monsieur", norm("il m'a dit"), "il m a dit")
eq("metres win over monsieur", norm("1,75 m de haut"), "1 75 metres de haut")
# A spaced comma between two numbers is an enumeration, not a decimal.
eq("enumeration is not a decimal", norm("deux, trois patients"), "2 3 patients")
# The tail of a phone/serial number must not be eaten as a thousands separator: the
# rule wants a digit, then ONE space, then exactly three digits.
eq("four digits are not thousands", norm("code 12 3456"), "code 12 3456")
# Dropping the decimal separator must not also drop the zero behind it: a leading zero
# is silent in a date ("le 05 janvier"), audible in a dose.
eq("date leading zero", norm("le 05 janvier"), "le 5 janvier")
if norm("3,05 mg") == norm("3,5 mg"):
    _failures.append("3,05 and 3,5 must stay distinct")
# A number hanging off a letter is a clinical code, not a measurement: the metatarsals
# M1M2 must not be read as square metres just because the code ends in "m2".
eq("anatomical code is not an area", norm("l'angle M1M2"), "l angle m1m2")
eq("vertebral code is not a unit", norm("arthrodese T1L2"), "arthrodese t1l2")
# French plural/gender endings are inaudible but are NOT folded: doing so would also
# hide a TTS that dropped a word, and they cost 1-2 characters.
if not (cer("les antalgiques", "les antalgique") or 0) > 0:
    _failures.append("plural difference should still count")

# --- the folds added from the measured cost ranking ---------------------------------
# Each line is a (label, transcript) pair that used to cost CER for writing the same
# spoken thing two ways. Ranked by what they cost over 60k scored clips.
for _label, _hyp, _why in [
    ("0,5 milligrammes par kilogramme", "0,5 mg/kg", "plural on the last word only"),
    ("175 unites internationales par kilogramme", "175 UI/kg", "UI/kg missing"),
    ("750 centimetres cubes", "750 cm3", "cubed unit"),
    ("2 millimetres carres", "2 mm2", "squared unit"),
    ("1,7 metres carres", "1,7 m2", "surface area"),
    ("15 gigabecquerels", "15 GBq", "nuclear medicine activity"),
    ("189 giga par litre", "189 G/L", "giga vs grammes, case is the only clue"),
    ("2,8 giga par litre", "2,8 gigas par litre", "plural of giga"),
    ("zero virgule cinq", "0,5", "zero was the one accented cardinal"),
    ("le premier mars", "le 1er mars", "ordinal word vs digit"),
    ("la troisieme cure", "la 3eme cure", "ordinal suffixes"),
    ("le statut HER2 est a 1 plus", "le statut HER2 est a 1+", "+ is spoken"),
    ("du Vicryl 3 barre 0", "du Vicryl 3/0", "slash read as barre"),
    ("du Vicryl 3 0", "du Vicryl 3,0", "the suture gauge Whisper writes as a decimal"),
    ("tension 120 sur 80", "tension 120/80", "slash read as sur"),
    ("a J plus 2", "a J2", "day offset"),
    ("2 heures 5 minutes", "2h05", "compact time"),
    ("11500 leucocytes", "11 500 leucocytes", "thousands separator"),
    ("le 5 janvier", "le 05 janvier", "leading zero"),
    # Rate tails: the number sits on the head, so only the head folds by itself.
    ("15 milligrammes par kilogramme", "15 milligrammes par kg", "unit after par"),
    ("3 grammes par millilitre", "3 grammes par ml", "same, per volume"),
    ("2 litres par minute", "2 litres par min", "same, per time"),
    ("50 millijoules par centimetre carre", "50 mJ par cm2", "laser fluence"),
]:
    approx0(f"free: {_why} ({_label!r} vs {_hyp!r})", cer(_label, _hyp))

# --- letter-spelled acronyms (07_acronyms support) ---
# The label writes the sigle joined, Whisper may write the spelling it heard; a chain
# of >= 3 single letters joined by "-" or "." folds to the joined form on both sides.
eq("letter-dash chain folds", norm("E-S-A-T"), "esat")
eq("letter-dot chain folds", norm("E.S.A.T."), "esat")
eq("elided chain folds", norm("l'E-S-A-T"), "l esat")
approx0("spelled sigle is free", cer("La GGT est élevée.", "la G-G-T est elevee"))
approx0("dotted sigle is free", cer("le bilan ORL", "le bilan O.R.L."))
# The guards: what must NOT fold.
eq("euphonic t survives", norm("y a-t-il des anomalies"), "y a t il des anomalies")
eq("inverted verb survives", norm("que dira-t-elle"), "que dira t elle")
eq("two letters never fold", norm("vitamine B. A demain"), "vitamine b a demain")
eq("digit chains stay spelled", norm("la cure de 5-F-U"), "la cure de 5 f u")
eq("gene dot digit still spaced", norm("le gène L.M.B.R.1"), "le gene l m b r 1")
eq("hyphen compound untouched", norm("cet infarctus ST-plus"), "cet infarctus st plus")
eq("word chunks untouched", norm("l'e-sath du secteur"), "l e sath du secteur")

# --- compute_metrics: representational diffs must be free ---
approx0("accent-only diff is free", cer("hépatique", "hepatique"))
approx0("ligature diff is free", cer("cœur", "coeur"))
approx0("case/punct diff is free", cer("À l'hôpital.", "a l hopital"))
# A real word error must still be penalized (sanity: normalization is not too lenient):
if not (cer("hepatique", "renale") or 0) > 0.2:
    _failures.append("real word error should still score high CER")
# Empty reference -> None (nothing to score), unchanged contract:
eq("empty ref -> None cer", cer("", "anything"), None)

if _failures:
    print("FAILED:")
    for f in _failures:
        print("  -", f)
    raise SystemExit(1)
print("ALL PASSED")
