"""Stdlib test for utils/label_conventions.py and its use by the text generators.

The cases are real UltiMed v1 label fragments: every rewrite must reach the corpus
majority form, leave the look-alikes alone (``PCR M. tuberculosis``, ``le monsieur du
lit 4``, ``les 4H et 4T``, ``mille neuf cent soixante-quatorze grammes``, ``deux
comprimés``, ``deux trois jours``), and be idempotent. ``test_generator_parse`` checks that ``parse_asr_training_target`` (every
generator's parse step) applies them; it needs the LLM stack and is skipped when that
is not installed.

Run: python tests/test_label_conventions.py

This file was written by Claude Code.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from label_conventions import (  # noqa: E402
    DrugCaser, apply_label_conventions, default_drug_caser, expand_titles, normalize_clock,
    normalize_compounds, normalize_dates, normalize_quantities, normalize_spelling, normalize_staging, normalize_sutures,
    parse_french_number,
)


def check(fn, cases) -> None:
    for raw, expected in cases:
        got = fn(raw)
        assert got == expected, (raw, got, expected)
        assert fn(got) == got, ("not idempotent", got, fn(got))


def test_titles() -> None:
    check(expand_titles, [
        ("Compte rendu de M. Lefèvre, 45 ans.", "Compte rendu de Monsieur Lefèvre, 45 ans."),
        ("Ordonnance pour M. et Mme Petit.", "Ordonnance pour Monsieur et Madame Petit."),
        ("La patiente, Mme Dubois, 24 ans, et Mlle Girard.",
         "La patiente, Madame Dubois, 24 ans, et Mademoiselle Girard."),
        ("Transmission de soins pour M. Le Corre, 34 ans.",
         "Transmission de soins pour Monsieur Le Corre, 34 ans."),
        ("Adressée par le Dr Moreau. Pr Girard a validé.",
         "Adressée par le docteur Moreau. Professeur Girard a validé."),
        ("Dr Martin a vu la patiente.", "Docteur Martin a vu la patiente."),
        # Casing of the spelled-out titles before a name, mid-sentence.
        ("Le dossier de monsieur Girard et de madame Colin.",
         "Le dossier de Monsieur Girard et de Madame Colin."),
        ("Suivi pour le Docteur Lefèvre, médecin traitant : Professeur Dubois.",
         "Suivi pour le docteur Lefèvre, médecin traitant : professeur Dubois."),
        ("Docteur Lefèvre est informé.", "Docteur Lefèvre est informé."),
    ])
    for untouched in [
        "La PCR M. tuberculosis est négative.",
        "Madame M. présente une odynophagie.",
        "Plaque de taille M. La plaque n'est pas déplacée.",
        "Cellules en phase M. On propose une chimiothérapie.",
        "Le monsieur du lit 4 est sorti.",
        "Stade T2 N0 M0, madame, vous allez bien.",
        "Prescription pour M. le docteur Martin.",
    ]:
        assert expand_titles(untouched) == untouched, (untouched, expand_titles(untouched))
    print("test_titles: OK")


def test_dates() -> None:
    check(normalize_dates, [
        ("Opérée le quinze mars deux mille vingt.", "Opérée le 15 mars 2020."),
        ("Née le premier janvier mille neuf cent quatre-vingt-deux.", "Née le 1er janvier 1982."),
        ("Le trente et un octobre deux-mille-vingt-et-un, puis",
         "Le 31 octobre 2021, puis"),
        ("Né le 3 mai mil neuf cent soixante et onze.", "Né le 3 mai 1971."),
        ("Le 3 mai dix-neuf cent quatre-vingts.", "Le 3 mai 1980."),
        ("Hospitalisé du six au treize mai deux mille vingt-cinq.",
         "Hospitalisé du 6 au 13 mai 2025."),
        ("Du premier au trois juin.", "Du 1er au 3 juin."),
        ("Fracture en deux mille onze. Opéré depuis deux mille dix-neuf,",
         "Fracture en 2011. Opéré depuis 2019,"),
        ("Suivi en deux mille dix et deux mille onze.", "Suivi en 2010 et deux mille onze."),
        ("Bilan du premier mai 2025.", "Bilan du 1er mai 2025."),
    ])
    for untouched in [
        "Une dose de mille neuf cent soixante-quatorze grammes.",
        "Il vit depuis deux mille ans dans la légende.",
        "Tiré en deux mille exemplaires.",
        "Il reviendra dans deux mois.",
        "Un mars pluvieux.",
    ]:
        assert normalize_dates(untouched) == untouched, (untouched, normalize_dates(untouched))
    print("test_dates: OK")


def test_clock() -> None:
    check(normalize_clock, [
        ("Arrivée aux urgences à 14h30 pour brûlure.", "Arrivée aux urgences à 14 heures 30 pour brûlure."),
        ("Injecté à 10 h 15, la douleur cède.", "Injecté à 10 heures 15, la douleur cède."),
        ("Plan rouge à 09h45, décès à 02h15.", "Plan rouge à 9 heures 45, décès à 2 heures 15."),
        ("Reprendre à 8 h 00 avec le reste, alimentation à 17h00.",
         "Reprendre à 8 heures avec le reste, alimentation à 17 heures."),
        ("Antalgiques à 20h, constantes à 14 h : stables.",
         "Antalgiques à 20 heures, constantes à 14 heures : stables."),
        ("Pause de 1h puis 14H30.", "Pause de 1 heure puis 14 heures 30."),
    ])
    for untouched in [
        "Selon l'algorithme des 4H et 4T.",
        "Blocs 1a à 1h, 2, 3a.",
        "Une perfusion de 0,5 h par jour.",
        "Glycémie à J6h de la greffe.",
    ]:
        assert normalize_clock(untouched) == untouched, (untouched, normalize_clock(untouched))
    print("test_clock: OK")


def test_parse_french_number() -> None:
    for words, want in [
        ("deux cent cinquante", 250), ("quatre vingt dix sept", 97), ("soixante et onze", 71),
        ("trois mille cinq cents", 3500), ("cent quatre vingts", 180), ("vingt et une", 21),
        ("dix sept", 17), ("mille", 1000), ("zéro", 0),
        # Juxtaposed numbers are not one number: "deux trois jours" is 2 or 3 days.
        ("deux trois", None), ("vingt trente", None), ("cent cent", None),
        ("et deux", None), ("deux et", None), ("comprimé", None),
    ]:
        assert parse_french_number(words.split()) == want, (words, parse_french_number(words.split()))
    print("test_parse_french_number: OK")


def test_quantities() -> None:
    # Real drug_sentence references: the dose is a measure (digits), the count of
    # tablets is not (words), like the UltiMed majority.
    check(normalize_quantities, [
        ("On réduit la dose de silodosine à une gélule de quatre milligrammes par jour.",
         "On réduit la dose de silodosine à une gélule de 4 milligrammes par jour."),
        ("deux comprimés de deux cent cinquante microgrammes par jour",
         "deux comprimés de 250 microgrammes par jour"),
        ("une capsule de zéro virgule vingt-cinq microgrammes chaque matin",
         "une capsule de 0,25 microgrammes chaque matin"),
        ("un flacon de deux cent cinquante-sept virgule cinquante milligrammes",
         "un flacon de 257,50 milligrammes"),
        ("zéro virgule zéro cinq milligramme", "0,05 milligramme"),
        ("une goutte toutes les deux heures pendant dix-huit mois",
         "une goutte toutes les 2 heures pendant 18 mois"),
        ("Soixante-quinze milligrammes et quatre-vingt-dix-sept ans, trois mille cinq cents unités.",
         "75 milligrammes et 97 ans, 3500 unités."),
        ("vingt et une heures", "21 heures"),
        # A ratio converts both halves, never "5 jours sur sept".
        ("cinq jours sur sept, vingt-quatre heures sur vingt-quatre, trois semaines sur quatre",
         "5 jours sur 7, 24 heures sur 24, 3 semaines sur 4"),
        # Counts of times / sessions, range heads and scores: digits (author's choice).
        ("deux comprimés trois fois par jour, dix séances", "deux comprimés 3 fois par jour, 10 séances"),
        ("pendant deux à trois jours, un à 2 fois, et deux à 4 semaines",
         "pendant 2 à 3 jours, 1 à 2 fois, et 2 à 4 semaines"),
        ("douleur à deux sur dix, EVA zéro sur dix", "douleur à 2 sur 10, EVA 0 sur 10"),
        # A percentage before a count is not one number.
        ("0,10 gramme pour cent une fois par jour", "0,10 gramme pour cent une fois par jour"),
    ])
    for untouched in [
        "deux comprimés",
        "une fois par jour",
        "un milligramme, une heure, un an",
        "pendant deux trois jours",
        "Une dose de 4 milligrammes.",
    ]:
        assert normalize_quantities(untouched) == untouched, (untouched, normalize_quantities(untouched))
    print("test_quantities: OK")


def test_compounds() -> None:
    check(normalize_compounds, [
        ("À prendre au petit déjeuner.", "À prendre au petit-déjeuner."),
        ("Les petits déjeuners. Petit déjeuner léger.", "Les petits-déjeuners. Petit-déjeuner léger."),
        ("Voies biliaires extra hépatiques, rein multi lithiasique, Intra abdominal.",
         "Voies biliaires extra-hépatiques, rein multilithiasique, Intra-abdominal."),
        ("Lésions intra et extra-hépatiques, supra ou infra centimétriques.",
         "Lésions intra et extra-hépatiques, supra ou infracentimétriques."),
    ])
    assert normalize_compounds("un petit déjeune") == "un petit déjeune"
    print("test_compounds: OK")


def test_sutures_and_decimals() -> None:
    check(normalize_sutures, [
        ("Fermeture au Vicryl trois zéro.", "Fermeture au Vicryl 3-0."),
        ("Surjet au Monocryl 4 zéros, PDS rapide 3/0.", "Surjet au Monocryl 4-0, PDS rapide 3-0."),
        ("Points à l'Ethilon neuf zéro.", "Points à l'Ethilon 9-0."),
    ])
    # Not a gauge: no suture material before it, or a plain "Vicryl zéro".
    for kept in ("schéma zéro un zéro", "surjet de Vicryl zéro", "Monocryl 4-0."):
        assert normalize_sutures(kept) == kept, kept
    check(normalize_quantities, [
        ("un transducteur de 3 virgule 5 mégahertz", "un transducteur de 3,5 mégahertz"),
        ("cent vingt milligrays centimètres", "120 milligrays centimètres"),
        ("une IRM trois teslas", "une IRM 3 teslas"),
    ])
    # The TTS source keeps what voxtral said.
    assert apply_label_conventions("Vicryl trois zéro", tts_source=True) == "Vicryl trois zéro"
    print("test_sutures_and_decimals: OK")


def test_spelling() -> None:
    check(normalize_spelling, [
        ("Le cœur, l'Œdème, une manœuvre.", "Le coeur, l'Oedème, une manoeuvre."),
        ("Pancréatite aigüe, lésions subaigües, ambigüité.", "Pancréatite aiguë, lésions subaiguës, ambiguïté."),
        ("Compte-rendu opératoire, les comptes-rendus.", "Compte rendu opératoire, les comptes rendus."),
        ("Bêta-bloquants, bêtabloquant, bêta bloqueur.", "Bétabloquants, bétabloquant, bétabloqueur."),
        ("Anévrysme, urèthre, sténose uréthrale, périuréthral.", "Anévrisme, urètre, sténose urétrale, périurétral."),
        ("A surveiller. A jeun. A l'examen, A renouveler.", "À surveiller. À jeun. À l'examen, A renouveler."),
        # The verb avoir, mid-sentence, ambiguous: unchanged.
        ("A présenté une fièvre. A bien toléré. A encore mal. Il A voir.",
         "A présenté une fièvre. A bien toléré. A encore mal. Il A voir."),
    ])
    assert normalize_spelling("Docteur Argüelles") == "Docteur Argüelles"
    print("test_spelling: OK")


def test_drug_caser() -> None:
    caser = DrugCaser(
        caps={"PRIMPERAN": "Primpéran", "PARACETAMOL": "paracétamol", "UVEDOSE": "Uvedose",
              "FORTE": "forte"},
        variants={"primperan": "Primpéran", "skénan": "Skenan", "paracetamol": "paracétamol"},
    )
    check(caser, [
        ("PRIMPERAN et Primperan puis primperan.", "Primpéran et Primpéran puis Primpéran."),
        ("Prendre PARACETAMOL 1 g, puis paracetamol. PARACETAMOL encore.",
         "Prendre paracétamol 1 g, puis paracétamol. Paracétamol encore."),
        ("UVEDOSE une ampoule, l'UVEDOSE FORTE aussi.", "Uvedose une ampoule, l'Uvedose forte aussi."),
        ("Du Skénan.", "Du Skenan."),
    ])
    # Canonical forms of the committed lexicon, and a protected acronym.
    default = default_drug_caser()
    assert default("PARACETAMOL LP et KARDEGIC.") == "Paracétamol LP et Kardégic.", default("PARACETAMOL LP et KARDEGIC.")
    assert default("un comprimé SANS SUCRE") == "un comprimé SANS sucre", default("un comprimé SANS SUCRE")
    assert default("paracétamol codéiné") == "paracétamol codéiné"
    assert default("une note de menthe") == "une note de menthe"
    print("test_drug_caser: OK")


def test_staging() -> None:
    check(normalize_staging, [
        ("Cancer de stade IIIb, grade I à II, type I et II.", "Cancer de stade 3b, grade 1 à 2, type 1 et 2."),
        ("Antalgique de palier deux, classe III NYHA, NYHA II.", "Antalgique de palier 2, classe 3 NYHA, NYHA 2."),
        ("Diabète de type 2, stades IVB.", "Diabète de type 2, stades 4B."),
    ])
    # Roman numerals that are names, letters that happen to be Roman, "un peu".
    for kept in ("angiotensine II", "APACHE II", "métaphase II", "type C", "type XX", "un type un peu particulier"):
        assert normalize_staging(kept) == kept, kept
    print("test_staging: OK")


def test_all() -> None:
    raw = ("M. Dupont prend du DOLIPRANE à 14h30 depuis le quinze mars deux mille vingt, "
           "deux comprimés de cinq cents milligrammes au petit déjeuner.")
    want = ("Monsieur Dupont prend du Doliprane à 14 heures 30 depuis le 15 mars 2020, "
            "deux comprimés de 500 milligrammes au petit-déjeuner.")
    assert apply_label_conventions(raw) == want, apply_label_conventions(raw)
    assert apply_label_conventions(want) == want
    print("test_all: OK")


def test_generator_parse() -> None:
    try:
        from _pipeline_shared import parse_asr_training_target
    except ImportError as exc:
        print(f"test_generator_parse: SKIPPED ({exc})")
        return
    raw = "<t>Mme Petit prend du KARDEGIC.</t><t>Vu le premier mai à 9h.</t>"
    assert parse_asr_training_target(raw, expected=2) == [
        "Madame Petit prend du Kardégic.", "Vu le 1er mai à 9 heures."], parse_asr_training_target(raw, expected=2)
    print("test_generator_parse: OK")


if __name__ == "__main__":
    test_titles()
    test_dates()
    test_clock()
    test_parse_french_number()
    test_quantities()
    test_compounds()
    test_sutures_and_decimals()
    test_spelling()
    test_staging()
    test_drug_caser()
    test_all()
    test_generator_parse()
