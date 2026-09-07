# Building the drug stage's inputs

`02_drugs/01_generate_drug_texts.py` reads three committed files. These are the scripts that produce them, vendored here from two standalone sibling projects so the drug stage is reproducible end to end rather than starting from opaque data.

Both subfolders are copies. They keep their original numbering and their original READMEs, and each still resolves its paths relative to its own directory, so run them from inside their own folder.

## The chain

```
                    BDPM / medicaments-api                OPEN_MEDIC 2025 (CNAM)   RETROCEDAM 2025 (CNAM)
                       database.json                        OPEN_MEDIC_2025.CSV      2017-2025 xlsx
                            |                                       |                       |
   base_de_donnee_medicament/                                       |                       |
     filter.py               -> filtered.json                       |                       |
     extract_unique_values.py -> uniqued_filtered.json              |                       |
     create_drug_db.py       -> drug_db.jsonl  ------------------.  |                       |
                                                                 |  |                       |
   drugs_frequency/                                              |  |                       |
     01_enrich_open_medic.py    joins OPEN_MEDIC to BDPM on CIP13 <--'-----------------------'
                                -> 01_OPEN_MEDIC_2025_enriched.csv
     02_combine_datasets.py     -> 02_drugs_by_substance_2025.jsonl
                                   02_drugs_by_brand_2025.jsonl
     03_score_terms.py          -> 03_drug_terms_2025.jsonl
     04_combine_freq_dosage.py  -> 04_drug_freq_dosage.jsonl
```

## What lands where

| Produced here | Committed in `02_drugs/` as | Role |
|---|---|---|
| `base_de_donnee_medicament/outputs/drug_db.jsonl` | `drugs_dosages.jsonl` | 410 substances with their real pharmaceutical forms and dosages |
| `drugs_frequency/03_drug_terms_2025.jsonl` | `drugs_frequency_2025.jsonl` | every brand and substance with a 0-10 importance score |
| `drugs_frequency/04_drug_freq_dosage.jsonl` | `drugs_freq_dosages.jsonl` | the two joined: the actual stage input |

The committed copies were taken from a 2025-12-31 snapshot of the sources, so re-running these scripts today gives slightly different counts.

## Why the score comes from sales data, not an LLM

This is the one place in the pipeline where an LLM is not the ranking signal, and it is the better design wherever you can manage it. `03_score_terms.py` scores each term by its real 2025 dispensing volume, log-normalized into 0-10, so `DOLIPRANE` earns 11 spoken variants and a drug nobody prescribes earns 2. The dictionary stage has to fall back on an LLM judging usefulness precisely because no equivalent usage data exists for a word like `blépharospasme`.

That script also drops brands that merely restate their active substance (`PARACETAMOL TEVA` adds nothing over `PARACETAMOL`) while keeping real brand names.

## Getting the raw inputs

None of the raw source data is committed here. It is all French public open data under the Etalab Open Licence 2.0:

- **BDPM** via the [medicaments-api](https://github.com/Giygas/medicaments-api) project: `curl https://medicaments-api.giygas.dev/database | tee database.json`. `sample.json` in `base_de_donnee_medicament/` shows one record's shape.
- **OPEN_MEDIC 2025**: <https://www.assurance-maladie.ameli.fr/etudes-et-donnees/open-medic-base-complete-depenses-medicaments>
- **RETROCEDAM 2025**: <https://www.assurance-maladie.ameli.fr/etudes-et-donnees/medicaments-retrocession-hospitaliere-retrocedam>

Credit: République française, produced by the ANSM (BDPM) and the Caisse nationale de l'Assurance Maladie (OPEN_MEDIC, RETROCEDAM).

## The dosage padding fix

BDPM pads decimal parts inconsistently, so one strength arrived spelled several ways for a single form: `500 mg`, `500,0 mg` and `500,00 mg`. Fed to the LLM as a presentation hint, that reads as three distinct doses of the same drug.

`create_drug_db.py` now fixes it, in `canonical_dosage` and `dedupe_dosages`. Two things were needed, not one. De-duplication handles the 65 presentation lists that carried a padded value *alongside* its clean twin. Canonicalization handles the other case, which plain de-duplication would have missed entirely: 88 presentations carried a padded value with **no** twin to fold into, so `BENZYLTHIOURACILE` shipped `25,00 mg` and nothing else, and would have gone on being read aloud as "vingt-cinq virgule zero zero milligrammes".

Only trailing zeros of the fraction and runs of whitespace are touched. The integer part keeps its grouping (`1 000 mg` stays `1 000 mg`), the decimal separator keeps whichever character was used, and the unit is untouched. Units are compared case-folded and whitespace-collapsed but **not** stripped of punctuation, so `M UI` and `M.U.I.` stay distinct rather than being guessed to be the same thing.

Measured on the real 410-substance input: 1075 dosage entries become 952, 113 substances change, and zero padded values remain. Verified that no substance, no presentation heading and no distinct (form, dose value, unit) triple was lost or gained. Covered by `tests/test_drug_dosage_dedup.py` (`uv run tests/test_drug_dosage_dedup.py`).

**The committed jsonl files are deliberately not regenerated.** They are the exact inputs that built UltiMed-v1, so re-running them through the fixed script would desync the repo from the published dataset. The fix applies to the next build.

## What was left behind

`base_de_donnee_medicament/drug_db_to_text.py` is not copied here. It was the original one-off LLM sentence generator for drugs and it has been superseded by `02_drugs/01_generate_drug_texts.py`, which does the same job through the shared engine in `utils/text_generation_engine.py`. Copying it in would have put two generators in the repo with no way to tell which one built the dataset.
