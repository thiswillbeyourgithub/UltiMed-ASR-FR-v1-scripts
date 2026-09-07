# OPEN_MEDIC enrichment

Source CSV: <https://www.assurance-maladie.ameli.fr/etudes-et-donnees/open-medic-base-complete-depenses-medicaments>

`01_enrich_open_medic.py` joins each row of `OPEN_MEDIC_2025.CSV` to the
[medicaments-api](https://github.com/Giygas/medicaments-api) (BDPM) on the
`CIP13` code, adds human-readable columns, and writes
`01_OPEN_MEDIC_2025_enriched.csv` sorted by `BOITES` descending (most-prescribed
presentations first).

Generated with the help of Claude Code.

## Run

```bash
python 01_enrich_open_medic.py
```

The full medicaments export is fetched once and cached locally as
`medicaments_export.json`.

## Columns

### Original OPEN_MEDIC 2025 (CNAM)

| Column | Meaning |
| --- | --- |
| `ATC1` / `l_ATC1` | ATC level 1 — anatomical main group (code + label) |
| `ATC2` / `L_ATC2` | ATC level 2 — therapeutic subgroup |
| `ATC3` / `L_ATC3` | ATC level 3 — pharmacological subgroup |
| `ATC4` / `L_ATC4` | ATC level 4 — chemical subgroup |
| `ATC5` / `L_ATC5` | ATC level 5 — chemical substance |
| `CIP13` / `l_cip13` | 13-digit presentation code + label of the drug box |
| `TOP_GEN` | Generic flag (0 = not in a generic group, 1 = reference, 2 = generic, …) |
| `GEN_NUM` | Generic group number |
| `AGE` | Age band of the patient (0, 20, 60, 100, …) |
| `sexe` | 1 = male, 2 = female, 9 = unknown |
| `BEN_REG` | Beneficiary's region code |
| `PSP_SPE` | Speciality of the prescriber |
| `BOITES` | Number of boxes dispensed (volume metric) |
| `REM` | Amount reimbursed by Assurance Maladie (€) |
| `BSE` | Reimbursement base (€, what reimbursement is computed from) |

### Added from medicaments-api (BDPM)

| Column | Meaning |
| --- | --- |
| `drug_name` | Full drug name (e.g. `DOLIPRANE 1000 mg, comprimé`) |
| `forme` | Pharmaceutical form (comprimé, gélule, solution…) |
| `voies` | Route(s) of administration, `\|`-separated |
| `substances` | Active ingredients + dosage, `\|`-separated (`natureComposant = SA`) |
| `titulaire` | Marketing authorisation holder (lab) |
| `presentation_libelle` | Packaging description (e.g. "plaquette de 16 comprimés") |
| `prix_eur` | Public price in € |
| `taux_remboursement` | Reimbursement rate (15%, 30%, 65%, 100%) |
| `etat_commercialisation` | Commercialisation status (Commercialisée, Arrêt…) |

`BOITES × prix_eur` gives an approximate spend per row; aggregating `BOITES`
by `drug_name` answers "how much of each drug is prescribed in France".

## Hospital retrocession (RETROCEDAM)

Source: <https://www.assurance-maladie.ameli.fr/etudes-et-donnees/medicaments-retrocession-hospitaliere-retrocedam>

The single xlsx (`2017-a-2025_retroced-am_serie-annuelle..xlsx`) covers
hospital-retroceded drugs from 2017 to 2025. We only use the 2025 column
(`Unités 2025`). Volume is counted in **UCD** (Unité Commune de Dispensation,
the single dispensation unit), **not in boxes** — so it is *not* directly
comparable to `BOITES` from OPEN_MEDIC. Sheet 2 of the workbook has the data;
the join key on the hospital side is `cod_ucd`.

## Combine OPEN_MEDIC + RETROCEDAM

```bash
python 02_combine_datasets.py
```

`02_combine_datasets.py` reads `01_OPEN_MEDIC_2025_enriched.csv` (must be generated
first via `01_enrich_open_medic.py`) and downloads the RETROCEDAM xlsx (cached as
`retrocedam_2017_2025.xlsx`). It produces two JSONL files for 2025, sorted by
combined volume descending:

- `02_drugs_by_substance_2025.jsonl` — one entry per ATC5 chemical substance
  (e.g. `N02BE01` = paracetamol). Top entry on the current data is
  `PARACETAMOL` with ~419M ambulatory boxes.
- `02_drugs_by_brand_2025.jsonl` — one entry per commercial brand. The brand
  comes from the RETROCEDAM `Produit` column on the hospital side and is
  extracted from BDPM `drug_name` on the OPEN_MEDIC side (leading run of
  uppercase letters/hyphens/spaces, stopping at the first digit or comma —
  e.g. `DOLIPRANE 1000 mg, comprimé` → `DOLIPRANE`, `PARACETAMOL TEVA 500 mg`
  → `PARACETAMOL TEVA`). Top entry is `DOLIPRANE`.

Each JSONL row has these fields:

| Field | Meaning |
| --- | --- |
| `atc5` / `atc5_label` *(substance only)* | ATC5 code + label |
| `brand` *(brand only)* | Uppercased commercial name |
| `ambulatory_boxes` | Sum of `BOITES` from OPEN_MEDIC 2025 |
| `hospital_units` | Sum of `Unités 2025` from RETROCEDAM (UCD count) |
| `total` | `ambulatory_boxes + hospital_units` |
| `hospital_only` | `true` when the entry exists only in RETROCEDAM (no ambulatory boxes) |

⚠ `total` mixes two different units (boxes vs UCD units) and is intended as a
rough ranking signal, not an exact count. For the TTS sampling use case this
is acceptable: hospital volumes are tiny next to ambulatory volumes for the
top entries, and `hospital_only` flags the niche substances/brands that exist
only in the hospital data.

## Score terms

```bash
python 03_score_terms.py
```

`03_score_terms.py` flattens both JSONL files into a single scored vocabulary
`03_drug_terms_2025.jsonl` for TTS/ASR sample weighting. Each row has exactly
three keys:

| Field | Meaning |
| --- | --- |
| `term` | The brand name or substance label |
| `type` | `brand` or `substance` (which source it came from) |
| `category` | Always `drugs` (constant, for merging with other term lists) |
| `score` | `int` in [0, 10]: `log(total)` min-max normalized in log space across all terms, so the single most-sold term is 10 and the least-sold is 0 |

Brands that merely restate their active substance are dropped (e.g.
`FLUOXETINE MILAN` or `PARACETAMOL TEVA` add nothing over the substance
`FLUOXETINE` / `PARACETAMOL`): a brand is removed when any of its name words
(3+ chars, ignoring French connectors) matches a word from its `substances`.
Real brand names like `DOLIPRANE` are kept. The script also prints a terminal
histogram of how many terms fall in each score bin. Generated with help from
Claude Code.
