# NOTICE

This file records the attribution and licensing obligations for the third-party
sources this dataset is derived from. It ships alongside the dataset so the
credits travel with the data, as the upstream licenses require. It is separate
from the dataset's own `LICENSE`.

This NOTICE was assembled with the help of Claude Code.

## The derived dataset

This is a French medical Automatic Speech Recognition (ASR) dataset: short,
dictation-style French sentences rich in medical vocabulary, paired with audio
synthesized from them by a local text-to-speech model. It was built by
Olivier Cornelis.

- Derived dataset page: <https://huggingface.co/datasets/Olicorne/UltiMed-ASR-FR-v1>
- Derived dataset license: CC BY 4.0 for the main corpus (dictionary, drugs,
  PARHAF, acronyms subsets); the separate test-only PARROT subset is
  CC BY-NC-SA 4.0 (see its block below).

The text was produced by large-language-model rewriting/generation over the
sources below; the audio was produced by text-to-speech synthesis over that
text. In every case the underlying source material has been **modified**
(reformatted and synthesized to audio); see each block's "Changes" line.

## Attribution for upstream sources

### French public medicines data (drug names, forms, dosages, frequencies)

- Sources:
  - Base de donnees publique des medicaments (BdPM),
    <https://base-donnees-publique.medicaments.gouv.fr/>, accessed via the
    data.gouv API <https://www.data.gouv.fr/dataservices/api-medicaments-fr>
    and the medicaments-api project
    <https://github.com/Giygas/medicaments-api> (snapshot 2025-12-31)
  - OPEN_MEDIC 2025 (CNAM / Assurance Maladie), ambulatory reimbursement volumes,
    <https://www.assurance-maladie.ameli.fr/etudes-et-donnees/open-medic-base-complete-depenses-medicaments>
  - RETROCEDAM 2025 (CNAM), hospital retrocession volumes,
    <https://www.assurance-maladie.ameli.fr/etudes-et-donnees/medicaments-retrocession-hospitaliere-retrocedam>
- License: Licence Ouverte / Open License 2.0 (Etalab),
  <https://github.com/etalab/licence-ouverte/blob/master/LO.md>
- Credit: Republique francaise, produced by the ANSM (BdPM) and the
  Caisse nationale de l'Assurance Maladie (OpenMedic).
- Changes: Modified. Drug names, forms, dosages and prescription frequencies were
  turned into short French medication-dictation sentences and synthesized to audio.

### PARHAF

- Source: HealthDataHub / PARHAF on Hugging Face,
  <https://huggingface.co/datasets/HealthDataHub/PARHAF>
- Version / date used: HF revision `e5e3b433965c26054d283402dfac6e60ad62c2dc`
  (the latest commit as of this release), downloaded 2026-05-12.
- Licensor / credit: HealthDataHub / Plateforme des Donnees de Sante.
- License: dual-licensed under
  Creative Commons Attribution 4.0 (CC BY 4.0),
  <https://creativecommons.org/licenses/by/4.0/>, AND
  Licence Ouverte / Open License 2.0 (Etalab),
  <https://github.com/etalab/licence-ouverte/blob/master/LO.md>.
- Citation: Tannier et al., "PARHAF, a human-authored corpus of clinical reports
  for fictitious patients in French", arXiv:2603.20494,
  <https://arxiv.org/abs/2603.20494> (see the BibTeX `tannier2026parhaf` on the
  dataset card).
- Changes: Modified. PARHAF clinical documents were rewritten into faithful
  dictation-style French paragraphs and synthesized to audio.
- Scope: Only the PARHAF **training set** is used and redistributed here. The
  PARHAF **test set is under embargo and is NOT included or redistributed.**
- No endorsement: Neither HealthDataHub nor the Plateforme des Donnees de Sante
  endorses this derived dataset or any model trained on it (an explicit
  requirement of the Etalab 2.0 license).

### Wikipedia list of medical abbreviations (acronyms subset)

- Source: French Wikipedia, "Liste d'abreviations en sante",
  <https://fr.wikipedia.org/wiki/Liste_d%27abr%C3%A9viations_en_sant%C3%A9>
- License: Wikipedia article text is CC BY-SA 4.0,
  <https://creativecommons.org/licenses/by-sa/4.0/>. Only the acronyms and
  their expansions (uncopyrightable facts) were taken from the list; no article
  text, definitions or prose are reproduced or redistributed.
- Credit: Wikipedia contributors.
- Changes: Modified. The list was hand-filtered by the author down to 511 common
  medical acronyms, each acronym was given its French pronunciation(s), and
  new LLM-generated French dictation sentences containing the acronyms were
  written and synthesized to audio.

### PARROT

- Source: PARROT radiology reports (French rows only), version 1.0,
  <https://github.com/PARROT-reports/PARROT_v1.0>
- Licensor / credit: the PARROT authors (Le Guellec, Kuchcinski, Bressem et al.).
- License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0
  (CC BY-NC-SA 4.0), <https://creativecommons.org/licenses/by-nc-sa/4.0/>.
- Citation: Le Guellec et al., "PARROT, an open multilingual radiology reports
  dataset", European Journal of Radiology Artificial Intelligence, 2026,
  <https://doi.org/10.1016/j.ejrai.2025.100066> (see the BibTeX
  `leguellec2026parrot` on the dataset card).
- Changes: Modified. French PARROT radiology reports were rewritten into faithful
  dictation-style French paragraphs and synthesized to audio.
- Scope: PARROT is shipped as a **separate, test-only subset** kept under its own
  CC BY-NC-SA 4.0 license, distinct from the rest of the release. Because it is
  NonCommercial + ShareAlike, it is not merged into the main-license corpus.

## Disclaimers

- **Research use only.** This dataset is released for research purposes only. It
  is **not** validated or intended for clinical decision-making, clinical
  validation, or clinical deployment. Do not use it, or any model trained on it,
  to make medical decisions.
- **No clinical claims.** No performance or fitness claim is made for any clinical
  setting.
- **No endorsement.** None of the sources or organizations credited above endorse
  this derived dataset or any model trained on it.
