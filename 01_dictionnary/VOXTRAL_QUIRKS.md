# voxtral-tts pronunciation quirks (single source of truth)

Registry of how our local **voxtral-tts** engine handles the tricky tokens in
the French medical ASR corpus. This drives the SOURCE-pass normalization
(the `asr_training_source` step). Keep every decision here, with the sample
that proves it, so contradicting rules are easy to spot.

**Philosophy for voxtral: pass-through by default.** voxtral has strong built-in
text normalization and reads most written medical acronyms, abbreviations,
numbers and units correctly *as written*. So we transform **only** the tokens
listed as `FIX` below. Every `KEEP RAW` row is a rule the old SOURCE pass
(tuned for OpenAI gpt-4o-mini-tts) applied that we should **drop** for voxtral,
because respelling a token voxtral already knows makes it worse (e.g. `T S H`).

> The old SOURCE pass was an LLM prompt tuned for gpt-4o-mini-tts that
> over-normalized for voxtral. It has been removed (prompt file deleted, LLM
> pass dropped); `../voxtral_normalize.py` is its deterministic replacement and
> this file is the voxtral-specific spec that drives it.

> **Decision (2026-07): Architecture A.** The label (`asr_training_target`) keeps
> spelled-out units ("milligrammes par litre"); the TARGET prompt is unchanged.
> The SOURCE *LLM* pass is replaced by the deterministic `../voxtral_normalize.py`
> (Roman + ARNm; units already spelled by the TARGET LLM). The residual detector
> is the safety net that catches any unit the TARGET LLM fails to spell out.

## Legend

- **KEEP RAW** = voxtral says the written form correctly, apply NO transform.
- **FIX** = voxtral mispronounces it, respell per the "Decided rule" column.
- **?** = untested, run `voxtral_sweep.txt` and record the result here.

## Confirmed

| Written form | Category | voxtral behaviour | Verdict | Settles (old SOURCE rule) |
|---|---|---|---|---|
| `TSH` | letter-acronym | natural, better than spaced | **KEEP RAW** | drop letter-spacing (`T S H`) |
| `SIDA` | word-acronym | correct | **KEEP RAW** | (was already a no-op) |
| `ARA2` | acronym+digit | correct | **KEEP RAW** | drop trailing-digit spelling (`A R A deux`) |
| `CIM-10` | acronym-hyphen-digit | correct | **KEEP RAW** | keep hyphenated form whole |
| `DSM-V` | acronym-hyphen-roman | correct | **KEEP RAW** | drop Roman-numeral spelling here |
| `95 %` | percent | correct | **KEEP RAW** | drop `% -> pourcent` |
| `RAS` | word-acronym (rare, FR-clinical) | read as the French word "race" | **FIX** | needs a targeted respelling |

### Open FIX decisions

- **`RAS`**: voxtral reads it as "race". Candidate fixes (pick after more data):
  1. respell so it reads as letters (test `R.A.S.` vs `R A S` vs `err-a-ess`), or
  2. expand to the full phrase `rien à signaler`. Note: option 2 changes the
     audio-vs-label relationship (the ASR label would still say `RAS` while the
     audio says the full phrase), so it is a different decision, not just a
     pronunciation fix.

## Sweep conclusion

34/50 probes are KEEP RAW. Every FIX except three one-offs is a **unit written
with `/`, `µ` or `°`**. So the deterministic normalizer only has to handle:

1. **Units containing `/`, `µ`, `°`** -> spell out in French. Bare units that
   read fine stay raw (`mmHg`, bare `mg`), as do dates, numbers, `%`, decimals.
2. **Roman numeral after a staging/anatomy word** (`stade IV`, `nerf X`) ->
   French number word. NOT `type II` / `Henri IV` / `en IV`, which read correct
   raw (we normalize the staging cases anyway, since it is harmless and voxtral
   is inconsistent across contexts).
3. **`ARNm`** -> `ARN-m`.
4. **`RAS`** -> TODO (open decision above).

This is implemented deterministically in `../voxtral_normalize.py` (NOT an LLM).
Its residual detector flags any `/ µ °` that survives (a unit not yet in the
table) to a review queue, so a novel unit goes to human review, never to bad
audio. That backstop is why the unit table does not need to be exhaustive.

### FIX list (decided rules)

| Trigger | Rule | Example |
|---|---|---|
| unit with `/` | `X/Y` -> `X par Y`, both parts spelled | `mg/L` -> `milligrammes par litre` |
| micro `µ` | `µ` -> `micro` prefix | `µg` -> `microgrammes`; `µmol/L` -> `micromoles par litre` |
| `°C` | -> `degrés Celsius` | `38 °C` -> `38 degrés Celsius` |
| staging/anatomy + Roman | Roman -> FR cardinal word | `stade IV` -> `stade quatre`; `nerf X` -> `nerf dix` |
| `ARNm` | -> `ARN-m` | |
| `RAS` | **TODO** (reads as "race") | see open decision |

**KEEP RAW (confirmed, do NOT transform):** all letter-acronyms
(ECG/IRM/ADN/VIH/NFS/CRP/BPCO/AVC/IEC/LCR/ECBU/EEG/ORL/HbA1c), word-acronyms
(SIDA/SAMU/SMUR/OMS/AVK), acronym+digit (ARA2/DSM-V/DSM-5/CIM-10/SpO2/T4/CO2/IgG/ASA 3),
`mmHg`, bare `mg`, decimals, `%`, all date formats, `8h`/`14h30`, `type II`,
`Henri IV`, `en IV`.

## Sweep results (voxtral_sweep.txt)

50 probes run through voxtral; verdicts recorded below. `voxtral_sweep.txt` is
generated from the Probe column of this table, do not hand-edit it.

| Probe | Category | Verdict |
|---|---|---|
| L'ECG est sans particularité. | letter-acronym | correct |
| L'IRM cérébrale est normale. | letter-acronym | correct |
| Une analyse de l'ADN est demandée. | letter-acronym | correct |
| La sérologie VIH est négative. | letter-acronym | correct |
| La NFS est normale. | letter-acronym | correct |
| La CRP reste élevée. | letter-acronym | correct |
| Patient suivi pour une BPCO. | letter-acronym | correct |
| Antécédent d'AVC ischémique. | letter-acronym | correct |
| Introduction d'un IEC. | letter-acronym | correct |
| Ponction du LCR réalisée. | letter-acronym | correct |
| L'ECBU est stérile. | letter-acronym | correct |
| L'EEG de veille est normal. | letter-acronym | correct |
| Un avis ORL est demandé. | letter-acronym | correct |
| Le taux d'HbA1c est élevé. | mixed-case token | correct |
| Le SAMU est intervenu rapidement. | word-acronym | correct |
| Transfert médicalisé par le SMUR. | word-acronym | correct |
| Selon les recommandations de l'OMS. | word-acronym | correct |
| Patient anticoagulé par AVK. | word-acronym | correct |
| Les critères du DSM-5 sont remplis. | acronym-hyphen-digit | correct |
| La SpO2 est abaissée. | acronym+digit | correct |
| Le dosage de la T4 libre est normal. | acronym+digit | correct |
| Les anticorps IgG sont positifs. | acronym+lowercase-suffix | correct |
| Vaccin à ARNm à jour. | acronym+lowercase-suffix | "ARN-m" is better |
| Le taux de CO2 est normal. | acronym+digit | correct |
| Patient classé ASA 3. | acronym+spaced-digit | correct |
| La CRP est à 42 mg/L. | unit-symbol | hallucinated the /L |
| Lévothyroxine 75 µg par jour. | unit-symbol | hard but sometimes reads correctly |
| Créatinine à 120 µmol/L. | unit-symbol | hallucinated umol |
| Tension à 140 mmHg. | unit-symbol | correct |
| Température à 38 °C. | unit-symbol | hallucinated the unit |
| Clairance à 60 mL/min. | unit-symbol | wrong unit |
| TSH à 0,3 mUI/L. | unit-symbol | wrong unit |
| Glycémie à 5,5 mmol/L. | unit-symbol | wrong /L |
| Vancomycine 15 mg/kg. | unit-symbol | wrong /kg |
| Protéinurie à 2 g/L. | unit-symbol | wrong /l |
| Phosphatases à 80 UI/L. | unit-symbol | wrong /l |
| Dexaméthasone 0,5 mg le matin. | decimal-comma | correct |
| Fièvre à 37,5 ce matin. | decimal-comma | correct |
| Saturation à 95 %. | percent (pourcent probe) | correct |
| Metformine 1000 mg deux fois par jour. | large-number | correct |
| Opéré le 12/07/1998. | date DD/MM/YYYY | correct |
| Consultation le 3 mars 2024. | date long-form | correct |
| Contrôle prévu le 2024-04-15. | date ISO | correct |
| Corticoïdes administrés à 8h. | time no-space | correct |
| Admis à 14h30. | time with minutes | read 14 heure 30 minutes |
| Cancer diagnostiqué au stade IV. | Roman numeral | wrong IV |
| Diabète de type II. | Roman numeral | correct 2 |
| Atteinte du nerf X. | Roman / letter ambiguity | read X instead of 10 |
| Antibiotiques administrés en IV. | IV = intraveineuse (ambiguity) | correct |
| Hôpital Henri IV. | Roman in proper noun | read 4 |

## Contradiction watch

Note here any FIX whose rule would clash with a KEEP RAW, so we never ship
mutually inconsistent transforms:

- **Acronym spacing.** `TSH` proves "never space letter-acronyms". So the `RAS`
  fix must NOT be a generic "space all acronyms" rule, it has to target `RAS`
  (and its confirmed peers) specifically.
- **`IV` is context-dependent.** `stade IV` / `type II` (Roman -> number) vs
  `en IV` (intraveineuse). If voxtral gets the raw form wrong in either context,
  the fix stays context-aware, not a blanket `IV` substitution.
