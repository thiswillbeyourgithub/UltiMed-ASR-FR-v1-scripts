---
license:
    - cc-by-4.0
    - cc-by-nc-sa-4.0
    # A YAML list of two licences is intentional and renders fine on the HF Hub
    # (verified live): cc-by-4.0 covers the main corpus (dictionary + drugs + PARHAF + acronyms),
    # cc-by-nc-sa-4.0 covers the separate PARROT radiology subset. Keep both entries;
    # do NOT collapse to a single value or to `license: other`. See "Licensing" below.
language:
  - fr
task_categories:
  - automatic-speech-recognition
pretty_name: "UltiMed-ASR-FR-v1"
size_categories:
  - 100K<n<1M
tags:
  - medical
  - french
  - asr
  - speech
  - synthetic-speech
  - text-to-speech
  - voxtral
  - parakeet
  - nemo
  - parrot
  - parhaf
  - healthdatahub
  - radiology
  - surgery
  - anatomy
  - drugs
  - acronyms
  - technical
# The whole corpus ships as sharded Parquet with the FLAC bytes embedded, built by
# scripts/build_parquet.py, one subset per source. The per-subset licence is encoded
# in the config (subset) name with underscores (e.g. dictionary_CC_BY_4.0 and
# parrot_CC_BY-NC-SA_4.0), so it shows in the viewer's subset dropdown. Config names
# must be valid identifiers (letters / digits / _ / - / .): spaces and parentheses
# make the HF dataset viewer error out, but underscores, hyphens and dots are fine.
# The HF Data Viewer and
# load_dataset(...) work on the ENTIRE dataset, not a sample. Declaring configs
# also disables HF's automatic data-file detection, so the split globs below are
# authoritative. To rebuild the NeMo training layout (loose FLAC or tarred shards)
# from these Parquets, see scripts/parquet_to_nemo.py and "Loading and training".
configs:
  - config_name: dictionary_CC_BY_4.0
    default: true
    data_files:
      - split: train
        path: dictionary/train-*.parquet
      - split: val
        path: dictionary/val-*.parquet
      - split: test
        path: dictionary/test-*.parquet
  - config_name: drugs_CC_BY_4.0
    data_files:
      - split: train
        path: drugs/train-*.parquet
      - split: val
        path: drugs/val-*.parquet
      - split: test
        path: drugs/test-*.parquet
  - config_name: parhaf_CC_BY_4.0
    data_files:
      - split: train
        path: parhaf/train-*.parquet
      - split: val
        path: parhaf/val-*.parquet
      - split: test
        path: parhaf/test-*.parquet
  - config_name: acronyms_CC_BY_4.0
    data_files:
      - split: train
        path: acronyms/train-*.parquet
      - split: val
        path: acronyms/val-*.parquet
      - split: test
        path: acronyms/test-*.parquet
  - config_name: parrot_CC_BY-NC-SA_4.0   # non-commercial, eval-only, separate subset
    data_files:
      - split: test
        path: parrot/test-*.parquet
---

# UltiMed-ASR-FR-v1

*A large, fully documented French medical speech dataset for evaluating or training models, plus an open recipe to rebuild it in any language or topic.*

## Contents

- [Changelog](#changelog)
- [What is UltiMed-v1](#what-is-ultimed-v1)
- [Why I made UltiMed](#why-i-made-ultimed)
- [Who made UltiMed](#who-made-ultimed)
- [Quick start (TL;DR)](#quick-start-tldr)
- [How I made UltiMed](#how-i-made-ultimed)
  - [Sources](#sources)
  - [Breakdown by source](#breakdown-by-source)
  - [Text generation (LLM)](#text-generation-llm)
  - [Audio synthesis (TTS)](#audio-synthesis-tts)
  - [Hardware and conditions](#hardware-and-conditions)
  - [Repository layout on the Hub](#repository-layout-on-the-hub)
  - [Row format and the two text fields](#row-format-and-the-two-text-fields)
  - [Loading and training](#loading-and-training)
  - [Splits (combined release-wide re-split)](#splits-combined-release-wide-re-split)
- [Evaluation as a standardized benchmark](#evaluation-as-a-standardized-benchmark)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Audio quality](#audio-quality)
  - [The single-voice drawback](#the-single-voice-drawback)
  - [Other limitations](#other-limitations)
  - [Privacy and PII](#privacy-and-pii)
- [Licensing](#licensing)
  - [Main corpus: CC BY 4.0](#main-corpus-cc-by-40)
  - [PARROT subset: CC BY-NC-SA 4.0](#parrot-subset-cc-by-nc-sa-40)
- [Citations](#citations)
  - [UltiMed-ASR-FR-v1](#ultimed-asr-fr-v1)
  - [PARHAF (source, please cite)](#parhaf-source-please-cite)
  - [PARROT (source, please cite)](#parrot-source-please-cite)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Changelog

- **v1.0** (2026-08-19): initial release. Main corpus 601,338 clips / 3,105.0 h / 254.66 GB (dictionary + drugs + PARHAF + acronyms, CC BY 4.0), plus a separate **test-only** 1,549-clip / 10.1 h PARROT radiology subset (CC BY-NC-SA 4.0). [Voxtral][voxtral] `fr_female`.

## What is UltiMed-v1

A 3000+ hours corpus of **dictation-style French medical sentences** spoken by a high-quality open-weights TTS model. As far as I know, the largest French medical speech dataset of its kind. Each clip pairs a written French medical transcript with its synthesized speech, so the audio trains and evaluates ASR on the technical vocabulary (anatomy, pathology, drug names, clinical phrasing) that general-purpose ASR handles badly. A small radiology subset from [PARROT][parrot-paper] ships separately under a different licence (see [Licensing](#licensing)).

| | |
|---|---|
| Language | French (`fr`) |
| Task | Automatic Speech Recognition (training **and** evaluation) |
| Audio | **24 kHz, mono, FLAC (PCM 16-bit), straight out of [Voxtral][voxtral]**, unprocessed (see [Audio synthesis](#audio-synthesis-tts)) |
| Packaging | **Sharded Parquet with embedded FLAC** (one subset per source, split into `<source>/<split>-*.parquet`); rebuild NeMo loose/tarred data with `scripts/parquet_to_nemo.py`; see [Repository layout](#repository-layout-on-the-hub) |
| Clips | **601,338** (main corpus) + 1,549 (PARROT subset) |
| Duration | **3,105.0 h** (main corpus) + 10.1 h (PARROT subset) |
| On-disk size | **254.66 GB** (main corpus) + 851 MB (PARROT subset) |
| Splits | train / val / test, target 80 / 10 / 10, balanced by **audio duration** |
| Sources | public French medical dictionaries (Wiktionary + others), French drug names, PARHAF, common medical acronyms (Wikipedia-sourced, hand-filtered) (main corpus, CC BY 4.0); PARROT radiology (separate CC BY-NC-SA 4.0, test-only subset) |
| Voice | a single voice (`fr_female`) everywhere -- see [the drawback](#the-single-voice-drawback) |

Numbers regenerate from the source NeMo manifests (the working master the Parquet is built from) with:

```bash
uv run scripts/get_statistics.py --source combined   # the actual train/val/test
```

## Why I made UltiMed

As a resident psychiatrist and developer, I think medicine vastly underuses large-scale data, and a big part of the blame is **input**: getting information in and out of a computer is painfully slow for clinicians. Speech is high-bandwidth and already how clinicians pass much information to each other, yet ASR is barely used in medicine because general-purpose models mis-transcribe exactly the technical vocabulary clinical work is made of. A good medical ASR model is a concrete first step, and nobody seemed to be building it, so I did.

It is also a **general recipe**: a Github repository contains all the useful code needed to easily replicate this on new domain or languages: [github repo](https://github.com/thiswillbeyourgithub/UltiMed-ASR-FR-v1-scripts).

A finetuned release of [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) for French medical ASR is available: [Olicorne/parakeet-tdt-0.6b-v3-UltiMed-onnx](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-UltiMed-onnx)

## Who made UltiMed

Made entirely by **Olivier Cornelis** ([olicorne.org](https://olicorne.org), Hugging Face [@Olicorne](https://huggingface.co/Olicorne)), French resident psychiatrist and developer. Fully self-funded, no conflicts of interest to declare.

- Version: **v1** (this repo is the frozen v1 corpus: a single voice throughout). A future **v2** will be the same corpus re-synthesized with several voices, published as its own repo; this one stays as it is.
- Fine-tuned model: [Olicorne/parakeet-tdt-0.6b-v3-UltiMed-onnx](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-UltiMed-onnx) (a [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) fine-tune, released separately, trained with NeMo)
- Multilingual baseline it was fine-tuned from: [Olicorne/parakeet-tdt-0.6b-v3-optimized-onnx](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-optimized-onnx) (the upstream ONNX rebuilt for int8 accuracy and browser speed; use it when you do not need French medical vocabulary)
- In-browser app that runs either model: [Parakeet Web](https://github.com/thiswillbeyourgithub/parakeet_web)
- Reproduction scripts: [github repo](https://github.com/thiswillbeyourgithub/UltiMed-ASR-FR-v1-scripts) (the full pipeline, released to make it easier to adapt to another domain or language)

## Quick start (TL;DR)

If you just want the data and an opinion on whether it fits, this is the whole thing in one screen. Everything after it is detail (how it was built, exact row format, splits, licensing).

<details>
<summary><b>Expand</b>: a runnable <code>load_dataset</code> snippet, what you get, the subsets, and the one caveat.</summary>

```python
from datasets import load_dataset

# main corpus (dictionary + drugs + PARHAF + acronyms), CC BY 4.0
ds = load_dataset("Olicorne/UltiMed-ASR-FR-v1", "dictionary_CC_BY_4.0", split="train")
row = ds[0]
row["audio"]  # 24 kHz mono waveform, decoded from the embedded FLAC
row["text"]   # the transcript, and the field you train an ASR model on
```

- **What you get**: ~3,105 h (601,338 clips) of clean, studio-quality **synthetic** French medical speech. Each clip is a short dictation-style sentence (anatomy, pathology, drug names, clinical phrasing) spoken by a single high-quality voice and paired with its exact transcript. Built to train **and** evaluate medical ASR.
- **Train on the `text` field.** Everything else (`asr_training_source`, provenance columns) is there for transparency, not as a label.
- **Subsets, picked by config name**: `dictionary_CC_BY_4.0`, `drugs_CC_BY_4.0`, `parhaf_CC_BY_4.0`, `acronyms_CC_BY_4.0` are the main corpus (**CC BY 4.0**); `parrot_CC_BY-NC-SA_4.0` is a small, separate, **evaluation-only** radiology set under a **non-commercial** licence. Download or skip each independently.
- **`val` / `test` are ready-made in-domain benchmarks**: held out group-disjoint from `train`, so report WER on `test` to benchmark technical French medical speech.
- **The one caveat to know up front**: a **single voice** everywhere, so no speaker / accent / noise diversity, and a model can overfit its acoustics. See [the single-voice drawback](#the-single-voice-drawback).
- **Training with NeMo?** NeMo does not read Parquet directly; rebuild the loose/tarred NeMo layout with `scripts/parquet_to_nemo.py`. See [Loading and training](#loading-and-training).

</details>

## How I made UltiMed

Three-stage pipeline (source vocabulary -> written transcript -> synthesized audio), packaged as sharded Parquet with the FLAC bytes embedded (one subset per source). Scripts are released ([github repo](https://github.com/thiswillbeyourgithub/UltiMed-ASR-FR-v1-scripts)). Each stage is detailed below.

<details>
<summary><b>Expand</b>: the full build detail, stage by stage (sources, LLM text generation, TTS, hardware, repo layout, row format, loading, splits), plus the standardized benchmark protocol.</summary>

### Sources

<details>
<summary><b>Expand</b>: the five source corpora, and how the drug vocabulary was built and ranked.</summary>

| Source | What it contributes | Link |
|--------|--------------------|------|
| Medical and french dictionaries | ~60k French medical terms (eg: `blépharospasme`) -> dictation sentences | various public sources ([Wiktionnaire](https://fr.wiktionary.org/) and assorted medical dictionaries / glossaries) |
| Drug names | the exhaustive 2025 list of French drug names, i.e. the DCI (INN, active-substance names) plus the commercial brand names -> LLM-written sentences that name a medication and often a real dosage | built by the author from French open data (below) |
| [PARHAF](https://huggingface.co/datasets/HealthDataHub/PARHAF) | French clinical documents -> faithful LLM rewrites, one spoken-style paragraph per chunk | [HealthDataHub/PARHAF](https://huggingface.co/datasets/HealthDataHub/PARHAF) |
| Medical acronyms | 511 common medical acronyms (eg: `BPCO`, `CPAP`), hand-filtered from French Wikipedia's list of health abbreviations -> dictation sentences, spoken with their real French pronunciation | [Wikipedia][wiki-abbrev] |
| [PARROT][parrot-paper] (French rows only) | radiology reports, given the same rewrite treatment | [PARROT-reports/PARROT_v1.0](https://github.com/PARROT-reports/PARROT_v1.0) |

**The drug vocabulary is original work.** It was built from **official 2025 France
data** and ranked by real prescription frequency:

- **[OPEN_MEDIC 2025][openmedic]** (CNAM / Assurance Maladie): ambulatory reimbursement volumes.
- **[RETROCEDAM 2025][retrocedam]** (CNAM): hospital retrocession volumes (in UCD units).
- **BDPM** (Base de donnees publique des medicaments) via [data.gouv open data][bdpm-opendata] and the [medicaments-api](https://github.com/Giygas/medicaments-api) project (its live [API](https://medicaments-api.giygas.dev/), snapshot 2025-12-31): drug names, pharmaceutical forms, active substances and dosages.

Joined on drug codes, aggregated per substance and per brand, then scored 0-10 by combined sales (log-normalized): the most prescribed drugs (paracetamol, DOLIPRANE) get up to 11 sentences, the long tail as few as 2.

</details>

### Breakdown by source

<details>
<summary><b>Expand</b>: per-source clip counts, hours, size and mean clip length.</summary>

| Source | Clips | % clips | Hours | % duration | Size | Mean clip |
|--------|------:|------:|------:|-----------:|-----:|----------:|
| dictionary | 534,385 | 88.9% | 2,656.7 | 85.6% | 217.69 GB | 17.9 s |
| drugs | 21,901 | 3.6% | 77.0 | 2.5% | 6.27 GB | 12.7 s |
| PARHAF | 43,456 | 7.2% | 367.2 | 11.8% | 30.35 GB | 30.4 s |
| acronyms | 1,596 | 0.3% | 4.1 | 0.1% | 351 MB | 9.3 s |
| **main corpus** | **601,338** | 100% | **3,105.0** | 100% | **254.66 GB** | 18.6 s |

Shipped separately as the `parrot` subset under CC BY-NC-SA 4.0, evaluation-only:

| Source | Clips | Hours | Size | Mean clip | Subset |
|--------|------:|------:|-----:|----------:|--------|
| PARROT | 1,549 | 10.1 | 851 MB | 23.6 s | `parrot` (eval-only, test split) |

The dictionary dominates by count; PARHAF clips are fewer but longer (whole rewritten clinical paragraphs, median 29.9 s against the dictionary's 17.9 s), so it carries 11.8% of the duration on 7.2% of the clips. The acronyms subset is deliberately tiny: a targeted patch of short sentences (mean 9.3 s) teaching the sigles the other sources miss. Full per-split tables, a duration histogram and the truncation audit come from `scripts/get_statistics.py`.

That audit is worth reading before training: a TTS generation stopped by its token limit is cut off mid-sentence at exactly the cap, so the transcript no longer matches the audio, and the only observable is a pile-up of clips on one duration. The current corpus is clean (every subset has a single clip at its maximum, no pile-up). An earlier build was not: PARHAF was generated against a 2048-frame cap and 42% of its clips were pinned at 163.84 s, which is why the chunking was tightened and the audio regenerated.

</details>

### Text generation (LLM)

<details>
<summary><b>Expand</b>: the models used, how many sentences each term gets, and how documents are rewritten.</summary>

The written transcripts were generated with **[DeepSeek V4 Pro][deepseek]** (`deepseek/deepseek-v4-pro`) via **[OpenRouter][openrouter]**. Three shapes were used: `Term`, `Acronym` and `Document`

- **Term -> N sentences** (dictionary, drugs): one call per term produces N short, natural, dictation-style sentences, each of which must actually contain the term.

  Varying N spends more audio on the terms that matter and less on the long tail. Each term gets an **importance signal** (`0-10`) that sets its N:

  | Importance (0-10) | Number of spoken samples |
  |------:|---------:|
  | 0-1 | 2 |
  | 2-3 | 4 |
  | 4 | 5 |
  | 5 | 6 |
  | 6 | 7 |
  | 7 | 8 |
  | 8-10 | 11 |

  The two sources compute that signal differently, because only one has real usage data:

  - **Drugs have sales volumes**, so importance is just the **official 2025 sales figure** (log-normalized to `0-10`): common drugs are heard up to 11 times, rare ones only 2. No LLM needed. Drug sentences often also mention a **real dosage** (eg: a sentence naming both `Doliprane` and `comprimés de 1000 mg`).
  - **Dictionary terms have no such data** (a word like `blépharospasme` has no sales count). There is nothing to rank them by, so the **LLM stands in for the missing frequency data**: it scores each term `0-10` for how useful it is to teach the model (roughly, how likely a clinician is to dictate it). A proxy for the signal drugs get for free, not an extra filter on top.

- **Acronym -> 3 sentences** (acronyms): 511 common medical acronyms missing from the other sources, hand-filtered from French Wikipedia's [list of health abbreviations][wiki-abbrev], expanded to 532 (acronym, pronunciation) pairs because some have several accepted French readings (`gamma-GT` letter-spelled or read as a word). Each pair gets 3 sentences containing the acronym verbatim, exactly one of which also glosses its meaning in French. The TTS text swaps the written acronym for its pronunciation (`CAMSP` -> `came-S-P`) so the voice reads it the intended way while the written label keeps the acronym.

- **Document -> one faithful paragraph** (PARHAF, PARROT): each document is split into <= 500-char chunks, each rewritten into one faithful, dictation-ready French paragraph (the originals are written-style: typos, headers, newlines). The chunk cap is a clip-length cap: one chunk becomes one spoken clip.

  Measured on the built corpus, the cap lands PARHAF at a **median clip of 29.9 s** (p90 43.7, p95 48.8, p99 59.8, max 151.5) and PARROT at 23.9 s. Note the cap bounds the **source chunk**, not the spoken text: rewriting spells numbers and units out, so the paragraph actually read aloud averages 507 characters and reaches 1,644, at roughly 0.060 s per character. That is why **8.4% of PARHAF clips (3,635 clips, 14.3% of its duration) run past 45 s**, the ceiling ASR fine-tuning configs commonly impose on `max_duration`: those clips are silently skipped by such a trainer unless the ceiling is raised. Lower the chunk cap if you need all of PARHAF to fit under 45 s.

  The longest paragraphs all come from the same kind of chunk: near-pure shorthand (lab panels, biology tables), where a faithful spell-out of every value legitimately runs to 3x the source length (`Na+ 142meq/L` becomes "le sodium est à cent quarante-deux milliéquivalents par litre"). A length sanity check on the rewrite initially rejected those, leaving 41 of PARHAF's 43,456 chunks unwritten; they were regenerated with the check's upper bound lifted, so **every chunk of every source document is present**.

Resulting volumes:

| Source | Terms / chunks in | Generated texts out |
|--------|------------------:|--------------------:|
| dictionary | 62,448 terms | 534,385 (88.6%) |
| drugs | 3,252 terms | 21,901 (3.6%) |
| PARHAF | 43,456 chunks | 43,456 (7.2%) |
| acronyms | 532 (acronym, pronunciation) pairs | 1,596 (0.3%) |
| PARROT | 1,549 chunks | 1,549 (0.3%) |

**The text is deliberately tailored to Parakeet TDT v3.** Every source term and every generated transcript is checked against the [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) vocabulary, and anything its tokenizer would turn into `<unk>` is fixed before the text is kept. Characters the model does not know are rewritten to the form it does (en-dash and Unicode hyphen to the ASCII `-`, `ɶ` to `œ`, `Ō` to `O`, `ğ` to `g`), symbols that have no spoken form are excluded (`[`, `]`, `→`, `;`), and the two symbols that do are written as the words they are read as (`CD4+` to `CD4 plus`, `T2*` to `T2 étoile`). The result is still ordinary French, but it is normalized for one tokenizer's quirks rather than being source-faithful punctuation: if you train a model with a different vocabulary, expect the label set to be narrower than yours (no bracket, no arrow, no semicolon, no `+`).

Total LLM cost was **under $100**, thanks to a **95%+ prompt-cache hit rate** on the shared system prompt. Each written `target` then gets a deterministically derived TTS `source` (Roman numerals after a staging word, plus the rarer `/ µ °` units and `ARNm`, spelled out for the voice, no second LLM call); the `target` is the shipped label. Both are kept (see [Row format](#row-format-and-the-two-text-fields)).

</details>

### Audio synthesis (TTS)

<details>
<summary><b>Expand</b>: the TTS model, its exact settings, and what was (not) done to the audio.</summary>

Every `asr_training_source` was spoken locally by **[Voxtral][voxtral]** (`mistralai/Voxtral-4B-TTS-2603`), served with **[vLLM](https://vllm.ai/) 0.22.0** in a docker container with a single RTX 3090 Ti.

TTS settings:

| Setting | Value |
|---|---|
| Voice | `fr_female` |
| Euler steps | 16 |
| CFG alpha | 1.3 |
| Acoustic temperature | 0.7 |
| Seed | 42 |
| Output | 24 kHz mono FLAC (PCM 16-bit) |

**The clips are the raw FLAC the model emitted**: no trimming, silence removal, loudness normalization, or resampling.

</details>

### Hardware and conditions

<details>
<summary><b>Expand</b>: the GPU, its power cap, and the room it all ran in.</summary>

The entire dataset (all ~3,115 hours of audio) was generated on **a single NVIDIA RTX 3090 Ti, power-capped to 100-200 W with its memory downclocked by 100 MHz**, running in a **Parisian apartment with no air conditioning during the 2026 French heatwave**.

</details>

### Repository layout on the Hub

<details>
<summary><b>Expand</b>: the file tree on the Hub and what each folder and column holds.</summary>

The whole corpus ships as **sharded Parquet with the original FLAC bytes embedded** (one subset per source), so the HF Data Viewer and `load_dataset(...)` work on the **entire** dataset with no manifest to pair, and it is a few hundred Parquet files instead of ~603k loose FLACs (which would throttle any transfer to a crawl):

```
UltiMed-ASR-FR-v1/
  README.md
  dictionary/  train-00000-of-00174.parquet .. val-*.parquet .. test-*.parquet   # config "dictionary_CC_BY_4.0"
  drugs/       train-*.parquet   val-*.parquet   test-*.parquet                    # config "drugs_CC_BY_4.0"
  parhaf/      train-*.parquet   val-*.parquet   test-*.parquet                    # config "parhaf_CC_BY_4.0"
  acronyms/    train-*.parquet   val-*.parquet   test-*.parquet                    # config "acronyms_CC_BY_4.0"
  parrot/      test-*.parquet                                                      # config "parrot_CC_BY-NC-SA_4.0", eval-only
  scripts/     build_parquet.py   parquet_to_nemo.py
  preview_samples/  dictionary/  drugs/  parhaf/  acronyms/  parrot/                # 5 clips + manifest.jsonl each
```

- **`<source>/<split>-NNNNN-of-NNNNN.parquet`**: ~2,500 clips each. Every row embeds the **original FLAC bytes verbatim** (byte-identical round-trip) in the `datasets` `Audio` feature, plus `text`, `duration`, `category`, provenance (`group_id`, `group_mode`, `item_index`), `filename` and the quality-control columns (`cer`, `cer_tail`, `stt_transcript`, `stt_model`, `n_stt_check`, `cfg_alpha`, `regenerated`, `qc_status`). Rebuild with `uv run scripts/build_parquet.py`, after `uv run 03_sync_hotfix_results.py --apply` has carried stage 06's results into the manifests.
- **One subset per source**, each a declared config, browsable and downloadable independently. The licence is encoded in the config name (`dictionary_CC_BY_4.0`, `drugs_CC_BY_4.0`, `parhaf_CC_BY_4.0`, `acronyms_CC_BY_4.0`, `parrot_CC_BY-NC-SA_4.0`) so it shows in the viewer's subset dropdown; the Parquet still lives under the plain `dictionary/`, `drugs/`, `parhaf/`, `acronyms/`, `parrot/` folders.
- **`dictionary` + `drugs` + `parhaf` + `acronyms`** are the main corpus (CC BY 4.0); **`parrot`** is separate (CC BY-NC-SA 4.0, **test-only**), download or skip independently.
- **`scripts/`**: `build_parquet.py` (how the Parquet was built) and `parquet_to_nemo.py` (rebuilds the NeMo layout, loose or tarred; see [Loading and training](#loading-and-training)).
- **`preview_samples/`**: a few-MB offline preview, one folder per subset with 5 random `.flac` clips and a 5-line NeMo `manifest.jsonl`. Grab just this to listen and read transcripts without pulling the full corpus (or if the viewer is down). It is not a split; do not train on it.

</details>

### Row format and the two text fields

<details>
<summary><b>Expand</b>: the columns, why there are two text fields, and a worked example.</summary>

Each Parquet row is one clip, with these columns:

- `audio`: original FLAC bytes (24 kHz mono) via the `datasets` `Audio` feature; decoded by `load_dataset` or read raw from `row["audio"]["bytes"]`. Byte-identical to the synthesized FLAC.
- `text`: the written ASR label (`asr_training_target`), what the model should output, and **the field to train on.**
- `asr_training_source`: the exact text spoken to the TTS engine, derived **deterministically** from `text` (no second LLM call): the same sentence with a few tokens respelled for the voice. Shipped for transparency, **not** the training label.
- `duration`: clip length in seconds (from the FLAC header).
- `category`: the source (`dictionary`, `drugs`, `parhaf`, `acronyms`, or `parrot`).
- `group_id` / `group_mode` / `item_index`: provenance used by the split logic (`distribute` for dictionary / drug / acronym term variants, `atomic` for PARHAF / PARROT document chunks); ignored during training.
- `filename`: the clip's source basename, e.g. `000014_0001_abaissement_de_la_cataracte.flac`.

Every clip was also transcribed back and scored, and those results ship with it:

- `cer`: character error rate of `stt_transcript` against `text`, after folding away differences that are only notation (a unit spelled out on one side and abbreviated on the other, `1er` against `premier`, and so on). **This is a quality signal, not a defect list**: the corpus median is around 0.02, and most of what remains at 0.05 to 0.10 is the STT model's spelling of rare medical terms or French agreement it cannot hear, not audio that is wrong. Filter on it if you want a stricter subset.
- `cer_tail`: the same score over the last ~30 seconds only, `null` on clips shorter than that. It catches endings a whole-clip average dilutes (a chunk truncated at the TTS output cap, a voice that derails or loops on an enumeration).
- `stt_transcript`: what the STT model actually heard, so every `cer` above can be checked rather than trusted.
- `stt_model`: which model produced them, `whisper.cpp/ggml-large-v3-turbo` (Whisper large-v3-turbo served by whisper.cpp through an OpenAI-compatible endpoint). Spelled out on purpose: the API alias the client sends is `whisper-1`, which says nothing about what actually ran.
- `n_stt_check`: how many readings a clip needed. `null` for the vast majority (read once); set when a bad score triggered a re-transcription at a higher temperature to rule out an STT hallucination, in which case the best reading is the one scored.
- `cfg_alpha`: the classifier-free guidance the audio was drawn at (1.3 for the original pass, 1.0 to 3.0 for a regenerated clip).
- `regenerated` / `qc_status`: whether the quality pass replaced this clip. `qc_status` is `improved` (a better draw replaced the original), `exhausted` (the clip tripped a gate but no redraw was good enough, so **the original audio ships**: these are the known-weak clips, 6 of them), or `original` (never tripped a gate). Across the corpus 946 clips were replaced and the mean CER of the shipped audio against its label is **0.012**.

**On seeds.** The per-clip TTS seed is deliberately **not** shipped. The serving stack ignored the per-request seed for most of the corpus (it reached a sampler that this model does not use, while the flow-matching noise came from a global RNG), so a seed column would have implied a reproducibility that does not exist. Regenerating a byte-identical clip from the released text is not possible; the audio itself is the artifact.

**Why two fields.** `text` is the clean transcript the model must produce; `asr_training_source` is the same sentence adjusted only where the voice would mispronounce a token: Roman numerals after a staging word (`type II`, `grade I`, `stade II`) written out, and the rarer `/`, `µ`, `°` units and `ARNm` spelled out. Everything else, plain digits included (`16 mg`, `4 semaines`), stays as written. Both are French, restricted to characters the [Parakeet TDT v3][parakeet-v3] tokenizer can represent (see [Text generation](#text-generation-llm)), and free of banned non-spoken symbols.

Worked example (a real `drugs` clip):

| field | value |
|-------|-------|
| `text` (the label you train on) | Instauration de KENZEN 16 milligrammes le matin pour une hypertension artérielle de grade **I**, contrôle tensionnel à 4 semaines. |
| `asr_training_source` (what was spoken) | Instauration de KENZEN 16 milligrammes le matin pour une hypertension artérielle de grade **un**, contrôle tensionnel à 4 semaines. |

Only `grade I` became `grade un`; `16 milligrammes` and `4 semaines` stayed as written. You train the model on `text`; the audio in `audio` is what `asr_training_source` produced.

</details>

### Loading and training

<details>
<summary><b>Expand</b>: loading with <code>datasets</code>, rebuilding the NeMo layout, and reading the raw FLAC bytes.</summary>

**Browsing / general use** (`datasets`): load any subset by config name; audio and transcript come back together, no manifest to pair.

```python
from datasets import load_dataset
ds = load_dataset("Olicorne/UltiMed-ASR-FR-v1", "dictionary_CC_BY_4.0", split="train")
ds[0]["audio"]  # {'array': ..., 'sampling_rate': 24000}, decoded from the embedded FLAC
ds[0]["text"]   # the transcript label
```

**With NeMo** (the intended training path): NeMo does not read Parquet directly, so rebuild the training layout from it with the shipped `scripts/parquet_to_nemo.py`, which decodes the embedded FLAC verbatim and writes a NeMo manifest.

```bash
# loose FLAC + manifest.jsonl per split (drop-in for a manifest_filepath config)
uv run scripts/parquet_to_nemo.py --format loose

# or NeMo tarred (WebDataset) shards, for an is_tarred config
uv run scripts/parquet_to_nemo.py --format tarred
```

By default it merges `dictionary` + `drugs` + `parhaf` + `acronyms` into `train` / `val` / `test` and keeps `parrot` as its own eval-only group (`--per-source` keeps every subset separate). Loose manifests plug into `model.train_ds.manifest_filepath=...`; tarred output into `is_tarred: true` + `tarred_audio_filepaths=<group>/tarred/'audio__OP_0..N_CL_.tar'`.

**Without NeMo**: read the Parquet with any Arrow / `datasets` reader and pull the raw bytes from `row["audio"]["bytes"]` (a complete FLAC file) if you want the original encoded clip rather than a decoded waveform.

</details>

### Splits (combined release-wide re-split)

<details>
<summary><b>Expand</b>: the 80 / 10 / 10 duration-balanced tables and the grouping rules that prevent leakage.</summary>

**For both training and evaluation.** `val` and `test` are held out group-disjoint from `train` (no term or document leaks), so they are ready-made **in-domain benchmarks**, not just training monitors: report WER on `test` for technical French medical speech. The separate `parrot` subset is an **out-of-domain radiology** benchmark (below).

Main corpus balanced 80 / 10 / 10 by **audio duration**, group-aware so nothing leaks:

| Split | Clips | % clips | Hours | % duration | Size |
|-------|------:|------:|------:|-----------:|-----:|
| train | 486,007 | 80.8% | 2,484.0 | 80.0% | 203.71 GB |
| val | 57,712 | 9.6% | 310.5 | 10.0% | 25.48 GB |
| test | 57,619 | 9.6% | 310.5 | 10.0% | 25.47 GB |
| **all** | **601,338** | 100% | **3,105.0** | 100% | **254.66 GB** |

These splits are the **main corpus** (dictionary + drugs + PARHAF + acronyms). PARROT is **not** mixed in: it ships as its own eval-only `parrot` subset (all 1,549 clips as one `test` split) under its own licence, **test-only on purpose** because the source authors ask that it not be trained on, and one untouched `test` split keeps it a clean out-of-domain benchmark (not split into val/test: the main corpus already has a large `val`, and halving 10.1 h would only add noise). Counts run slightly above 80% while duration lands exactly at 80% because the dictionary's short clips are the most numerous. Grouping rules:

- **Dictionary / drug / acronym terms**: the variants of one term are *distributed* across splits (at least one variant in train; one per split when a term has >= 3 variants), so no term is memorized in train and only tested in test.
- **PARHAF documents** (and, in its own subset, **PARROT** documents): all chunks of one source document stay in the **same** split (*atomic*), so no document appears in both train and test.

</details>

---

## Evaluation as a standardized benchmark

<details>
<summary><b>Expand</b>: what to report, the fixed normalization recipe, and a runnable scoring snippet.</summary>

`val` and `test` are held out group-disjoint from `train` (no term or document leaks), so they are ready-made in-domain benchmarks. Please report them with the recipe below rather than an ad-hoc setup, so numbers stay comparable across models.

**What to report**

- **Metrics**: WER (primary) and CER (secondary).
- **Reference text**: the `text` field (the training label), never `asr_training_source`.
- **Splits**: the main-corpus `test` split (in-domain) and the `parrot` `test` split (out-of-domain radiology). Report them separately; do not average or mix them.
- **Per subset**: report `dictionary`, `drugs`, `parhaf` and `acronyms` on their own as well as combined, since their difficulty and clip length differ a lot (see [Breakdown](#breakdown-by-source)).

**Normalization**, applied identically to reference and hypothesis, in this order:

1. lowercase
2. French number words to digits, so the spoken "seize" scores against the written "16" (e.g. `text_to_num.alpha2digit(s, lang="fr")`)
3. strip punctuation
4. collapse whitespace

Step 2 is not optional here: [Voxtral][voxtral] speaks a written "16 mg" as "seize milligrammes", so without it every digit in the label counts as an error. It does not canonicalize spelled-out unit words ("milligrammes" vs "mg"); that is a known residual mismatch, so keep the four steps fixed as above rather than adding per-model tweaks.

```python
# uv run --with datasets --with jiwer --with text_to_num eval.py
import re
import jiwer
from datasets import load_dataset
from text_to_num import alpha2digit

def normalize(s: str) -> str:
    s = alpha2digit(s.lower(), lang="fr", relaxed=True)  # "seize" -> "16"
    s = re.sub(r"[^\w\s]", " ", s)                        # strip punctuation
    return re.sub(r"\s+", " ", s).strip()                 # collapse whitespace

ds = load_dataset("Olicorne/UltiMed-ASR-FR-v1", "dictionary_CC_BY_4.0", split="test")
refs = [normalize(r["text"]) for r in ds]
hyps = [normalize(transcribe(r["audio"])) for r in ds]   # plug in your ASR model

print(f"WER {jiwer.wer(refs, hyps):.4f}  CER {jiwer.cer(refs, hyps):.4f}")
```

**With NeMo**: rebuild the layout with `scripts/parquet_to_nemo.py` (see [Loading and training](#loading-and-training)) and score with NeMo's built-in WER, applying the **same** four normalization steps to hypotheses and references so the numbers line up with the snippet above.

When you publish results, state the model, the exact subset and split, and that you followed this recipe.

</details>

</details>

---

## Considerations for Using the Data

<details>
<summary><b>Expand</b>: audio quality, the single-voice drawback, the other limitations, and privacy.</summary>

### Audio quality

<details>
<summary><b>Expand</b>: what the clips actually sound like.</summary>

High-fidelity 24 kHz mono: clean, studio-style speech, no background noise, channel effects, or clipping. The `fr_female` voice reads technical vocabulary (drug names, anatomy, pathology) accurately, exactly what an in-domain medical set needs. The flip side (one uniform voice, no real-world noise) is covered below.

</details>

### The single-voice drawback

<details>
<summary><b>Expand</b>: why a single voice was kept, and what it costs you.</summary>

The most important limitation: **a single voice (`fr_female`) is used for every clip**, kept because its quality and jargon accuracy were far ahead of the alternatives. So the corpus has **no speaker, accent, gender, or recording-condition diversity**, and a model trained only on it may overfit this voice's acoustics; take measures against that.

Voxtral-4B-TTS-2603 has two voices designed for French: `fr_female` and `fr_male`.

- `fr_female`: subjectively excellent and, crucially, **rarely mispronounces technical jargon** (drug names, anatomy, pathology), exactly what an in-domain medical read must get right. The clips are the model's direct output.
- `fr_male`: excellent too (better than the best voices of every other local TTS I tried), but it makes more mistakes on technical words, so I did not use it.

If a better TTS comes along, or Voxtral releases its voice embedder, I might redo a pass to add voice diversity. The only other models that could pronounce this jargon were either very expensive or about to be sunset (`gpt-4o-mini-tts-2025-12-15`).

</details>

### Other limitations

<details>
<summary><b>Expand</b>: synthetic audio, LLM-written text, tokenizer-normalized punctuation, known unknowns.</summary>

- **Synthetic audio**: no background noise, microphone variation, or spontaneous disfluency.
- **LLM-generated text**: sentences are model-written and validated by automatic gates, but may still contain occasional artifacts.
- **Text normalized for one tokenizer**: the labels were gated on the [Parakeet TDT v3][parakeet-v3] vocabulary, so a handful of characters were rewritten or excluded to avoid `<unk>` (details in [Text generation](#text-generation-llm)). Harmless for most models, but the punctuation is not source-faithful.
- **Known unknowns**: the automatic gates cannot catch everything, and some blind spots are certainly still in there. If you hit a wrong transcript, a mispronounced term, a systematic artifact, or anything else worth fixing, please report it (see [Contact](#contact)). Feedback is what a v2 would be built on.

</details>

### Privacy and PII

<details>
<summary><b>Expand</b>: why every source is PII-free.</summary>

**Only PII-free sources were used, and no personal data was added.** Dictionary terms are from public vocabularies, the drug data is public aggregate statistics (no patient-level data), PARHAF is public de-identified clinical data, the acronyms come from a public Wikipedia list, and PARROT reports are **completely fictional** per their authors. Transcripts are LLM-generated, so any string resembling a real person is **coincidental** or already present in DeepSeek's training data or a public source. None was introduced by me.

</details>

</details>

---

## Licensing

<details>
<summary><b>Expand</b>: the two licence tiers and the per-source attribution to ship.</summary>

**This release ships in two licence tiers, so the bulk stays permissive while the one non-commercial source is kept separable:**

- **Main corpus** -- dictionary + drugs + PARHAF + acronyms, i.e. the `dictionary`, `drugs`, `parhaf` and `acronyms` subsets: **[CC BY 4.0][ccby]**.
- **PARROT radiology subset** -- the `parrot` subset, shipped on its own and **not** mixed into the main-corpus subsets: **[CC BY-NC-SA 4.0][ccbyncsa]**.

Ship a `NOTICE` with the per-source attributions below, **mark the data as modified** (text synthesized to audio), and keep the no-endorsement statement.

### Main corpus: [CC BY 4.0][ccby]

- **Medical terms** -- drawn from various public French sources (Wiktionary and assorted medical dictionaries / glossaries). Only individual terms are used; no definitions or article text are redistributed.
- **PARHAF** -- [CC BY 4.0][ccby] + [Etalab Open Licence 2.0][etalab]. Credit HealthDataHub / Plateforme des Donnees de Sante, link the [dataset][parhaf-ds] and version, cite the paper (below), note the data was **modified** (synthesized to audio), no endorsement, **training set only** (test set embargoed, not redistributed).
- **Drug names** -- French **open data** under the [Licence Ouverte / Etalab 2.0][etalab]: [OPEN_MEDIC][openmedic] and [RETROCEDAM][retrocedam] (CNAM / Assurance Maladie) and the [BDPM public medicines database][bdpm-opendata]. Only names and public dosages are used; credit CNAM / Assurance Maladie and the BDPM.
- **Acronyms** -- hand-filtered from French Wikipedia's [list of health abbreviations][wiki-abbrev]. Only the acronyms and their expansions (uncopyrightable facts) are used; no article text is redistributed. The sentences containing them are LLM-generated.

### PARROT subset: [CC BY-NC-SA 4.0][ccbyncsa]

The PARROT clips are **LLM-based rewrites** of PARROT's fictional radiology reports, so this subset carries PARROT's own licence, [CC BY-NC-SA 4.0][ccbyncsa]: credit the [PARROT authors][parrot-paper] (Le Guellec et al., *European Journal of Radiology Artificial Intelligence*, 2026; [dataset repo][parrot-repo]), non-commercial use only, and any adaptation stays under the same licence. Please [cite the paper](#parrot-source-please-cite). It ships as a **test-only** evaluation subset, separate from the training data. The reports are fictional.

</details>

## Citations

### UltiMed-ASR-FR-v1

```bibtex
@misc{cornelis_ultimed_asr_fr_2026,
  title        = {UltiMed-ASR-FR-v1: a large synthetic French medical speech dataset for ASR},
  author       = {Cornelis, Olivier},
  year         = {2026},
  howpublished = {Hugging Face},
  url          = {https://huggingface.co/datasets/Olicorne/UltiMed-ASR-FR-v1}
}
```

### PARHAF (source, please cite)

<!-- The arXiv id is intentionally omitted from this citation. Any contiguous form
of it (an arxiv.org URL, the DOI, or a bibtex eprint field) makes the HF Hub
auto-derive an arxiv:<id> tag that makes PARHAF's paper look like THIS dataset's
own paper. Readers reach the paper through the PARHAF dataset page linked below. -->

```bibtex
@misc{tannier2026parhaf,
  title        = {PARHAF, a human-authored corpus of clinical reports for fictitious patients in French},
  author       = {Tannier, Xavier and Abbara, Salam and Flicoteaux, R{\'e}mi and Khalil, Youness and N{\'e}v{\'e}ol, Aur{\'e}lie and Zweigenbaum, Pierre and Bacry, Emmanuel},
  year         = {2026},
  howpublished = {arXiv preprint; see the PARHAF dataset page},
  url          = {https://huggingface.co/datasets/HealthDataHub/PARHAF}
}
```

### PARROT (source, please cite)

```bibtex
@article{leguellec2026parrot,
  title   = {PARROT, an open multilingual radiology reports dataset},
  author  = {Le Guellec, Bastien and Adambounou, Kokou and Adams, Lisa C. and Agripnidis, Thibault and Ahn, Sung Soo and Ait Chalal, Radhia and Akinci D'Antonoli, Tugba and Amouyel, Philippe and Andersson, Henrik and Bentegeac, Rapha{\"e}l and Benzoni, Claudio and Blandino, Antonino Andrea and Busch, Felix and Can, Elif and Cau, Riccardo and Cavallo, Armando Ugo and Chavihot, Christelle and Chiquete, Erwin and Cuocolo, Renato and Divjak, Eugen and Dziadkowiec-Macek, Barbara and Elogne, Armel and Fanni, Salvatore Claudio and Ferrarotti, Carlos and Fossataro, Claudia and Fossataro, Federica and Fu{\l}ek, Katarzyna and Fu{\l}ek, Micha{\l} and Ga{\'c}, Pawe{\l} and Gachowska, Martyna and Garc{\'i}a-Ju{\'a}rez, Ignacio and Gatti, Marco and Gorelik, Natalia and Goulianou, Alexia Maria and Hamroun, Aghiles and Herinirina, Nicolas and Holay, Quentin and Ivanac, Gordana and Kitamura, Felipe and Klontzas, Michail E. and Kompanowska, Anna and Kompanowski, Rafa{\l} and Kraik, Krzysztof and Krupka, Dominik and Lef{\`e}vre, Alexandre and Lemke, Tristan and Lindholz, Maximilian and Macek, Piotr and Makowski, Marcus and Mannacio, Luigi and Meddeb, Aymen and M{\"u}ller, Lukas and Natale, Antonio and Nguema Edzang, B{\'e}atrice and Ojeda, Adriana and Park, Yae Won and Piccione, Federica and Ponsiglione, Andrea and Por{\k{e}}ba, Ma{\l}gorzata and Por{\k{e}}ba, Rafa{\l} and Prucker, Philipp and Pruvo, Jean-Pierre and Pugliesi, Rosa alba and Rabemanorintsoa, Feno Hasina and Rafailidis, Vasileios and Resler, Katarzyna and Rotkegel, Jan and Saba, Luca and Siebert, Ezann and Stanzione, Arnaldo and Tekin, Ali Fuat and Toapanta-Yanchapaxi, Liz and Triantafyllou, Matthaios and Tsaoulia, Ekaterini and Urban, Szymon and Vassalou, Evangelia and Vernuccio, Federica and Wang, Weilang and Wass{\'e}lius, Johan and W{\l}odarczak, Adrian and W{\l}odarczak, Szymon and Wysocki, Andrzej and Xu, Lina and Zato{\'n}ski, Tomasz and Zhang, Shuhang and Ziegelmayer, Sebastian and Kuchcinski, Gr{\'e}gory and Bressem, Keno K.},
  journal = {European Journal of Radiology Artificial Intelligence},
  year    = {2026},
  volume  = {5},
  pages   = {100066},
  doi     = {10.1016/j.ejrai.2025.100066},
  url     = {https://doi.org/10.1016/j.ejrai.2025.100066}
}
```

## Acknowledgements

*In no particular order*

- [Corentin Sautier](https://csautier.github.io/), PhD (Hugging Face [@csautier](https://huggingface.co/csautier)).
- [Adrien Parrot](https://www.linkedin.com/in/adrien-parrot-doc/), MD, at [InterHop](https://interhop.org).
- [Bastien Leguellec](https://bleguellec.org/), MD and currently pursuing a PhD.

## Contact

Olivier Cornelis, feel free to reach me at [olicorne.org](https://olicorne.org)

**Please report errors and blind spots.** Bad clips, mispronunciations, missing vocabulary, anything that looks off: open a discussion on this dataset page or reach out directly. Corrections are welcome and will feed a future version.

[parakeet-v3]: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
[ccby]: https://creativecommons.org/licenses/by/4.0/
[ccbyncsa]: https://creativecommons.org/licenses/by-nc-sa/4.0/
[etalab]: https://github.com/etalab/licence-ouverte/blob/master/LO.md
[openmedic]: https://www.assurance-maladie.ameli.fr/etudes-et-donnees/open-medic-base-complete-depenses-medicaments
[retrocedam]: https://www.assurance-maladie.ameli.fr/etudes-et-donnees/medicaments-retrocession-hospitaliere-retrocedam
[bdpm-opendata]: https://www.data.gouv.fr/dataservices/api-medicaments-fr
[parhaf-ds]: https://huggingface.co/datasets/HealthDataHub/PARHAF
[parrot-repo]: https://github.com/PARROT-reports/PARROT_v1.0
[parrot-paper]: https://doi.org/10.1016/j.ejrai.2025.100066
[wiki-abbrev]: https://fr.wikipedia.org/wiki/Liste_d%27abr%C3%A9viations_en_sant%C3%A9
[voxtral]: https://huggingface.co/mistralai/Voxtral-4B-TTS-2603
[openrouter]: https://openrouter.ai/
[deepseek]: https://openrouter.ai/deepseek/deepseek-v4-pro

