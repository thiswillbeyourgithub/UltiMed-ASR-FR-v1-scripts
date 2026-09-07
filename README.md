# UltiMed-ASR-FR-v1-scripts

The build scripts behind **[UltiMed-ASR-FR-v1](https://huggingface.co/datasets/Olicorne/UltiMed-ASR-FR-v1)**, a 3,105-hour French medical speech-recognition dataset, and the fine-tuned ASR model trained on it.

This repository is the **recipe, not the product**. It documents how a single person turned public text sources into a 600k-clip domain-specific ASR corpus on one consumer GPU, so that the same approach can be re-pointed at another specialty, another language, or another ASR model.

> **This is documentation, not a package.** It is deliberately not batteries-included: there is no installer, no orchestrator, and no bundled data. The expected way to use it is to read it, or to point a coding agent at it, and then rebuild the parts you need for your own domain. Every script here ran for real, against real sources, and produced the published dataset.

## Contents

- [Disclaimer and scope](#disclaimer-and-scope)
- [The published artifacts](#the-published-artifacts)
- [The core idea](#the-core-idea)
- [The pipeline](#the-pipeline)
- [Architecture: one engine, five sources](#architecture-one-engine-five-sources)
- [Adapting this to another domain or language](#adapting-this-to-another-domain-or-language)
- [Data you have to bring yourself](#data-you-have-to-bring-yourself)
- [What is deliberately absent](#what-is-deliberately-absent)
- [Running the scripts](#running-the-scripts)
- [Scale and cost](#scale-and-cost)
- [Known issues in v1](#known-issues-in-v1)
- [Licensing](#licensing)
- [Third-party attribution](#third-party-attribution)
- [Credits](#credits)

## Disclaimer and scope

**Research use only.** Neither this code, nor the dataset it builds, nor any model trained on that dataset is validated or intended for clinical decision-making, clinical validation, or clinical deployment. Do not use any of it to make medical decisions. No performance or fitness claim is made for any clinical setting.

**No source data is distributed here.** This repository contains code, prompts and documentation. It does not redistribute the medical dictionary, the PARHAF corpus, the PARROT corpus, the drug databases, or any generated audio. You obtain each source yourself from its upstream and comply with its own licence. See [Data you have to bring yourself](#data-you-have-to-bring-yourself).

**The generated text is LLM output.** Every transcript in the dataset was written or rewritten by a large language model and was not reviewed clinically. It is training material for a speech recognizer, which only ever has to learn how the words *sound*, and it should not be read as medically accurate prose.

**No endorsement.** None of the organizations credited in [Third-party attribution](#third-party-attribution) endorse this code, the derived dataset, or any model trained on it. This is an explicit requirement of the Etalab 2.0 licence covering some of the sources.

**No warranty.** This software is provided on an "AS IS" basis, without warranties or conditions of any kind, express or implied. See sections 7 and 8 of the [LICENSE](LICENSE).

## The published artifacts

| Artifact | What it is |
|---|---|
| [Olicorne/UltiMed-ASR-FR-v1](https://huggingface.co/datasets/Olicorne/UltiMed-ASR-FR-v1) | The dataset. 601,338 clips / 3,105 h main corpus, plus a separate eval-only PARROT subset. Sharded Parquet with FLAC bytes embedded. |
| [Olicorne/parakeet-tdt-0.6b-v3-UltiMed-onnx](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-UltiMed-onnx) | The fine-tuned model: `parakeet-tdt-0.6b-v3` trained on the above, exported to ONNX. |
| [Olicorne/parakeet-tdt-0.6b-v3-optimized-onnx](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-optimized-onnx) | The re-quantized upstream baseline the fine-tune builds on. |
| This repository | The scripts that produced all of the above, stages 1 to 4 of the plan below. |
| [UltiMed-ASR-FR-v1-Voxtral](https://github.com/thiswillbeyourgithub/UltiMed-ASR-FR-v1-Voxtral) | The Voxtral TTS container that spoke every clip: Dockerfile, tuning, and the vllm-omni patches. |
| [Parakeet Web](https://github.com/thiswillbeyourgithub/parakeet_web) | The in-browser ASR app that loads either ONNX model. Not part of this pipeline, but it is where the models end up. |

The plan, end to end:

1. Take text sources rich in domain vocabulary.
2. Use LLM calls to turn those terms and documents into short, natural, dictation-style sentences (the written ASR transcript).
3. Speak those sentences with a local TTS model to produce audio.
4. Package and publish the resulting audio dataset.
5. Fine-tune the ASR model on it. **This step is not in this repo**: it happens in a separate [NeMo](https://github.com/NVIDIA/NeMo) checkout with a standard NeMo training config.
6. Release the fine-tuned model.

## The core idea

Two things carry most of the value here, and both survive translation to any other domain.

**Every row has two text fields, not one.** `asr_training_target` is the written label, what the model should output. `asr_training_source` is the text handed to the TTS engine. The source is derived from the target **deterministically**, with no second LLM call, by [`utils/voxtral_normalize.py`](utils/voxtral_normalize.py): it spells out the units, digits and Roman numerals that the TTS mispronounces, and it flags anything it does not recognize to a review queue instead of quietly producing bad audio. Because the transform is code and not a model, the two fields cannot drift apart semantically, which is exactly the failure mode that poisons synthetic ASR corpora.

**Synthetic audio is scored and regenerated, not trusted.** Stage 06 transcribes every generated clip with Whisper, scores the transcript against the label by character error rate, and regenerates the bad ones, keeping the best draw. The subtle part is the scoring: a representational difference is not an error. When the TTS spells "milligrammes" out loud and Whisper writes it back as "mg", that is a round trip, not a mistake. [`06_hotfixes/01_compute_stt.py`](06_hotfixes/01_compute_stt.py) folds one written form per spoken thing (units, titles, percent, ordinals, staging Roman numerals, letter-spelled acronyms) by importing the same tables `voxtral_normalize` uses in the other direction, so a fold added on one side is understood by both. Every fold in that list was justified by replaying it over the already-scored clips and counting how many it improved against how many it made worse, not by intuition.

## The pipeline

Top-level numbered folders are ordered pipeline stages. Files numbered inside a folder are that stage's ordered sub-steps.

| Stage | Role |
|---|---|
| [`01_dictionnary/`](01_dictionnary/) | ~62.6k French medical dictionary terms. An LLM scores each term 0 to 10 for usefulness, the score sets how many sentences it gets, then one call per term produces those sentences. Bring your own term list: see [Data you have to bring yourself](#data-you-have-to-bring-yourself). |
| [`02_drugs/`](02_drugs/) | French and international drug names to medication-dictation sentences. Importance comes from real 2025 sales volumes rather than an LLM, and sentences name real dosages. [`02_drugs/sources/`](02_drugs/sources/) holds the scripts that build the stage's inputs from the public drug databases. |
| [`03_PARHAF/`](03_PARHAF/) | PARHAF clinical documents, chunked and rewritten into faithful spoken-French paragraphs. |
| [`04_PARROT/`](04_PARROT/) | PARROT radiology reports, French rows only, same chunk-and-rewrite path. |
| [`05_generate_audio/`](05_generate_audio/) | Runs a local TTS server over every stage's text output. Resumable, with a truncation guard. |
| [`06_hotfixes/`](06_hotfixes/) | The quality loop: score every clip against its transcript, regenerate the bad ones, keep the best draw. [`manifest_listener/`](06_hotfixes/manifest_listener/) is the human-in-the-loop complement, a small app for listening to clips and flagging them. |
| [`07_acronyms/`](07_acronyms/) | 511 hand-filtered medical acronyms, each spoken with its real pronunciation while the label keeps the written form. |
| [`99_hf_release/`](99_hf_release/) | Build NeMo manifests, package as sharded Parquet, publish. |
| [`utils/`](utils/) | Everything shared by more than one stage. |
| [`tests/`](tests/) | Stdlib and `uv run` tests for the shared logic. |

Two stage docs are worth reading on their own: [`06_hotfixes/README_alternating_improvement.md`](06_hotfixes/README_alternating_improvement.md) (how the STT and TTS passes alternate when they cannot share one GPU) and [`05_generate_audio/README.md`](05_generate_audio/README.md).

[`docs/BUILD_LOG.md`](docs/BUILD_LOG.md) is the running log kept while building: the actual commands, the actual score distributions, the actual costs, and the reasoning behind the decisions that did not work out the first time.

[`CLAUDE.md`](CLAUDE.md) is the architecture guide. It was written for coding agents and it is the densest description of how the pieces fit together. **If you are pointing an agent at this repository, that is the file to give it first.**

## Architecture: one engine, five sources

All five text stages plug into a single run engine, [`utils/text_generation_engine.py`](utils/text_generation_engine.py). It owns everything that is identical across sources: the score-based variant policy, the resume scanner, the thread pool, deduplication, run statistics, the review and skip queues, the cost pre-plan, and the deterministic source pass.

A stage supplies a `StageAdapter` with only the four things that actually differ: how to build the user prompt, how to validate a generated variant, how to shape the output row, and which `call_llm` to use. There are two families:

- **Term to N sentences**: dictionary, drugs, acronyms. One call per term yields N variants, each of which must genuinely contain the term.
- **Document to one paragraph**: PARHAF, PARROT. One chunk becomes one faithful rewritten paragraph, via a second adapter in [`utils/text_rewrite.py`](utils/text_rewrite.py).

Adding a source means writing an adapter, never forking the loop. The rest of `utils/` is deliberately small and single-purpose:

| Module | Responsibility |
|---|---|
| [`_pipeline_shared.py`](utils/_pipeline_shared.py) | LLM plumbing: litellm wrapper, prompt-cache markers, retry, finish-reason guard, cost tracking, the `<t>` block parser, the soft validators. |
| [`text_generation_engine.py`](utils/text_generation_engine.py) | The source-agnostic run engine described above. |
| [`text_rewrite.py`](utils/text_rewrite.py) | The document-rewrite adapter and its validators. |
| [`text_chunking.py`](utils/text_chunking.py) | Deterministic, stdlib-only paragraph and sentence chunking. |
| [`voxtral_normalize.py`](utils/voxtral_normalize.py) | Target to source: the deterministic spoken-form transform. |
| [`parakeet_tokenizer.py`](utils/parakeet_tokenizer.py) | Would the target ASR tokenizer map this text to `<unk>`? Also the shared character audit CLI. |
| [`nemo_manifest.py`](utils/nemo_manifest.py) | Duration-aware, group-aware train/val/test splitting and NeMo manifest IO. |

One detail in `text_chunking.py` generalizes badly if you miss it: **the 500-character chunk cap is a clip-length cap, not a context cap.** One chunk becomes one spoken clip, and NeMo's `train_ds.max_duration` silently *drops* clips longer than the limit. The cap was derived from measured audio (0.065 s per raw character, p99 0.093) against the trainer's 45 s ceiling. If you change your trainer's `max_duration`, re-derive the cap; do not copy 500.

## Adapting this to another domain or language

Roughly two thirds of this code is domain-agnostic and language-agnostic. Here is the honest split.

**Reusable as-is**: `utils/text_generation_engine.py`, `utils/text_rewrite.py`, `utils/text_chunking.py`, `utils/nemo_manifest.py`, `utils/_pipeline_shared.py`, all of `05_generate_audio/`, the state machine in `06_hotfixes/01_recursive_improvement.py`, and all of `99_hf_release/`.

**Must be rewritten for your target**:

1. **The prompts.** Every `PROMPT_*.md` file encodes French dictation conventions and medical register. These are the highest-leverage files in the repo and the first thing to rewrite.
2. **`utils/voxtral_normalize.py`.** Its unit table, number words and Roman-numeral rules are French, and its quirk list is specific to one TTS model. Expect to rebuild it empirically: synthesize a few hundred clips, listen to the failures, and add a rule per failure class. [`01_dictionnary/VOXTRAL_QUIRKS.md`](01_dictionnary/VOXTRAL_QUIRKS.md) is the human-readable registry that this code is the executable form of; keep the two in sync.
3. **`utils/parakeet_vocab.txt`.** Extracted from one specific ASR tokenizer. Regenerate it from whichever model you are fine-tuning, or the `<unk>` gate will be checking the wrong alphabet.
4. **The scoring folds** in `06_hotfixes/01_compute_stt.py`. Same language-specific problem as `voxtral_normalize`, in the opposite direction. Do not port these blind: replay any fold you add over your own already-scored clips and check the improved-versus-worsened count before keeping it.
5. **The chunk cap** in `utils/text_chunking.py`, re-derived from your trainer's `max_duration` and your TTS speaking rate.

**A rough order of operations for a new domain:**

1. Find text sources rich in the vocabulary you care about, and get a signal for which terms matter (usage frequency beats an LLM's guess whenever it exists).
2. Gate every term through the tokenizer audit before you spend anything on it. `uv run utils/parakeet_tokenizer.py --input yourfile.jsonl --field term` groups the untokenizable characters by how many entries they hit, so you normalize the alphabet before generating, not after.
3. Write the target prompt, generate a few hundred sentences, and read them. Iterate on the prompt, not on the code.
4. Build the normalizer against your TTS by listening to failures.
5. Generate everything, then run the CER loop until the tail stops moving.
6. Build manifests, split by duration rather than row count, and package.

## Data you have to bring yourself

None of these are in this repository. Each has its own licence, which is yours to comply with.

| Source | Where | Licence |
|---|---|---|
| French medical term list (~62.6k terms) | Bring your own. [Wiktionnaire](https://fr.wiktionary.org/) is a good CC BY-SA starting point for French medical vocabulary. | Depends which source you pick. |
| PARHAF clinical documents | [HealthDataHub/PARHAF](https://huggingface.co/datasets/HealthDataHub/PARHAF) | CC BY 4.0 **and** Etalab Open Licence 2.0. Training set only; the test set is under embargo and must not be redistributed. |
| PARROT radiology reports | [PARROT_v1.0](https://github.com/PARROT-reports/PARROT_v1.0) | CC BY-NC-SA 4.0. **NonCommercial and ShareAlike**: keep it in a separate subset and do not merge it into a differently-licensed corpus. |
| Raw French Wikipedia acronym scrape | [Liste d'abréviations en santé](https://fr.wikipedia.org/wiki/Liste_d%27abr%C3%A9viations_en_sant%C3%A9) | CC BY-SA 4.0. Only the 511-row hand-filtered CSV ships here; the raw scrape carries verbatim article prose and does not. |

The drug data in [`02_drugs/`](02_drugs/) **is** included, because it is French public open data under Etalab Open Licence 2.0. See [Third-party attribution](#third-party-attribution).

## What is deliberately absent

- **All generated corpora and audio.** Hundreds of gigabytes, and derived from sources with licences this repository does not carry.
- **`99_hf_release/scripts/upload_to_hf.py`.** Kept out of version control on purpose: it resolves paths through a local symlink to an external drive. The `.gitignore` entry explains how to bring it back.
- **The fine-tuning config.** Training happens in a separate NeMo checkout. `99_hf_release/scripts/parquet_to_nemo.py` rebuilds the NeMo training layout from the published Parquet, in either loose-manifest or tarred form, and that is the handoff point.
- **Sample rows.** There is no `samples/` folder, because real rows would mean redistributing NonCommercial and ShareAlike source text under this repository's licence. Field names and semantics are documented in `CLAUDE.md` and in each script's docstring instead.

## Running the scripts

Every script is a self-contained [PEP 723](https://peps.python.org/pep-0723/) script: its dependencies are declared in a `# /// script` header, so there is nothing to install.

```bash
uv run 01_dictionnary/01_llm_scoring.py --help
```

Notes that will save you an hour:

- Some scripts resolve paths relative to their own file, others relative to the working directory. When in doubt, run a script from inside its own stage folder.
- LLM calls go through [litellm](https://github.com/BerriAI/litellm). Set the provider key in the environment (`OPENROUTER_API_KEY` for the defaults). Pin `--provider` on OpenRouter models or your prompt-cache hit rate collapses, and with it your budget.
- If you add an import from `utils/_pipeline_shared`, its transitive dependencies must be added to the calling script's `dependencies` block too.
- Tests are plain scripts: `python tests/test_voxtral_normalize.py`, `uv run tests/test_acronyms_stage.py`, and so on.

## Scale and cost

For calibration, since the whole point is that this is reproducible by one person.

| | |
|---|---|
| Text generation | 62,461 dictionary terms to 534,441 sentences, roughly **$50 to $60** total, at a ~95% prompt-cache hit rate |
| LLM | DeepSeek V4 Pro via OpenRouter, 32 parallel jobs |
| TTS | Voxtral 4B TTS on vLLM, single `fr_female` voice, 24 kHz mono FLAC, no post-processing |
| Quality loop | 599,740 clips transcribed with Whisper and scored; median CER **0.0095** |
| Hardware | **One RTX 3090 Ti**, power-capped to 100-200 W, in a Paris apartment with no air conditioning during the 2026 heatwave |
| Output | 601,338 clips, 3,105 hours, 254 GB |

The CER reports are committed: [`06_hotfixes/improved/full.stt.statistics.md`](06_hotfixes/improved/full.stt.statistics.md) and [`06_hotfixes/improved_parrot/full.stt.statistics.md`](06_hotfixes/improved_parrot/full.stt.statistics.md).

## Known issues in v1

Both of these were found **after** UltiMed-ASR-FR-v1 was published and after the fine-tune had already been trained on it. The shipped audio and the released model carry them. They are recorded here so anyone reusing this recipe starts from the fixed version, and so nobody rediscovers them the hard way.

<details>
<summary><b>Expand</b>: the <code>RAS</code> mispronunciation and the dosage zero-padding.</summary>

### `RAS` is spoken as the word "race"

`RAS` (*rien à signaler*, the French clinical shorthand for "nothing to report") is a word-acronym, so the TTS reads it as a word rather than as letters, and Voxtral lands on "race". The written label is correct; only the audio is wrong.

**Scope: 341 clips of 601,338, or 0.06% of the main corpus** (282 dictionary, 15 drugs, 44 PARHAF, none in acronyms or PARROT).

**Not fixed.** It is logged as an open decision in [`01_dictionnary/VOXTRAL_QUIRKS.md`](01_dictionnary/VOXTRAL_QUIRKS.md), because the two candidate remedies are not equivalent. Respelling it so the voice reads letters (`R.A.S.`, `R A S`, `err-a-ess`) is a pure pronunciation fix. Expanding it to `rien à signaler` is not: the audio would then say the full phrase while the label still says `RAS`, which deliberately changes the audio-to-label relationship. Picking between them needs listening data that does not exist yet.

If you are adapting this pipeline, the transferable lesson is that word-acronyms are the dangerous class. Letter-acronyms (`TSH`, `ECG`, `IRM`) all read correctly raw; it is the ones that happen to look like a word in the target language that a TTS will mispronounce, and no amount of CER scoring catches it, because the label and the transcript agree while the audio does not.

### Dosage strings carried BDPM's zero padding

The public drug database pads decimal parts inconsistently, so a single strength arrived spelled several ways for one form: `500 mg`, `500,0 mg` and `500,00 mg`. Those went into the LLM's presentation hint unchanged, where they read as three distinct doses of the same drug. Worse, 88 presentations carried a padded value with **no** clean twin at all, so `BENZYLTHIOURACILE` shipped `25,00 mg` and nothing else and was read aloud as "vingt-cinq virgule zéro zéro milligrammes".

**Fixed** in [`90a6ed6`](https://github.com/thiswillbeyourgithub/UltiMed-ASR-FR-v1-scripts/commit/90a6ed612ac7ab33f82ad881f1ae3403f8004db8), *"canonicalize and de-duplicate BDPM dosage strings"*, which canonicalizes the padding away and then de-duplicates whatever becomes equal, in `02_drugs/sources/base_de_donnee_medicament/create_drug_db.py`. On the real 410-substance input that takes 1075 dosage entries down to 952 across 113 substances, with zero padded values left and no distinct dose lost. Covered by `tests/test_drug_dosage_dedup.py`.

The committed `02_drugs/*.jsonl` files are **deliberately not regenerated**: they are the exact inputs that built v1, and re-running them through the fixed script would desync this repository from the published dataset. The fix lands on the next build.

</details>

## Licensing

**The code in this repository is [Apache-2.0](LICENSE).** That covers every `.py`, `.sh`, and `.md` file here, including the prompts.

Two deliberate exceptions, both data files rather than code, included by aggregation and licensed by their upstream:

- [`07_acronyms/wikipedia_acronyms.filtered.authorfiltered.csv`](07_acronyms/wikipedia_acronyms.filtered.authorfiltered.csv) and the `wikipedia_acronyms.expanded.jsonl` derived from it. The `TERM` and `MEANING` columns come from French Wikipedia and remain under **CC BY-SA 4.0**, credited to Wikipedia contributors. The selection of the 511 entries and the `PRONOUNCED_AS` column are original work by the author.
- The `.jsonl` files in [`02_drugs/`](02_drugs/), derived from French public medicines data under the **Etalab Open Licence 2.0**.

Licences elsewhere in the project, for completeness:

- The **dataset** is CC BY 4.0 for the main corpus, with the PARROT subset separately under CC BY-NC-SA 4.0.
- The **models** are CC BY 4.0.
- Using a permissive licence for code and a Creative Commons licence for data and weights is the normal split, and there is no conflict between them: they cover different works.

If you keep this repository under Apache-2.0, note that a dataset you produce by *running* these scripts is not a derivative work of the scripts. Your output is yours, subject only to the licences of the source data you fed in.

## Third-party attribution

Full detail, in the form that ships with the dataset, is in [`99_hf_release/NOTICE.md`](99_hf_release/NOTICE.md). In summary:

- **PARHAF**: HealthDataHub / Plateforme des Données de Santé. Dual-licensed CC BY 4.0 and Etalab 2.0. Cite Tannier et al., arXiv:2603.20494. The data was modified (rewritten and synthesized to audio). Training set only; the test set is embargoed. No endorsement is implied.
- **PARROT**: Le Guellec, Kuchcinski, Bressem et al. CC BY-NC-SA 4.0. Cite Le Guellec et al., *European Journal of Radiology Artificial Intelligence*, 2026. Kept as a separate, non-commercial, evaluation-only subset.
- **French public medicines data**: BdPM (ANSM) and OPEN_MEDIC / RETROCEDAM (CNAM), République française, Etalab Open Licence 2.0.
- **Acronyms**: French Wikipedia, [Liste d'abréviations en santé](https://fr.wikipedia.org/wiki/Liste_d%27abr%C3%A9viations_en_sant%C3%A9), CC BY-SA 4.0, credited to Wikipedia contributors.
- **Base ASR model**: `nvidia/parakeet-tdt-0.6b-v3`.
- **TTS**: `mistralai/Voxtral-4B-TTS-2603`.

## Credits

Built by Olivier Cornelis. The code, the prompts and the documentation in this repository were written with **[Claude Code](https://claude.com/claude-code)**, with some earlier passes in [aider](https://aider.chat/); the individual file docstrings note which where it matters.

Repository: <https://github.com/thiswillbeyourgithub/UltiMed-ASR-FR-v1-scripts>
