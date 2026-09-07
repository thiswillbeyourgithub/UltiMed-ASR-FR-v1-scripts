# CLAUDE.md

Guidance for working in this repository. Read this first, then the stage READMEs
and script docstrings for detail. This file was written with Claude Code.

## What this project is

This repo builds a **French medical ASR training dataset**. The end goal is:

1. Take good text sources rich in medical terms (dictionary, drug names, PARHAF,
   PARROT).
2. Use LLM API calls to turn those terms into short, natural, dictation-style
   French sentences (the written ASR transcript).
3. Speak those sentences with a local TTS model to produce audio.
4. Publish the resulting **medical audio dataset** on Hugging Face under the
   author's name.
5. Use that audio to fine-tune the `parakeet-tdt-v3` multilingual ASR model so it
   handles technical medical vocabulary better (fine-tuning lives in the separate
   `../NeMo/` checkout, already present alongside this repo).
6. Release the fine-tuned model.

The repo you are in covers steps 1 to 4 (text generation, audio, release).
Fine-tuning (step 5) is done in `../NeMo/`, not here.

## The pipeline is the folder numbering

**Top-level numbered folders are ordered pipeline stages.** They are meant to be
run in order. Files numbered inside a folder (`01_`, `02_`, ...) are the ordered
sub-steps of that stage.

| Stage | Status | Role |
|------|--------|------|
| `01_dictionnary/` | active | French medical dictionary terms to text pairs |
| `02_drugs/` | active | French + international drug names to dictation sentences |
| `03_PARHAF/` | active | PARHAF French clinical documents (rewrite path wired) |
| `04_PARROT/` | active | PARROT radiology reports, French (rewrite path wired) |
| `05_generate_audio/` | active | run local TTS over every stage's text output |
| `06_hotfixes/` | active | score every clip against its transcript (Whisper CER) and regenerate the bad ones |
| `07_acronyms/` | active | common medical acronyms (Wikipedia-sourced, hand-filtered) to dictation sentences, with per-pronunciation TTS sources |
| `99_hf_release/` | active | build NeMo manifests + package as Parquet, upload to HF (all done; the repo is private, making it public is the remaining step) |

`99_hf_release/` first builds the **NeMo-format JSONL manifests** from each stage's
`generated_dataset.jsonl` + its `.flac` clips: `01_build_nemo_manifest.py`
(per dataset) and `02_combine_nemo_manifests.py` (release-wide combine), both thin
wrappers over `utils/nemo_manifest.py`. `03_sync_hotfix_results.py` then carries stage
06's results into every manifest: it refreshes the `duration` of clips whose audio was
regenerated (a rebuild would NOT, since `probe_durations` caches by file name with no
mtime) and attaches the QC columns the release ships (`cer`, `cer_tail`,
`stt_transcript`, `stt_model`, `n_stt_check`, `cfg_alpha`, `regenerated`, `qc_status`).
It joins on the resolved absolute audio path, so manifests that store it relative to
different directories all match, and it is idempotent. Run it after any improvement run
and before the parquet build. The audio lives under the `data/` symlink
(`data/{dictionary,drugs,PARHAF,PARROT}/`, an external SSD) and manifests are
written to `data/NeMO_files/`; neither the symlink nor the generated manifests are
committed (the SSD path embeds the username).

The **release format is Parquet, not tar**: `scripts/build_parquet.py` packages the
whole corpus into sharded Parquet with the FLAC bytes embedded (one subset per
source: dictionary / drugs / parhaf / acronyms / parrot), so the HF Data Viewer and
`load_dataset` work on the entire dataset. `scripts/parquet_to_nemo.py` rebuilds the
NeMo training layout from that Parquet (`--format loose` for a `manifest_filepath`
config, `--format tarred` for `is_tarred`, reusing NeMo's own converter).
`scripts/upload_to_hf.py` (git-ignored, points at the local `data` symlink) pushes
the Parquet + those scripts to the Hub and deletes any stale remote tar first, then
**purges the orphan Parquet shards of earlier builds** (`purge_orphan_parquet`).
That purge is not optional housekeeping: a rebuild that changes a split's shard
COUNT renames every shard of it (`test-00000-of-00020` -> `test-00000-of-00021`),
so the new files land BESIDE the old ones instead of overwriting them, and both
sets still match the dataset card's `dictionary/test-*.parquet` globs, i.e. the
Data Viewer and `load_dataset` read two inconsistent builds of one split.
`data/hf_parquet/<subset>/` is authoritative (a subset with no local shards is
skipped, never wiped), and the purge runs AFTER the upload so a failed upload
cannot leave a split short of shards. Deleted shards still occupy the storage
quota until `--squash-history` collapses the history.
`scripts/get_statistics.py` reports over the local manifests. The earlier tarred /
preview approach was retired (`build_tarred.py` / `build_preview_parquet.py`
deleted). The dataset is uploaded (NOTICE + card shipped) to the private repo
`Olicorne/UltiMed-ASR-FR-v1`; making it public is the remaining step. The `-v1`
suffix is deliberate: a future v2 re-synthesizes the same corpus with several
voices (v1 is single-voice `fr_female` throughout) and gets its OWN repo, so this
one is frozen. The repo name is hardcoded as `REPO_ID` in both
`scripts/upload_to_hf.py` and `scripts/remove_arxiv_tag.py`, and appears in the
dataset card (title, `pretty_name`, the `load_dataset` examples, the BibTeX) and
in `NOTICE.md`; the GitHub link stays unsuffixed because the pipeline repo builds
any version.

## Shared code lives in `utils/`

Anything used by more than one stage lives in `utils/`. Do not duplicate it into a
stage folder. The modules are plain `uv run` scripts (not a package), imported by
adding `utils/` to `sys.path`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from _pipeline_shared import call_llm, PricingTracker  # noqa: E402
```

- **`_pipeline_shared.py`** is the LLM pipeline infrastructure shared by the text
  generators: `call_llm` (litellm wrapper with prompt-cache markers, transient
  retry, finish-reason guard, `<think>`-block stripping, optional `temperature`
  / `top_p` / `reasoning` pass-through, `api_base`, and OpenRouter `provider`
  pinning), `PricingTracker` (live + projected cost), the `<t>` block parser,
  the soft validators, `check_term_in_variants` (and its non-raising boolean core
  `term_present_in_variant`, for callers that OR several candidate anchors),
  the retry-with-validation loop,
  the OpenRouter helpers,
  `sanitize_definition` (+ its `DEFINITION_SUBSTITUTIONS` table: the glyph and
  Greek-letter spell-outs applied to every `Definition:` hint; moved here from the
  dictionary stage so the acronyms stage shares it),
  and the error classes (`LLMError`, `CountMismatchError`,
  `BlockCountError`, `TermMissingError`, `ValidationError`). A wrong `<t>` block
  count is a retryable/skippable `BlockCountError`; `CountMismatchError` is
  reserved for the fatal internal target/source length invariant.
- **`text_generation_engine.py`** is the source-agnostic run engine every text
  stage plugs into (dictionary, drugs, acronyms, PARHAF, PARROT). It owns the
  orchestration that is identical across sources: the score-based n-variants
  policy (`parse_n_variants` / `resolve_n_variants`), the JSONL input iterator
  (auto-assigns a stable `index` to rows that lack one), the resume-state
  scanner, `build_target_user_prompt` (the shared `Term:/Definition:/Produce N`
  user message; its `extra_lines` slot inserts stage lines after Term+Definition
  so the cacheable prefix stays common, e.g. the acronyms `Pronounced as:` line),
  the per-item `generate_variants_for_term` (one LLM call to N
  `<t>` targets, then the deterministic voxtral source pass), and `run` (thread
  pool, target/source dedup, run statistics, review / skip queues, pricing
  pre-plan). A stage supplies a `StageAdapter` with the four things that differ:
  `build_user_prompt`, `validate_fn`, `make_row`, and `call_llm_fn` (injected so
  a stage wrapper's module-global `call_llm` stays monkeypatchable in tests),
  plus its sampling knobs and system prompt. The adapter also carries
  `require_score` (default `True`): scored stages abort on missing scores, the
  rewrite and acronyms stages set it `False`; and an optional `source_transform`
  hook applied to each validated target BEFORE `voxtral_normalize` in the source
  pass (the acronyms stage swaps the written sigle for its spoken pronunciation
  there; `None` for every other stage).
- **`text_chunking.py`** is the stdlib-only deterministic preprocessing for the
  rewrite stages: `strip_parens` (drop every nested `(...)`, per the agreed
  policy) and `chunk_text` (split a raw document into coherent `<=500`-char
  chunks on paragraph then sentence boundaries). Kept free of the LLM stack so the
  `03_chunk_for_rewrite.py` preprocessors import it cheaply. Covered by
  `tests/test_text_chunking.py` (`python tests/test_text_chunking.py`).
  **The cap is a CLIP-LENGTH cap, not a context cap**: one chunk becomes one
  spoken clip, and the fine-tuning config (`../NeMo/perso/training_config.yaml`,
  `train_ds.max_duration`) silently drops clips past 45 s, so the ceiling is set
  from measured audio (0.065 s per raw chunk char, p99 0.093) rather than from the
  model's context. It is hard: `_rebalance_short_chunks` re-splits a sub-`min_chars`
  fragment against its neighbour instead of merging past the cap. If you change
  either default, re-derive it from the trainer's `max_duration`, and remember that
  changing it invalidates every downstream artefact (see the stage sections).
- **`text_rewrite.py`** is the rewrite-stage adapter over the engine (PARHAF +
  PARROT): `build_rewrite_adapter` (`require_score=False`, one paragraph per
  chunk), `rewrite_validate` (reuses the shared granular checks, plus no-colon,
  Parakeet-tokenizable, and a length-sanity ratio vs the source chunk), and
  `run_rewrite` (the shared entry point both stage wrappers call). Its system
  prompt is composed like the drugs stage: the shared base rules in
  `utils/PROMPT_REWRITE_TO_PARAGRAPH.md` (kept separate from the scored stages'
  prompt so their cache key stays untouched) plus one per-stage worked example
  appended at load time (`load_system_prompt(base, example)`), a clinical-document
  example for PARHAF (`03_PARHAF/PROMPT_REWRITE_PARHAF_EXAMPLES.md`) and a
  radiology example for PARROT (`04_PARROT/PROMPT_REWRITE_PARROT_EXAMPLES.md`). The
  rules stay shared (no duplication); only the example differs per stage.
- **`parakeet_tokenizer.py`** + **`parakeet_vocab.txt`** answer one question: would
  the Parakeet TDT v3 tokenizer map this text to `<unk>`? Used to gate terms and
  generated text to characters the model can actually represent. The vocab path
  resolves next to the module, so keep the two files together. It is also the
  shared, source-agnostic **character audit** every stage points at: `audit_texts`
  (importable) plus a CLI that reads a plain term-per-line file, an inline
  `--text`, or a JSONL field via `--field` (e.g.
  `uv run utils/parakeet_tokenizer.py --input 03_PARHAF/02_parhaf_texts.jsonl --field text`)
  and groups the still-untokenizable characters by how many entries they hit, so
  you know what to normalize next. Do not fork this per stage.
- **`voxtral_normalize.py`** deterministically turns an `asr_training_target`
  (written label) into the `asr_training_source` (text fed to the local
  voxtral-tts engine): it applies only the small, proven set of fixes voxtral
  needs (units with `/ µ °` spelled out, Roman numerals after a staging word,
  `ARNm`), and flags any uncovered unit to a review queue instead of producing
  bad audio. It is the executable form of `01_dictionnary/VOXTRAL_QUIRKS.md`;
  keep the two in sync. Its `UNIT_SPOKEN` table (plus `compile_unit_rules`,
  `STAGING_ROMAN_RE`, `roman_to_int`, `FR_CARDINAL`) is **public on purpose**: the CER
  scorer in `06_hotfixes/01_compute_stt.py` folds the same symbol/spoken pairs the other
  way (the TTS spells a unit out, then Whisper abbreviates it back, and neither
  difference is an error), and imports them instead of keeping a second copy. Add a unit
  here and both sides learn it; the scorer's own `_SCORING_ONLY_UNITS` holds what voxtral
  reads correctly and so is absent from this table (bare units, the squared / cubed forms,
  becquerels), `_CASE_UNITS` holds the pairs only case tells apart (`G/L` giga against
  `g/L` grammes, folded before the lowercasing), and a set of notation rules folds what the
  generators speak but Whisper writes as a symbol (decimals, ordinals, `1+`, `3/0`, `J+2`,
  `2h05`). Those were chosen from a measured cost ranking of every surviving diff, not
  guessed, then replayed over a 30k-clip sample to check none of them made a clip worse;
  do both before adding more. That replay is why number separators are dropped rather than
  spoken: the label writes a suture gauge `3 0` where Whisper writes `3,0` or `3/0`, and
  deletion is the only form all three reach.
- **`tests/test_voxtral_normalize.py`** is its stdlib test: `python tests/test_voxtral_normalize.py`.
- **`nemo_manifest.py`** is the stage-99 NeMo-manifest core shared by
  `01_build_nemo_manifest.py` and `02_combine_nemo_manifests.py`: split-spec
  parsing, the grouped duration-aware `assign_splits` (distribute mode for
  dictionary/drugs term variants, atomic mode for PARHAF/PARROT document chunks;
  per-split coverage of >=3-variant terms is best-effort by default in the CLI,
  `--strict-coverage` restores the hard >=1-per-split guarantee),
  the FLAC-header duration probe (cached), the stage-05 filename `audio_index`, and
  relative-path/jsonl io. Import path is stdlib-only (`soundfile`/`tqdm` imported
  lazily) so `tests/test_nemo_manifest.py`
  (`python tests/test_nemo_manifest.py`) exercises the splitter without those
  wheels. Do not fork the split logic into a stage.

## Core data model

Every text generator emits pairs built around two fields:

- **`asr_training_target`**: the written ASR label, what the model should output.
  It must be tokenizable by Parakeet (no `<unk>`), French, and free of the banned
  non-spoken symbols.
- **`asr_training_source`**: the text handed to the local TTS engine to synthesize
  the audio. It is derived **deterministically** from the target by
  `voxtral_normalize` (no second LLM call), so the two stay semantically identical
  and only differ in how units, digits and Roman numerals are spelled out.

The audio (stage `05`) is synthesized from `asr_training_source`; the transcript
label shipped to the ASR trainer is `asr_training_target`.

## Stage detail

### `01_dictionnary/` (active)
Source: a ~62.6k-entry French medical dictionary (`original_dictionnary.jsonl`).
The default input/output paths chain, so the pipeline reproduces by running the
numbered scripts in order with no arguments (`01b` is optional):

    uv run 01_llm_scoring.py     # original_dictionnary.jsonl -> .scored.jsonl
    uv run 02_token_check.py     # .scored.jsonl -> .scored.normalized.jsonl
    uv run 03_generate_texts.py  # .scored.normalized.jsonl -> generated_dataset.jsonl

Sub-steps:
- `01_llm_scoring.py`: score each term 0 to 10 for ASR-usefulness, resumable.
  Input defaults to `original_dictionnary.jsonl`, writes
  `original_dictionnary.scored.jsonl`. Failed rows (truncated/censored/no-tag,
  guarded by `finish_reason`) are omitted rather than written null, so `--resume`
  retries them. (Written with aider.)
- `01b_filter_scored.py`: optional. Show the score distribution (`--stat`) or carve
  a subset by score (`--above` / `--under`, printed to stdout). Not a mandatory
  stage: `03`'s `--n-variants` score policy already includes/excludes by score.
- `02_token_check.py`: normalize `term` glyphs and gate on the Parakeet `<unk>`
  detector; passes `score` (and every other field) through unchanged. Reads
  `original_dictionnary.scored.jsonl`, writes
  `original_dictionnary.scored.normalized.jsonl` only when every term is clean.
- `03_generate_texts.py`: the main generator. Reads
  `original_dictionnary.scored.normalized.jsonl` (both scored and tokenizable);
  aborts if any row lacks a score. One LLM call per term produces N
  `asr_training_target` variants (N scaled by score via `--n-variants`), then
  `voxtral_normalize` derives each `asr_training_source`. Output:
  `generated_dataset.jsonl`. Each run appends config + cost + retry stats to
  `run_statistics.jsonl` (and mirrors the loguru stream to `run_statistics.log`).
- Supporting: `PROMPT_GENERATE_ASR_TRAINING_TARGET.md` (the target system prompt),
  `VOXTRAL_QUIRKS.md` (TTS quirk registry, single source of truth),
  `test_models.sh`.

### `02_drugs/` (active)
`01_generate_drug_texts.py` is a thin **adapter over `utils/text_generation_engine.py`**
(the same engine the dictionary stage uses): it produces N
`asr_training_target` variants per drug (short medication-dictation sentences
that name the drug) and the deterministic `asr_training_source` pair, sharing the
engine's dedup, resumability, run statistics, review / skip queues and pricing
pre-plan. Input is `drugs_freq_dosages.jsonl` (`term`, `type` substance|brand,
`category` "drugs", `score`, `substances`, `dosages`). The system prompt is the
shared base target prompt plus `01_dictionnary/PROMPT_GENERATE_ASR_TRAINING_TARGET_DRUGS.md`
(the drugs addendum), so the model emits the same `<t>` blocks with the same
validators; the per-drug user prompt passes the drug's real forms + dosages as
the `Definition:` presentation hint. Each variant must name the drug's concrete
anchor(s): `presence_needle` reduces the label first, so ATC combination filler
("EN ASSOCIATION", "ET DIURETIQUES", "INHIBITEUR D'ENZYME") is not required, a
real "+"/"ET" combination requires both active drugs (order-independent), and a
purely generic class label ("ASSOCIATIONS") skips the presence check. A row's
`substances` active ingredients are added as ALTERNATIVE anchors
(`_acceptable_needles`): a variant is valid if it names EITHER the reduced label
OR the molecule(s), which rescues truncated brand presentation codes the model
can only speak by their ingredient names (e.g. `EMTRICIT/TENOF.MYL200/245` ->
`TENOFOVIR DISOPROXIL ET EMTRICITABINE`). When several anchors apply the OR check
uses the shared boolean `term_present_in_variant`; a single anchor still goes
through `check_term_in_variants` so its fuzzy-match warnings survive. Rows whose
`substances` just echoes a generic label (substance-type combos) get no extra
anchor from it.
**Data-quality TODO:** `drugs_freq_dosages.jsonl` dosage lists still contain
near-duplicate artifacts (e.g. `500 mg`, `500,0 mg`, `500,00 mg` for one form);
the upstream builder that produces this file should de-duplicate them so the
presentation hint is clean. `drugs_dosages.jsonl` / `drugs_frequency_2025.jsonl`
are the other committed inputs; the file that builds `drugs_freq_dosages.jsonl`
from them is not committed yet.

### `03_PARHAF/` (data prepped)
`01_parquet_to_jsonl.py` extracts `id`, `local_id`, `documents` from the PARHAF
parquet into `01_parhaf_documents.jsonl` (run from inside the folder).
`02_clean_split_texts.py` is now **audit-only**: a deterministic line cleaner kept
for its alphanumeric-loss accounting and `<unk>` character audit
(`02_parhaf_texts.jsonl` is a diagnostic artefact, no longer the audio input).
The dataset text is produced by the **rewrite path**: `03_chunk_for_rewrite.py`
strips parentheses and splits each raw document into `<=500`-char chunks
(`03_parhaf_rewrite_chunks.jsonl`), then `04_generate_texts.py` (a thin wrapper
over `utils/text_rewrite.run_rewrite`) rewrites each chunk into ONE faithful
French paragraph via the shared engine (score gate off, one paragraph per chunk)
into `generated_dataset.jsonl`.
PARHAF carries specific attribution obligations (see README, "release" below).

### `04_PARROT/` (data prepped)
`01_filter_french.py <in> <out>` keeps only French-language rows from the PARROT
radiology dataset (`PARROT_v1_0.jsonl` to `PARROT_v1_0_french.jsonl`).
`02_clean_split_texts.py` is now **audit-only** (same demotion as PARHAF). The
dataset text is produced by the **rewrite path**: `03_chunk_for_rewrite.py`
(parens stripped, chunked to `03_parrot_rewrite_chunks.jsonl`) then
`04_generate_texts.py`, the PARROT twin of the PARHAF wrapper, rewriting each
chunk into ONE faithful paragraph via the shared engine.

### `05_generate_audio/` (active)
`01_generate_audio.py` is a minimal client for the local TTS server on `:8003`
(`POST /v1/audio/speech`), simplified from `CrispASR/tts.py` in the separate
CrispASR repo. Text mode (one clip) or jsonl batch mode: each row's
`asr_training_source` becomes `{term_index}_{variant_index}_{normalized_term}.wav`
in the output folder, resumable via skip-if-exists (`--overwrite` to redo).
Backend-agnostic: covers crispasr-tts (TADA), voxtral-tts (vLLM Voxtral) and
qwen3-tts VoiceDesign; the backend choice is still being investigated. Knobs
are sent only when explicitly set so server startup defaults apply otherwise.
A clip that comes back pinned at the backend's frame cap (`--max-audio-seconds`)
stopped for LENGTH, not at the end of the text, so it is rejected instead of
written and the row counts as failed (see the frame-cap note under "known
duplication"). `tests/test_audio_truncation.py` covers that guard on both sides.

### `06_hotfixes/` (active)
Quality control over the synthesized clips: transcribe each one with Whisper, score
the transcript against the label (CER), regenerate the bad clips, keep the best draw.
- `01_compute_stt.py`: the shared core (transcription client, `normalize_for_scoring` /
  `compute_metrics` / `tail_cer`, resumable `process_file` with atomic writes and a
  duty-cycled flush). Scoring folds one written form per spoken thing (units, titles,
  percent, numbers, staging Roman numerals, and letter-spelled sigles: a chain of
  >= 3 single letters joined by `-` or `.` folds to the joined form, so Whisper's
  `G-G-T` is not an error against the label `GGT`; guards keep `a-t-il`, `5-F-U`,
  `L.M.B.R.1` and hyphen compounds untouched) so a representational difference is never
  counted as an error, reusing `utils/voxtral_normalize`'s tables rather than copying
  them. Every fold is justified by a replay over the already-scored clips (the
  letter-chain fold: 297 improved / 52 worsened, none of the 52 unfairly). Tested by
  `test_normalize_for_scoring.py`.
- `01_recursive_improvement.py`: the state machine on top of it
  (`pending_tts` -> `pending_stt` -> `improved` / `exhausted`), split into `--mode stt`
  and `--mode tts` because the STT and TTS servers cannot share the GPU. The `stt` pass
  only picks a winner from a COMPLETE candidate set (the pick is final): a set short of
  `--n-improv` goes back to `pending_tts`, where short means the tts pass did not deliver
  every draw it was asked for, not one file per draw (a duplicate / truncated draw is
  skipped deterministically, so requiring it would ping-pong forever). `--rescore` /
  `--rescore-only` re-derive stored scores (and re-judge the regeneration queue) after a
  scoring-rule change; `--retry-exhausted` additionally reopens the clips a previous run
  gave up on, which are otherwise final, dropping the marker on the ones a scoring change
  makes clean and requeueing the rest. Tested by
  `tests/test_recursive_improvement_modes.py`.
- `driver.sh`: alternates the two modes, one dataset at a time, switching the docker
  server between passes; env-var driven (`MODE`, `RESCORE`, `RESCORE_ONLY`,
  `RETRY_EXHAUSTED`, `CATEGORY`, `CER_THRESHOLD`, ...). See
  `README_alternating_improvement.md`, the stage's real doc.
  Its loop is tested by `tests/test_driver_alternate_mode.py` (runs the real script with
  the passes stubbed), which pins that an `stt` pass with nothing to do (exit 11) makes
  the alternating run continue to the `tts` pass instead of dropping the dataset, and that
  `RETRY_EXHAUSTED` rides the first `stt` pass of each dataset only (otherwise the retry
  would reopen the markers it just wrote and never converge).
- `02_statistics.py`: report over a scored file (CER by category, duration ceiling
  audit). `03_collect_suspicious.py`: copy every clip the gates flagged into one local
  folder with an index, so they can be listened to rather than trusted to a number.

### `07_acronyms/` (active)
Common medical acronyms missing from the other sources, from a Wikipedia list,
hand-filtered into `wikipedia_acronyms.filtered.authorfiltered.csv`
(`TERM,MEANING,PRONOUNCED_AS`). `02_filter_existing_terms.py` is the provenance
tool that dropped acronyms already present as dictionary/drug terms.
`03_generate_texts.py` is a thin adapter over the shared engine
(`require_score=False`): the CSV expands to ONE ENGINE ENTRY PER
(term, pronunciation) pair (`;` separates several valid pronunciations, each
getting its own `--n-texts` variants; an empty PRONOUNCED_AS derives a default
per hyphen segment, `AAA -> A-A-A`, `gamma-GT -> gamma-G-T`) into
`wikipedia_acronyms.expanded.jsonl`, which is committed and drift-checked on
re-runs because its `index` keys both the engine resume state and stage 05's
file names. The LLM writes only `asr_training_target` variants (the acronym
verbatim, exact-case word-boundary validation; exactly one variant must gloss
the MEANING in French, enforced by fuzzy containment with a cognate-word
fallback for English meanings). The `asr_training_source` derives
deterministically via the engine's `source_transform` hook (acronym ->
pronunciation, then `voxtral_normalize`), and a round-trip guard
(pronunciation substituted back must read like the normalized target) queues
misses fail-open to `roundtrip_review_queue.jsonl`. System prompt = the shared
base target prompt + `PROMPT_GENERATE_ASR_TRAINING_TARGET_ACRONYMS.md` (the
`Pronounced as:` line's semantics: it only steers articles/elision, and is
never written in a variant). Tested by `tests/test_acronyms_stage.py`
(`uv run tests/test_acronyms_stage.py`).

## Running scripts

- Scripts are self-contained **PEP 723 `uv run` scripts**: dependencies are
  declared in the `# /// script` header, so run them with `uv run <path>.py`.
  When you add an import from `utils/_pipeline_shared`, its transitive deps
  (`litellm`, `tiktoken`, `rapidfuzz`, `tenacity`, `loguru`, `click`) must be in
  that script's `dependencies` block, or the `uv run` env will be missing them.
- Some scripts resolve paths relative to their own file (`__file__`), others use
  relative CWD paths (e.g. `03_PARHAF`). When in doubt, run a script from inside
  its own stage folder.
- LLM calls go through litellm. Set the provider key in the environment
  (`OPENROUTER_API_KEY` for the default OpenRouter models, or the relevant
  provider key). Pin `--provider` on OpenRouter models for reliable prompt-cache
  hits.
- Use `python` (not `python3`) for ad-hoc scripts, per the global preference.

## Release obligations (for stage `99`)

The derived dataset must ship correct attribution. Summary (full detail in
`README.md`):
- PARHAF is dual-licensed CC BY 4.0 + Etalab 2.0: credit HealthDataHub /
  Plateforme des Donnees de Sante, link the HF dataset page and version, cite the
  paper, state both licenses in a NOTICE, note the data was modified (synthesized
  to audio), and add "no endorsement".
- Ship a "research use only, not for clinical deployment" disclaimer.
- Do **not** redistribute the PARHAF test set (embargoed); training set only.

## Known duplication and gotchas

Flagging these because avoiding silent duplication is a hard rule in this repo:

- **Cross-repo glyph rules (deliberate copy).** `01_dictionnary/02_token_check.py`
  keeps `NORMALIZATION_RULES` / `EXCLUDE_CHARS` and the `+`/`*` spoken rewrites as
  a **deliberate copy** of the same rules in `create_french_medical.py` in the
  separate `parakeet_web_phrase_boosting` repo. The two live in different repos
  and each is a standalone tool, so they are not shared through `utils/`. If you
  change one side, ask before changing the other; they must agree or the biasing
  list and the `<unk>` gate disagree about what is tokenizable.
- **`voxtral_normalize.py` vs `VOXTRAL_QUIRKS.md`.** The code is the executable
  form of the quirks table. Keep them in sync when you add or remove a TTS fix.
- **`VOXTRAL_VOICES` (deliberate copy).** `05_generate_audio/01_generate_audio.py`
  copies the 20 built-in Voxtral voice-embedding names from `CrispASR/tts.py`
  (separate repo, standalone tool). If the served voices change, update both
  sides.
- **The TTS frame cap (deliberate copy of a serving constant).** Same file keeps
  `VOXTRAL_FRAME_RATE_HZ = 12.5` and `VOXTRAL_DEFAULT_MAX_NEW_TOKENS = 4096` (a
  327.68 s ceiling, matching the served `VOXTRAL_MAX_TOKENS`, not the packaged
  2048) alongside `is_truncated`, because `/v1/audio/speech` has no
  `finish_reason`: a generation that stopped for LENGTH answers 200 with a clip cut
  off mid-sentence, and a duration pinned at the cap is the only way to see it. Stage
  05 refuses to write such a clip and stage 06 refuses to promote such a draw (both
  via `--max-audio-seconds`, which 06 imports from 05 along with `payload_duration`,
  so the check exists once). If the served `VOXTRAL_MAX_TOKENS` changes, pass the
  matching `--max-audio-seconds` / `MAX_AUDIO_SECONDS` or update the default here.
  Note the compose file's `VOXTRAL_MAX_MODEL_LEN` is shared between prompt tokens
  and generated frames, so it must stay above `VOXTRAL_MAX_TOKENS` + the prompt or
  the real cap is lower than this constant (today: 6144 vs 4096, so 4096 binds).
- **Text stages share one engine.** All five stages plug into
  `utils/text_generation_engine.py` via a `StageAdapter`; do not copy the run loop
  / dedup / stats / source-pass into a stage. Dictionary + drugs + acronyms are the
  term->N-sentences family (acronyms adds a `source_transform` and
  `require_score=False`); PARHAF + PARROT are the raw-text->one-paragraph rewrite
  family (adapter in `utils/text_rewrite.py`, `require_score=False`). Add an
  adapter, never a fork. The uncommitted builder that produces
  `02_drugs/drugs_freq_dosages.jsonl` from the raw drug files is still a TODO.
- **Rewrite paren-strip vs the cleaners' paren peel.** `utils/text_chunking.py`
  and the audit-only `02_clean_split_texts.py` cleaners each peel `(...)` with the
  same `_PAREN` regex, but the cleaner does it per-line inside a larger
  spell-out-units pass while `text_chunking` does it over the whole document; the
  surrounding logic differs, so they are not shared. If the paren rule itself
  changes, change both.

## Working agreements (from the global preferences)

- **Never duplicate code.** If shared logic appears in two stages, move it to
  `utils/` and import it. Tell the user when you spot duplication.
- **Commit small and often**: one commit per feature / per loop turn.
- **No em-dashes anywhere** (code, comments, docs, commit messages). Use commas,
  colons, or parentheses.
- Mark placeholders with `TODO` and tell the user they exist.
- When a project has tests, a bug fix should come with a test covering it.
- Never hardcode the author's GitHub username (typically in absolute paths).
- Never create/switch git branches or add entries to `.gitignore` without asking.
