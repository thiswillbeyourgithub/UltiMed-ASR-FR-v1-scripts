
*Here, I intend to keep track of how I'm creating and curating the dataset for ASR finetuning*
# Dataset creation logs

---

# 01_dictionnary

## filtering

Generated using:

```bash
uv run 01_llm_scoring.py original_dictionnary.jsonl --model "openrouter/openai/gpt-oss-120b" --resume --limit 1000
```

- **Entries:** 62601
- **Estimated input tokens:** ~47,924,801 (62601 entries × ~628 system tokens + entry text)

### Score distribution (after scoring 62569 entries)

| Score | Count | % |
|------:|------:|------:|
| 0 | 2494 | 3.99 |
| 1 | 794 | 1.27 |
| 2 | 4601 | 7.35 |
| 3 | 1332 | 2.13 |
| 4 | 1459 | 2.33 |
| 5 | 4233 | 6.77 |
| 6 | 7916 | 12.65 |
| 7 | 6714 | 10.73 |
| 8 | 11362 | 18.16 |
| 9 | 21652 | 34.60 |
| 10 | 12 | 0.02 |
| **Total** | **62569** | **100.00** |

We ended up *not* filtering out low-score entries. Instead, the score is fed
into `03_generate_texts.py` to decide how many text samples to generate per
term (via the `--n-variants` range-policy), so lower-quality entries simply
contribute fewer variants rather than being dropped entirely.

## Text generation

Generated the texts with:
```
uv run 03_generate_texts.py --n-jobs 32 --model="openrouter/deepseek/deepseek-v4-pro" --provider="deepseek" --n-variants '{"0-1":2,"2-3":4,"4":5,"5":6,"6":7,"7":8,"8-10":11}' -v
```

With a cache hit rate of about 95%, the total cost to transform the 62 461 terms into 534 441 text samples cost between $50 and $60.

# Audio generation

I decided to only generate audios using the female voice as it's very high quality, subjectively much higher than `fr_male`.
```
uv run ../05_generate_audio/01_generate_audio.py --input generated_dataset.jsonl --output voxtral_audios_euler_and_cfg_max --voice "fr_female" --seed 42 --concurrency=10
```

---

# 02_DRUGS

## Text generation

```bash
uv run 01_generate_drug_texts.py --n-jobs 32 --model="openrouter/deepseek/deepseek-v4-pro" --provider="deepseek" --n-variants '{"0-1":2,"2-3":4,"4":5,"5":6,"6":7,"7":8,"8-10":11}' -v
```

Same engine and score policy as the dictionary stage: reads `drugs_freq_dosages.jsonl`, emits N `asr_training_target` variants per drug (each must contain the drug name) plus the deterministic `asr_training_source`, into `generated_dataset.jsonl`.

---

# 03_PARHAF

Source: [HealthDataHub/PARHAF on Hugging Face](https://huggingface.co/datasets/HealthDataHub/PARHAF).

Dual-licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and [Etalab Open License 2.0](https://github.com/etalab/licence-ouverte/blob/master/LO.md). Both allow commercial use, modification, and redistribution of derivative works (text-to-speech audio counts as a derivative). The only real obligation is attribution.

## What I have to do when releasing the derived ASR dataset

- **Credit the source.** Name the licensor (HealthDataHub / Plateforme des Données de Santé), link back to the [HF dataset page](https://huggingface.co/datasets/HealthDataHub/PARHAF), and mention the date / version of PARHAF used.
- **Cite the paper.** PARHAF, arXiv [2603.20494](https://arxiv.org/abs/2603.20494) (BibTeX key `parhaf2025` on the dataset card).
- **State both licenses** of the upstream data (CC BY 4.0 + Etalab 2.0), keep them in a NOTICE / README shipped with the derived dataset, and indicate that the data has been modified (text synthesized to audio), as CC BY 4.0 requires.
- **No endorsement.** Do not suggest HealthDataHub endorses the derived dataset or any model trained on it (explicit requirement of Etalab 2.0).
- **No clinical claims.** PARHAF's dataset card lists clinical decision-making, clinical validation, performance claims and clinical deployment as out-of-scope uses. The release must carry a "research use only, not for clinical deployment" disclaimer.
- **Do not redistribute the test set.** Only the training set is publicly released; test sets are under embargo.

## parquet to jsonl

```bash
uv run 01_parquet_to_jsonl.py
```

Reads `train-00000-of-00001.parquet` and writes `01_parhaf_documents.jsonl`, keeping only `id`, `local_id` and `documents` per row (4254 entries).

Then `02_clean_split_texts.py` turns that into `02_parhaf_texts.jsonl`: one coherent longish French text per entry, line-filtered (keeps lines ending in `.`/`?`, no colon, at least 15 chars, not all-caps, starting with a capital then a non-capital), parentheses and `°` normalized away, contiguous survivors joined and blocks under 80 chars dropped. Each entry gets `id = {orig_id}-t{text_index}-s{split_index}` and `category = "parhaf"`. This file is now audit-only; the dataset text comes from the rewrite path below.

## Text generation

Chunk the raw documents, then rewrite each chunk into one faithful French paragraph (fixed one paragraph per chunk, so no `--n-variants`):

```bash
uv run 03_chunk_for_rewrite.py
uv run 04_generate_texts.py --n-jobs 32 --model="openrouter/deepseek/deepseek-v4-pro" --provider="deepseek" -v
```

The rewrite prompt is the shared base rules plus a PARHAF clinical-document example (`PROMPT_REWRITE_PARHAF_EXAMPLES.md`), written into `generated_dataset.jsonl`.

### The 40 chunks the length cap rejected: `--no-max-len-ratio`

That run left 40 chunks out of 43456 unwritten, all logged to `term_missing_skips.jsonl` as `validation_after_retries`. Every one of them failed the same validator, the rewrite length upper bound (`MAX_LEN_RATIO = 2.0` in `utils/text_rewrite.py`), at 201% to 318% of the source length.

They are not bad rewrites. They are the chunks that are almost pure shorthand, lab panels and biology tables (half of them are over 10% digit characters), where the mandatory spell-out mechanically triples the character count: `Na+ 142meq/L; K+ 3,6meq/L` becomes "le sodium est à cent quarante-deux milliéquivalents par litre, le potassium à trois virgule six milliéquivalents par litre". No faithful rewrite of those fits under any cap worth keeping for the other 99.9% of chunks.

So the cap is turned off for a targeted re-run instead of being raised. `--no-max-len-ratio` drops the upper bound only, the lower bound (too much content dropped) still applies. The run resumes from `generated_dataset.jsonl`, so it picks up exactly those 40 pending chunks and nothing else:

```bash
uv run 04_generate_texts.py --no-max-len-ratio --n-jobs 8 --model="openrouter/deepseek/deepseek-v4-pro" --provider="deepseek" -v
```

`03_PARHAF/skipped_chunks_inspect.jsonl` (uncommitted, rebuildable) holds those 40 source chunks with their failure ratios, for eyeballing what they look like. The same flag exists on `04_PARROT/04_generate_texts.py`, which shares the rewrite engine.

---

# 04_PARROT

Source: [PARROT radiology reports](https://doi.org/10.1016/j.ejrai.2025.100066), French rows only. Source link and license note are done, in `99_hf_release/NOTICE.md`: CC BY-NC-SA 4.0, shipped as a separate non-commercial eval-only subset.

## Text generation

`01_filter_french.py` keeps the French rows, then the same chunk + rewrite path as PARHAF (one paragraph per chunk):

```bash
uv run 03_chunk_for_rewrite.py
uv run 04_generate_texts.py --n-jobs 32 --model="openrouter/deepseek/deepseek-v4-pro" --provider="deepseek" -v
```

Same shared base rules, but with a PARROT radiology example (`PROMPT_REWRITE_PARROT_EXAMPLES.md`), written into `generated_dataset.jsonl`.

---

# 99_hf_release

Final pipeline stage: assemble the generated audio + transcripts into NeMo ASR
manifests, then publish to Hugging Face. Both the manifest builders below and the
Parquet packaging / upload are now done. Written with Claude Code.

## NeMo-format manifests

Stage `05_generate_audio` only writes the audio clips (one `.flac` per row, named
`{term_index:06d}_{variant_index:04d}_{normalized_term}.flac`); it emits no
manifest. Two scripts here turn each stage's `generated_dataset.jsonl` + its clips
into the **NeMo-format JSONL** the fine-tuning pipeline in `../NeMo/` consumes
(one object per line, `{"audio_filepath": "...", "duration": 3.21, "text": "..."}`,
plus a few provenance fields NeMo ignores). The shared split logic lives in
`utils/nemo_manifest.py` so neither script forks it; it is covered by
`tests/test_nemo_manifest.py`.

### Per dataset: `01_build_nemo_manifest.py`

Writes `data/NeMO_files/<name>/{full,train,val,test}.jsonl`.

```bash
uv run 01_build_nemo_manifest.py --all           # all four known stages
# or one explicitly:
uv run 01_build_nemo_manifest.py \
    --input ../01_dictionnary/generated_dataset.jsonl \
    --audio-dir data/dictionary --name dictionary
```

- `text` is `asr_training_target` (the written label), not the TTS source.
- Clips are matched to rows by the stage-05
  `{term_index:06d}_{variant_index:04d}` filename prefix (reusing that naming, not
  re-deriving the slug); rows whose audio is missing are dropped.
- `duration` is read from the FLAC header (`soundfile`) and cached per dataset in
  `.duration_cache.json`, so re-runs are cheap.
- `audio_filepath` is stored **relative** to the manifest, never absolute: an
  absolute path under the audio SSD would bake in the home/username.
- `--split` (default `80/10/10`) sets the ratio; a `0` disables that split.
- `--stratify` (default on) keeps groups sensible. dictionary/drugs term variants
  are *distributed* (>=1 variant in train; >=1 in every split when the term has
  >=3 variants); PARHAF/PARROT document chunks are *atomic* (all chunks of one
  `source_id` land in the same split, so no document leaks train<->test).
  Auto-detected: rows with a `source_id` are atomic-by-document, the rest
  distribute-by-term.
- `--stratify-duration` (default on) balances the split by summed clip duration
  instead of row count.
- `--strict-coverage` / `--best-effort-coverage` (default best-effort) trades term
  coverage against ratio accuracy. Best-effort guarantees only >=1 variant in
  train, so val/test hit 80/10/10 (dictionary and drugs both land at 10% by
  duration). Strict additionally forces every >=3-variant term into all three
  splits: each split then sees the term, but that floors val/test above 10%
  (drugs ~15%). The atomic stages (PARHAF/PARROT) are unaffected either way.

### Combined: `02_combine_nemo_manifests.py`

Reads every per-dataset `full.jsonl` and writes the release-wide
`data/NeMO_files/{full,train,val,test}.jsonl`, re-basing each `audio_filepath` for
the shallower parent location (a plain concat would point one `../` too far).

```bash
uv run 02_combine_nemo_manifests.py                   # global duration re-split
uv run 02_combine_nemo_manifests.py --no-stratify-duration   # plain concat
uv run 02_combine_nemo_manifests.py --exclude PARROT  # leave a subset out of the combine
```

`--stratify-duration` (default on) re-derives the split over the pooled rows so the
combined manifest hits the target ratio by **total audio duration** (the proxy for
how much speech each split holds), staying group-aware (atomic documents whole,
term guarantees preserved). Off just concatenates the per-dataset splits and
reports the duration skew. The same `--strict-coverage` / `--best-effort-coverage`
(default best-effort) applies to the global re-split. `--exclude NAME` (repeatable)
leaves a named subdataset out of the combined manifests, so a differently-licensed
subset (the release ships PARROT as a separate, test-only CC BY-NC-SA 4.0 subset,
built with `--split 0/0/100`) can be kept separable while the rest stays under one
licence.

### Statistics: `scripts/get_statistics.py`

Prints a Markdown report over the manifests: character / word / duration
min/mean/median/max, file and duration counts, and a duration histogram, broken
down per split, per dataset, and per dataset x split.

```bash
uv run scripts/get_statistics.py                    # per-dataset splits, to stdout
uv run scripts/get_statistics.py --source combined  # the release-wide re-split
uv run scripts/get_statistics.py --output stats.md
```

## Release obligations

This stage also carries the attribution / license / disclaimer requirements
already documented in the **03_PARHAF** section above (credit the source, cite
the paper, state both licenses in a NOTICE, mark the data as modified, no
endorsement, research-use-only disclaimer, do **not** redistribute the PARHAF
test set). PARROT's source link and license note are done, see `99_hf_release/NOTICE.md`.

---

# 07_acronyms

`wikipedia_acronyms.csv` (columns `ACRONYM,MEANING`) is scraped from the French
Wikipedia article
[Liste d'abréviations en santé](https://fr.wikipedia.org/wiki/Liste_d%27abr%C3%A9viations_en_sant%C3%A9),
one row per `* SIGLE : signification` entry across its A-Z (plus Symboles /
Chiffres / Lettres grecques) sections. The wikitext was fetched via the MediaWiki
`action=raw` API and its markup (links, templates, `<ref>` tags, italics) stripped
with `mwparserfromhell`. Wikipedia text is licensed CC BY-SA; credit the article
if this list ships in the release. Built with Claude Code.
