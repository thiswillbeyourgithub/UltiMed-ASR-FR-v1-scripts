# 08_drug_asr_rules: regex fixes for drug names the ASR model misspells

Goal: transcribe every clip of the UltiMed `drugs` subset with the fine-tuned Parakeet, collect how each drug name gets misspelled (`"mirtazapine": {"myrtazapine": 15, "mire tazapine": 6}`), and turn that into an ordered list of regex rules that fix drug names in ASR output while rarely overcorrecting.

Written with Claude Code.

## 1. Transcribe (NeMo repo, GPU)

Run it after training, when the GPU is free: the trainer uses about 22 GB of the 24 GB card, and CPU inference takes about 3.5 s per clip in fp32 (about 21 hours for the 21.9k drug clips). From the NeMo repo root (`UltiMed-ASR-FR-v1-NeMo_training_scripts`):

```bash
BS=16 ./.venv/bin/python perso/transcribe_manifests.py <final.ckpt or .nemo> drugs_hyps.jsonl \
    ../UltiMed-ASR-FR-v1-scripts/99_hf_release/data/NeMO_files/drugs/{train,val,test}.jsonl
```

It is resumable (rerun the same command after an interruption) and writes `{"audio": <abs path>, "hyp": <text>}` per line.

Caveat: the model was trained on the train split, so its train-clip errors are underestimated. val and test give the honest picture, and train mostly adds more occurrences of the same drugs.

## 2. Collect the misspellings

```bash
uv run 08_drug_asr_rules/01_extract_drug_errors.py drugs_hyps.jsonl \
    99_hf_release/data/NeMO_files/drugs/{train,val,test}.jsonl
```

This writes `08_drug_asr_rules/drug_asr_errors.json`, most errors first. Any manifest works (the oli or PARHAF sets too): every drug word in a label counts, whatever the source. The script docstring details how drug words are spotted and how a hypothesis stretch is attributed to one drug.

## 3. Build the rules

```bash
uv run 08_drug_asr_rules/02_build_fix_rules.py drugs_hyps.jsonl
```

This writes `08_drug_asr_rules/drug_fix_rules.jsonl` (one rule per line, apply top to bottom) and `drug_fix_rules.rejected.jsonl` (with the reason, for review). Scanning the 600k corpus labels for `real_text` takes about 1.5 min. The guards (`short`, `contains_target`, `other_term` (called `other_drug` in the builds below), `real_text`, `french_word`, `ambiguous`, `imprecise`, `rare`, `far`, `common_words`) and their thresholds are in the script docstring and `--help`. Both scripts take a repeated `--lexicon <file.jsonl>` (any JSONL with a `term` field, e.g. `01_dictionnary/original_dictionnary.jsonl`): the words of its terms rarer in French than `--max-term-zipf` (default 3.0, wordfreq Zipf) are tracked like drug names, so the same pipeline mines medical-term misspellings on any subset. With the dictionary the lexicon grows from 2,999 drug words to 32,094 ("syndrome de Brugada" adds `brugada`, not `syndrome`). The acronym list adds nothing: no acronym has 5 letters or more without a digit. Several models can feed one rule set: give each model's hyps file as HYPS and its 01 report with a repeated `--errors`, and `merge_reports` sums them. The `rare` and `common_words` thresholds count distinct clips (the most any single model wrote the variant), not the sum: every model transcribes the same audio, so the same slip from several models on one clip is one clip of evidence.

Apply the rules with `fix = compile_rules(rules)` then `fix(text)` from `02_build_fix_rules.py` (`load_rules(path)` reads the JSONL). `compile_rules` indexes each rule by its anchor, the longest accent-folded word of its variant, so a text only runs the few rules whose anchor it contains: same output as `apply_rules(text, rules)` (every rule in order, the reference implementation) but about 150x faster (0.11 ms against 16.9 ms per text, measured with 1.6k rules). You can also copy the patterns into any regex engine that supports lookbehind. Each pattern is case-insensitive: compile it with the `i` flag, plus `u` in JavaScript (`new RegExp(pattern, 'giu')`), and when the matched text starts with a capital and the replacement does not, capitalise the replacement (`Myrtazapine` -> `Mirtazapine`). The word bounds spell out their letter class instead of using `\w`, so the patterns behave the same in Python and JavaScript. `scripts/bench/regex-rescore.mjs` in the [parakeet-tdt-0.6b-v3-ultra-onnx](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx) repo is a JavaScript port, anchor index included (0 mismatches against Python on 41,743 transcripts).

## Committed rules (2026-10-01, all splits)

The committed `drug_fix_rules.jsonl` (14,377 rules) is learned from ALL splits of the drugs subset (train, val AND test) and from four models: `drug_asr_errors.json` is the released UltiMed model (run 1.7.0, steps 13255 and 14255 averaged), `drug_asr_errors.ultra.json` the base parakeet-ultra (fp32 ONNX), `drug_asr_errors.ultra-int8.json` / `drug_asr_errors.ultra-w4a8.json` its int8 and w4a8 ONNX builds (transcribed on CUDA with onnx-asr; the w4a8 encoder runs with the int8 decoder).

Why test is included: the rules are not a model, they are a list of fixes for every drug-name mistake the models make on this dataset. The claim they support is "fixes every seen mistake without overcorrection", so the test drug numbers below show coverage, not generalization (the held-out train+val build further down shows how well such rules generalize to unseen clips). Overcorrection is measured where it can be: on every correct label of every subset and split, and on texts the rules never saw.

The build is scripted in the parakeet-ultra ONNX model repo (`scripts/drug-rules/`, written with Claude Code): `transcribe-drugs.sh` transcribes the drugs subset with one ONNX variant, `build-rules.sh` runs 01 and 02 on each model and then both checks (`evalrules.py` and `compare-rulesets.sh`):

```bash
# from the parakeet-tdt-0.6b-v3-ultra-onnx repo; each dir holds a drugs_hyps.jsonl
TEXTS="local/voxpopuli-ignore_backups/fr-validation.txt local/voxpopuli-ignore_backups/en-validation.txt" \
  bash scripts/drug-rules/build-rules.sh <out dir> UltiMed=<dir> ultra-fp32=<dir> ultra-int8=<dir> ultra-w4a8=<dir>
```

Results (test drug clips: WER, then clips better / worse; corpora: texts the rules change, any change is an overcorrection):

| rules | rules count | UltiMed | ultra fp32 | ultra int8 | ultra w4a8 | correct labels changed (all splits) | VoxPopuli fr | VoxPopuli en |
|---|---|---|---|---|---|---|---|---|
| none | 0 | 3.69 % | 10.79 % | 11.00 % | 11.59 % | | | |
| previous (train+val, R4) | 13,626 | 2.63 % (584 / 0) | 9.12 % (811 / 0) | 9.35 % (802 / 0) | 9.96 % (795 / 0) | 40 of 602,813 | 6 of 1,662 | 20 of 1,695 |
| committed (all splits) | 14,377 | 2.04 % (864 / 0) | 8.28 % (1152 / 0) | 8.43 % (1149 / 0) | 9.04 % (1156 / 0) | 0 of 602,813 | 0 of 1,662 | 5 of 1,695 |

- The 5 English VoxPopuli changes (`artificial` -> `artificielles` x3, `strength` -> `Strensiq`, `Cuban` -> `Kuvan`) are expected: the rules are meant for French output only, which is how the benchmarks apply them.
- On boosted UltiMed int8 browser transcripts, test_drugs goes 3.56 % (no rules) -> 2.70 % (R4) -> 2.32 % (committed); FLEURS fr is unchanged and FLEURS en moves by at most 0.05 WER.
- Two 02 fixes came with this build. Words are keyed with hyphens split, like `variant_pattern` matches them, so `sous antidote` in a label now blocks the `sous-antidote` variant (`real_text`; before, it rewrote the label). A one-word variant that is a common French word (`zipf_frequency >= --max-word-zipf`, default 2.5, e.g. `tienne`) is rejected as `french_word`: such words never appeared in a medical label, but VoxPopuli caught them.
- Rejected (in `drug_fix_rules.rejected.jsonl`): far 706, short 525, common_words 317, real_text 278, imprecise 273, french_word 38, other_drug 24, ambiguous 22, contains_target 3.

### Earlier train+val build (R4), held out

Before the all-splits decision the rules were built on train+val only, keeping the test split as an honest check. Those numbers are the generalization estimate:

```bash
# per model: drugs_hyps.jsonl from step 1, drugs_hyps.trainval.jsonl = the same minus the test clips
uv run 08_drug_asr_rules/01_extract_drug_errors.py drugs_hyps.trainval.jsonl \
    99_hf_release/data/NeMO_files/drugs/{train,val}.jsonl --out <drug_asr_errors.json or drug_asr_errors.ultra.json>
uv run 08_drug_asr_rules/02_build_fix_rules.py {ultimed,ultra,ultra-int8,ultra-w4a8}/drugs_hyps.trainval.jsonl \
    --errors 08_drug_asr_rules/drug_asr_errors.json --errors 08_drug_asr_rules/drug_asr_errors.ultra.json \
    --errors 08_drug_asr_rules/drug_asr_errors.ultra-int8.json --errors 08_drug_asr_rules/drug_asr_errors.ultra-w4a8.json \
    --labels 99_hf_release/data/NeMO_files/train.jsonl --labels 99_hf_release/data/NeMO_files/val.jsonl
```

The `--labels` matter: the default `full.jsonl` includes the test labels. Result: 13,626 rules covering 51,539 train+val errors (rejected: 647 `far`, 483 `short`, 270 `common_words`, 242 `real_text`, 158 `imprecise`, 23 `ambiguous`, 23 `other_drug`, 3 `contains_target`). 1,171 of them restore more than the drug name: a word the misspelling swallowed (`soufflue oxétine` -> `sous fluoxétine`, `souvenent la vaccine` -> `sous venlafaxine`, `paraclasta` -> `par aclasta`) or a glued article (`létoposide` -> `l'étoposide`).

How the defaults were chosen: every candidate rule set was applied in full to held-out test transcripts (word error rate, clips better / worse) and to every correct test label (59,151 across the five subsets, where any change is an overcorrection). On the 2,059 test drug clips:

| rules learned from | settings | rules | UltiMed WER | parakeet-ultra WER | correct labels changed |
|---|---|---|---|---|---|
| none | | 0 | 3.73 % | 10.98 % | |
| UltiMed | `--min-count 2 --min-ratio 0` (earlier default) | 1,623 | 3.00 % (419 better, 0 worse) | | 0 |
| UltiMed | defaults | 5,372 | 2.81 % (521 / 0) | 10.08 % (475 / 0) | 0 |
| parakeet-ultra | `--min-count-words 1` | 7,210 | 2.92 % (448 / 0) | 9.49 % (731 / 0) | 5 |
| both | `--min-count-words 1` | 9,980 | 2.72 % (559 / 0) | 9.44 % (759 / 0) | 5 |
| both | defaults, before swallowed-word targets | 9,807 | 2.73 % (557 / 0) | 9.47 % (753 / 0) | 3 |
| both | defaults | 10,030 | 2.70 % (566 / 0) | 9.45 % (758 / 0) | 3 |

Adding the parakeet-ultra int8 and w4a8 ONNX transcripts (2026-10-01), scored the same way with a slightly different text normalization (so compare within this table only; int8 and w4a8 are the ultra ONNX builds):

| rules learned from | rules | UltiMed | ultra fp32 | ultra int8 | ultra w4a8 | correct labels changed |
|---|---|---|---|---|---|---|
| none | 0 | 3.69 % | 10.79 % | 11.00 % | 11.59 % | |
| UltiMed + ultra fp32 | 10,030 | 2.66 % (570 / 0) | 9.22 % (769 / 0) | 9.47 % (750 / 0) | 10.15 % (713 / 0) | 3 |
| + int8 | 11,691 | 2.65 % (577 / 0) | 9.17 % (788 / 0) | 9.41 % (778 / 0) | 10.08 % (747 / 0) | 3 |
| + int8 + w4a8 (committed) | 13,626 | 2.63 % (584 / 0) | 9.12 % (811 / 0) | 9.35 % (802 / 0) | 9.96 % (795 / 0) | 3 |

On the boosted UltiMed int8 browser transcripts the committed rules bring test_drugs from 3.56 % to 2.70 % (10,030 rules: 2.73 %) and the drug sentences from 3.59 % to 2.82 % (3.02 %); FLEURS French is unchanged and FLEURS English moves by at most 0.05 WER point (one extra rewrite, `program` -> `Prograf`; apply the rules to French only).

- `--min-count` 1 against 2 or 3 and `--min-ratio` 0 to 0.8 all changed 0 labels with the UltiMed rules, so the loosest won; the 0.5 ratio floor only drops garbled one-offs (`reea iutis aeec et are` -> `oméga`) for 3 clips.
- Before the distinct-clip count, a third model pushed the same one-off slips over `--min-count-words` (`lait unique` 3 times = 1 clip x 3 models) and the two overcorrections below came back.
- `--min-count-words 2` removes the two real overcorrections of the parakeet-ultra rules (`café au lait unique` -> `Levunique`, `alpha sur bêta estimé` -> `bétahistine`). The 3 labels still changed name a drug the label spells another way (`méronème` -> `Meronem`, `médroxy-progestérone`, `alpha-calcidol` -> `alfacalcidol`).
- On the UltiMed test_dictionary transcripts the rules also fix 50 clips (WER 2.82 % -> 2.79 %) and worsen none.
- Before `contains_target`, 3 test clips got worse (`anti-TNF-alpha` -> `alpha`).

## Tests

```bash
uv run --with click --with loguru --with litellm --with tiktoken --with tqdm --with tenacity --with rapidfuzz --with wordfreq tests/test_drug_asr_rules.py
```

## Smoke test (2026-09-29)

Run 1.7.0 step 7000 on the 205 private oli clips found 211 drug occurrences (198 drugs) and 59 misspellings, for example `phélodipine` (félodipine), `d'hexaméthasone` (dexaméthasone) and `létoposide` (target `l'étoposide`, so the article is kept). 02 kept 57 rules and rejected 2 as `real_text`.
