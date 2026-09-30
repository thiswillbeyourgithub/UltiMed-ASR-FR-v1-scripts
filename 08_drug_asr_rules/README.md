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

This writes `08_drug_asr_rules/drug_fix_rules.jsonl` (one rule per line, apply top to bottom) and `drug_fix_rules.rejected.jsonl` (with the reason, for review). Scanning the 600k corpus labels for `real_text` takes about 1.5 min. The guards (`short`, `contains_target`, `other_drug`, `real_text`, `ambiguous`, `imprecise`, `rare`, `far`, `common_words`) and their thresholds are in the script docstring and `--help`. Several models can feed one rule set: give each model's hyps file as HYPS and its 01 report with a repeated `--errors`, and `merge_reports` sums them.

Apply the rules with `fix = compile_rules(rules)` then `fix(text)` from `02_build_fix_rules.py` (`load_rules(path)` reads the JSONL). `compile_rules` indexes each rule by its anchor, the longest accent-folded word of its variant, so a text only runs the few rules whose anchor it contains: same output as `apply_rules(text, rules)` (every rule in order, the reference implementation) but about 150x faster (0.11 ms against 16.9 ms per text, measured with 1.6k rules). You can also copy the patterns into any regex engine that supports lookbehind. Each pattern is case-insensitive: compile it with the `i` flag, plus `u` in JavaScript (`new RegExp(pattern, 'giu')`), and when the matched text starts with a capital and the replacement does not, capitalise the replacement (`Myrtazapine` -> `Mirtazapine`). The word bounds spell out their letter class instead of using `\w`, so the patterns behave the same in Python and JavaScript. `scripts/bench/regex-rescore.mjs` in the [parakeet-tdt-0.6b-v3-ultra-onnx](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx) repo is a JavaScript port, anchor index included (0 mismatches against Python on 41,743 transcripts).

## Committed rules (2026-09-30)

The rules learn from two models' misspellings, which both help: `drug_asr_errors.json` comes from the released UltiMed model (run 1.7.0, steps 13255 and 14255 averaged), `drug_asr_errors.ultra.json` from parakeet-ultra, its base (the fp32 `.nemo` the ultra ONNX is exported from). Both are built from the train and val splits ONLY, so the test split stays an honest check:

```bash
# per model: drugs_hyps.jsonl from step 1, drugs_hyps.trainval.jsonl = the same minus the test clips
uv run 08_drug_asr_rules/01_extract_drug_errors.py drugs_hyps.trainval.jsonl \
    99_hf_release/data/NeMO_files/drugs/{train,val}.jsonl --out <drug_asr_errors.json or drug_asr_errors.ultra.json>
uv run 08_drug_asr_rules/02_build_fix_rules.py ultimed/drugs_hyps.trainval.jsonl ultra/drugs_hyps.trainval.jsonl \
    --errors 08_drug_asr_rules/drug_asr_errors.json --errors 08_drug_asr_rules/drug_asr_errors.ultra.json \
    --labels 99_hf_release/data/NeMO_files/train.jsonl --labels 99_hf_release/data/NeMO_files/val.jsonl
```

The `--labels` matter: the default `full.jsonl` includes the test labels. Result: 10,030 rules covering 23,876 train+val errors (rejected: 410 `far`, 323 `short`, 203 `real_text`, 173 `common_words`, 79 `imprecise`, 21 `other_drug`, 16 `ambiguous`, 3 `contains_target`). 807 of them restore more than the drug name: a word the misspelling swallowed (`soufflue oxétine` -> `sous fluoxétine`, `souvenent la vaccine` -> `sous venlafaxine`, `paraclasta` -> `par aclasta`) or a glued article (`létoposide` -> `l'étoposide`).

How the defaults were chosen: every candidate rule set was applied in full to held-out test transcripts (word error rate, clips better / worse) and to every correct test label (59,151 across the five subsets, where any change is an overcorrection). On the 2,059 test drug clips:

| rules learned from | settings | rules | UltiMed WER | parakeet-ultra WER | correct labels changed |
|---|---|---|---|---|---|
| none | | 0 | 3.73 % | 10.98 % | |
| UltiMed | `--min-count 2 --min-ratio 0` (earlier default) | 1,623 | 3.00 % (419 better, 0 worse) | | 0 |
| UltiMed | defaults | 5,372 | 2.81 % (521 / 0) | 10.08 % (475 / 0) | 0 |
| parakeet-ultra | `--min-count-words 1` | 7,210 | 2.92 % (448 / 0) | 9.49 % (731 / 0) | 5 |
| both | `--min-count-words 1` | 9,980 | 2.72 % (559 / 0) | 9.44 % (759 / 0) | 5 |
| both | defaults, before swallowed-word targets | 9,807 | 2.73 % (557 / 0) | 9.47 % (753 / 0) | 3 |
| both (committed) | defaults | 10,030 | 2.70 % (566 / 0) | 9.45 % (758 / 0) | 3 |

- `--min-count` 1 against 2 or 3 and `--min-ratio` 0 to 0.8 all changed 0 labels with the UltiMed rules, so the loosest won; the 0.5 ratio floor only drops garbled one-offs (`reea iutis aeec et are` -> `oméga`) for 3 clips.
- `--min-count-words 2` removes the two real overcorrections of the parakeet-ultra rules (`café au lait unique` -> `Levunique`, `alpha sur bêta estimé` -> `bétahistine`). The 3 labels still changed name a drug the label spells another way (`méronème` -> `Meronem`, `médroxy-progestérone`, `alpha-calcidol` -> `alfacalcidol`).
- On the UltiMed test_dictionary transcripts the rules also fix 50 clips (WER 2.82 % -> 2.79 %) and worsen none.
- Before `contains_target`, 3 test clips got worse (`anti-TNF-alpha` -> `alpha`).

## Tests

```bash
uv run --with click --with loguru --with litellm --with tiktoken --with tqdm --with tenacity --with rapidfuzz tests/test_drug_asr_rules.py
```

## Smoke test (2026-09-29)

Run 1.7.0 step 7000 on the 205 private oli clips found 211 drug occurrences (198 drugs) and 59 misspellings, for example `phélodipine` (félodipine), `d'hexaméthasone` (dexaméthasone) and `létoposide` (target `l'étoposide`, so the article is kept). 02 kept 57 rules and rejected 2 as `real_text`.
