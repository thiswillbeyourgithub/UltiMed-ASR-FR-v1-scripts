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

This writes `08_drug_asr_rules/drug_fix_rules.json`: `rules` (ordered, apply top to bottom) and `rejected` (with the reason, for review). Scanning the 600k corpus labels for `real_text` takes about 1.5 min. The guards (`short`, `contains_target`, `other_drug`, `real_text`, `ambiguous`, `imprecise`, `rare`) and their thresholds are in the script docstring and `--help`.

Apply the rules with `apply_rules(text, rules)` from `02_build_fix_rules.py`, or copy the patterns into any regex engine that supports lookbehind. Each pattern is case-insensitive: compile it with the `i` flag, plus `u` in JavaScript (`new RegExp(pattern, 'giu')`), and when the matched text starts with a capital and the replacement does not, capitalise the replacement (`Myrtazapine` -> `Mirtazapine`). The word bounds spell out their letter class instead of using `\w`, so the patterns behave the same in Python and JavaScript. `regex-rescore.mjs` in the [parakeet-tdt-0.6b-v3-optimized-onnx](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-optimized-onnx) repo is a JavaScript port.

## Committed rules (2026-09-30)

`drug_asr_errors.json` and `drug_fix_rules.json` come from the released UltiMed model (run 1.7.0, steps 13255 and 14255 averaged) over the drugs subset, built from the train and val splits ONLY so the test split stays an honest check:

```bash
uv run 08_drug_asr_rules/01_extract_drug_errors.py drugs_hyps.jsonl 99_hf_release/data/NeMO_files/drugs/{train,val}.jsonl
# drugs_hyps.trainval.jsonl: drugs_hyps.jsonl minus the test clips (02 measures precision on it)
uv run 08_drug_asr_rules/02_build_fix_rules.py drugs_hyps.trainval.jsonl \
    --labels 99_hf_release/data/NeMO_files/train.jsonl --labels 99_hf_release/data/NeMO_files/val.jsonl
```

The `--labels` matter: the default `full.jsonl` includes the test labels. Result: 1,623 rules covering 5,955 of the 9,963 train+val errors (rejected: 3,974 `rare`, 161 `short`, 133 `real_text`, 110 `ambiguous`, 51 `contains_target`, 20 `other_drug`, 19 `imprecise`).

On the 2,059 unseen test clips (3,496 drug mentions), the rules cut misspelled drug names from 1,193 to 755 and word error rate from 3.73 % to 3.00 %: 419 clips improve, none gets worse. Before `--min-count 2` and `contains_target`, 3 test clips got worse (`anti-TNF-alpha` -> `alpha`).

## Tests

```bash
uv run --with click --with loguru --with litellm --with tiktoken --with tqdm --with tenacity --with rapidfuzz tests/test_drug_asr_rules.py
```

## Smoke test (2026-09-29)

Run 1.7.0 step 7000 on the 205 private oli clips found 211 drug occurrences (198 drugs) and 59 misspellings, for example `phélodipine` (félodipine), `d'hexaméthasone` (dexaméthasone) and `létoposide` (target `l'étoposide`, so the article is kept). 02 kept 57 rules and rejected 2 as `real_text`.
