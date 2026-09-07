> **Historical note.** These are the original project notes for the standalone
> `base_de_donnee_medicament` project, kept for provenance. Steps 1 and 2 below are
> still accurate and are what this folder does. Steps 3 to 5 describe an early plan
> that was replaced: sentence generation moved to `02_drugs/01_generate_drug_texts.py`
> on the shared engine, and the TTS is a local Voxtral server rather than OpenAI.
> This file was named `CLAUDE.md` upstream and is renamed here so it is not picked up
> as live instructions.

# Drug Database for ASR Finetuning

## Project Goal

Build a pipeline to finetune an ASR (Automatic Speech Recognition) model (Parakeet)
so it better understands doctors dictating medication notes in French.

### Pipeline

1. **Extract & filter** pharmaceutical data from French drug databases (`filter.py`, `extract_unique_values.py`)
2. **Build drug DB** — `create_drug_db.py` produces `outputs/drug_db.jsonl` (410 drugs with forms/dosages)
3. **Generate sentences** — `drug_db_to_text.py` calls an LLM (via litellm) to produce N realistic
   dictation-style sentences per drug, saved as JSONL. One LLM call per drug for token efficiency.
   The output uses XML format with a `<thinking>` section (discarded) and `<sentences>` section.
   For each drug, Python randomizes which pharmaceutical presentation (form + dosage) is included
   in the prompt, ensuring variety across the N sentences (one sentence always omits presentation).
   The LLM must never abbreviate drug names.
4. **TTS** — (future) Use OpenAI TTS models (11 voices) to generate audio from those sentences.
5. **Finetune** — (future) Finetune Parakeet ASR on the generated audio.

### Data shapes

`outputs/drug_db.jsonl` — one JSON object per line:
```json
{"substance": "ÉZÉTIMIBE", "forms": {"comprimé": [["un comprimé", "10 mg"]], "gélule": [["une gélule", "10 mg"]]}}
```
Each form key (e.g. "comprimé") maps to a list of presentation lists. Each presentation list
starts with an article+form string (e.g. "un comprimé") followed by available dosages.

`outputs/drug_sentences.jsonl` — output of `drug_db_to_text.py`, one JSON object per line:
```json
{"substance": "ÉZÉTIMIBE", "sentences": ["sentence1", "sentence2", ...]}
```

## Tech choices

- Python, single-file scripts with PEP 723 inline metadata (run via `uv run`)
- CLI: `click`, logging: `loguru`, LLM calls: `litellm`
- Default model: `openrouter/anthropic/claude-sonnet-4-5` via OpenRouter
- Created with assistance from Claude Code.
