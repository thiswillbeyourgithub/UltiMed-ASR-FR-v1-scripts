# 05_generate_audio

Stage 05 of the pipeline: speak the text produced by stages 01 to 04 with the
local TTS server. Both candidate services bind `:8003` (only one up at a
time) and the client covers whichever answers: `crispasr-tts` (HumeAI TADA 3B
multilingual), `voxtral-tts` (vLLM serving mistralai/Voxtral-4B-TTS-2603), or
a qwen3-tts VoiceDesign deployment. The final backend choice is still under
investigation. Written with Claude Code.

## `01_generate_audio.py`

A minimal client for `POST /v1/audio/speech`, simplified from the full
`CrispASR/tts.py` (which absorbed the old voxtral query client). Run it with
`uv run 01_generate_audio.py` (PEP 723 script).

Two modes, chosen by `--input`:

- **Text mode**: `--input "some sentence"` synthesizes once and writes the
  audio to `--output` (a file, default `tts_out.wav`).
- **Batch mode**: `--input path/to/file.jsonl` reads one JSON object per line,
  synthesizes each row's `asr_training_source`, and writes into the
  `--output` folder as `{term_index}_{variant_index}_{normalized_term}.wav`
  (term slug: accents stripped, lowercased, non-alphanumerics to `_`).

Batch runs are resumable: existing files are skipped unless `--overwrite` is
given, a tqdm bar tracks only the remaining work, per-row server errors are
logged and skipped (exit code 1 if any row failed), and an unreachable server
aborts the run so you can re-run to resume. Rows missing the required keys
(`term_index`, `variant_index`, `term`, `asr_training_source`) are logged and
counted as bad.

Sanity check the server first:

```bash
uv run 01_generate_audio.py --check
```

Typical batch run over the stage 01 output:

```bash
uv run 01_generate_audio.py \
  --input ../01_dictionnary/generated_dataset.jsonl \
  --output ./audio \
  --seed 42
```

Per-request knobs (`--instruction`, `--voice`, `--seed`, `--speed`,
`--model`, and the talker sampling flags `--temperature`, `--top-p`,
`--top-k`, `--repetition-penalty`, `--do-sample`) are sent only when set;
anything unset keeps the server's startup default. That contract is also what
keeps TADA-only fields out of Voxtral requests: Voxtral honours only
`input` / `voice` / `response_format` / `speed` per request (its sampling and
seed are startup env vars). Backend cheat sheet:

- **TADA (crispasr-tts)**: all knobs work; `--voice` takes a
  `tada-ref-*.gguf` path or registry name.
- **Voxtral (voxtral-tts)**: pass `--voice` one of the 20 built-in embedding
  names (`fr_male`, `fr_female`, `neutral_female`, ...; see `VOXTRAL_VOICES`
  in the script, a deliberate copy from `CrispASR/tts.py`), leave the
  sampling flags unset, and add `--model mistralai/Voxtral-4B-TTS-2603` if
  vLLM rejects a modelless request. Formats: `wav/flac/mp3/aac/opus/pcm`.
- **Qwen3 VoiceDesign**: `--instruction` carries the voice-direction prose.

Clip durations (stdlib `wave`, wav responses only) are logged per file at
DEBUG level and totalled in the batch summary. Logs go to stderr and are
appended to `tts_generate.log` next to the script (override with
`--log-file`).
