#!/usr/bin/env bash
# Audit the cleaned PARHAF texts for characters the Parakeet TDT v3 tokenizer
# maps to <unk>, so you know what still needs normalizing. Thin wrapper over the
# shared audit tool (utils/parakeet_tokenizer.py); extra args pass through, e.g.
#   ./02_token_check.sh --max-report 0
set -euo pipefail
cd "$(dirname "$0")"
uv run ../utils/parakeet_tokenizer.py --input 02_parhaf_texts.jsonl --field text "$@"
