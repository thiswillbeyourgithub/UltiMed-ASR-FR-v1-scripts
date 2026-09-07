#!/usr/bin/env bash
# Audit the cleaned PARROT texts for characters the Parakeet TDT v3 tokenizer
# maps to <unk>, so you know what still needs normalizing. Thin wrapper over the
# shared audit tool (utils/parakeet_tokenizer.py); extra args pass through, e.g.
#   ./02b_token_check.sh --max-report 0
set -euo pipefail
cd "$(dirname "$0")"
uv run ../utils/parakeet_tokenizer.py --input 02_parrot_texts.jsonl --field text "$@"
