#!/usr/bin/env bash
set -euo pipefail

models=(
    # "openrouter/google/gemini-3.1-flash-lite"
    # "openrouter/moonshotai/kimi-k2.6"
    "openrouter/deepseek/deepseek-v4-pro"
    # "openrouter/z-ai/glm-5.1"
)

mkdir -p test_output

for model in "${models[@]}"; do
    name="${model#*/}"
    name="${name#*/}"
    log="test_output/${name}.txt"
    out="test_output/${name}.jsonl"
    uv run 03_generate_texts.py \
        --n-variants '{"0": 1, "1-3": 2,"4-6":4, "7": 7, "8-10":11}' \
        --n-jobs 1 \
        --limit 10 \
        --model="$model" \
        --output-path="$out" \
        --provider=deepseek \
        2>&1 | tee "$log"
done
