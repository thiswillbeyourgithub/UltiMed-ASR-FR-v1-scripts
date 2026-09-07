#!/bin/zsh

uv run filter.py --input database.json --output outputs/filtered.json &&\
  uv run extract_unique_values.py --input outputs/filtered.json --output outputs/uniqued_filtered.json &&\
  uv run create_drug_db.py --input outputs/uniqued_filtered.json --output outputs/drug_db.jsonl


