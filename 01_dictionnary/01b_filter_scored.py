#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "click",
# ]
# ///
"""Filter a scored JSONL dictionary by score value.

Examples
--------
Only scores >= 8::

    uv run filter_scored.py original_dictionnary.scored.jsonl --above 8

Scores between 5 and 8 (inclusive)::

    uv run filter_scored.py original_dictionnary.scored.jsonl --above 5 --under 9

Show score distribution without filtering::

    uv run filter_scored.py original_dictionnary.scored.jsonl --stat

Pipe filtered output to a file::

    uv run filter_scored.py original_dictionnary.scored.jsonl --above 8 > high_scores.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import click

# Default input: the stage-1 scored file next to this script, so
# `uv run 01b_filter_scored.py --stat` inspects the score distribution with no
# arguments. This is an optional inspection / subsetting tool, not a mandatory
# pipeline stage: it prints to stdout (redirect to a file to carve a subset).
HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "original_dictionnary.scored.jsonl"


@click.command()
@click.argument(
    "input_path",
    required=False,
    default=str(DEFAULT_INPUT),
    type=click.Path(exists=True, path_type=Path),
)
@click.option("--above", type=int, default=None, help="Keep only entries with score >= N.")
@click.option("--under", type=int, default=None, help="Keep only entries with score < M.")
@click.option("--stat", is_flag=True, default=False, help="Print score distribution to stderr.")
def main(
    input_path: Path,
    above: Optional[int],
    under: Optional[int],
    stat: bool,
) -> None:
    scores: list[int] = []
    filtered_lines: list[str] = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: skipping malformed JSON line", file=sys.stderr)
                continue

            score = obj.get("score")
            if score is None:
                continue
            try:
                score_val = int(score)
            except (ValueError, TypeError):
                continue

            scores.append(score_val)

            if above is not None and score_val < above:
                continue
            if under is not None and score_val >= under:
                continue

            filtered_lines.append(json.dumps(obj, ensure_ascii=False))

    if stat:
        total = len(scores)
        if total == 0:
            print("No scored entries found.", file=sys.stderr)
        else:
            counter = Counter(scores)
            print(f"{'Score':>5} | {'Count':>6} | {'%':>6}", file=sys.stderr)
            print("-" * 24, file=sys.stderr)
            for s in sorted(counter):
                count = counter[s]
                pct = 100.0 * count / total
                print(f"{s:>5} | {count:>6} | {pct:>6.2f}", file=sys.stderr)
            print("-" * 24, file=sys.stderr)
            print(f"{'Total':>5} | {total:>6} | {'100.00':>6}", file=sys.stderr)

    for out_line in filtered_lines:
        print(out_line)


if __name__ == "__main__":
    main()
