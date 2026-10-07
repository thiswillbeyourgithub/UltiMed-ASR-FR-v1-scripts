#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "litellm", "tiktoken", "tqdm", "tenacity", "rapidfuzz", "wordfreq"]
# ///
"""Build ``manual_fix_rules.jsonl`` from the hand-written ``manual_fixes.tsv``.

Each TSV line is ``variant<TAB>replacement`` (``#`` starts a comment). The patterns come
from 02's ``variant_pattern`` and are merged per replacement exactly as 03_merge_rules.py
does, so the file has the same format as the learned rules. Consumers apply it FIRST,
before the drug and term rules, so a hand fix wins over a learned one on the same words
(``lodose`` -> ``lowdose`` instead of the learned ``Lodoz``).

Usage: uv run 08_drug_asr_rules/05_build_manual_rules.py [TSV] [OUT.jsonl]
This file was written by Claude Code.
"""
import importlib.util
import json
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("b", _HERE / "02_build_fix_rules.py")
b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b)
_mspec = importlib.util.spec_from_file_location("m", _HERE / "03_merge_rules.py")
m = importlib.util.module_from_spec(_mspec)
_mspec.loader.exec_module(m)


def build_manual_rules(lines) -> list[dict]:
    """Merged rules from ``variant<TAB>replacement`` lines.

    >>> rules = build_manual_rules(["# c", "lodose\\tlowdose", "lodoses\\tlowdose", "ceresta\\tSeresta"])
    >>> [(r.get("variants") or [r["variant"]], r["replacement"]) for r in rules]
    [(['lodose', 'lodoses'], 'lowdose'), (['ceresta'], 'Seresta')]
    >>> fix = b.compile_rules(rules)
    >>> fix("deux Lodoses et du céresta, la dose")  # a capital initial is kept
    'deux Lowdose et du Seresta, la dose'
    """
    rules = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        variant, replacement = line.split("\t")
        rules.append({"variant": variant, "pattern": b.variant_pattern(variant), "replacement": replacement})
    return [r for _, r in m.merge_rules(rules)]


@click.command()
@click.argument("tsv", type=click.Path(exists=True, path_type=Path), default=_HERE / "manual_fixes.tsv")
@click.argument("out", type=click.Path(path_type=Path), default=_HERE / "manual_fix_rules.jsonl")
def main(tsv: Path, out: Path) -> None:
    rules = build_manual_rules(tsv.read_text(encoding="utf-8").splitlines())
    with out.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rules)
    logger.info(f"{tsv} -> {out}: {len(rules)} rules")


if __name__ == "__main__":
    main()
