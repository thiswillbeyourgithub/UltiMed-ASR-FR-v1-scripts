# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Filter wikipedia_acronyms.csv down to acronyms NOT already present as a *term*
in the dataset.

We only compare against the `term` field of the term-source files (dictionary +
drugs), i.e. an acronym is dropped only when it IS itself a dataset term, never
when it merely appears inside some term's definition / generated text.

Matching is exact after casefolding + whitespace collapse, so "AAA" matches a
term "aaa" but not a term that merely contains "AAA".

Written with Claude Code.
"""
import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Source-of-truth term lists (the `term` field only).
TERM_FILES = [
    ROOT / "01_dictionnary" / "original_dictionnary.jsonl",
    ROOT / "02_drugs" / "drugs_freq_dosages.jsonl",
]

IN_CSV = HERE / "wikipedia_acronyms.csv"
OUT_CSV = HERE / "wikipedia_acronyms.filtered.csv"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


def load_terms() -> set[str]:
    terms: set[str] = set()
    for path in TERM_FILES:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                term = json.loads(line).get("term")
                if term:
                    terms.add(norm(term))
    return terms


def main() -> None:
    terms = load_terms()

    with IN_CSV.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)

    kept, dropped = [], []
    for row in rows:
        acronym = row[0]
        (dropped if norm(acronym) in terms else kept).append(row)

    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(kept)

    print(f"term universe: {len(terms)} unique terms from {len(TERM_FILES)} files")
    print(f"input:   {len(rows)} acronyms")
    print(f"dropped: {len(dropped)} already a dataset term")
    print(f"kept:    {len(kept)} -> {OUT_CSV.name}")
    if dropped:
        preview = ", ".join(r[0] for r in dropped[:20])
        print(f"examples dropped: {preview}")


if __name__ == "__main__":
    main()
