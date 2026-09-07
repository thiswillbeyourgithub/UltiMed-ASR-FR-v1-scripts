"""
Turn the per-substance and per-brand JSONL files into a single flat
vocabulary of scored terms for TTS/ASR sample weighting.

Each output row is one term (a brand name or a substance label) with a
`score` in [0, 100]: the log of its combined sales, min-max normalized in
log space so the single most-sold term is 100 and the least-sold is 0.
The log makes the slope between the two ends logarithmic rather than
linear, and the score is stored as an int.

Every row also carries a `substances` list naming the active substance(s)
the term maps to (its own label for substance rows, the source
`substances` list for brand rows). Downstream 04_combine_freq_dosage.py
uses this to look each term's dosage up by substance.

Also prints a terminal histogram of how many terms fall in each score bin.
"""

from pathlib import Path
import argparse
import json
import math
import re

HERE = Path(__file__).parent
IN_SUBSTANCE = HERE / "02_drugs_by_substance_2025.jsonl"
IN_BRAND = HERE / "02_drugs_by_brand_2025.jsonl"
OUT_TERMS = HERE / "03_drug_terms_2025.jsonl"

# Where the human-readable term lives in each source file.
NAME_KEY = {"substance": "atc5_label", "brand": "brand"}
CATEGORY = "drugs"  # constant `category` key stamped on every output row
SCORE_MAX = 10  # scores span [0, SCORE_MAX]

# French connector words that appear inside `substances` strings (e.g.
# "PARACETAMOL EN ASSOCIATION AVEC DES ...") and must not count as a
# substance-name match when deciding whether a brand is redundant.
STOPWORDS = {
    "EN", "ET", "AVEC", "DES", "DE", "DU", "LA", "LE", "LES", "AU", "AUX",
    "ASSOCIATION", "AUTRES",
}


def words(text: str) -> set[str]:
    """Uppercased word tokens, dropping connectors and tokens under 3 chars."""
    toks = re.split(r"[^0-9A-Za-zÀ-ÿ]+", text.upper())
    return {t for t in toks if len(t) >= 3 and t not in STOPWORDS}


def read_terms() -> list[dict]:
    """Load both sources into raw {term, type, total} dicts (total > 0 only)."""
    terms: list[dict] = []
    skipped = 0
    redundant = 0
    for kind, path in (("substance", IN_SUBSTANCE), ("brand", IN_BRAND)):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                term = (row.get(NAME_KEY[kind]) or "").strip()
                total = row.get("total") or 0
                if not term or total <= 0:
                    skipped += 1
                    continue
                # A substance row maps to itself; a brand row maps to the
                # active substance(s) listed in the source.
                if kind == "brand":
                    substances = [s.strip() for s in (row.get("substances") or []) if s.strip()]
                    # Drop brands that merely restate their active substance,
                    # e.g. "FLUOXETINE MILAN" -> already covered by "FLUOXETINE".
                    sub_words = words(" ".join(substances))
                    if words(term) & sub_words:
                        redundant += 1
                        continue
                else:
                    substances = [term]
                terms.append({
                    "term": term,
                    "type": kind,
                    "total": total,
                    "substances": substances,
                })
    if skipped:
        print(f"Skipped {skipped} rows with no term or total <= 0.")
    if redundant:
        print(f"Dropped {redundant} brands that restate their substance.")
    return terms


def score_terms(terms: list[dict]) -> list[dict]:
    """Add an int `score` in [0, SCORE_MAX] = log(total) min-max normalized (global)."""
    logs = [math.log(t["total"]) for t in terms]
    lo, hi = min(logs), max(logs)
    span = hi - lo
    scored: list[dict] = []
    for t, lg in zip(terms, logs):
        score = SCORE_MAX * (lg - lo) / span if span else SCORE_MAX
        scored.append({
            "term": t["term"],
            "type": t["type"],
            "category": CATEGORY,
            "score": int(round(score)),
            "substances": t["substances"],
        })
    return scored


def print_histogram(scored: list[dict]) -> None:
    """Terminal histogram of term counts per integer score in [0, SCORE_MAX]."""
    counts = [0] * (SCORE_MAX + 1)
    for t in scored:
        counts[t["score"]] += 1

    peak = max(counts) or 1
    width = 50
    print(f"\nScore distribution ({len(scored)} terms):")
    for score, c in enumerate(counts):
        bar = "#" * round(c / peak * width)
        print(f"  {score:2d} | {bar} {c}")


def main() -> None:
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()

    terms = read_terms()
    print(f"Loaded {len(terms)} terms from both sources.")
    scored = score_terms(terms)
    scored.sort(key=lambda t: t["score"], reverse=True)

    with OUT_TERMS.open("w", encoding="utf-8") as f:
        for t in scored:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"Wrote {len(scored)} terms -> {OUT_TERMS.name}")
    if scored:
        print(f"Top: {scored[0]}")
        print(f"Bottom: {scored[-1]}")

    print_histogram(scored)


if __name__ == "__main__":
    main()
