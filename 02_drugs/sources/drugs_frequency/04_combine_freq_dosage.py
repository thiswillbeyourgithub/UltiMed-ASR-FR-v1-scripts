"""
Enrich the scored term vocabulary (03_drug_terms_2025.jsonl) with the
dosage information from drugs_dosages.jsonl.

Each scored term carries a `substances` list (see 03_score_terms.py). For
every term we look each of its substances up in the dosage table and, when
found, attach the matching `forms` dict under a `dosages` key keyed by the
substance name as it appears in the term's `substances` list.

Matching is accent- and case-insensitive because the OPEN MEDIC substance
labels are unaccented uppercase (e.g. "PARACETAMOL") while the dosage table
keeps the accented French spelling (e.g. "PARACÉTAMOL").

Terms with no dosage match are written through unchanged (no `dosages`
key), so 04_drug_freq_dosage.jsonl is a superset of 03's rows.
"""

from pathlib import Path
import argparse
import json
import unicodedata

HERE = Path(__file__).parent
IN_TERMS = HERE / "03_drug_terms_2025.jsonl"
IN_DOSAGES = HERE / "drugs_dosages.jsonl"
OUT_COMBINED = HERE / "04_drug_freq_dosage.jsonl"


def norm(text: str) -> str:
    """Accent-stripped, uppercased key for matching substance names.

    The dosage table and the OPEN MEDIC labels disagree on accents, so we
    fold both to a common form (NFKD, drop combining marks, uppercase).
    """
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.upper().strip()


def load_dosages() -> dict[str, dict]:
    """Map normalized substance name -> its `forms` dict."""
    dosages: dict[str, dict] = {}
    with IN_DOSAGES.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            substance = (row.get("substance") or "").strip()
            forms = row.get("forms")
            if substance and forms:
                dosages[norm(substance)] = forms
    return dosages


def combine(terms_path: Path, dosages: dict[str, dict]) -> list[dict]:
    """Read scored terms and attach a `dosages` key where any substance matches."""
    combined: list[dict] = []
    with terms_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # Keyed by the substance name as it appears on the row so the
            # unaccented OPEN MEDIC spelling stays the visible label.
            found = {
                substance: dosages[norm(substance)]
                for substance in row.get("substances") or []
                if norm(substance) in dosages
            }
            if found:
                row["dosages"] = found
            combined.append(row)
    return combined


def main() -> None:
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()

    dosages = load_dosages()
    print(f"Loaded {len(dosages)} substances with dosages.")

    combined = combine(IN_TERMS, dosages)
    with_dosage = sum(1 for r in combined if "dosages" in r)

    with OUT_COMBINED.open("w", encoding="utf-8") as f:
        for row in combined:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(combined)} terms -> {OUT_COMBINED.name}")
    print(f"{with_dosage} of them got a dosage ({100 * with_dosage // max(len(combined), 1)}%).")


if __name__ == "__main__":
    main()
