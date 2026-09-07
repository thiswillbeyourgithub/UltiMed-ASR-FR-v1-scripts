# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "click",
# ]
# ///

"""
Convert pharmaceutical unique values JSON to JSONL drug database.

This script reads the output from extract_unique_values.py and creates a JSONL
database where each line represents one drug substance with its pharmaceutical
forms and dosages.
Created with assistance from aider.chat.
"""

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import click

# A dosage string is a French number followed by a unit: "500 mg", "0,5 mg",
# "1 000 mg", "2 000 000 UI". The thousands separator is a space (plain,
# non-breaking or narrow no-break), the decimal separator a comma or a dot.
_DOSAGE_RE = re.compile(
    r"^(\d{1,3}(?:[\s\u00a0\u202f]\d{3})+|\d+)"  # integer part, maybe grouped
    r"(?:([.,])(\d+))?"  # optional separator + decimal part
    r"\s*(.*)$"  # unit, whatever is left
)


def canonical_dosage(dosage: str) -> str:
    """Trim BDPM's zero padding from one dosage string.

    BDPM pads the decimal part inconsistently, and does it even when there is no
    other spelling of the same strength to disambiguate: `BENZYLTHIOURACILE`
    ships `25,00 mg` and nothing else. Fed to the LLM as a presentation hint
    that reads as "vingt-cinq virgule zero zero milligrammes", so the padding
    has to go whether or not a duplicate exists.

    Only trailing zeros of the fraction and runs of whitespace are touched. The
    integer part keeps its original grouping (`1 000 mg` stays `1 000 mg`), the
    decimal separator keeps whichever character was used, and the unit is passed
    through untouched. A string that does not parse is returned with its
    whitespace collapsed and nothing else changed.

    >>> canonical_dosage("25,0000  mg")
    '25 mg'
    >>> canonical_dosage("12,500 mg")
    '12,5 mg'
    """
    collapsed = " ".join(dosage.split())
    match = _DOSAGE_RE.match(collapsed)
    if match is None:
        return collapsed
    integer, separator, fraction, unit = match.groups()
    fraction = (fraction or "").rstrip("0")
    number = f"{integer}{separator}{fraction}" if fraction else integer
    return f"{number} {unit}" if unit else number


def dosage_key(dosage: str) -> tuple[Decimal, str] | None:
    """Canonical (value, unit) for a dosage string, or None if unparseable.

    Comparing on the numeric VALUE rather than the string collapses `500 mg`,
    `500,0 mg` and `500,00 mg` into one dose.

    The unit is only case-folded and whitespace-collapsed, deliberately not
    stripped of punctuation: `M UI` and `M.U.I.` are left as distinct units
    rather than guessed to be the same thing.
    """
    match = _DOSAGE_RE.match(" ".join(dosage.split()))
    if match is None:
        return None
    integer = re.sub(r"[\s\u00a0\u202f]", "", match.group(1))
    fraction = match.group(3) or "0"
    unit = match.group(4).casefold()
    return Decimal(f"{integer}.{fraction}"), unit


def dedupe_dosages(presentation: list[str]) -> list[str]:
    """Canonicalize and de-duplicate the dosages of one presentation list.

    A presentation is `["un comprime", "500 mg", "500,00 mg", ...]`: the article
    plus form first, then its available strengths. The first element is never
    touched. Every dosage is canonicalized, then those equal by `dosage_key`
    collapse to one entry, in first-appearance order. Empty entries are dropped.
    Unparseable entries survive, deduped only when identical, so an unexpected
    format is never silently lost.
    """
    head, rest = presentation[:1], presentation[1:]
    seen: dict[Any, str] = {}
    for dosage in rest:
        if not dosage.strip():
            continue
        key = dosage_key(dosage)
        canonical = canonical_dosage(dosage)
        # Unparseable strings key on themselves, so they only match a twin.
        bucket = key if key is not None else ("raw", canonical)
        seen.setdefault(bucket, canonical)
    return head + list(seen.values())


def dedupe_forms(forms: dict[str, list[list[str]]]) -> tuple[dict, int]:
    """Apply `dedupe_dosages` across every presentation of every form.

    Returns the cleaned forms and how many dosage entries were dropped.
    """
    cleaned: dict[str, list[list[str]]] = {}
    removed = 0
    for form, presentations in forms.items():
        out: list[list[str]] = []
        for presentation in presentations:
            deduped = dedupe_dosages(presentation)
            removed += len(presentation) - len(deduped)
            if deduped not in out:
                out.append(deduped)
        cleaned[form] = out
    return cleaned, removed


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to input JSON file from extract_unique_values.py.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Path where to write the output JSONL file.",
)
def main(input_path: Path, output_path: Path) -> None:
    """
    Convert unique values JSON to JSONL drug database.

    Reads the JSON output from extract_unique_values.py and extracts the
    denominationSubstance field, writing each drug as a single JSONL line
    with its pharmaceutical forms and dosages.

    Parameters
    ----------
    input_path : Path
        Path to input JSON file from extract_unique_values.py.
    output_path : Path
        Path to output JSONL file.
    """
    # Load input JSON file
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract denominationSubstance section
    # This contains the drug substances with their forms and dosages
    drugs = data.get("denominationSubstance", {})

    # Write to JSONL format - one drug per line
    total_removed = 0
    with output_path.open("w", encoding="utf-8") as f:
        for drug_name, forms in drugs.items():
            # BDPM lists one strength several ways ("500 mg", "500,00 mg"). Left
            # alone they all reach the LLM's presentation hint, which then reads
            # as three distinct doses of the same drug.
            forms, removed = dedupe_forms(forms)
            total_removed += removed
            # Create a JSON object for each drug with its name and forms
            drug_entry = {
                "substance": drug_name,
                "forms": forms,
            }
            # Write as single line JSON
            f.write(json.dumps(drug_entry, ensure_ascii=False) + "\n")

    click.echo(f"Successfully processed {len(drugs)} drug substances.")
    click.echo(f"Removed {total_removed} duplicate dosage entries.")
    click.echo(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
