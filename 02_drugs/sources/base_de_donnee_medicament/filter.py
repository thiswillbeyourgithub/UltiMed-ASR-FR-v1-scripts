# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "click",
# ]
# ///

"""
Filter pharmaceutical database to keep only active and commercialized entries.

This script processes a JSON file containing pharmaceutical entries and filters
them to keep only entries with statusAutorisation set to "Autorisation active"
and etatComercialisation set to "Commercialisée".
Created with assistance from aider.chat.
"""

import json
import re
from pathlib import Path
from typing import Any

import click


# Substance filtering configuration
# These lists define patterns that will exclude pharmaceutical entries
# from the filtered database based on denominationSubstance values.
# All checks are case-insensitive (substance names are converted to uppercase).

# Substances starting with these prefixes are excluded
EXCLUDED_PREFIXES = [
    "STREPTOCOCCUS",
    "VIBRIO",
    "TEINTURE",
    "TARTRATE",
    "SULPHATE",
    "SULFATE",
    "SUCCINATE",
    "SEL ",
    "VIRUS",
    "HUILE",
    "EXTRAIT",
    "EXTRACTION",
    "PHOSPHATE",
    "PEROXYDE",
    "CHLORHYDRATE",
    "PROTÉINE",
    "POLYOSIDE",
    "OLIGOSIDE",
    "MÉSILATE",
    "MALÉATE",
    "INSULINE",
]

# Substances ending with these suffixes are excluded
EXCLUDED_SUFFIXES = [
    "INACTIVÉ",
]

# Substances containing these strings or matching these patterns are excluded
# Can contain strings (for simple substring matching) or compiled regex patterns
# String checks are done on uppercase version; regex patterns should account for this
EXCLUDED_CONTAINS = [
    "NATUROPA",
    "SOLUTION",
    "(",  # Exclude substances with parentheses
    ")",
    "[",
    "]",
    re.compile(r"HOM[EÉ]OPATHIQUE"),  # Homeopathic medications
    re.compile(
        r"^\S+ATE\s+D"
    ),  # Patterns like "TOSYLATE DE ..." (first word ends with ATE, followed by space and D)
]

# Substances that exactly match these values are excluded
EXCLUDED_EXACT_MATCH = [
    "OR",
    "URÉE",
]

# Allowed pharmaceutical forms
# Only entries whose formePharmaceutique contains at least one of these forms
# will be kept. Check is case-insensitive.
ALLOWED_PHARMACEUTICAL_FORMS = [
    "capsule",
    "collyre",
    "comprimé",
    "gélule",
    "microgranule gastro-résistant en gélule",
    "microsphère et solution pour usage parentéral ou à libération prolongée",
    "ovule",
    "suppositoire",
]


def should_exclude_substance(denomination: str) -> bool:
    """
    Check if a substance should be excluded based on filtering rules.

    This function checks the substance name against multiple exclusion criteria:
    - Exact matches
    - Prefix matches
    - Suffix matches
    - Substring or pattern matches (supports both strings and compiled regex patterns)

    Parameters
    ----------
    denomination : str
        The denominationSubstance to check.

    Returns
    -------
    bool
        True if the substance should be excluded, False otherwise.
    """
    denomination_upper = denomination.upper()

    # Check for exact matches (case-insensitive)
    if denomination_upper in (item.upper() for item in EXCLUDED_EXACT_MATCH):
        return True

    # Check if starts with any excluded prefix
    if any(denomination_upper.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return True

    # Check if ends with any excluded suffix
    if any(denomination_upper.endswith(suffix) for suffix in EXCLUDED_SUFFIXES):
        return True

    # Check contains patterns (both strings and regex patterns)
    for pattern in EXCLUDED_CONTAINS:
        if isinstance(pattern, re.Pattern):
            # Compiled regex pattern - search in uppercase version
            if pattern.search(denomination_upper):
                return True
        else:
            # String pattern - check if substring exists in uppercase version
            if pattern.upper() in denomination_upper:
                return True

    return False


def clean_forme_pharmaceutique(value: str) -> str:
    """
    Clean formePharmaceutique value.

    This function performs the following operations:
    - Strips leading/trailing whitespace
    - Removes content in parentheses (including the parentheses)
    - Replaces multiple consecutive whitespaces with a single space

    Parameters
    ----------
    value : str
        The formePharmaceutique string to clean.

    Returns
    -------
    str
        Cleaned formePharmaceutique string.
    """
    # Strip leading/trailing whitespace
    cleaned = value.strip()

    # Remove content in parentheses (including the parentheses)
    cleaned = re.sub(r"\([^)]*\)", "", cleaned)

    # Replace multiple whitespaces with a single space
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Strip again in case removing parentheses left trailing spaces
    cleaned = cleaned.strip()

    return cleaned


def identify_derivative_substances(entries: list[dict[str, Any]]) -> set[str]:
    """
    Identify which substances are derivatives of base substances.

    A substance is considered a derivative if it has more than one word.
    For example:
    - "LISINOPRIL ANHYDRE" is a derivative (multi-word)
    - "ÉTHANOL À 70 POUR CENT" is a derivative (multi-word)
    - "LISINOPRIL" is a base substance (single-word)

    This filtering ensures that only single-word base substances are kept,
    removing all multi-word derivative forms.

    Parameters
    ----------
    entries : list[dict[str, Any]]
        List of pharmaceutical entry dictionaries.

    Returns
    -------
    set[str]
        Set of substance names that are derivatives (all multi-word substances).
    """
    # Collect all unique substance names across all entries
    all_substances = set()
    for entry in entries:
        composition = entry.get("composition") or []
        for component in composition:
            denomination = component.get("denominationSubstance", "")
            if denomination:
                all_substances.add(denomination)

    # Identify derivative substances
    # Any substance with more than one word is considered a derivative
    derivatives = set()
    for substance in all_substances:
        words = substance.split()
        if len(words) > 1:
            derivatives.add(substance)

    return derivatives


def filter_active_commercialized(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Filter pharmaceutical entries to keep only active and commercialized ones.

    This function filters entries based on two criteria:
    - statusAutorisation must be "Autorisation active"
    - etatComercialisation must be "Commercialisée"
    Both conditions must be met for an entry to be included in the output.

    Additionally filters out homeopathic and naturopathic medications by checking
    if any denominationSubstance contains 'HOMÉOPATHIQUE', 'HOMEOPATHIQUE', or
    'NATUROPA' (case-insensitive).

    Also filters out entries where any denominationSubstance contains parentheses.

    Additionally excludes entries where any denominationSubstance starts with:
    STREPTOCOCCUS, VIRUS, PHOSPHATE, SULPHATE, PEROXYDE, CHLORHYDRATE, PROTÉINE,
    POLYOSIDE, OLIGOSIDE, MÉSILATE, MALÉATE, or INSULINE.

    Also excludes entries where any denominationSubstance ends with INACTIVÉ,
    is exactly "OR", or contains "SOLUTION".

    Additionally excludes entries containing derivative substances (substances
    with multiple words when a base form with just the first word exists).

    Additionally, cleans the formePharmaceutique field by stripping whitespace,
    normalizing multiple spaces, and removing parenthetical content.

    Finally, filters to keep only entries whose formePharmaceutique contains at
    least one of the allowed pharmaceutical forms: capsule, collyre, comprimé,
    gélule, microgranule gastro-résistant en gélule, microsphère et solution
    pour usage parentéral ou à libération prolongée, ovule, or suppositoire.

    Parameters
    ----------
    entries : list[dict[str, Any]]
        List of pharmaceutical entry dictionaries.

    Returns
    -------
    list[dict[str, Any]]
        Filtered list containing only entries matching both criteria.
    """
    # Identify derivative substances across all entries
    # This is done first to have a complete view of all substances
    derivative_substances = identify_derivative_substances(entries=entries)
    filtered = []

    for entry in entries:
        # Both conditions must be met to include the entry
        # Using .get() to safely handle missing fields
        if (
            entry.get("statusAutorisation") == "Autorisation active"
            and entry.get("etatComercialisation") == "Commercialisée"
        ):
            # Check each substance in composition against exclusion rules
            should_exclude = False
            composition = entry.get("composition") or []

            if composition:
                for component in composition:
                    denomination = component.get("denominationSubstance", "")
                    if should_exclude_substance(denomination=denomination):
                        should_exclude = True
                        break

                    # Filter out if elementPharmaceutique contains "solution"
                    element_pharma = component.get("elementPharmaceutique", "")
                    if element_pharma and "solution" in element_pharma.lower():
                        should_exclude = True
                        break

            if should_exclude:
                continue

            # Filter out entries containing derivative substances
            # Check if any substance in the composition is a derivative
            contains_derivative = False
            for component in composition:
                denomination = component.get("denominationSubstance", "")
                if denomination in derivative_substances:
                    contains_derivative = True
                    break

            if contains_derivative:
                continue

            # Clean formePharmaceutique if present
            if "formePharmaceutique" in entry and entry["formePharmaceutique"]:
                entry["formePharmaceutique"] = clean_forme_pharmaceutique(
                    entry["formePharmaceutique"]
                )

            # Filter by pharmaceutical form
            # Only keep entries whose formePharmaceutique contains at least one allowed form
            forme_pharma = entry.get("formePharmaceutique", "").lower()
            if not any(
                allowed_form.lower() in forme_pharma
                for allowed_form in ALLOWED_PHARMACEUTICAL_FORMS
            ):
                continue

            filtered.append(entry)

    return filtered


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to input JSON file containing list of pharmaceutical entries.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Path where to write the filtered JSON file.",
)
def main(input_path: Path, output_path: Path) -> None:
    """
    Filter pharmaceutical database to keep only active and commercialized entries.

    Reads a JSON file containing pharmaceutical entries and filters them to keep
    only those with statusAutorisation="Autorisation active" and
    etatComercialisation="Commercialisée". The filtered entries are written to
    the output file in the same format.

    Parameters
    ----------
    input_path : Path
        Path to input JSON file.
    output_path : Path
        Path to output JSON file.
    """
    # Load input JSON file
    with input_path.open("r", encoding="utf-8") as f:
        entries = json.load(f)

    # Filter entries based on authorization and commercialization status
    filtered_entries = filter_active_commercialized(entries=entries)

    # Dump to string first to allow character replacements
    # Using ensure_ascii=False to preserve special characters (accents, etc.)
    json_string = json.dumps(filtered_entries, ensure_ascii=False, indent=2)

    # Replace encoding artifacts with correct characters
    json_string = json_string.replace("Ã©", "é")

    # Write the corrected JSON string to file
    with output_path.open("w", encoding="utf-8") as f:
        f.write(json_string)

    click.echo(f"Processed {len(entries)} entries.")
    click.echo(f"Kept {len(filtered_entries)} entries matching criteria.")
    click.echo(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
