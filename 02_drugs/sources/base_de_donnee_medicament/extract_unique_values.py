# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "click",
# ]
# ///

"""
Extract unique field values from pharmaceutical database entries.

This script processes a JSON file containing pharmaceutical entries and extracts
unique values for specific fields, outputting them as a sorted dictionary.
Created with assistance from aider.chat.
"""

import json
from pathlib import Path
from typing import Any

import click


def get_base_forme(forme: str) -> str:
    """
    Get the base form of a pharmaceutical form by extracting its first word.

    This reduces derivative forms to their base form. For example:
    - "capsule dure" -> "capsule"
    - "collyre en solution" -> "collyre"
    - "comprimé" -> "comprimé"

    Parameters
    ----------
    forme : str
        Pharmaceutical form string.

    Returns
    -------
    str
        Base form (first word) of the pharmaceutical form.
    """
    if not forme:
        return ""
    return forme.split()[0]


def merge_derivative_formes(formes: list[str]) -> list[str]:
    """
    Merge derivative pharmaceutical forms into their base forms.

    For example, "collyre", "collyre en suspension", "collyre en solution"
    will be merged to just "collyre". The logic is to group values by their
    first word, then keep only the base form (the value that equals the first word).
    If no exact base form exists, the shortest variant is kept.

    Parameters
    ----------
    formes : list[str]
        Sorted list of pharmaceutical forms.

    Returns
    -------
    list[str]
        List with derivative forms merged into base forms.
    """
    if not formes:
        return []

    # Group formes by their first word
    # This allows us to identify derivatives that share the same base word
    groups = {}
    for forme in formes:
        first_word = forme.split()[0] if forme else ""
        if first_word:
            if first_word not in groups:
                groups[first_word] = []
            groups[first_word].append(forme)

    # For each group, keep only the base form (the one equal to first_word)
    # If the base form doesn't exist, keep the shortest variant as a fallback
    result = []
    for first_word, variants in groups.items():
        # Check if the base form exists (exactly matches first_word)
        if first_word in variants:
            result.append(first_word)
        else:
            # Keep the shortest variant as the base
            result.append(min(variants, key=len))

    return sorted(result)


def extract_substance_dosages(
    entries: list[dict[str, Any]], filtered_substances: set[str]
) -> dict[str, dict[str, list[list[str]]]]:
    """
    Extract dosage information for filtered substances organized by pharmaceutical form.

    For each substance in filtered_substances, collect all unique dosages
    organized by their base pharmaceutical form (elementPharmaceutique reduced
    to base form). Dosages with the same referenceDosage are merged into a single
    list. Only active substances (natureComposant="SA") are processed.

    Parameters
    ----------
    entries : list[dict[str, Any]]
        List of pharmaceutical entry dictionaries.
    filtered_substances : set[str]
        Set of denominationSubstance values to include.

    Returns
    -------
    dict[str, dict[str, list[list[str]]]]
        Nested dictionary structure:
        - First level keys: denominationSubstance values
        - Second level keys: base elementPharmaceutique values
        - Values: list of [referenceDosage, dosage1, dosage2, ...] lists,
          where dosages with the same reference are merged together, sorted
    """
    # Use sets for deduplication during collection
    # Structure: {denomination: {base_element: {reference: set(dosages)}}}
    # This groups dosages by their referenceDosage for merging
    temp_result = {}

    for entry in entries:
        if "composition" in entry and entry["composition"]:
            for comp in entry["composition"]:
                # Only process active substances that are in our filtered set
                if comp.get("natureComposant") == "SA":
                    denomination = comp.get("denominationSubstance", "").strip()

                    # Only process if this substance is in our filtered set
                    if denomination not in filtered_substances:
                        continue

                    element = comp.get("elementPharmaceutique", "").strip()
                    dosage = comp.get("dosage", "").strip()
                    reference = comp.get("referenceDosage", "").strip()

                    if element:  # element is required
                        # Get base form of elementPharmaceutique
                        base_element = get_base_forme(element)

                        # Initialize nested structure if needed
                        if denomination not in temp_result:
                            temp_result[denomination] = {}
                        if base_element not in temp_result[denomination]:
                            temp_result[denomination][base_element] = {}
                        if reference not in temp_result[denomination][base_element]:
                            temp_result[denomination][base_element][reference] = set()

                        # Add dosage to the set for this reference
                        # This automatically deduplicates dosages
                        temp_result[denomination][base_element][reference].add(dosage)

    # Convert nested dicts to sorted lists
    # Each entry becomes [referenceDosage, dosage1, dosage2, ...]
    result = {}
    for denomination in sorted(temp_result.keys()):
        result[denomination] = {}
        for base_element in sorted(temp_result[denomination].keys()):
            # Convert {reference: set(dosages)} to [[reference, dosage1, dosage2, ...]]
            reference_dosages = []
            for reference in sorted(temp_result[denomination][base_element].keys()):
                dosages = sorted(temp_result[denomination][base_element][reference])
                # Create list with reference as first element, followed by all dosages
                reference_dosages.append([reference] + dosages)
            result[denomination][base_element] = reference_dosages

    return result


def extract_unique_values(entries: list[dict[str, Any]]) -> dict[str, list[str]]:
    """
    Extract unique values for specific fields from pharmaceutical entries.

    This function collects all unique values for formePharmaceutique,
    voiesAdministration, statusAutorisation, etatComercialisation,
    surveillanceRenforce, denominationSubstance, and natureComposant fields.
    voiesAdministration is a list in each entry, so we flatten all lists and
    collect unique values. denominationSubstance and natureComposant are found
    in the composition list items.

    Parameters
    ----------
    entries : list[dict[str, Any]]
        List of pharmaceutical entry dictionaries.

    Returns
    -------
    dict[str, list[str]]
        Dictionary with field names as keys and sorted lists of unique values.
    """
    # Initialize sets to collect unique values
    # Using sets automatically handles uniqueness
    formes = set()
    voies = set()
    status = set()
    etats = set()
    surveillances = set()
    denominations = set()
    natures = set()

    for entry in entries:
        # Extract formePharmaceutique - single string value
        if "formePharmaceutique" in entry and entry["formePharmaceutique"]:
            formes.add(entry["formePharmaceutique"].strip())

        # Extract voiesAdministration - list of strings, need to flatten
        if "voiesAdministration" in entry and entry["voiesAdministration"]:
            for voie in entry["voiesAdministration"]:
                voies.add(voie.strip())

        # Extract statusAutorisation - single string value
        if "statusAutorisation" in entry and entry["statusAutorisation"]:
            status.add(entry["statusAutorisation"].strip())

        # Extract etatComercialisation - single string value
        if "etatComercialisation" in entry and entry["etatComercialisation"]:
            etats.add(entry["etatComercialisation"].strip())

        # Extract surveillanceRenforce - single string value
        if "surveillanceRenforce" in entry and entry["surveillanceRenforce"]:
            surveillances.add(entry["surveillanceRenforce"].strip())

        # Extract denominationSubstance and natureComposant from composition list
        # composition is a list of dicts, each may have these fields
        # Only process composition items where natureComposant is "SA" (active substance)
        # This filters out "FT" (fraction thérapeutique) which are not the active ingredients
        if "composition" in entry and entry["composition"]:
            for comp in entry["composition"]:
                # Only process if this is an active substance (SA), not a fraction (FT)
                if comp.get("natureComposant") == "SA":
                    if (
                        "denominationSubstance" in comp
                        and comp["denominationSubstance"]
                    ):
                        denominations.add(comp["denominationSubstance"].strip())
                    if "natureComposant" in comp and comp["natureComposant"]:
                        natures.add(comp["natureComposant"].strip())

    # Convert sets to sorted lists for output
    # For formePharmaceutique, merge derivative forms into base forms
    # to avoid redundancy like "collyre", "collyre en suspension", etc.
    # For denominationSubstance, build a nested dict with dosage information
    # organized by base elementPharmaceutique
    return {
        "formePharmaceutique": merge_derivative_formes(sorted(formes)),
        "voiesAdministration": sorted(voies),
        "statusAutorisation": sorted(status),
        "etatComercialisation": sorted(etats),
        "surveillanceRenforce": sorted(surveillances),
        "denominationSubstance": extract_substance_dosages(
            entries=entries, filtered_substances=denominations
        ),
        "natureComposant": sorted(natures),
    }


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
    help="Path where to write the output JSON file with unique values.",
)
def main(input_path: Path, output_path: Path) -> None:
    """
    Extract unique field values from pharmaceutical database.

    Reads a JSON file containing pharmaceutical entries and extracts unique
    values for formePharmaceutique, voiesAdministration, statusAutorisation,
    etatComercialisation, surveillanceRenforce, denominationSubstance, and
    natureComposant fields. Outputs a JSON dictionary with these fields as
    keys and sorted lists of unique values.

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

    # Extract unique values for specified fields
    result = extract_unique_values(entries=entries)

    # Write output JSON file with readable formatting
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    click.echo(f"Successfully processed {len(entries)} entries.")
    click.echo(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
