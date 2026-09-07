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
from pathlib import Path
from typing import Any

import click


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
    with output_path.open("w", encoding="utf-8") as f:
        for drug_name, forms in drugs.items():
            # Create a JSON object for each drug with its name and forms
            drug_entry = {
                "substance": drug_name,
                "forms": forms,
            }
            # Write as single line JSON
            f.write(json.dumps(drug_entry, ensure_ascii=False) + "\n")

    click.echo(f"Successfully processed {len(drugs)} drug substances.")
    click.echo(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
