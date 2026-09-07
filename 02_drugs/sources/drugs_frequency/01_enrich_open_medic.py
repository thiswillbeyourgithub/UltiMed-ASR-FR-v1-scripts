"""
Enrich OPEN_MEDIC_2025.CSV with human-readable drug info from
https://github.com/Giygas/medicaments-api

Joins on CIP13 and adds the drug name, pharmaceutical form, route(s),
active substances + dosages, holder, price and reimbursement rate.
"""

from pathlib import Path
import argparse
import csv
import json
import urllib.request

HERE = Path(__file__).parent
SRC_CSV = HERE / "OPEN_MEDIC_2025.CSV"
OUT_CSV = HERE / "01_OPEN_MEDIC_2025_enriched.csv"
EXPORT_JSON = HERE / "medicaments_export.json"
EXPORT_URL = "https://medicaments-api.giygas.dev/v1/medicaments/export"


def fetch_export() -> list[dict]:
    if not EXPORT_JSON.exists():
        print(f"Downloading {EXPORT_URL} ...")
        urllib.request.urlretrieve(EXPORT_URL, EXPORT_JSON)
    with EXPORT_JSON.open(encoding="utf-8") as f:
        return json.load(f)


def build_cip_index(meds: list[dict]) -> dict[str, dict]:
    """Map CIP13 (as 13-char zero-padded string) -> flat enrichment dict."""
    index: dict[str, dict] = {}
    for med in meds:
        substances = "|".join(
            f"{c.get('denominationSubstance','')} {c.get('dosage','')}".strip()
            for c in (med.get("composition") or [])
            if c.get("natureComposant") == "SA"
        )
        voies = "|".join(med.get("voiesAdministration") or [])
        for pres in med.get("presentation") or []:
            cip13 = pres.get("cip13")
            if cip13 is None:
                continue
            key = str(cip13).zfill(13)
            index[key] = {
                "drug_name": med.get("elementPharmaceutique", ""),
                "forme": med.get("formePharmaceutique", ""),
                "voies": voies,
                "substances": substances,
                "titulaire": med.get("titulaire", ""),
                "presentation_libelle": pres.get("libelle", ""),
                "prix_eur": pres.get("prix", ""),
                "taux_remboursement": pres.get("tauxRemboursement", ""),
                "etat_commercialisation": pres.get("etatComercialisation", ""),
            }
    return index


EXTRA_FIELDS = [
    "drug_name",
    "forme",
    "voies",
    "substances",
    "titulaire",
    "presentation_libelle",
    "prix_eur",
    "taux_remboursement",
    "etat_commercialisation",
]


def main() -> None:
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()
    meds = fetch_export()
    print(f"Loaded {len(meds)} médicaments from API export.")
    cip_index = build_cip_index(meds)
    print(f"Indexed {len(cip_index)} CIP13 presentations.")

    matched = 0
    rows: list[dict] = []
    with SRC_CSV.open(encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin, delimiter=";")
        src_fields = list(reader.fieldnames or [])
        for row in reader:
            cip = (row.get("CIP13") or "").strip().zfill(13)
            info = cip_index.get(cip)
            if info:
                matched += 1
                row.update(info)
            else:
                row.update({k: "" for k in EXTRA_FIELDS})
            rows.append(row)
            if len(rows) % 200_000 == 0:
                print(f"  ...{len(rows):,} rows processed ({matched:,} matched)")

    print(f"Sorting {len(rows):,} rows by BOITES desc...")
    def boites_key(r: dict) -> int:
        try:
            return int(r.get("BOITES") or 0)
        except ValueError:
            return 0
    rows.sort(key=boites_key, reverse=True)

    with OUT_CSV.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(
            fout, fieldnames=src_fields + EXTRA_FIELDS, delimiter=";"
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. {matched:,}/{len(rows):,} rows matched a CIP13.")
    print(f"Output: {OUT_CSV}")


if __name__ == "__main__":
    main()
