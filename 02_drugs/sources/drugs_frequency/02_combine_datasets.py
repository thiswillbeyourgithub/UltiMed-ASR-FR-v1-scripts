"""
Combine OPEN_MEDIC 2025 (ambulatory) + RETROCEDAM 2025 (hospital retrocession)
into per-substance and per-brand JSONL files for TTS sample weighting.

Volumes are kept separate (boxes vs UCD units — different granularities) and
also summed into a `total` field. `hospital_only=true` flags entries present
only in RETROCEDAM.
"""

from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET
import argparse
import csv
import json
import re
import urllib.request
import zipfile

from tqdm import tqdm

HERE = Path(__file__).parent
ENRICHED_CSV = HERE / "01_OPEN_MEDIC_2025_enriched.csv"
RETRO_XLSX = HERE / "retrocedam_2017_2025.xlsx"
RETRO_URL = (
    "https://www.assurance-maladie.ameli.fr/sites/default/files/"
    "2017-a-2025_retroced-am_serie-annuelle..xlsx"
)
OUT_SUBSTANCE = HERE / "02_drugs_by_substance_2025.jsonl"
OUT_BRAND = HERE / "02_drugs_by_brand_2025.jsonl"

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
YEAR = "2025"

BRAND_RE = re.compile(r"^([A-ZÉÈÊÀÂÔÎÛÇ\- ]+?)(?=\s*[0-9,]|$)")
_WORD_RE = re.compile(r"\b(\w+)\b")


def _molecule_words(text: str) -> list[str]:
    """Return all uppercase words from a substance string (split on '|')."""
    words = []
    for part in text.split("|"):
        words.extend(_WORD_RE.findall(part.upper()))
    return [w for w in words if len(w) > 3]  # skip tiny tokens


def brand_is_generic(brand: str, molecule_words: set[str]) -> str | None:
    """Return the matching INN word if brand contains one, else None."""
    brand_up = brand.upper()
    for w in molecule_words:
        if w in brand_up:
            return w
    return None


def fetch_retrocedam() -> Path:
    if not RETRO_XLSX.exists():
        print(f"Downloading {RETRO_URL} ...")
        req = urllib.request.Request(
            RETRO_URL,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req) as r, RETRO_XLSX.open("wb") as f:
            f.write(r.read())
    return RETRO_XLSX


def read_shared_strings(z: zipfile.ZipFile) -> list[str]:
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    out: list[str] = []
    for si in root.findall(NS + "si"):
        # concat all <t> descendants (handles rich text)
        out.append("".join((t.text or "") for t in si.iter(NS + "t")))
    return out


def cell_value(c: ET.Element, strings: list[str]) -> str:
    v = c.find(NS + "v")
    if v is None:
        return ""
    if c.get("t") == "s":
        return strings[int(v.text)]
    return v.text or ""


def parse_retrocedam_sheet2(path: Path) -> list[dict]:
    """Return list of {cod_ucd, nom, Produit, code_atc, classe_atc, units_2025}."""
    with zipfile.ZipFile(path) as z:
        strings = read_shared_strings(z)
        sheet = ET.fromstring(z.read("xl/worksheets/sheet2.xml"))

    rows = sheet.find(NS + "sheetData").findall(NS + "row")
    # find header row (first row whose first cell == "cod_ucd")
    header_idx = None
    header: list[str] = []
    for i, r in enumerate(rows):
        cells = [cell_value(c, strings) for c in r.findall(NS + "c")]
        if cells and cells[0] == "cod_ucd":
            header_idx = i
            header = cells
            break
    if header_idx is None:
        raise RuntimeError("RETROCEDAM: could not find header row")

    col = {name: idx for idx, name in enumerate(header)}
    units_col = col[f"Unités {YEAR}"]

    out = []
    for r in rows[header_idx + 1:]:
        cells = [cell_value(c, strings) for c in r.findall(NS + "c")]
        if len(cells) <= units_col:
            cells += [""] * (units_col + 1 - len(cells))
        if not cells[col["cod_ucd"]]:
            continue
        try:
            units = float(cells[units_col].replace(",", ".") or 0)
        except ValueError:
            units = 0
        if units <= 0:
            continue
        out.append({
            "cod_ucd": cells[col["cod_ucd"]],
            "nom": cells[col["nom"]],
            "produit": cells[col["Produit"]],
            "code_atc": cells[col["code_atc"]],
            "classe_atc": cells[col["classe_atc"]],
            "units_2025": units,
        })
    return out


def extract_brand(drug_name: str) -> str:
    if not drug_name:
        return ""
    m = BRAND_RE.match(drug_name.strip())
    if m:
        b = m.group(1).strip()
        if b:
            return b.upper()
    # fallback: first whitespace token
    return drug_name.strip().split()[0].upper() if drug_name.strip() else ""


def _build_inn_vocab(path: Path) -> set[str]:
    """First pass: collect INN words from the substances field only (true INNs from API)."""
    vocab: set[str] = set()
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            for w in _molecule_words(row.get("substances") or ""):
                vocab.add(w)
    return vocab


def aggregate_open_medic(
    path: Path,
) -> tuple[dict[str, dict], dict[str, dict], set[str], Counter]:
    """Return (by_atc5, by_brand, filtered_generics, exclusion_counts)."""
    by_atc: dict[str, dict] = defaultdict(lambda: {"label": "", "boxes": 0})
    by_brand: dict[str, dict] = defaultdict(lambda: {"boxes": 0, "substances": set()})
    filtered_generics: set[str] = set()
    exclusion_counts: Counter = Counter()

    inn_vocab = _build_inn_vocab(path)
    print(f"  INN vocabulary: {len(inn_vocab):,} words")

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in tqdm(reader, desc="OPEN_MEDIC", unit=" rows"):
            try:
                boxes = int(row.get("BOITES") or 0)
            except ValueError:
                boxes = 0
            if boxes <= 0:
                continue

            atc5 = (row.get("ATC5") or "").strip()
            label = (row.get("L_ATC5") or "").strip()
            if atc5:
                by_atc[atc5]["boxes"] += boxes
                if label and not by_atc[atc5]["label"]:
                    by_atc[atc5]["label"] = label

            drug_name = row.get("drug_name") or ""
            brand = extract_brand(drug_name)
            if brand:
                match = brand_is_generic(brand, inn_vocab)
                if match:
                    filtered_generics.add(brand)
                    exclusion_counts[match] += 1
                else:
                    by_brand[brand]["boxes"] += boxes
                    if label:
                        by_brand[brand]["substances"].add(label)
    return dict(by_atc), dict(by_brand), filtered_generics, exclusion_counts


def aggregate_retrocedam(
    rows: list[dict],
    inn_vocab: set[str],
    atc_to_inn: dict[str, str],
) -> tuple[dict[str, dict], dict[str, dict], set[str], Counter]:
    by_atc: dict[str, dict] = defaultdict(lambda: {"label": "", "units": 0.0})
    by_brand: dict[str, dict] = defaultdict(lambda: {"units": 0.0, "substances": set()})
    filtered_generics: set[str] = set()
    exclusion_counts: Counter = Counter()
    for r in rows:
        atc = r["code_atc"].strip()
        if atc:
            by_atc[atc]["units"] += r["units_2025"]
            if r["classe_atc"] and not by_atc[atc]["label"]:
                by_atc[atc]["label"] = r["classe_atc"].strip()
        brand = (r["produit"] or "").strip().upper()
        if brand:
            match = brand_is_generic(brand, inn_vocab)
            if match:
                filtered_generics.add(brand)
                exclusion_counts[match] += 1
            else:
                by_brand[brand]["units"] += r["units_2025"]
                inn = atc_to_inn.get(atc)
                if inn:
                    by_brand[brand]["substances"].add(inn)
    return dict(by_atc), dict(by_brand), filtered_generics, exclusion_counts


def round_units(x: float) -> int | float:
    # Units in RETROCEDAM are already integers in practice; keep ints when possible
    if abs(x - round(x)) < 1e-6:
        return int(round(x))
    return x


def merge_substance(
    om: dict[str, dict], rt: dict[str, dict]
) -> list[dict]:
    keys = set(om) | set(rt)
    out = []
    for k in keys:
        boxes = om.get(k, {}).get("boxes", 0)
        units = rt.get(k, {}).get("units", 0)
        label = om.get(k, {}).get("label") or rt.get(k, {}).get("label") or ""
        out.append({
            "atc5": k,
            "atc5_label": label,
            "ambulatory_boxes": boxes,
            "hospital_units": round_units(units),
            "total": boxes + round_units(units) if isinstance(round_units(units), int) else boxes + units,
            "hospital_only": boxes == 0 and units > 0,
        })
    out.sort(key=lambda r: r["total"], reverse=True)
    return out


def merge_brand(
    om: dict[str, dict], rt: dict[str, dict]
) -> list[dict]:
    keys = set(om) | set(rt)
    out = []
    for k in keys:
        boxes = om.get(k, {}).get("boxes", 0)
        units = rt.get(k, {}).get("units", 0)
        subs = om.get(k, {}).get("substances", set()) | rt.get(k, {}).get("substances", set())
        out.append({
            "brand": k,
            "substances": sorted(subs),
            "ambulatory_boxes": boxes,
            "hospital_units": round_units(units),
            "total": boxes + round_units(units) if isinstance(round_units(units), int) else boxes + units,
            "hospital_only": boxes == 0 and units > 0,
        })
    out.sort(key=lambda r: r["total"], reverse=True)
    return out


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()
    print("Fetching RETROCEDAM ...")
    fetch_retrocedam()
    print("Parsing RETROCEDAM sheet 2 ...")
    retro_rows = parse_retrocedam_sheet2(RETRO_XLSX)
    print(f"  {len(retro_rows):,} UCD rows with Unités {YEAR} > 0")

    print("Aggregating OPEN_MEDIC enriched ...")
    om_atc, om_brand, om_filtered, om_excl = aggregate_open_medic(ENRICHED_CSV)
    print(f"  by ATC5: {len(om_atc)}, by brand: {len(om_brand)}")

    print("Aggregating RETROCEDAM ...")
    inn_vocab = _build_inn_vocab(ENRICHED_CSV)
    atc_to_inn = {k: v["label"] for k, v in om_atc.items() if v.get("label")}
    rt_atc, rt_brand, rt_filtered, rt_excl = aggregate_retrocedam(retro_rows, inn_vocab, atc_to_inn)
    print(f"  by ATC5: {len(rt_atc)}, by brand: {len(rt_brand)}")

    print("Merging and writing JSONL outputs ...")
    sub_rows = merge_substance(om_atc, rt_atc)
    brand_rows = merge_brand(om_brand, rt_brand)
    write_jsonl(sub_rows, OUT_SUBSTANCE)
    write_jsonl(brand_rows, OUT_BRAND)

    print(f"Wrote {len(sub_rows):,} substances -> {OUT_SUBSTANCE.name}")
    print(f"Wrote {len(brand_rows):,} brands -> {OUT_BRAND.name}")
    if sub_rows:
        print(f"Top substance: {sub_rows[0]}")
    if brand_rows:
        print(f"Top brand: {brand_rows[0]}")

    all_filtered = sorted(om_filtered | rt_filtered)
    print(f"\nFiltered generic-named brands ({len(all_filtered):,} total):")
    for b in all_filtered:
        print(f"  {b}")

    combined_excl = om_excl + rt_excl
    heavy_hitters = [(w, n) for w, n in combined_excl.most_common() if n > 50]
    if heavy_hitters:
        print(f"\nINN words that excluded >50 distinct brand rows (may indicate a too-broad match):")
        for w, n in heavy_hitters:
            print(f"  {w}: {n}")


if __name__ == "__main__":
    main()
