#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   # <4.0 on purpose: the 4.x Audio backend needs torchcodec (torch + ffmpeg);
#   # the 2.x/3.x soundfile backend embeds the original FLAC bytes into Parquet
#   # verbatim (no re-encode, lossless round-trip) without pulling in torch.
#   "datasets>=2.19,<4.0",
#   "soundfile>=0.12",
#   "tqdm>=4.66",
# ]
# ///
"""
Build the FULL corpus as sharded Parquet with embedded FLAC, one subset per source.

This is the primary HF distribution format. The whole 564k-clip corpus ships as
Parquet: each row is the original FLAC bytes (embedded via the `datasets` Audio
feature, no re-encode) plus its transcript and provenance. So the HF Data Viewer
and `load_dataset(...)` work on the ENTIRE dataset, not a sample, and there is no
separate manifest to pair (the text lives in the same row as the audio). A small
companion script (scripts/parquet_to_nemo.py) turns these Parquets back into the
loose-flac / tarred layout NeMo trains on. Written with Claude Code.

Source of truth
    Reads the committed NeMo manifests on the `data` SSD symlink:
        data/NeMO_files/{train,val,test}.jsonl   (categories: dictionary, drugs, parhaf, acronyms)
        data/NeMO_files/PARROT/full.jsonl         (category: parrot, test-only)
    A row's `audio_filepath` is relative to its manifest's own directory
    (e.g. `../dictionary/x.flac`), so the FLAC resolves without any absolute path
    and the machine's home directory never leaks into the output.

Layout: one viewer SUBSET (config) per source, sharded so no single file is huge:

    hf_parquet/dictionary/{train,val,test}-NNNNN-of-NNNNN.parquet   # config "dictionary_CC_BY_4.0"
    hf_parquet/drugs/{train,val,test}-NNNNN-of-NNNNN.parquet         # config "drugs_CC_BY_4.0"
    hf_parquet/parhaf/{train,val,test}-NNNNN-of-NNNNN.parquet        # config "parhaf_CC_BY_4.0"
    hf_parquet/acronyms/{train,val,test}-NNNNN-of-NNNNN.parquet      # config "acronyms_CC_BY_4.0"
    hf_parquet/parrot/test-NNNNN-of-NNNNN.parquet                    # config "parrot_CC_BY-NC-SA_4.0"

The config names carry the licence with underscores (`dictionary_CC_BY_4.0`,
`parrot_CC_BY-NC-SA_4.0`); the README `configs:` block declares them and globs the
matching plain folder (`dictionary/train-*.parquet`). HF config names must be valid
identifiers (letters / digits / _ / - / .), so use underscores, hyphens and dots,
never a space or parenthesis, or the dataset viewer errors out.
PARROT is test-only and non-commercial, so it is its own subset and upload toggle.

Memory
    Written shard by shard: one shard's worth of FLAC bytes is held in RAM, the
    Parquet is flushed, then the buffers are freed before the next shard. Peak RAM
    is ~one shard (~1 GB at the default --shard-clips), never the whole corpus.

Output (under data/hf_parquet/, git-ignored via the `data` symlink; uploaded to the
Hub at the repo root as `<source>/<split>-*.parquet`).

Usage
    uv run scripts/build_parquet.py --dry-run          # show the shard plan, write nothing
    uv run scripts/build_parquet.py                    # build all subsets
    uv run scripts/build_parquet.py --only parhaf parrot
    uv run scripts/build_parquet.py --shard-clips 4000
"""

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../99_hf_release/scripts
RELEASE_DIR = HERE.parent                        # .../99_hf_release
DATA_DIR = RELEASE_DIR / "data"                  # symlink -> SSD audio_datasets
NEMO_DIR = DATA_DIR / "NeMO_files"               # committed split manifests
OUT_DIR = DATA_DIR / "hf_parquet"                # uploaded to HF at the repo root

# One viewer subset per source. `licence` seeds both the build log and the config
# name in the README (e.g. dictionary_CC_BY_4.0): encode it with underscores, never
# spaces or parentheses, or the HF viewer errors. The main sources are filtered
# out of the shared {train,val,test} split manifests by `category`; PARROT is a
# single test-only manifest. `manifest` is relative to DATA_DIR; `audio_filepath`
# inside it is relative to that manifest's own directory.
MAIN_SPLITS = {
    "train": NEMO_DIR / "train.jsonl",
    "val": NEMO_DIR / "val.jsonl",
    "test": NEMO_DIR / "test.jsonl",
}
SOURCES = {
    "dictionary": {"licence": "CC BY 4.0",       "category": "dictionary", "from": "main"},
    "drugs":      {"licence": "CC BY 4.0",       "category": "drugs",      "from": "main"},
    "parhaf":     {"licence": "CC BY 4.0",       "category": "parhaf",     "from": "main"},
    "acronyms":   {"licence": "CC BY 4.0",       "category": "acronyms",   "from": "main"},
    "parrot":     {"licence": "CC BY-NC-SA 4.0", "category": "parrot",     "from": "parrot",
                   "manifest": NEMO_DIR / "PARROT" / "full.jsonl", "split": "test"},
}

# Stage-06 quality control, attached to the manifests by 03_sync_hotfix_results.py (run
# it first, or these all come out null). Declared as name -> dtype rather than as
# datasets.Value objects because the datasets stack is imported lazily, so --dry-run stays
# free of it. `cer_tail` is null on clips shorter than 30 s and `n_stt_check` on clips that
# were read only once, both by design.
QC_COLUMNS = {
    "cer": "float32",
    "cer_tail": "float32",
    "stt_transcript": "string",
    "n_stt_check": "int32",
    "stt_model": "string",
    "cfg_alpha": "float32",
    "regenerated": "bool",
    "qc_status": "string",
}

DEFAULT_SHARD_CLIPS = 2500  # ~1 GB/shard at the corpus' ~408 KB mean FLAC size


def read_manifest(path: Path) -> list[dict]:
    """Read a NeMo JSONL manifest into a list of rows (small dicts)."""
    if not path.is_file():
        sys.exit(f"error: manifest not found: {path}\n"
                 f"       is the SSD mounted / does the `data` symlink resolve?")
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_audio(manifest: Path, row: dict) -> Path:
    """Resolve a row's manifest-relative audio_filepath to an absolute FLAC path."""
    rel = os.path.normpath(os.path.join(manifest.parent, row["audio_filepath"]))
    return Path(rel)


def plan_shards(n: int, shard_clips: int) -> tuple[int, int]:
    """(n_shards, per_shard) for an even-ish split of n rows into <=shard_clips."""
    if n == 0:
        return 0, 0
    n_shards = max(1, math.ceil(n / shard_clips))
    per = math.ceil(n / n_shards)
    return n_shards, per


def write_shard(out: Path, manifest: Path, chunk: list[dict]) -> None:
    # Imported here so --dry-run does not need the (heavy) datasets stack.
    from datasets import Audio, Dataset, Features, Value

    audio, text, source, duration, category = [], [], [], [], []
    group_id, group_mode, item_index, filename = [], [], [], []
    # Quality control, from 03_sync_hotfix_results.py. Flat columns rather than a nested
    # struct so the HF Data Viewer can sort and filter on them directly.
    qc: dict[str, list] = {k: [] for k in QC_COLUMNS}
    for r in chunk:
        path = resolve_audio(manifest, r)
        if not path.is_file():
            sys.exit(f"error: missing audio for parquet: {path}")
        member = os.path.basename(r["audio_filepath"])
        audio.append({"bytes": path.read_bytes(), "path": member})
        text.append(r.get("text", ""))
        source.append(r.get("asr_training_source", ""))
        duration.append(float(r.get("duration", 0.0)))
        category.append(r.get("category", ""))
        group_id.append(str(r.get("group_id", "")))
        group_mode.append(str(r.get("group_mode", "")))
        item_index.append(int(r.get("item_index", 0)))
        filename.append(member)
        for key in QC_COLUMNS:
            qc[key].append(r.get(key))

    feats = Features({
        "audio": Audio(),
        "text": Value("string"),
        "asr_training_source": Value("string"),
        "duration": Value("float32"),
        "category": Value("string"),
        "group_id": Value("string"),
        "group_mode": Value("string"),
        "item_index": Value("int32"),
        "filename": Value("string"),
        **{name: Value(dtype) for name, dtype in QC_COLUMNS.items()},
    })
    ds = Dataset.from_dict(
        {"audio": audio, "text": text, "asr_training_source": source,
         "duration": duration, "category": category,
         "group_id": group_id, "group_mode": group_mode, "item_index": item_index,
         "filename": filename, **qc},
        features=feats,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(out))
    del ds, audio, text, source, duration, category, group_id, group_mode, item_index
    del filename, qc
    gc.collect()


def build_source_split(source: str, split: str, manifest: Path, rows: list[dict],
                       shard_clips: int, dry_run: bool) -> None:
    """Shard one (source, split) row set into Parquet files, streaming per shard."""
    rows = sorted(rows, key=lambda r: r["audio_filepath"])  # deterministic sharding
    n = len(rows)
    n_shards, per = plan_shards(n, shard_clips)
    out_dir = OUT_DIR / source
    print(f"    {split:5} {n:6} clips -> {n_shards:3} shard(s)  ({out_dir.relative_to(DATA_DIR)}/)")
    if dry_run or n == 0:
        return

    # Clear stale shards for this (source, split) so a changed shard count never
    # leaves orphaned `-of-<old>` files behind.
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(f"{split}-*.parquet"):
        stale.unlink()

    from tqdm import tqdm
    for i in tqdm(range(n_shards), desc=f"  {source}/{split}", unit="shard"):
        chunk = rows[i * per:(i + 1) * per]
        out = out_dir / f"{split}-{i:05d}-of-{n_shards:05d}.parquet"
        write_shard(out, manifest, chunk)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build the full-corpus HF Parquet subsets (embedded FLAC + text), "
                    "one per source.",
        allow_abbrev=False)
    ap.add_argument("--only", nargs="+", choices=list(SOURCES), metavar="SOURCE",
                    help=f"build only these sources (default: all {list(SOURCES)})")
    ap.add_argument("--shard-clips", type=int, default=DEFAULT_SHARD_CLIPS, metavar="N",
                    help=f"max clips per Parquet shard (default {DEFAULT_SHARD_CLIPS}, "
                         "~1 GB); also bounds peak RAM")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the per-subset shard plan, write nothing")
    args = ap.parse_args()

    if not DATA_DIR.exists():
        sys.exit(f"error: {DATA_DIR} does not resolve; is the SSD mounted?")

    sources = [s for s in SOURCES if not args.only or s in args.only]
    print(f"data root : {DATA_DIR.resolve()}")
    print(f"out dir   : {OUT_DIR.relative_to(DATA_DIR)}  (uploaded to HF at '<source>/')")
    print(f"shard     : up to {args.shard_clips} clips/shard\n")

    # The main sources share the {train,val,test} manifests, so read each
    # split file once and bucket its rows by category (avoids reading train.jsonl
    # once per source). PARROT has its own single test-only manifest.
    main_sources = [s for s in sources if SOURCES[s]["from"] == "main"]
    if main_sources:
        wanted = {SOURCES[s]["category"]: s for s in main_sources}
        for split, manifest in MAIN_SPLITS.items():
            all_rows = read_manifest(manifest)
            buckets: dict[str, list[dict]] = {s: [] for s in main_sources}
            for r in all_rows:
                s = wanted.get(r.get("category"))
                if s is not None:
                    buckets[s].append(r)
            del all_rows
            for s in main_sources:
                meta = SOURCES[s]
                print(f"[{s} ({meta['licence']})]  split {split}")
                build_source_split(s, split, manifest, buckets[s],
                                   args.shard_clips, args.dry_run)
            del buckets
            gc.collect()

    if "parrot" in sources:
        meta = SOURCES["parrot"]
        manifest = meta["manifest"]
        rows = [r for r in read_manifest(manifest) if r.get("category") == meta["category"]]
        print(f"[parrot ({meta['licence']})]  split {meta['split']} (test-only)")
        build_source_split("parrot", meta["split"], manifest, rows,
                           args.shard_clips, args.dry_run)

    if args.dry_run:
        print("\ndry-run: nothing written.")
    else:
        print("\ndone. Upload with:  uv run scripts/upload_to_hf.py")
        print("The README `configs:` block globs each subset at <source>/<split>-*.parquet.")


if __name__ == "__main__":
    main()
