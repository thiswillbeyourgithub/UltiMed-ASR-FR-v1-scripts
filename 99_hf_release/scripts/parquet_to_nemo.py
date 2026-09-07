#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pyarrow>=14",
#   "tqdm>=4.66",
# ]
# ///
"""
Reconstruct NeMo training data from the HF Parquet corpus.

The dataset ships on the Hub as sharded Parquet with the FLAC bytes embedded (see
scripts/build_parquet.py). This script turns those Parquets back into the layout
NeMo trains on: it decodes each row's embedded FLAC straight to disk (verbatim, no
re-encode) and writes a NeMo JSONL manifest next to it. This is the companion that
makes the Parquet-only release trainable, and it is uploaded to the Hub alongside
the data. Written with Claude Code.

Two output formats
    --format loose   (default) loose `.flac` files + a `manifest.jsonl`
                     (audio_filepath / duration / text ...). Drop-in for a NeMo
                     `manifest_filepath` config (what training_config.yaml loads).
    --format tarred  additionally packs each group into NeMo tarred (WebDataset)
                     shards via NeMo's own converter, for an `is_tarred: true`
                     config. Needs a NeMo checkout (sibling by default,
                     --nemo-converter to point elsewhere).

Grouping
    --combine  (default) merge dictionary + drugs + parhaf + acronyms per split into
               train / val / test (the release's combined main corpus); PARROT
               stays in its own eval-only `parrot/` group (different licence).
    --per-source  keep every subset separate: `dictionary_train/`, `drugs_val/`, ...

Only the embedded FLAC bytes are read (via pyarrow), never decoded, so this needs
no audio/ML stack. Shard count adapts to clip count (~2500 clips/shard by default);
NeMo tarred shards must all hold the same sample count, so tarred mode lets the
converter drop the tiny trailing remainder of each group (a training reconstruction
loses nothing meaningful). Loose mode keeps every clip.

Disk note: tarred mode writes the loose FLAC first, then the tars, so it needs
roughly 2x a group's audio size free while converting.

Usage
    uv run scripts/parquet_to_nemo.py --dry-run                  # show the plan
    uv run scripts/parquet_to_nemo.py                            # loose, combined
    uv run scripts/parquet_to_nemo.py --format tarred            # NeMo tar shards
    uv run scripts/parquet_to_nemo.py --per-source --only drugs parrot
    uv run scripts/parquet_to_nemo.py --parquet-dir /path/to/download --out /path/to/nemo
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../99_hf_release/scripts
RELEASE_DIR = HERE.parent                        # .../99_hf_release
DATA_DIR = RELEASE_DIR / "data"                  # symlink -> SSD audio_datasets
DEFAULT_PARQUET_DIR = DATA_DIR / "hf_parquet"    # what build_parquet.py writes
DEFAULT_OUT_DIR = DATA_DIR / "nemo_from_parquet"

# Sibling NeMo checkout (…/<parent>/NeMo), overridable with --nemo-converter.
DEFAULT_CONVERTER = (RELEASE_DIR.parent.parent
                     / "NeMo/scripts/speech_recognition/convert_to_tarred_audio_dataset.py")
# The converter's light standard-path deps, pulled in via a nested `uv run`.
CONVERTER_DEPS = ["numpy", "soundfile", "joblib", "omegaconf", "tabulate", "tqdm"]

# Sources and their splits (must match build_parquet.py's layout on disk / the Hub).
MAIN_SOURCES = ["dictionary", "drugs", "parhaf", "acronyms"]   # CC BY 4.0, combined
MAIN_SPLITS = ["train", "val", "test"]
PARROT_SOURCE = "parrot"                            # CC BY-NC-SA 4.0, test-only
ALL_SOURCES = MAIN_SOURCES + [PARROT_SOURCE]

TARGET_CLIPS_PER_SHARD = 2500
SHUFFLE_SEED = 42
# Non-audio columns carried from the Parquet into the reconstructed manifest
# (asr_training_source is the deterministic TTS input alongside text; the rest is
# provenance). NeMo ignores all of them during training.
PROV_COLS = ["asr_training_source", "category", "group_id", "group_mode", "item_index"]


def parquet_files(parquet_dir: Path, source: str, split: str) -> list[Path]:
    """The Parquet shards for one (source, split), sorted."""
    return sorted((parquet_dir / source).glob(f"{split}-*.parquet"))


def build_groups(sources: list[str], combine: bool, parquet_dir: Path) -> list[dict]:
    """Resolve the requested sources into output groups.

    Each group is {name, files: [Path], shuffle: bool}. `files` is the list of
    Parquet shards feeding that group; empty groups are dropped (with a warning).
    """
    groups: list[dict] = []
    main = [s for s in sources if s in MAIN_SOURCES]
    if combine:
        for split in MAIN_SPLITS:
            files = [f for s in main for f in parquet_files(parquet_dir, s, split)]
            if files:
                groups.append({"name": split, "files": files, "shuffle": split == "train"})
    else:
        for s in main:
            for split in MAIN_SPLITS:
                files = parquet_files(parquet_dir, s, split)
                if files:
                    groups.append({"name": f"{s}_{split}", "files": files,
                                   "shuffle": split == "train"})
    if PARROT_SOURCE in sources:
        files = parquet_files(parquet_dir, PARROT_SOURCE, "test")
        if files:
            groups.append({"name": "parrot", "files": files, "shuffle": False})
    for s in sources:
        has = any(parquet_files(parquet_dir, s, sp) for sp in MAIN_SPLITS + ["test"])
        if not has:
            print(f"  warning: no Parquet shards for source '{s}' under {parquet_dir}; skipping.")
    return groups


def iter_rows(files: list[Path]):
    """Yield (member, flac_bytes, duration, text, provenance_dict) across shards.

    Reads in batches so only a batch of FLAC bytes is resident at a time."""
    import pyarrow.parquet as pq
    for f in files:
        pf = pq.ParquetFile(str(f))
        for batch in pf.iter_batches(batch_size=256):
            cols = batch.column_names
            audio = batch.column("audio").to_pylist()
            text = batch.column("text").to_pylist()
            duration = batch.column("duration").to_pylist()
            filename = (batch.column("filename").to_pylist()
                        if "filename" in cols else [a.get("path") for a in audio])
            prov = {c: batch.column(c).to_pylist() for c in PROV_COLS if c in cols}
            for i in range(len(audio)):
                member = filename[i] or audio[i].get("path") or f"clip_{i}.flac"
                pdict = {c: prov[c][i] for c in prov}
                yield member, audio[i]["bytes"], float(duration[i] or 0.0), text[i] or "", pdict


def count_rows(files: list[Path]) -> int:
    import pyarrow.parquet as pq
    return sum(pq.ParquetFile(str(f)).metadata.num_rows for f in files)


def reconstruct_loose(group: dict, out_root: Path, abs_paths: bool) -> Path:
    """Decode a group's FLAC to <out>/<name>/audio/ and write manifest.jsonl.

    Returns the group dir. audio_filepath is absolute (robust for a NeMo
    manifest_filepath) unless --relative, in which case it is `audio/<member>`."""
    from tqdm import tqdm
    name = group["name"]
    gdir = out_root / name
    adir = gdir / "audio"
    adir.mkdir(parents=True, exist_ok=True)
    manifest = gdir / "manifest.jsonl"
    n = count_rows(group["files"])
    written = 0
    with manifest.open("w", encoding="utf-8") as mf:
        for member, flac, duration, text, prov in tqdm(
                iter_rows(group["files"]), total=n, desc=f"  {name}", unit="clip"):
            dst = adir / member
            dst.write_bytes(flac)
            ap = str(dst.resolve()) if abs_paths else f"audio/{member}"
            entry = {"audio_filepath": ap, "duration": duration, "text": text, **prov}
            mf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1
    print(f"    wrote {written} clips + manifest.jsonl -> {gdir}")
    return gdir


def run_nemo_tar(group: dict, gdir: Path, shards: int, converter: Path,
                 workers: int, max_duration: float) -> None:
    """Pack a reconstructed group into NeMo tarred shards via NeMo's converter.

    Writes a relative _tar_src.jsonl (audio_filepath = `audio/<member>`) and runs
    the converter with cwd=<group dir> so the tar member names stay clean and no
    absolute path is baked in. Reuses the same hermetic nested-uv-run approach as
    the old build_tarred.py."""
    manifest = gdir / "manifest.jsonl"
    tar_src = gdir / "_tar_src.jsonl"
    # Rewrite the manifest to relative paths for clean tar member names.
    with manifest.open(encoding="utf-8") as fin, tar_src.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            e["audio_filepath"] = f"audio/{os.path.basename(e['audio_filepath'])}"
            fout.write(json.dumps(e, ensure_ascii=False) + "\n")

    target = gdir / "tarred"
    conv_args = [
        "--manifest_path", str(tar_src.name),
        "--target_dir", str(target),
        "--num_shards", str(shards),
        "--max_duration", str(max_duration),
        "--workers", str(workers),
    ]
    if group["shuffle"]:
        conv_args += ["--shuffle", "--shuffle_seed", str(SHUFFLE_SEED)]

    cmd = ["uv", "run", "--no-project"]
    for dep in CONVERTER_DEPS:
        cmd += ["--with", dep]
    cmd += ["python", str(converter), *conv_args]
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env.pop("CONDA_PREFIX", None)
    result = subprocess.run(cmd, cwd=str(gdir), env=env)

    # The converter writes all tars + the manifest, THEN an optional
    # dynamic-bucketing metadata step imports lhotse/NeMo and fails in this light
    # env. The dataset is already complete, so verify outputs rather than the rc.
    tars = sorted(target.glob("audio_*.tar"))
    tmanifest = target / "tarred_audio_manifest.json"
    if len(tars) != shards or not tmanifest.is_file():
        sys.exit(f"error: {group['name']} tarring incomplete: {len(tars)}/{shards} tars, "
                 f"manifest={'ok' if tmanifest.is_file() else 'MISSING'} "
                 f"(converter rc={result.returncode}); see the output above.")
    note = (" (ignored the expected lhotse/NeMo metadata-step failure)"
            if result.returncode != 0 else "")
    print(f"    tarred: {len(tars)} shards + manifest OK{note} -> {target}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reconstruct NeMo training data (loose FLAC or tarred shards) "
                    "from the HF Parquet corpus.",
        allow_abbrev=False)
    ap.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR,
                    help="root holding <source>/<split>-*.parquet "
                         f"(default {DEFAULT_PARQUET_DIR.name}/ under data)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                    help=f"output root (default {DEFAULT_OUT_DIR.name}/ under data)")
    ap.add_argument("--format", choices=["loose", "tarred"], default="loose",
                    help="loose FLAC + manifest (default) or NeMo tarred shards")
    ap.add_argument("--only", nargs="+", choices=ALL_SOURCES, metavar="SOURCE",
                    help=f"reconstruct only these sources (default: all {ALL_SOURCES})")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--combine", dest="combine", action="store_true", default=True,
                     help="merge dictionary+drugs+parhaf+acronyms per split (default)")
    grp.add_argument("--per-source", dest="combine", action="store_false",
                     help="keep every (source, split) as its own group")
    ap.add_argument("--relative", action="store_true",
                    help="loose manifest audio_filepath as `audio/<file>` instead of absolute")
    ap.add_argument("--num-shards", type=int, default=None, metavar="N",
                    help="override the per-group tar shard count (tarred only)")
    ap.add_argument("--workers", type=int, default=4, metavar="N",
                    help="NeMo converter worker processes (tarred only; default 4)")
    ap.add_argument("--max-duration", type=float, default=100000.0, metavar="SECS",
                    help="drop clips longer than this in tarred mode (default: keep all)")
    ap.add_argument("--nemo-converter", type=Path, default=None, metavar="PATH",
                    help="path to convert_to_tarred_audio_dataset.py (default: sibling NeMo)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan (groups, clips, shards), write nothing")
    args = ap.parse_args()

    parquet_dir = args.parquet_dir.resolve()
    if not parquet_dir.is_dir():
        sys.exit(f"error: parquet dir not found: {parquet_dir}\n"
                 f"       build it first: uv run scripts/build_parquet.py")
    converter = (args.nemo_converter.resolve() if args.nemo_converter else DEFAULT_CONVERTER)
    if args.format == "tarred" and not converter.is_file():
        sys.exit(f"error: NeMo converter not found at {converter}\n"
                 f"       pass --nemo-converter <path> to point at it.")

    sources = [s for s in ALL_SOURCES if not args.only or s in args.only]
    groups = build_groups(sources, args.combine, parquet_dir)
    if not groups:
        sys.exit(f"error: no Parquet shards found under {parquet_dir} for {sources}.")

    print(f"parquet in : {parquet_dir}")
    print(f"out root   : {args.out.resolve()}")
    print(f"format     : {args.format}   grouping: {'combined' if args.combine else 'per-source'}\n")
    for g in groups:
        n = count_rows(g["files"])
        shards = (args.num_shards if args.num_shards
                  else max(1, round(n / TARGET_CLIPS_PER_SHARD)))
        extra = f" -> {shards} tar shard(s)" if args.format == "tarred" else ""
        print(f"  {g['name']:16} {n:7} clips  ({len(g['files'])} parquet shard(s)){extra}")
    if args.dry_run:
        print("\ndry-run: nothing written.")
        return

    for g in groups:
        print(f"\n=== {g['name']} ===")
        gdir = reconstruct_loose(g, args.out, abs_paths=not args.relative)
        if args.format == "tarred":
            n = count_rows(g["files"])
            shards = (args.num_shards if args.num_shards
                      else max(1, round(n / TARGET_CLIPS_PER_SHARD)))
            run_nemo_tar(g, gdir, shards, converter, args.workers, args.max_duration)

    print("\ndone.")
    if args.format == "loose":
        print("Point NeMo at each group's manifest.jsonl (model.train_ds.manifest_filepath=...).")
    else:
        print("Point NeMo at each group's tarred/ (is_tarred=true, "
              "tarred_audio_filepaths=<group>/tarred/'audio__OP_0..N_CL_.tar').")


if __name__ == "__main__":
    main()
