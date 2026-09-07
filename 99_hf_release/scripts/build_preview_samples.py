# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Build a tiny, human-inspectable preview of the release: one folder per subset,
each holding 5 random FLAC clips and a 5-line NeMo manifest.

Layout produced (default --out data/preview_samples):

    preview_samples/
      dictionary/  <5 .flac> + manifest.jsonl   # 5 nemo lines
      drugs/       <5 .flac> + manifest.jsonl
      parhaf/      <5 .flac> + manifest.jsonl
      parrot/      <5 .flac> + manifest.jsonl

Each manifest line is a standard NeMo record pointing at the FLAC by basename
(so the folder is self-contained): {"audio_filepath", "duration", "text"}. The
`text` is the training label (asr_training_target), same as the Parquet `text`
column. No splits: this is for eyeballing a few clips + transcripts, e.g. to send
to someone or to inspect the data when the HF dataset viewer is unavailable.

Source of truth is the per-subset NeMo manifest built by 01_build_nemo_manifest.py
(data/NeMO_files/<SUBSET>/full.jsonl), whose audio_filepath is relative to that
manifest dir. Sampling is a seeded reservoir sample, so it is reproducible and
does not load the whole (500k-line) manifest into memory.

Usage:
    uv run scripts/build_preview_samples.py                 # -> data/preview_samples/
    uv run scripts/build_preview_samples.py --n 5 --seed 42 --out data/preview_samples

Written with Claude Code.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

# (manifest folder under data/NeMO_files, output folder name = HF subset name)
SUBSETS = [
    ("dictionary", "dictionary"),
    ("drugs", "drugs"),
    ("PARHAF", "parhaf"),
    ("acronyms", "acronyms"),
    ("PARROT", "parrot"),
]

REPO = Path(__file__).resolve().parent.parent          # 99_hf_release/
MANIFEST_ROOT = REPO / "data" / "NeMO_files"


def reservoir_sample(path: Path, k: int, rng: random.Random) -> list[str]:
    """Algorithm R: uniform k-sample of the non-empty lines in one pass."""
    sample: list[str] = []
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if len(sample) < k:
                sample.append(line)
            else:
                j = rng.randint(0, n)      # 0..n inclusive
                if j < k:
                    sample[j] = line
            n += 1
    return sample


def build_subset(manifest_sub: str, out_name: str, n: int, out_root: Path,
                 rng: random.Random) -> int:
    manifest = MANIFEST_ROOT / manifest_sub / "full.jsonl"
    if not manifest.exists():
        print(f"  [skip] {out_name}: no manifest at {manifest}")
        return 0
    rows = [json.loads(l) for l in reservoir_sample(manifest, n, rng)]
    out_dir = out_root / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    copied = 0
    for row in rows:
        src = (manifest.parent / row["audio_filepath"]).resolve()
        if not src.exists():
            print(f"  [warn] {out_name}: missing audio {src}")
            continue
        dst = out_dir / src.name
        shutil.copy2(src, dst)
        lines.append(json.dumps(
            {"audio_filepath": src.name, "duration": row["duration"], "text": row["text"]},
            ensure_ascii=False,
        ))
        copied += 1
    (out_dir / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  {out_name}: {copied} clips -> {out_dir}")
    return copied


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=5, help="clips per subset (default 5)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "preview_samples",
                    help="output dir (default data/preview_samples)")
    args = ap.parse_args()

    if not MANIFEST_ROOT.exists():
        sys.exit(f"no NeMo manifests at {MANIFEST_ROOT}; run 01_build_nemo_manifest.py first")

    rng = random.Random(args.seed)
    print(f"building preview samples: {args.n}/subset, seed {args.seed} -> {args.out}")
    total = sum(build_subset(ms, out, args.n, args.out, rng) for ms, out in SUBSETS)
    print(f"done: {total} clips across {len(SUBSETS)} subsets in {args.out}")


if __name__ == "__main__":
    main()
