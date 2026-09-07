#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru"]
# ///
"""Combine the per-dataset NeMo manifests into one release-wide manifest.

Reads every ``data/NeMO_files/<name>/full.jsonl`` written by
``01_build_nemo_manifest.py`` and writes, at the parent
``data/NeMO_files/`` level:

    full.jsonl   every row of every dataset
    train.jsonl
    val.jsonl
    test.jsonl

Each row's ``audio_filepath`` is rewritten to be correct from the parent folder:
the per-dataset manifests store it relative to their own subfolder (one level
deeper), so a plain concat would point one ``../`` too far. Every path is
re-derived relative to the parent manifest here (still relative, never absolute,
so no home path leaks).

``--stratify-duration`` (default on) controls the split:

* **on**: pool all rows and re-derive train/val/test *globally* so the combined
  split hits the target ratio by total audio duration (the proxy for how much
  speech each split holds), not just row count. The re-split stays group-aware
  (a PARHAF/PARROT document's chunks stay together; a dictionary/drugs term keeps
  a variant in train, and one per split when it has >=3), reusing the same
  ``assign_splits`` as the per-dataset builder.
* **off**: plain concat of each dataset's own train/val/test (already ~ratio by
  count), only fixing the paths, and report how far off the duration balance is.

    uv run 02_combine_nemo_manifests.py                 # global duration re-split
    uv run 02_combine_nemo_manifests.py --no-stratify-duration   # plain concat

Written with Claude Code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from nemo_manifest import (  # noqa: E402
    SPLITS,
    assign_splits,
    format_summary,
    parse_split,
    read_jsonl,
    relativize,
    summarize,
    write_jsonl,
    write_splits,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR / "data" / "NeMO_files"


def discover_subdatasets(root: Path) -> list[Path]:
    """Subfolders of ``root`` that hold a per-dataset ``full.jsonl``."""
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / "full.jsonl").is_file())


def load_full(subdir: Path) -> list[dict]:
    """Rows of a subfolder ``full.jsonl`` with an absolute audio path attached
    (``_audio_abs``), resolved from the subfolder-relative ``audio_filepath``."""
    rows = read_jsonl(subdir / "full.jsonl")
    for r in rows:
        r["_audio_abs"] = (subdir / r["audio_filepath"]).resolve()
    return rows


def refix_row(row: dict, subdir: Path, parent_dir: Path) -> dict:
    """Copy a sub-split row with its ``audio_filepath`` re-based from ``subdir``
    to ``parent_dir`` (the only field that changes when moving up one level)."""
    out = dict(row)
    audio_abs = (subdir / row["audio_filepath"]).resolve()
    out["audio_filepath"] = relativize(audio_abs, parent_dir)
    return out


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--root", type=click.Path(path_type=Path), default=DEFAULT_ROOT,
              show_default=True, help="Folder holding the per-dataset subfolders.")
@click.option("--split", "split_spec", default="80/10/10", show_default=True,
              help="Target train/val/test ratio for the global re-split.")
@click.option("--stratify-duration/--no-stratify-duration", default=True, show_default=True,
              help="On: global duration-balanced re-split. Off: concat per-dataset splits.")
@click.option("--stratify/--no-stratify", default=True, show_default=True,
              help="Keep groups (documents/terms) intact in the global re-split.")
@click.option("--strict-coverage/--best-effort-coverage", "strict_coverage", default=False,
              show_default=True,
              help="Strict: every >=3-variant term keeps >=1 in each split (floors "
                   "val/test above 10%). Best-effort (default): only >=1-in-train is "
                   "guaranteed, so val/test hit the target ratio.")
@click.option("--exclude", "exclude", multiple=True, metavar="NAME",
              help="Subdataset name(s) to leave out of the combined manifests, e.g. a "
                   "differently-licensed subset shipped on its own (repeatable).")
@click.option("--absolute", is_flag=True,
              help="Write absolute audio_filepath (embeds the machine path; off by default).")
def main(root: Path, split_spec: str, stratify_duration: bool, stratify: bool,
         strict_coverage: bool, exclude: tuple[str, ...], absolute: bool):
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> <level>{message}</level>")

    root = root.resolve()
    subdirs = discover_subdatasets(root)
    if exclude:
        skip = set(exclude)
        present = {d.name for d in subdirs}
        missing = sorted(skip - present)
        if missing:
            logger.warning("--exclude name(s) not found under {}: {}", root, ", ".join(missing))
        dropped = sorted(present & skip)
        if dropped:
            logger.info("excluding from the combined manifests: {}", ", ".join(dropped))
        subdirs = [d for d in subdirs if d.name not in skip]
    if not subdirs:
        raise click.UsageError(f"no <name>/full.jsonl found under {root}; run 01_build_nemo_manifest.py first")
    logger.info("combining {} datasets: {}", len(subdirs), ", ".join(d.name for d in subdirs))
    ratios = parse_split(split_spec)

    if stratify_duration:
        # Global re-split: pool every row, re-derive the split so the combined
        # manifest is balanced by duration across all datasets at once.
        all_rows: list[dict] = []
        for sd in subdirs:
            rows = load_full(sd)
            all_rows.extend(rows)
            logger.info("  loaded {} rows from {}", len(rows), sd.name)
        labels = assign_splits(all_rows, ratios, stratify=stratify, by_duration=True,
                               strict_coverage=strict_coverage)
        counts = write_splits(root, all_rows, labels, absolute=absolute)
        agg = summarize(all_rows, labels)
        logger.info("global duration re-split -> full={} train={} val={} test={}\n{}",
                    counts["full"], counts["train"], counts["val"], counts["test"],
                    format_summary(agg))
        return

    # Plain concat: keep each dataset's own split, only fix the paths, then
    # report the duration balance that fell out of the count-based sub-splits.
    def rebased(r: dict) -> dict:
        out = {k: v for k, v in r.items() if not k.startswith("_")}
        out["audio_filepath"] = (str(r["_audio_abs"]) if absolute
                                 else relativize(r["_audio_abs"], root))
        return out

    per_split: dict[str, list[dict]] = {s: [] for s in SPLITS}
    write_jsonl(root / "full.jsonl",
                (rebased(r) for sd in subdirs for r in load_full(sd)))
    labels = []
    concat_rows = []
    for s in SPLITS:
        for sd in subdirs:
            f = sd / f"{s}.jsonl"
            if not f.is_file():
                continue
            for r in read_jsonl(f):
                fixed = refix_row(r, sd, root)
                per_split[s].append(fixed)
                concat_rows.append(fixed)
                labels.append(s)
    for s in SPLITS:
        n = write_jsonl(root / f"{s}.jsonl", per_split[s])
        logger.info("  concat {}.jsonl: {} rows", s, n)
    agg = summarize(concat_rows, labels)
    logger.info("plain concat (count-based sub-splits); resulting duration balance:\n{}",
                format_summary(agg))


if __name__ == "__main__":
    main()
