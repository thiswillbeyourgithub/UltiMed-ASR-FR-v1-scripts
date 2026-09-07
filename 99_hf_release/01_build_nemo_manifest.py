#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["soundfile", "tqdm", "click", "loguru"]
# ///
"""Build a NeMo ASR manifest (+ train/val/test split) for one text stage.

Generic over the text stages: it takes any stage's ``generated_dataset.jsonl``
and its folder of synthesized ``.flac`` clips and writes, under
``data/NeMO_files/<name>/``:

    full.jsonl   every row that has audio, as {audio_filepath, duration, text, ...}
    train.jsonl
    val.jsonl    the split, default 80/10/10 (see --split)
    test.jsonl

Each output row is a NeMo manifest line: ``text`` is the written label
(``asr_training_target``), ``duration`` is measured from the FLAC header, and
``audio_filepath`` is stored *relative to the manifest file* (so nothing leaks an
absolute home path). Rows whose clip is missing (skipped/failed in stage 05) are
dropped rather than pointing at a non-existent file.

Clips are matched to rows by the stage-05 filename prefix
``{term_index:06d}_{variant_index:04d}`` (reusing that naming, not re-deriving the
term slug), so this stays correct whatever the slug did.

Splitting (``utils/nemo_manifest.assign_splits``, shared with the parent
combiner):

* ``--stratify`` (default on) keeps a term's variants and a document's chunks
  sensible across splits. dictionary/drugs/acronyms terms are *distributed* (a term with
  >=2 variants keeps one in train; with >=3, one in every split, so each split
  sees the term). PARHAF/PARROT documents are *atomic* (all chunks of a
  ``source_id`` land in the same split, so no document leaks train<->test). The
  mode is auto-detected: rows with a ``source_id`` are atomic-by-document, the
  rest distribute-by-term. ``--group-by`` / ``--group-mode`` override it.
* ``--stratify-duration`` (default on) balances the split by summed clip duration
  (the proxy for how much speech each split holds) instead of by row count.

In ``--all`` mode a stage may pin its own split in the ``STAGES`` table: PARROT is a
differently-licensed radiology subset shipped for evaluation only, so all of its
clips go to ``test`` (``0/0/100``), never train or val.

Run one stage explicitly:

    uv run 01_build_nemo_manifest.py --input ../01_dictionnary/generated_dataset.jsonl \\
        --audio-dir data/dictionary --name dictionary

or all known stages at once (paths resolved relative to this repo):

    uv run 01_build_nemo_manifest.py --all

Written with Claude Code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from nemo_manifest import (  # noqa: E402
    assign_splits,
    audio_index,
    format_summary,
    parse_split,
    probe_durations,
    summarize,
    write_splits,
    _stem_key,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUT = SCRIPT_DIR / "data" / "NeMO_files"

# The known stages, resolved relative to the repo so no absolute/home path
# is baked in. `audio` is a subfolder of the stage-99 `data/` symlink (the audio
# SSD). Mode/group are auto-detected from the rows, so they are not listed here.
# An optional `split` pins that stage's ratio in --all mode: PARROT is a
# differently-licensed radiology subset shipped for evaluation only, so all of its
# clips go to test (0/0/100), never train or val.
STAGES = [
    {"name": "dictionary", "input": "01_dictionnary/generated_dataset.jsonl", "audio": "dictionary"},
    {"name": "drugs", "input": "02_drugs/generated_dataset.jsonl", "audio": "drugs"},
    {"name": "PARHAF", "input": "03_PARHAF/generated_dataset.jsonl", "audio": "PARHAF"},
    {"name": "PARROT", "input": "04_PARROT/generated_dataset.jsonl", "audio": "PARROT", "split": "0/0/100"},
    {"name": "acronyms", "input": "07_acronyms/generated_dataset.jsonl", "audio": "acronyms"},
]


def detect_grouping(sample: dict, group_by: str | None, group_mode: str | None) -> tuple[str, str]:
    """Return (group_field, mode). Auto: a ``source_id`` present means the rows
    are document chunks (group by document, atomic); otherwise they are term
    variants (group by term_index, distribute). Either can be forced."""
    if group_by:
        field = group_by
    else:
        field = "source_id" if "source_id" in sample else "term_index"
    if group_mode:
        mode = group_mode
    else:
        mode = "atomic" if field == "source_id" else "distribute"
    return field, mode


def build_dataset(
    name: str,
    input_path: Path,
    audio_dir: Path,
    out_root: Path,
    ratios: dict,
    stratify: bool,
    by_duration: bool,
    strict_coverage: bool,
    jobs: int,
    absolute: bool,
    group_by: str | None,
    group_mode: str | None,
    limit: int | None,
) -> None:
    if not input_path.is_file():
        logger.error("{}: input not found: {}", name, input_path)
        return
    if not audio_dir.is_dir():
        logger.error("{}: audio dir not found: {}", name, audio_dir)
        return

    out_dir = out_root / name
    logger.info("[{}] indexing audio in {}", name, audio_dir)
    index = audio_index(audio_dir)
    logger.info("[{}] {} audio clips found", name, len(index))

    rows: list[dict] = []
    missing_audio = bad_rows = 0
    field = mode = None
    with input_path.open() as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            src = json.loads(line)
            if field is None:
                field, mode = detect_grouping(src, group_by, group_mode)
                logger.info("[{}] grouping by '{}' ({})", name, field, mode)
            try:
                key = _stem_key(src["term_index"], src["variant_index"])
            except (KeyError, ValueError, TypeError):
                bad_rows += 1
                continue
            path = index.get(key)
            if path is None:
                missing_audio += 1
                continue
            text = (src.get("asr_training_target") or "").strip()
            if not text:
                bad_rows += 1
                continue
            # The deterministic TTS input this clip was synthesized from. Carried
            # through so the published manifest (and the Parquet built from it) ship
            # both the written label (text) and the spoken source.
            source = (src.get("asr_training_source") or "").strip()
            raw_group = src.get(field, src.get("term_index"))
            category = src.get("category", name)
            rows.append({
                "_audio_abs": path,
                "text": text,
                "asr_training_source": source,
                "category": category,
                # Namespaced so the parent combiner never merges a term_index/
                # source_id shared by two datasets into one group.
                "group_id": f"{name}:{raw_group}",
                "group_mode": mode,
                "item_index": src.get("chunk_index", src.get("variant_index")),
            })
            if limit and len(rows) >= limit:
                break

    if not rows:
        logger.error("[{}] no rows with audio, nothing written", name)
        return

    logger.info("[{}] {} rows matched audio ({} missing audio, {} bad rows); probing durations",
                name, len(rows), missing_audio, bad_rows)
    durations = probe_durations(
        [r["_audio_abs"] for r in rows],
        cache_path=out_dir / ".duration_cache.json",
        jobs=jobs,
    )
    kept = []
    dropped_dur = 0
    for r in rows:
        dur = durations.get(r["_audio_abs"].name)
        if dur is None or dur <= 0:
            dropped_dur += 1
            continue
        r["duration"] = dur
        kept.append(r)
    if dropped_dur:
        logger.warning("[{}] dropped {} rows with unreadable/zero duration", name, dropped_dur)

    labels = assign_splits(kept, ratios, stratify=stratify, by_duration=by_duration,
                           strict_coverage=strict_coverage)
    counts = write_splits(out_dir, kept, labels, absolute=absolute)
    agg = summarize(kept, labels)
    total_h = sum(a["duration"] for a in agg.values()) / 3600.0
    logger.info("[{}] wrote {} -> full={} train={} val={} test={} ({:.1f} h audio)\n{}",
                name, out_dir, counts["full"], counts["train"], counts["val"], counts["test"],
                total_h, format_summary(agg))


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--input", "input_path", type=click.Path(path_type=Path),
              help="A stage's generated_dataset.jsonl (single-dataset mode).")
@click.option("--audio-dir", type=click.Path(path_type=Path),
              help="Folder of that stage's .flac clips (single-dataset mode).")
@click.option("--name", help="Output subfolder name under --outdir (single-dataset mode).")
@click.option("--all", "do_all", is_flag=True,
              help="Build all known stages (paths resolved relative to the repo).")
@click.option("--outdir", type=click.Path(path_type=Path), default=DEFAULT_OUT,
              show_default=True, help="Parent folder for the per-dataset subfolders.")
@click.option("--split", "split_spec", default="80/10/10", show_default=True,
              help="train/val/test ratio, e.g. 80/10/10 (a 0 disables that split).")
@click.option("--stratify/--no-stratify", default=True, show_default=True,
              help="Keep term variants / document chunks sensible across splits.")
@click.option("--stratify-duration/--no-stratify-duration", default=True, show_default=True,
              help="Balance the split by summed clip duration instead of row count.")
@click.option("--strict-coverage/--best-effort-coverage", "strict_coverage", default=False,
              show_default=True,
              help="Strict: every >=3-variant term keeps >=1 in each split (floors "
                   "val/test above 10%). Best-effort (default): only >=1-in-train is "
                   "guaranteed, so val/test hit the target ratio.")
@click.option("--group-by", default=None,
              help="Force the grouping field (default: source_id if present, else term_index).")
@click.option("--group-mode", type=click.Choice(["atomic", "distribute"]), default=None,
              help="Force the grouping mode (default: atomic for source_id, else distribute).")
@click.option("--jobs", type=int, default=8, show_default=True,
              help="Threads used to probe FLAC durations.")
@click.option("--absolute", is_flag=True,
              help="Write absolute audio_filepath (embeds the machine path; off by default).")
@click.option("--limit", type=int, default=None, help="Debug: cap rows per dataset.")
def main(input_path, audio_dir, name, do_all, outdir, split_spec, stratify,
         stratify_duration, strict_coverage, group_by, group_mode, jobs, absolute, limit):
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> <level>{message}</level>")
    ratios = parse_split(split_spec)
    logger.info("split {} -> train {:.1%} val {:.1%} test {:.1%} | stratify={} by_duration={} "
                "coverage={}", split_spec, ratios["train"], ratios["val"], ratios["test"],
                stratify, stratify_duration, "strict" if strict_coverage else "best-effort")

    if do_all:
        for stage in STAGES:
            stage_ratios = ratios
            if stage.get("split"):
                stage_ratios = parse_split(stage["split"])
                logger.info("[{}] split override {} (this stage only)", stage["name"], stage["split"])
            build_dataset(
                stage["name"],
                REPO_ROOT / stage["input"],
                SCRIPT_DIR / "data" / stage["audio"],
                outdir, stage_ratios, stratify, stratify_duration, strict_coverage, jobs,
                absolute, group_by, group_mode, limit,
            )
        return

    if not (input_path and audio_dir and name):
        raise click.UsageError("give --all, or all of --input / --audio-dir / --name")
    build_dataset(name, input_path, audio_dir, outdir, ratios, stratify,
                  stratify_duration, strict_coverage, jobs, absolute, group_by, group_mode, limit)


if __name__ == "__main__":
    main()
