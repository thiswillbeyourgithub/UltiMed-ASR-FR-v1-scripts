#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click"]
# ///
"""Report Markdown statistics over the NeMo manifests built by stage 99.

Reads the `{train,val,test}.jsonl` manifests under `data/NeMO_files/` and prints,
as Markdown, the character / word / duration distribution, file & duration counts,
and on-disk audio **size** (MB/GB), broken down **per split**, **per dataset**, and
**per dataset x split**.

Two sources (the same rows, partitioned differently):

* `--source per-dataset` (default): each dataset's own split under
  `<root>/<dataset>/{train,val,test}.jsonl` (dataset = subfolder name). "By split"
  is then the concatenation of every dataset's split.
* `--source combined`: the release-wide global re-split at `<root>/{train,val,test}.jsonl`
  (dataset = each row's `category`). Use this to describe the actual combined
  train/val/test the model trains on.

Per-dataset character/word/duration stats are identical under either source (a
dataset's row set is the same); only the split-level breakdown differs.

    uv run scripts/get_statistics.py                     # Markdown to stdout
    uv run scripts/get_statistics.py --source combined
    uv run scripts/get_statistics.py --output stats.md
    uv run scripts/get_statistics.py --no-sizes          # skip on-disk size probe

Stdlib only (`statistics`, `os`, `concurrent.futures`). Durations come from the
manifest; the audio itself is never decoded. Sizes are read with `os.stat` on the
resolved (relative) `audio_filepath` and cached in `<root>/.size_cache.json`, so
`--sizes` needs the clips present; `--no-sizes` drops the size columns. Written
with Claude Code.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

import click

SCRIPT_DIR = Path(__file__).resolve().parent
# The truncation audit is shared with 06_hotfixes/02_statistics.py, so it lives in utils/.
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "utils"))
from nemo_manifest import CEILING_TOL_S, duration_ceiling  # noqa: E402

DEFAULT_ROOT = SCRIPT_DIR.parent / "data" / "NeMO_files"
SPLITS = ("train", "val", "test")
# Duration histogram edges (seconds); last bucket is the open-ended tail.
DUR_EDGES = [0, 1, 2, 3, 5, 8, 12, 20, 40, 80, float("inf")]
DEFAULT_JOBS = 16


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _iter_rows(path: Path):
    with path.open() as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _triple(row: dict) -> tuple[float, int, int]:
    """(duration_seconds, n_chars, n_words) for one manifest row."""
    text = row.get("text", "") or ""
    return float(row.get("duration") or 0.0), len(text), len(text.split())


def _abspath(base: Path, row: dict) -> str | None:
    """Resolve a row's (relative) audio_filepath against its manifest dir."""
    ap = row.get("audio_filepath")
    if not ap:
        return None
    return os.path.normpath(os.path.join(base, ap))


def _iter_records(root: Path, source: str):
    """Yield (dataset, split, abspath_or_None, dur, nchars, nwords) per row.

    The manifest file's own directory is the base for resolving the row's
    relative audio_filepath (per-dataset paths are relative to the subfolder,
    combined paths to the root)."""
    if source == "combined":
        for split in SPLITS:
            f = root / f"{split}.jsonl"
            if not f.is_file():
                continue
            for r in _iter_rows(f):
                yield (r.get("category", "?"), split, _abspath(f.parent, r), *_triple(r))
    else:  # per-dataset
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            for split in SPLITS:
                f = sub / f"{split}.jsonl"
                if not f.is_file():
                    continue
                for r in _iter_rows(f):
                    yield (sub.name, split, _abspath(f.parent, r), *_triple(r))


def probe_sizes(paths: set[str], cache_path: Path, jobs: int) -> dict[str, int | None]:
    """Byte size per audio path, cached in ``cache_path`` (filename -> bytes).

    Missing/unreadable files map to ``None`` and are not cached (cheap to retry).
    The cache lives under the untracked ``data/`` tree, so absolute keys are fine."""
    cache: dict[str, int] = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            cache = {}

    todo = [p for p in paths if p not in cache]
    if todo:
        def _stat(p: str):
            try:
                return p, os.path.getsize(p)
            except OSError:
                return p, None

        with cf.ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
            for p, sz in ex.map(_stat, todo):
                if sz is not None:
                    cache[p] = sz
        try:
            cache_path.write_text(json.dumps(cache))
        except OSError:
            pass  # caching is best-effort; do not fail the report over it

    return {p: cache.get(p) for p in paths}


def load_cells(root: Path, source: str, show_sizes: bool, jobs: int):
    """Return ({(dataset, split): [(dur, nchars, nwords, size_bytes), ...]}, missing)."""
    records = list(_iter_records(root, source))
    sizes: dict[str, int | None] = {}
    if show_sizes:
        uniq = {r[2] for r in records if r[2]}
        sizes = probe_sizes(uniq, root / ".size_cache.json", jobs)

    cells: dict[tuple[str, str], list] = {}
    missing = 0
    for ds, split, ap, dur, nchars, nwords in records:
        sz = 0
        if show_sizes:
            got = sizes.get(ap) if ap else None
            if got is None:
                missing += 1
            else:
                sz = got
        cells.setdefault((ds, split), []).append((dur, nchars, nwords, sz))
    return cells, missing


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def describe(quads: list[tuple[float, int, int, int]]) -> dict | None:
    """min/mean/median/max of duration, chars, words + file/duration/byte totals."""
    if not quads:
        return None
    durs = [t[0] for t in quads]
    chars = [t[1] for t in quads]
    words = [t[2] for t in quads]

    def mmmm(xs):
        return (min(xs), statistics.mean(xs), statistics.median(xs), max(xs))

    return {
        "files": len(quads),
        "dur_total": sum(durs),
        "bytes_total": sum(t[3] for t in quads),
        "dur": mmmm(durs),
        "chars": mmmm(chars),
        "words": mmmm(words),
    }


def histogram(quads: list[tuple[float, int, int, int]]) -> list[dict]:
    """Per duration-bucket file count, summed duration, and summed byte size."""
    bins = [{"lo": DUR_EDGES[i], "hi": DUR_EDGES[i + 1], "files": 0, "dur": 0.0, "bytes": 0}
            for i in range(len(DUR_EDGES) - 1)]
    for dur, _, _, sz in quads:
        for b in bins:
            if b["lo"] <= dur < b["hi"]:
                b["files"] += 1
                b["dur"] += dur
                b["bytes"] += sz
                break
    return bins


# --------------------------------------------------------------------------- #
# markdown rendering
# --------------------------------------------------------------------------- #
def _i(x) -> str:
    return f"{x:,.0f}"


def _h(sec: float) -> str:
    return f"{sec / 3600:.2f}"


def _size(nbytes: float) -> str:
    if nbytes >= 1024 ** 3:
        return f"{nbytes / 1024 ** 3:.2f} GB"
    if nbytes >= 1024 ** 2:
        return f"{nbytes / 1024 ** 2:.1f} MB"
    if nbytes >= 1024:
        return f"{nbytes / 1024:.1f} KB"
    return f"{nbytes:.0f} B"


def _pct(part: float, whole: float) -> str:
    return f"{(100.0 * part / whole):.1f}%" if whole else "-"


def _mmmm_int(t) -> str:
    return f"{t[0]:,.0f} / {t[1]:,.1f} / {t[2]:,.1f} / {t[3]:,.0f}"


def _mmmm_dur(t) -> str:
    return f"{t[0]:.2f} / {t[1]:.2f} / {t[2]:.2f} / {t[3]:.2f}"


def _header(label: str, show_sizes: bool) -> str:
    cols = ["Files", "% files", "Dur (h)", "% dur"]
    if show_sizes:
        cols += ["Size", "% size"]
    cols += ["Chars (min/mean/med/max)", "Words (min/mean/med/max)", "Dur s (min/mean/med/max)"]
    return "| " + label + " | " + " | ".join(cols) + " |"


def _rule(show_sizes: bool) -> str:
    n = 1 + 4 + (2 if show_sizes else 0) + 3
    return "|" + "|".join(["---"] * n) + "|"


def _row(label: str, d: dict, tot_files: int, tot_dur: float, tot_bytes: int,
         show_sizes: bool) -> str:
    cells = [_i(d["files"]), _pct(d["files"], tot_files),
             _h(d["dur_total"]), _pct(d["dur_total"], tot_dur)]
    if show_sizes:
        cells += [_size(d["bytes_total"]), _pct(d["bytes_total"], tot_bytes)]
    cells += [_mmmm_int(d["chars"]), _mmmm_int(d["words"]), _mmmm_dur(d["dur"])]
    return "| " + label + " | " + " | ".join(cells) + " |"


def render(cells: dict, root: Path, source: str, show_sizes: bool, missing: int) -> str:
    datasets = sorted({ds for ds, _ in cells})
    all_quads = [t for v in cells.values() for t in v]
    grand = describe(all_quads)
    tot_files, tot_dur, tot_bytes = grand["files"], grand["dur_total"], grand["bytes_total"]

    out: list[str] = []
    out.append("# NeMo manifest statistics")
    out.append("")
    out.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    out.append(f"- Root: `{root}`")
    out.append(f"- Source: **{source}** "
               f"({'per-dataset splits' if source != 'combined' else 'combined release re-split'})")
    out.append(f"- Datasets: {', '.join(datasets)}")
    total_line = f"- Total: **{_i(tot_files)} files**, **{_h(tot_dur)} h** audio"
    if show_sizes:
        total_line += f", **{_size(tot_bytes)}** on disk"
    out.append(total_line)
    if show_sizes and missing:
        out.append(f"- Note: {_i(missing)} clip(s) had no readable file on disk; "
                   f"counted as 0 bytes in the size columns.")
    out.append("")
    out.append("Column groups `min / mean / median / max`. `Dur s` is per-clip "
               "duration in seconds; `Size` is on-disk audio; `% files` / `% dur` / "
               "`% size` are shares of the grand total unless noted.")
    out.append("")

    # By dataset
    out.append("## By dataset")
    out.append("")
    out.append(_header("Dataset", show_sizes))
    out.append(_rule(show_sizes))
    for ds in datasets:
        d = describe([t for (dds, _), v in cells.items() if dds == ds for t in v])
        out.append(_row(ds, d, tot_files, tot_dur, tot_bytes, show_sizes))
    out.append(_row("**all**", grand, tot_files, tot_dur, tot_bytes, show_sizes))
    out.append("")

    # Truncation audit: is any dataset's duration distribution sitting on a ceiling?
    out.append("## Duration ceiling check")
    out.append("")
    out.append("A TTS generation stopped by its token limit is cut off mid-sentence at "
               "exactly the cap, so truncated clips pile up on one duration instead of "
               "tailing off. `At max` counts the clips within one 12.5 Hz frame "
               f"({CEILING_TOL_S:.2f} s) of the longest one: more than a couple, and the "
               "dataset was generated against a cap, not to the end of its text.")
    out.append("")
    out.append("| Dataset | Longest clip | At max | Share | Verdict |")
    out.append("|---|---:|---:|---:|---|")
    capped_any = []
    for ds in datasets + ["**all**"]:
        quads = (all_quads if ds == "**all**"
                 else [t for (dds, _), v in cells.items() if dds == ds for t in v])
        c = duration_ceiling([t[0] for t in quads])
        if not c:
            continue
        if c["capped"] and ds != "**all**":
            capped_any.append(ds)
        verdict = "**CAPPED, clips are truncated**" if c["capped"] else "no ceiling"
        out.append(f"| {ds} | {c['max']:.2f} s | {_i(c['at_max'])} | "
                   f"{c['share'] * 100:.3f}% | {verdict} |")
    out.append("")
    if capped_any:
        out.append(f"> **Warning:** {', '.join(capped_any)} shows a duration ceiling. Those "
                   "clips stopped for length, not at the end of their text, so their audio "
                   "does not match their transcript. Raise the server's max_new_tokens or "
                   "shorten the source text, then regenerate them.")
        out.append("")

    # By split
    out.append("## By split")
    out.append("")
    out.append(_header("Split", show_sizes))
    out.append(_rule(show_sizes))
    for split in SPLITS:
        quads = [t for (_, sp), v in cells.items() if sp == split for t in v]
        d = describe(quads)
        if d:
            out.append(_row(split, d, tot_files, tot_dur, tot_bytes, show_sizes))
    out.append(_row("**all**", grand, tot_files, tot_dur, tot_bytes, show_sizes))
    out.append("")

    # By dataset x split (percentages are within each dataset, so splits sum to 100%)
    out.append("## By dataset x split")
    out.append("")
    out.append("`% files` / `% dur` / `% size` are shares **within the dataset** (so the "
               "three splits sum to 100%), which shows each dataset's split ratio.")
    out.append("")
    out.append(_header("Dataset / split", show_sizes))
    out.append(_rule(show_sizes))
    for ds in datasets:
        ds_quads = [t for (dds, _), v in cells.items() if dds == ds for t in v]
        ds_d = describe(ds_quads)
        for split in SPLITS:
            v = cells.get((ds, split))
            if not v:
                continue
            d = describe(v)
            out.append(_row(f"{ds} / {split}", d, ds_d["files"], ds_d["dur_total"],
                            ds_d["bytes_total"], show_sizes))
    out.append("")

    # Duration distribution (histogram)
    out.append("## Duration distribution")
    out.append("")
    hcols = ["Bucket (s)", "Files", "% files", "Dur (h)", "% dur"]
    if show_sizes:
        hcols += ["Size", "% size"]
    out.append("| " + " | ".join(hcols) + " |")
    out.append("|" + "|".join(["---"] * len(hcols)) + "|")
    for b in histogram(all_quads):
        hi = "inf" if b["hi"] == float("inf") else f"{b['hi']:g}"
        label = f"{b['lo']:g}-{hi}"
        cells_h = [label, _i(b["files"]), _pct(b["files"], tot_files),
                   _h(b["dur"]), _pct(b["dur"], tot_dur)]
        if show_sizes:
            cells_h += [_size(b["bytes"]), _pct(b["bytes"], tot_bytes)]
        out.append("| " + " | ".join(cells_h) + " |")
    out.append("")

    return "\n".join(out)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--root", type=click.Path(path_type=Path), default=DEFAULT_ROOT,
              show_default=True, help="Folder holding the NeMo manifests.")
@click.option("--source", type=click.Choice(["per-dataset", "combined"]), default="per-dataset",
              show_default=True, help="Which split files to read (see --help).")
@click.option("--sizes/--no-sizes", "show_sizes", default=True, show_default=True,
              help="Probe on-disk audio size (needs the clips present).")
@click.option("--jobs", type=int, default=DEFAULT_JOBS, show_default=True,
              help="Threads for the size probe.")
@click.option("--output", type=click.Path(path_type=Path), default=None,
              help="Write the Markdown here instead of stdout.")
def main(root: Path, source: str, show_sizes: bool, jobs: int, output: Path | None):
    root = root.resolve()
    if not root.is_dir():
        raise click.UsageError(f"root not found: {root} (run 01_build_nemo_manifest.py first)")
    cells, missing = load_cells(root, source, show_sizes, jobs)
    if not cells:
        raise click.UsageError(
            f"no {'/'.join(SPLITS)} manifests found under {root} for source={source!r}; "
            f"run 01_build_nemo_manifest.py (per-dataset) or 02_combine_nemo_manifests.py (combined)")
    md = render(cells, root, source, show_sizes, missing)
    if output:
        Path(output).write_text(md, encoding="utf-8")
        click.echo(f"wrote {output}", err=True)
    else:
        click.echo(md)


if __name__ == "__main__":
    main()
