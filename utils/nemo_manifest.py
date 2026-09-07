"""Shared NeMo-manifest assembly for the ``99_hf_release`` stage.

Turns each text stage's ``generated_dataset.jsonl`` plus its synthesized ``.flac``
clips into NeMo ASR manifests (one JSON object per line with ``audio_filepath``,
``duration``, ``text``, plus grouping provenance) and owns the grouped,
duration-aware train/val/test splitter that both stage-99 scripts share:

* ``01_build_nemo_manifest.py`` (per-dataset builder, one subfolder each), and
* ``02_combine_nemo_manifests.py`` (parent combiner, global re-split).

Kept in ``utils/`` so neither script forks the split policy. The split logic and
the small helpers (``parse_split``, ``assign_splits``, ``summarize``,
``relativize``, jsonl io) are stdlib-only *at import time*: ``soundfile`` and
``tqdm`` are imported lazily inside the audio-probing helpers, so the unit tests
can import and exercise the splitter without those wheels.

Two group behaviours, chosen per dataset by the builder:

* **distribute** (dictionary / drugs): the N ``asr_training_target`` variants of a
  term are spread across splits. A group of >=2 keeps at least one variant in
  train; under strict coverage a group of >=3 also keeps at least one in *every*
  enabled split, so each split sees the term. This is augmentation-style data,
  spreading it is wanted. The per-split coverage can be made best-effort (see
  ``assign_splits(strict_coverage=False)``) so val/test hit their target ratio.
* **atomic** (PARHAF / PARROT): every chunk of one source document (``source_id``)
  goes to the *same* split, so no clinical document leaks across train/test.

Singletons (a one-variant term, a one-chunk document) are placed by ratio like any
other free item. The greedy placement fills the currently most under-target split,
weighted by clip duration when ``by_duration`` is set (duration is the proxy for
"amount of speech / tokens" the split should balance), else by row count. The whole
procedure is deterministic (no RNG): same inputs give the same split.

Written with Claude Code.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Callable, Iterable, Sequence

log = logging.getLogger("nemo_manifest")

# Fixed split order; also the deterministic tiebreak when two splits are equally
# under-target. train first so an all-empty tie seeds train.
SPLITS: tuple[str, ...] = ("train", "val", "test")

# A NeMo training row reads only these three; the rest are provenance the parent
# combiner re-uses to re-split without re-reading the source datasets.
NEMO_CORE = ("audio_filepath", "duration", "text")


# --------------------------------------------------------------------------- #
# split-spec parsing
# --------------------------------------------------------------------------- #
def parse_split(spec: str) -> dict[str, float]:
    """Parse a ``"train/val/test"`` spec into normalised fractions.

    Accepts percentages or fractions in any scale (``"80/10/10"``,
    ``"8/1/1"``, ``"0.8/0.1/0.1"`` all give the same result); the three values
    are divided by their sum. A value may be ``0`` to disable that split
    (e.g. ``"90/10/0"`` for train/val only). Raises ``ValueError`` on anything
    that is not three non-negative numbers summing to > 0.
    """
    parts = [p for p in spec.replace(",", "/").split("/") if p.strip() != ""]
    if len(parts) != 3:
        raise ValueError(f"--split needs 3 values train/val/test, got {spec!r}")
    try:
        vals = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"--split values must be numbers, got {spec!r}") from exc
    if any(v < 0 for v in vals):
        raise ValueError(f"--split values must be >= 0, got {spec!r}")
    total = sum(vals)
    if total <= 0:
        raise ValueError(f"--split values must sum to > 0, got {spec!r}")
    return {s: v / total for s, v in zip(SPLITS, vals)}


# --------------------------------------------------------------------------- #
# the grouped, duration-aware splitter
# --------------------------------------------------------------------------- #
def assign_splits(
    rows: Sequence[dict],
    ratios: dict[str, float],
    *,
    stratify: bool = True,
    by_duration: bool = True,
    strict_coverage: bool = True,
) -> list[str]:
    """Assign each row to ``"train"`` / ``"val"`` / ``"test"``.

    ``rows`` is a sequence of dicts; each must carry ``group_id`` (hashable),
    ``group_mode`` (``"atomic"`` or ``"distribute"``) and ``duration`` (float).
    Returns a list of split labels parallel to ``rows``.

    * ``stratify=False`` ignores groups entirely: every row is placed
      independently by ratio (still duration-weighted when ``by_duration``).
    * ``stratify=True`` honours the per-group guarantees described in the module
      docstring: atomic groups stay whole in one split; distribute groups of >=2
      keep >=1 in train.
    * ``strict_coverage=True`` additionally forces a distribute group of >=3 to
      put >=1 in *every* enabled split. This guarantees each split sees the term,
      but floors val/test at one sample per such group, so a dataset with many
      small term-groups overshoots the target ratio (e.g. val/test ~15% instead
      of 10%). ``strict_coverage=False`` makes that per-split coverage
      best-effort: only >=1-in-train is kept as a hard guarantee and the rest of
      the group is placed greedily, so val/test converge on their target ratio
      (big groups still tend to span all splits naturally). ``by_duration`` and
      the atomic guarantee are unaffected.

    Deterministic: groups are processed largest-weight first (a
    longest-processing-time greedy that balances well), ties broken by
    ``str(group_id)``; within a group items are processed largest-first, ties
    broken by original position.
    """
    enabled = [s for s in SPLITS if ratios[s] > 0]
    if not enabled:  # parse_split guarantees at least one, but stay defensive
        raise ValueError("no split has a positive ratio")

    totals: dict[str, float] = {s: 0.0 for s in SPLITS}

    def weight(i: int) -> float:
        return float(rows[i]["duration"]) if by_duration else 1.0

    def pick(candidates: Iterable[str]) -> str:
        # Most under-target = smallest achieved/target ratio; SPLITS order breaks
        # ties (so the first placements seed train, then val, then test).
        return min(candidates, key=lambda s: (totals[s] / ratios[s], SPLITS.index(s)))

    labels: list[str | None] = [None] * len(rows)

    def place(i: int, s: str) -> None:
        totals[s] += weight(i)
        labels[i] = s

    if not stratify:
        # Every row independent; largest-first for balance.
        order = sorted(range(len(rows)), key=lambda i: (-weight(i), i))
        for i in order:
            place(i, pick(enabled))
        return [s for s in labels]  # type: ignore[misc]

    # Bucket rows into their groups (insertion order preserved for the tiebreak).
    groups: dict[object, list[int]] = {}
    modes: dict[object, str] = {}
    for i, row in enumerate(rows):
        gid = row["group_id"]
        groups.setdefault(gid, []).append(i)
        modes[gid] = row.get("group_mode", "distribute")

    def group_weight(idxs: list[int]) -> float:
        return sum(weight(i) for i in idxs)

    # Largest groups first (LPT); str(gid) tiebreak keeps it deterministic.
    ordered_gids = sorted(groups, key=lambda g: (-group_weight(groups[g]), str(g)))

    for gid in ordered_gids:
        idxs = sorted(groups[gid], key=lambda i: (-weight(i), i))
        mode = modes[gid]

        if len(idxs) == 1:
            place(idxs[0], pick(enabled))
            continue

        if mode == "atomic":
            # Whole document to one split (no chunk leakage across splits).
            s = pick(enabled)
            for i in idxs:
                place(i, s)
            continue

        # distribute, len >= 2
        if len(enabled) == 1:
            for i in idxs:
                place(i, enabled[0])
            continue

        # Coverage guarantee: >=1 in train always; and, under strict_coverage,
        # >=1 in every enabled split when the group has >=3 items. Assign the
        # largest items to the required splits, each to whichever required split
        # is currently most under-target. With strict_coverage off, only train is
        # reserved and the rest fall through to the greedy fill, so val/test are
        # not floored above their target ratio.
        cover_all = strict_coverage and len(idxs) >= 3
        required = list(enabled) if cover_all else (
            ["train"] if "train" in enabled else [enabled[0]]
        )
        covered = 0
        to_cover = list(required)
        for i in idxs:
            if not to_cover:
                break
            s = pick(to_cover)
            place(i, s)
            to_cover.remove(s)
            covered += 1
        for i in idxs[covered:]:
            place(i, pick(enabled))

    return [s for s in labels]  # type: ignore[misc]


def summarize(rows: Sequence[dict], labels: Sequence[str]) -> dict[str, dict[str, float]]:
    """Per-split ``{count, duration, count_pct, duration_pct}`` for logging."""
    agg = {s: {"count": 0, "duration": 0.0} for s in SPLITS}
    for row, s in zip(rows, labels):
        agg[s]["count"] += 1
        agg[s]["duration"] += float(row.get("duration") or 0.0)
    tot_c = sum(a["count"] for a in agg.values()) or 1
    tot_d = sum(a["duration"] for a in agg.values()) or 1.0
    for a in agg.values():
        a["count_pct"] = 100.0 * a["count"] / tot_c
        a["duration_pct"] = 100.0 * a["duration"] / tot_d
    return agg


# A TTS backend that stops on its token limit produces clips cut off at exactly the
# same length, because the limit is a frame count: Voxtral emits one audio frame every
# 1/12.5 s, so a truncated clip lands on the cap to the frame. The signature is a PILE-UP
# at the longest duration, where a healthy distribution tails off with a single clip at
# its max. This is the audit that would have caught the 163.84 s truncation, and it lives
# here so both the release statistics (99_hf_release/scripts/get_statistics.py) and the
# hotfix statistics (06_hotfixes/02_statistics.py) ask the question the same way.
CEILING_TOL_S = 0.08          # one Voxtral frame
CEILING_WARN_SHARE = 0.002    # 0.2% of a dataset sitting on its own maximum is not chance


def duration_ceiling(durs: Sequence[float]) -> dict | None:
    """Look for a hard cap at the top of a duration distribution. Returns the longest
    clip, how many clips sit within one frame of it, and whether that pile-up is big
    enough to mean a cap rather than coincidence."""
    if not durs:
        return None
    top = max(durs)
    at_top = sum(1 for d in durs if d >= top - CEILING_TOL_S)
    share = at_top / len(durs)
    return {"max": top, "at_max": at_top, "share": share,
            "capped": at_top > 1 and share >= CEILING_WARN_SHARE}


def format_summary(agg: dict[str, dict[str, float]]) -> str:
    """One-line-per-split human summary of a ``summarize`` result."""
    lines = []
    for s in SPLITS:
        a = agg[s]
        lines.append(
            f"  {s:5s}: {a['count']:>8d} rows ({a['count_pct']:5.1f}%)  "
            f"{a['duration'] / 3600.0:8.2f} h ({a['duration_pct']:5.1f}%)"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# audio index + duration probing
# --------------------------------------------------------------------------- #
def _stem_key(term_index: int, variant_index: int) -> str:
    """The stage-05 filename prefix, zero-padded exactly as
    ``05_generate_audio/01_generate_audio.py`` writes it. Six/four digits make
    the prefix fixed-width, so it can never be ambiguous between rows."""
    return f"{int(term_index):06d}_{int(variant_index):04d}"


def parse_clip_name(name: str) -> tuple[int, int, str] | None:
    """Split a stage-05 clip filename into ``(term_index, variant_index, slug)``.

    Stage 05 names every clip ``{term_index:06d}_{variant_index:04d}_{slug}.flac`` (see
    ``05_generate_audio/01_generate_audio.py``), which makes the name the only per-clip
    identity a NeMo manifest row carries: the manifest keeps ``group_id`` / ``item_index``
    and drops the source term. Returns ``None`` for anything that is not a clip (a
    ``*.flac.tmp`` leftover, a name without the two zero-padded index fields).
    """
    if not name.endswith(".flac"):
        return None
    parts = name[: -len(".flac")].split("_", 2)
    if len(parts) < 2 or not (
        len(parts[0]) == 6 and parts[0].isdigit()
        and len(parts[1]) == 4 and parts[1].isdigit()
    ):
        return None
    return int(parts[0]), int(parts[1]), parts[2] if len(parts) > 2 else ""


def audio_index(audio_dir: Path) -> dict[str, Path]:
    """Scan ``audio_dir`` once and map each clip's ``termidx_varidx`` prefix to
    its path, so a row is matched by its two indices rather than by re-deriving
    the term slug. A leftover ``*.tmp`` or a name without the two leading
    zero-padded index fields is skipped. First file wins on a duplicate prefix
    (warned)."""
    index: dict[str, Path] = {}
    for entry in os.scandir(audio_dir):
        name = entry.name
        # Only real clips: this also skips a "*.flac.tmp" leftover, a stray
        # sub-symlink (e.g. a self-referential loop) and dotfolders, without ever
        # stat-ing them (following a symlink loop would raise OSError).
        parsed = parse_clip_name(name)
        if parsed is None:
            continue
        try:
            if not entry.is_file():
                continue
        except OSError:
            continue  # broken/looping symlink named *.flac
        key = f"{parsed[0]:06d}_{parsed[1]:04d}"
        if key in index:
            log.warning("duplicate audio prefix %s: %s and %s", key, index[key].name, entry.name)
            continue
        index[key] = Path(entry.path)
    return index


def probe_durations(
    paths: Sequence[Path],
    cache_path: Path | None = None,
    jobs: int = 8,
    progress: bool = True,
) -> dict[str, float]:
    """Return ``{filename: duration_seconds}`` for ``paths``.

    Durations are read from the FLAC header only (``soundfile.info``), never a
    full decode. Results are cached by *file name* (location-independent, so the
    cache survives the audio folder being moved or re-symlinked) in
    ``cache_path`` as JSON; a re-run probes only the clips missing from it.
    Corrupt/unreadable clips are logged and omitted from the result.
    """
    import soundfile as sf  # lazy: keeps the module importable without the wheel

    cache: dict[str, float] = {}
    if cache_path and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("could not read duration cache %s, reprobing", cache_path)

    todo = [p for p in paths if p.name not in cache]
    if todo:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def one(p: Path) -> tuple[str, float | None]:
            try:
                info = sf.info(str(p))
                return p.name, (info.frames / info.samplerate if info.samplerate else None)
            except Exception as exc:  # noqa: BLE001 - any decode error drops the clip
                log.warning("cannot read duration of %s: %s", p.name, exc)
                return p.name, None

        bar = None
        if progress:
            try:
                from tqdm import tqdm
                bar = tqdm(total=len(todo), unit="clip", desc="probing durations")
            except ImportError:
                bar = None
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            for fut in as_completed(pool.submit(one, p) for p in todo):
                name, dur = fut.result()
                if dur is not None:
                    cache[name] = dur
                if bar is not None:
                    bar.update(1)
        if bar is not None:
            bar.close()
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
            tmp.write_text(json.dumps(cache))
            tmp.replace(cache_path)

    return {p.name: cache[p.name] for p in paths if p.name in cache}


# --------------------------------------------------------------------------- #
# path handling + jsonl io
# --------------------------------------------------------------------------- #
def relativize(audio_abs: str | Path, manifest_dir: Path) -> str:
    """POSIX path of ``audio_abs`` relative to the directory the manifest lives
    in. Relative (not absolute) on purpose: absolute paths under the audio SSD
    embed the machine's home/username, which must never be baked into a file
    that could be committed or published."""
    return Path(os.path.relpath(Path(audio_abs), manifest_dir)).as_posix()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    """Write ``rows`` as JSONL (utf-8, non-ascii kept). Returns the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
            n += 1
    tmp.replace(path)
    return n


def write_splits(
    out_dir: Path,
    rows: Sequence[dict],
    labels: Sequence[str],
    manifest_dir: Path | None = None,
    absolute: bool = False,
    audio_key: str = "_audio_abs",
) -> dict[str, int]:
    """Write ``full/train/val/test.jsonl`` into ``out_dir``.

    Each row is emitted with only the NeMo fields plus provenance, and its
    ``audio_filepath`` set from ``row[audio_key]`` (an absolute path kept
    internally): relative to ``manifest_dir`` (default ``out_dir``) unless
    ``absolute``. Returns ``{split_or_full: count}``.
    """
    manifest_dir = manifest_dir or out_dir

    def emit(row: dict) -> dict:
        audio_abs = row[audio_key]
        out = {
            "audio_filepath": str(audio_abs) if absolute else relativize(audio_abs, manifest_dir),
            "duration": round(float(row["duration"]), 3),
            "text": row["text"],
        }
        # The deterministic TTS input (asr_training_source) alongside the label
        # (text): passed through when present so downstream artifacts (the Parquet
        # release) ship both. NeMo training ignores it.
        if "asr_training_source" in row:
            out["asr_training_source"] = row["asr_training_source"]
        for k in ("category", "group_id", "group_mode", "item_index"):
            if k in row:
                out[k] = row[k]
        return out

    counts: dict[str, int] = {}
    counts["full"] = write_jsonl(out_dir / "full.jsonl", (emit(r) for r in rows))
    for s in SPLITS:
        sel = [emit(r) for r, lab in zip(rows, labels) if lab == s]
        counts[s] = write_jsonl(out_dir / f"{s}.jsonl", sel)
    return counts
