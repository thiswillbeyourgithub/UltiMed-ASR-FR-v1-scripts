#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.31",
#   "jiwer>=3.0",
#   "tqdm>=4.66",
#   "loguru>=0.7",
#   "click>=8.1",
# ]
# ///
"""Copy the clips this pipeline finds suspicious into one local folder, so they can be
LISTENED to instead of being trusted to a number.

Every gate in this stage is a proxy: a high CER usually means bad audio, but it can also
mean Whisper mishearing a rare drug name, and a clip nobody ever listens to can sit in
the release for months either way. This script takes the reasons the pipeline already
computes, gathers the audio behind them in one place with the text next to it, and gets
out of the way.

    uv run 03_collect_suspicious.py                       # every reason, default paths
    uv run 03_collect_suspicious.py --dry-run             # what it would copy, and how big
    uv run 03_collect_suspicious.py --reason worst-cer --limit 50
    uv run 03_collect_suspicious.py --match 'facteur V'   # hear one normalization case

Reasons (each capped independently by ``--limit``, 0 = no cap):

* ``worst-cer``   : whole-clip CER at or above ``--cer-threshold``, worst first.
* ``worst-tail``  : tail CER at or above ``--tail-cer-threshold`` (long clips whose END
  derailed, which a whole-clip CER dilutes into invisibility).
* ``pending-tts`` : flagged bad and waiting for regeneration.
* ``exhausted``   : regeneration gave up, so this audio ships as it is unless the text
  is fixed. The most important pile to actually listen to.
* ``ceiling``     : duration pinned at the top of the distribution, i.e. the shape a TTS
  that stopped on its token limit leaves behind (shares the release statistics'
  detector, ``utils/nemo_manifest.duration_ceiling``). Silent when there is no ceiling.
* ``stt-error``   : the transcript is missing or blank, so nothing was ever judged.
* ``match``       : label matching ``--match``, a plain listening sample rather than a
  defect (use it to check how a normalization rule sounds, e.g. Roman numerals).

Alongside the audio it writes ``index.md`` (readable: label, spoken text, transcript,
scores) and ``index.jsonl`` (the same, machine-readable). Filenames are prefixed with
the reason and the score so ``ls`` sorts by badness.

Scores come from the jsonl as it stands, so re-run after a ``RESCORE_ONLY=1`` pass if the
scoring rules changed. This file was written with Claude Code.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

import click
from loguru import logger
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "utils"))
from nemo_manifest import CEILING_TOL_S, duration_ceiling  # noqa: E402


def _load_module(path: Path, name: str):
    """Import a sibling script whose filename is not a valid module name."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# resolve_audio / read the scoring constants from script 1 rather than copying them.
_stt = _load_module(_HERE / "01_compute_stt.py", "compute_stt_mod")

REASONS = ("worst-cer", "worst-tail", "pending-tts", "exhausted", "ceiling",
           "stt-error", "match")


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def entry_of(rec: dict, model_key: str | None) -> dict:
    """The transcription entry to judge: the requested model, else the only one there."""
    tr = rec.get("transcriptions") or {}
    if model_key:
        return tr.get(model_key) or {}
    return next(iter(tr.values()), {}) if len(tr) == 1 else (tr.get("whisper-1") or {})


def safe_stem(rec: dict, audio_key: str) -> str:
    raw = rec.get(audio_key) or "clip"
    return re.sub(r"[^\w.-]", "_", Path(raw).stem)[:110]


def collect(rows: list[dict], cfg) -> dict[str, list[tuple[float, dict]]]:
    """Bucket the rows by reason. A row can land in several buckets: it is copied once,
    under the first reason that claimed it, and the index lists all of them."""
    picked: dict[str, list[tuple[float, dict]]] = {r: [] for r in REASONS}
    durations = [d for d in (r.get("duration") for r in rows)
                 if isinstance(d, (int, float)) and d > 0]
    ceiling = duration_ceiling(durations)
    match_re = re.compile(cfg.match, re.IGNORECASE) if cfg.match else None

    for rec in rows:
        entry = entry_of(rec, cfg.model)
        cer, tail = entry.get("cer"), entry.get("cer_tail")
        status = (rec.get("improvement") or {}).get("status")
        text = rec.get("text") or rec.get("asr_training_target") or ""

        if cer is not None and cer >= cfg.cer_threshold:
            picked["worst-cer"].append((-cer, rec))
        if tail is not None and tail >= cfg.tail_cer_threshold:
            picked["worst-tail"].append((-tail, rec))
        if status == "pending_tts":
            picked["pending-tts"].append((-(cer or 0.0), rec))
        if status == "exhausted":
            picked["exhausted"].append((-(cer or 0.0), rec))
        if entry.get("error") or (entry and not (entry.get("text") or "").strip()):
            picked["stt-error"].append((0.0, rec))
        # A ceiling is a pile-up, not simply the longest clip: one clip at the max is a
        # normal distribution and copying it would be noise.
        if ceiling and ceiling["capped"] and (rec.get("duration") or 0) >= ceiling["max"] - CEILING_TOL_S:
            picked["ceiling"].append((-(rec.get("duration") or 0.0), rec))
        if match_re and match_re.search(text):
            picked["match"].append((0.0, rec))

    for reason in picked:
        picked[reason].sort(key=lambda pair: pair[0])
        if cfg.limit:
            picked[reason] = picked[reason][:cfg.limit]
    return picked


def write_index(out_dir: Path, copied: list[dict], input_file: Path, cfg) -> None:
    (out_dir / "index.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in copied),
        encoding="utf-8")

    lines = [f"# Suspicious clips from `{input_file.name}`", "",
             f"{len(copied)} clip(s), gates: CER >= {cfg.cer_threshold}, "
             f"tail CER >= {cfg.tail_cer_threshold}."
             + (f" Label match: `{cfg.match}`." if cfg.match else ""), "",
             "Scores are whatever the jsonl held when this ran: re-run after a "
             "`RESCORE_ONLY=1` pass if the scoring rules have changed since.", "",
             "| File | Reasons | CER | tail | dur | category |",
             "|---|---|--:|--:|--:|---|"]
    for row in copied:
        cer = f"{row['cer']:.3f}" if row["cer"] is not None else "-"
        tail = f"{row['cer_tail']:.3f}" if row["cer_tail"] is not None else "-"
        lines.append(f"| `{row['file']}` | {', '.join(row['reasons'])} | {cer} | {tail} "
                     f"| {row['duration']:.1f}s | {row['category'] or '-'} |")
    lines.append("")
    lines.append("## What each clip should say, and what was heard")
    for row in copied:
        lines += [
            "", f"### `{row['file']}`",
            f"*{', '.join(row['reasons'])}, CER "
            + (f"{row['cer']:.3f}" if row["cer"] is not None else "n/a") + "*", "",
            f"- **label** (what the clip must say): {row['text']}",
        ]
        if row["asr_training_source"] and row["asr_training_source"] != row["text"]:
            lines.append(f"- **spoken** (what the TTS was given): {row['asr_training_source']}")
        lines.append(f"- **heard** (transcript): {row['transcript'] or '(nothing)'}")
    (out_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--input", "input_path", type=click.Path(path_type=Path),
              default=Path("improved/full.stt.jsonl"), show_default=True,
              help="Scored jsonl to read (the .stt.jsonl an stt pass writes).")
@click.option("--output", "out_dir", type=click.Path(path_type=Path),
              default=Path("suspicious_audio"), show_default=True,
              help="Local folder to fill with the clips and the index.")
@click.option("--audio-root", type=click.Path(path_type=Path),
              default=_HERE.parent / "99_hf_release" / "data" / "NeMO_files",
              help="Root for relative audio_filepath values. Defaults to the manifest "
                   "root the driver passes, because the paths in a .stt.jsonl are "
                   "relative to the MANIFEST it came from, not to itself.")
@click.option("--audio-key", default="audio_filepath", show_default=True)
@click.option("--model", default=None,
              help="Which transcription model's scores to read (default: the only one, else whisper-1).")
@click.option("--reason", "reasons", multiple=True, type=click.Choice(REASONS),
              help="Only these reasons (repeatable). Default: all of them.")
@click.option("--match", default=None,
              help="Also copy clips whose LABEL matches this regex, to listen to a "
                   "specific case (e.g. 'facteur V' for the Roman-numeral rule).")
@click.option("--cer-threshold", default=0.08, show_default=True, type=float)
@click.option("--tail-cer-threshold", default=0.12, show_default=True, type=float)
@click.option("--limit", default=0, show_default=True, type=int,
              help="Max clips per reason (0 = no cap).")
@click.option("--symlink", is_flag=True,
              help="Symlink instead of copying (no disk cost, but needs the audio drive mounted to play).")
@click.option("--dry-run", is_flag=True, help="Report the selection and its size, copy nothing.")
def main(input_path, out_dir, audio_root, audio_key, model, reasons, match,
         cer_threshold, tail_cer_threshold, limit, symlink, dry_run):
    input_file = Path(input_path).resolve()
    if not input_file.is_file():
        raise click.UsageError(f"input not found: {input_file}")
    cfg = click.get_current_context().params
    cfg = type("Cfg", (), {**cfg, "match": match, "model": model, "limit": limit,
                           "cer_threshold": cer_threshold,
                           "tail_cer_threshold": tail_cer_threshold})

    logger.info(f"reading {input_file}")
    rows = []
    with input_file.open(encoding="utf-8") as fh:
        for i, line in enumerate(tqdm(fh, desc="scan", unit="row")):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"skipping unreadable line {i}")
    logger.info(f"{len(rows)} rows")

    picked = collect(rows, cfg)
    wanted = set(reasons) if reasons else set(REASONS)
    for reason in REASONS:
        n = len(picked[reason])
        if n and reason not in wanted:
            logger.info(f"{reason:<12} {n:>6} clip(s)  (skipped, not in --reason)")
        elif reason in wanted:
            logger.info(f"{reason:<12} {n:>6} clip(s)")

    # One copy per clip, listing every reason it was picked for.
    order: dict[str, dict] = {}
    for reason in REASONS:
        if reason not in wanted:
            continue
        for _, rec in picked[reason]:
            key = rec.get(audio_key) or id(rec)
            slot = order.setdefault(key, {"rec": rec, "reasons": []})
            if reason not in slot["reasons"]:
                slot["reasons"].append(reason)

    root = Path(audio_root).resolve() if audio_root else None
    total_bytes, missing, plan = 0, 0, []
    for key, slot in order.items():
        rec = slot["rec"]
        src = _stt.resolve_audio(rec, input_file, audio_key, root)
        if src is None:
            missing += 1
            continue
        entry = entry_of(rec, model)
        cer = entry.get("cer")
        prefix = slot["reasons"][0]
        score = f"{cer:.3f}" if cer is not None else "none"
        name = f"{prefix}_{score}_{rec.get('category') or 'na'}_{safe_stem(rec, audio_key)}{src.suffix}"
        total_bytes += src.stat().st_size
        plan.append((src, name, rec, slot["reasons"], entry))

    logger.info(f"{len(plan)} clip(s) to {'link' if symlink else 'copy'}, "
                f"{human_size(total_bytes)}"
                + (f", {missing} with no audio on disk" if missing else ""))
    if dry_run:
        logger.info("--dry-run: nothing written")
        return
    if not plan:
        logger.info("nothing suspicious, nothing to listen to")
        return

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    copied = []
    for src, name, rec, why, entry in tqdm(plan, desc="copy", unit="clip"):
        dest = out / name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        if symlink:
            dest.symlink_to(src)
        else:
            shutil.copy2(src, dest)
        copied.append({
            "file": name, "reasons": why, "audio_filepath": rec.get(audio_key),
            "category": rec.get("category"), "duration": rec.get("duration") or 0.0,
            "cer": entry.get("cer"), "cer_tail": entry.get("cer_tail"),
            "status": (rec.get("improvement") or {}).get("status"),
            "text": rec.get("text") or rec.get("asr_training_target") or "",
            "asr_training_source": rec.get("asr_training_source") or "",
            "transcript": entry.get("text") or "",
        })

    write_index(out, copied, input_file, cfg)
    logger.success(f"wrote {len(copied)} clip(s) to {out}/ (start with index.md)")


if __name__ == "__main__":
    main()
