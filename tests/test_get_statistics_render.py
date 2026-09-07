#!/usr/bin/env python
"""`99_hf_release/scripts/get_statistics.py` must actually render its report.

The truncation audit was moved into `utils/nemo_manifest.py` and shared with
`06_hotfixes/02_statistics.py`, but only `duration_ceiling` was imported back: the
tolerance constant `CEILING_TOL_S` stayed a bare global reference in the report text.
Nothing failed at import time, so the script died with `NameError` only once it reached
the "Duration ceiling check" section, i.e. after the whole (multi-minute) manifest scan
and at the exact moment the release numbers were wanted.

Every other test here imports a module and calls a function, which would NOT have caught
this: the reference is inside a rendering branch. So this one runs the real script, end
to end, over a synthetic two-dataset manifest tree, and only checks that it succeeds and
reports what it read. `--no-sizes` keeps it off the disk probe, so no audio is needed.

    python tests/test_get_statistics_render.py

This file was written with Claude Code.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "99_hf_release" / "scripts" / "get_statistics.py"
SPLITS = ("train", "val", "test")


def write_manifest(path: Path, n: int, dur: float, *, pile_up: int = 0) -> None:
    """n rows of `dur` seconds, plus `pile_up` extra rows pinned at one longer value."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"audio_filepath": f"{path.stem}_{i}.flac", "duration": dur + i * 0.1,
         "text": "un texte de test", "category": path.parent.name}
        for i in range(n)
    ]
    rows += [
        {"audio_filepath": f"{path.stem}_cap_{i}.flac", "duration": 163.84,
         "text": "un texte tronque", "category": path.parent.name}
        for i in range(pile_up)
    ]
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def test_render_runs_over_a_synthetic_tree() -> None:
    if shutil.which("uv") is None:
        print("SKIP test_render_runs_over_a_synthetic_tree (uv not on PATH)")
        return
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "NeMO_files"
        # One healthy dataset and one whose clips pile up on the old 2048-frame cap, so
        # BOTH branches of the ceiling section are rendered (quiet verdict and warning).
        for split, n in zip(SPLITS, (40, 5, 5)):
            write_manifest(root / "dictionary" / f"{split}.jsonl", n, 2.0)
            write_manifest(root / "parhaf" / f"{split}.jsonl", n, 12.0, pile_up=n)
        out = Path(td) / "stats.md"
        proc = subprocess.run(
            ["uv", "run", str(SCRIPT), "--root", str(root),
             "--source", "per-dataset", "--no-sizes", "--output", str(out)],
            capture_output=True, text=True, cwd=str(REPO),
        )
        assert proc.returncode == 0, (
            f"get_statistics.py exited {proc.returncode}\n"
            f"--- stderr ---\n{proc.stderr[-3000:]}"
        )
        md = out.read_text(encoding="utf-8")
        # The section that held the undefined name, and the numbers around it.
        assert "Duration ceiling check" in md, md[-2000:]
        assert "0.08 s" in md, "the frame tolerance must be spelled out in the report"
        assert "163.84" in md, "the pile-up dataset must be reported at its cap"
        assert "dictionary" in md and "parhaf" in md, "both datasets must appear"
    print("OK test_render_runs_over_a_synthetic_tree")


if __name__ == "__main__":
    test_render_runs_over_a_synthetic_tree()
    print("all OK")
