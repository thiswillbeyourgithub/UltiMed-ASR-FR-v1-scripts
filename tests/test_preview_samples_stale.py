"""A re-sampled preview must not keep the previous sampling's clips.

Before the fix, ``build_subset`` only copied the new picks, so after the v1.3 drops the
dictionary preview folder held 15 FLACs for a 5-line manifest, and the extra clips were
uploaded to the Hub with it.

    python tests/test_preview_samples_stale.py

This file was written by Claude Code.
"""
from __future__ import annotations

import importlib.util
import json
import random
import tempfile
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "99_hf_release" / "scripts" / "build_preview_samples.py"
_spec = importlib.util.spec_from_file_location("build_preview_samples", _SCRIPT)
bps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bps)


def test_stale_clips_removed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "NeMO_files" / "dictionary"
        src.mkdir(parents=True)
        rows = []
        for i in range(3):
            (src / f"clip{i}.flac").write_bytes(b"fLaC")
            rows.append({"audio_filepath": f"clip{i}.flac", "duration": 1.0, "text": f"t{i}"})
        (src / "full.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        out = root / "preview"
        (out / "dictionary").mkdir(parents=True)
        (out / "dictionary" / "old_pick.flac").write_bytes(b"fLaC")
        bps.MANIFEST_ROOT = root / "NeMO_files"
        n = bps.build_subset("dictionary", "dictionary", 2, out, random.Random(0))
        flacs = sorted(p.name for p in (out / "dictionary").glob("*.flac"))
        assert n == 2 and len(flacs) == 2 and "old_pick.flac" not in flacs, flacs


if __name__ == "__main__":
    test_stale_clips_removed()
    print("ok")
