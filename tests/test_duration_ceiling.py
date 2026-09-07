#!/usr/bin/env python
"""The statistics scripts' truncation audit must actually fire.

`duration_ceiling` in `utils/nemo_manifest.py` is what catches a TTS backend that
stopped on its token limit: those clips are cut off at exactly the frame cap, so they
pile up on one duration instead of tailing off. Both statistics scripts import it
(`99_hf_release/scripts/get_statistics.py` and `06_hotfixes/02_statistics.py`). A
detector that silently never fires is worse than none, so this pins both directions: a
healthy distribution stays quiet, and the 163.84 s pile-up that the old 2048-frame cap
produced is reported.

    python tests/test_duration_ceiling.py

This file was written with Claude Code.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

MOD_PATH = Path(__file__).resolve().parent.parent / "utils" / "nemo_manifest.py"


def load_module():
    spec = importlib.util.spec_from_file_location("nemo_manifest_under_test", MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_duration_ceiling() -> None:
    mod = load_module()
    CAP = 163.84  # 2048 Voxtral frames at 12.5 Hz

    assert mod.duration_ceiling([]) is None

    # Healthy: durations tail off, exactly one clip at the maximum.
    healthy = [1.0 + i * 0.15 for i in range(1000)]  # 1.0 s to 150.8 s, all under the cap
    c = mod.duration_ceiling(healthy)
    assert c["at_max"] == 1 and not c["capped"], c

    # A single long clip is not a ceiling either (nothing piles up on it).
    c = mod.duration_ceiling([2.0, 5.0, 9.5, CAP])
    assert c["at_max"] == 1 and not c["capped"], c

    # Capped: the real shape of the old PARHAF audio, 42% of clips pinned at the cap.
    capped = healthy[:580] + [CAP] * 420
    c = mod.duration_ceiling(capped)
    assert c["capped"], c
    assert c["at_max"] == 420 and abs(c["max"] - CAP) < 1e-9, c

    # Frame-quantized durations land within a frame of each other, not exactly on it.
    near = healthy[:995] + [CAP, CAP - 0.04, CAP - 0.08, CAP - 0.02, CAP - 0.06]
    c = mod.duration_ceiling(near)
    assert c["at_max"] == 5 and c["capped"], "a pile-up within one frame still counts"

    # Just under the warning share (2 clips in 10000 = 0.02%... below 0.2%) stays quiet,
    # so a couple of coincidentally equal clips never cries wolf.
    quiet = [1.0 + i * 0.01 for i in range(10000)] + [500.0, 500.0]
    c = mod.duration_ceiling(quiet)
    assert c["at_max"] == 2 and not c["capped"], c
    print("OK test_duration_ceiling")


if __name__ == "__main__":
    test_duration_ceiling()
    print("all OK")
