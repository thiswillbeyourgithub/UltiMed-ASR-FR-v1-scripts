"""Stdlib test for 99_hf_release/04_drop_exhausted.py's row filter.

Checks that only `qc_status == "exhausted"` rows are dropped, that order is kept, and
that rows never synced by 03_sync_hotfix_results.py (no `qc_status` key) are kept but
counted, so the script can warn instead of calling them clean.

Run: python tests/test_drop_exhausted.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_sync_pass_guard import load_sync_module  # noqa: E402


def test_filter_rows() -> None:
    mod = load_sync_module(script="04_drop_exhausted.py")
    rows = [
        {"audio_filepath": "a.flac", "qc_status": "original"},
        {"audio_filepath": "b.flac", "qc_status": "exhausted"},
        {"audio_filepath": "c.flac", "qc_status": "improved"},
        {"audio_filepath": "d.flac"},
        {"audio_filepath": "e.flac", "qc_status": "exhausted"},
    ]
    kept, dropped, n_unsynced = mod.filter_rows(rows)
    assert [r["audio_filepath"] for r in kept] == ["a.flac", "c.flac", "d.flac"], kept
    assert [r["audio_filepath"] for r in dropped] == ["b.flac", "e.flac"], dropped
    assert n_unsynced == 1, n_unsynced
    # Idempotent: filtering the kept rows again drops nothing.
    assert mod.filter_rows(kept)[1] == []
    print("test_filter_rows: OK")


if __name__ == "__main__":
    test_filter_rows()
