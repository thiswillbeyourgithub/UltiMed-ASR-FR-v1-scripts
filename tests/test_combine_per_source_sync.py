"""Stdlib test for the per-source split sync of 99_hf_release/02_combine_nemo_manifests.py.

The combine re-splits the pooled rows into NEW top-level train/val/test files; the
per-dataset ``<name>/{train,val,test}.jsonl`` written by 01_build_nemo_manifest.py
used to keep their own, different partition (a v1 dictionary clip sat in
``dictionary/test.jsonl`` and in the top-level ``train.jsonl``). The sync must make
``<name>/<split>.jsonl`` exactly the ``<name>`` rows of ``<split>.jsonl``, keep the
rows' columns and subfolder-relative paths, leave excluded datasets alone, and
refuse when a per-source row is in no top-level split.

Run: python tests/test_combine_per_source_sync.py

This file was written by Claude Code.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_sync_pass_guard import load_sync_module  # noqa: E402

mod = load_sync_module(script="02_combine_nemo_manifests.py")


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _tree(td: str) -> Path:
    root = Path(td) / "NeMO_files"
    # Per-source rows: paths relative to the subfolder, an extra column kept.
    dic = [{"audio_filepath": f"../../dictionary/d{i}.flac", "text": f"d{i}", "cer": i} for i in range(4)]
    _write(root / "dictionary" / "full.jsonl", dic)
    # The stale per-source partition 01 wrote: everything in train.
    _write(root / "dictionary" / "train.jsonl", dic)
    _write(root / "dictionary" / "val.jsonl", [])
    _write(root / "dictionary" / "test.jsonl", [])
    parrot = [{"audio_filepath": "../../PARROT/p0.flac", "text": "p0"}]
    _write(root / "PARROT" / "full.jsonl", parrot)
    _write(root / "PARROT" / "test.jsonl", parrot)
    # Top-level split: paths relative to NeMO_files, d1 in val, d3 in test.
    top = lambda i: {"audio_filepath": f"../dictionary/d{i}.flac", "text": f"d{i}"}  # noqa: E731
    _write(root / "train.jsonl", [top(0), top(2)])
    _write(root / "val.jsonl", [top(1)])
    _write(root / "test.jsonl", [top(3)])
    return root


def test_sync_matches_top_level() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = _tree(td)
        counts = mod.sync_per_source_splits(root, [root / "dictionary"])
        assert counts == {"dictionary": {"train": 2, "val": 1, "test": 1}}, counts
        assert [r["text"] for r in _read(root / "dictionary" / "train.jsonl")] == ["d0", "d2"]
        assert _read(root / "dictionary" / "val.jsonl") == [
            {"audio_filepath": "../../dictionary/d1.flac", "text": "d1", "cer": 1}]
        assert [r["text"] for r in _read(root / "dictionary" / "test.jsonl")] == ["d3"]
        # PARROT was not passed (excluded from the combine): untouched.
        assert [r["text"] for r in _read(root / "PARROT" / "test.jsonl")] == ["p0"]
        # Idempotent.
        assert mod.sync_per_source_splits(root, [root / "dictionary"]) == counts
    print("test_sync_matches_top_level: OK")


def test_sync_refuses_orphan_rows() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = _tree(td)
        _write(root / "test.jsonl", [])  # d3 now in no top-level split
        try:
            mod.sync_per_source_splits(root, [root / "dictionary"])
        except mod.click.ClickException as exc:
            assert "1 rows of dictionary/full.jsonl" in str(exc), exc
        else:
            raise AssertionError("orphan row not refused")
        # Nothing was rewritten before the refusal.
        assert len(_read(root / "dictionary" / "train.jsonl")) == 4
    print("test_sync_refuses_orphan_rows: OK")


if __name__ == "__main__":
    test_sync_matches_top_level()
    test_sync_refuses_orphan_rows()
