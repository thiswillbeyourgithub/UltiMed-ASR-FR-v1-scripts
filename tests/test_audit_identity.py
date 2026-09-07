#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests>=2.31","jiwer>=3.0","tqdm>=4.66","loguru>=0.7","click>=8.1","soundfile>=0.12"]
# ///
"""What names a clip in the improvement audit trail.

Run: `uv run tests/test_audit_identity.py`

Every audit record used to store `index: null, term: null`, because it read `row["index"]`
/ `row["term"]` and the NeMo manifests the driver runs on carry neither (they keep
`group_id` / `item_index` and drop the source term). The fallback is the clip's own name,
which stage 05 built as `{term_index:06d}_{variant_index:04d}_{slug}.flac`, so the two
sides of that convention are pinned here: `utils/nemo_manifest.parse_clip_name` (shared
with the manifest builder's `audio_index`) and the improvement stage's `clip_identity`.

This file was written with Claude Code.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "utils"))
from nemo_manifest import parse_clip_name  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "improve_under_test", HERE.parent / "06_hotfixes" / "01_recursive_improvement.py"
)
_imp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_imp)

_failures: list[str] = []


def eq(label: str, got, want) -> None:
    if got != want:
        _failures.append(f"{label}: got {got!r} want {want!r}")


# --- the filename convention --------------------------------------------------------
eq("clip name splits into term index, variant index and slug",
   parse_clip_name("000014_0000_abaissement_de_la_cataracte.flac"),
   (14, 0, "abaissement_de_la_cataracte"))
eq("a chunk id is a slug like any other",
   parse_clip_name("000941_0002_anatomopathologie_00234_t1_c6.flac"),
   (941, 2, "anatomopathologie_00234_t1_c6"))
# The rejections matter: audio_index() walks a whole directory through this.
eq("a temp leftover is not a clip", parse_clip_name("000014_0000_x.flac.tmp"), None)
eq("unpadded indices are not a clip", parse_clip_name("14_0_x.flac"), None)
eq("a name without indices is not a clip", parse_clip_name("notes.flac"), None)

# --- what the audit records ---------------------------------------------------------
manifest_row = {"audio_filepath": "../drugs/000018_0000_paroxetine.flac",
                "group_id": "drugs:18", "item_index": 0}
eq("a manifest row is named from its clip", _imp.clip_identity(manifest_row), (18, "paroxetine"))
# A stage's own generated_dataset.jsonl has the real fields; they win, unparsed.
stage_row = {"audio_filepath": "../drugs/000018_0000_paroxetine.flac",
             "index": 18, "term": "PAROXETINE 20 mg"}
eq("the source fields win when present", _imp.clip_identity(stage_row), (18, "PAROXETINE 20 mg"))
eq("an unparseable path leaves nulls rather than inventing them",
   _imp.clip_identity({"audio_filepath": "../drugs/manual_take.flac"}), (None, None))
eq("a custom audio key is honoured",
   _imp.clip_identity({"clip": "../drugs/000018_0000_paroxetine.flac"}, "clip"),
   (18, "paroxetine"))

# --- repairing an audit file written before the fallback existed --------------------
with tempfile.TemporaryDirectory() as tmp:
    audit = Path(tmp) / "full.improved.jsonl"
    records = {
        "../dictionary/000014_0000_abaissement_de_la_cataracte.flac": {
            "audio_filepath": "../dictionary/000014_0000_abaissement_de_la_cataracte.flac",
            "index": None, "term": None, "new_cer": 0.01},
        "../drugs/000018_0000_paroxetine.flac": {
            "audio_filepath": "../drugs/000018_0000_paroxetine.flac",
            "index": 18, "term": "PAROXETINE", "new_cer": 0.02},
    }
    audit.write_text("".join(json.dumps(r) + "\n" for r in records.values()), encoding="utf-8")

    eq("only the null record is repaired", _imp.backfill_audit_identity(audit, records), 1)
    written = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines() if line]
    eq("the repair reached the file", (written[0]["index"], written[0]["term"]),
       (14, "abaissement_de_la_cataracte"))
    eq("an already-named record is untouched", (written[1]["index"], written[1]["term"]),
       (18, "PAROXETINE"))
    eq("every other field survives", written[0]["new_cer"], 0.01)
    # Idempotent, and it must not rewrite a file that has nothing to fix (a converged run
    # calls this on every dataset, every pass).
    mtime = audit.stat().st_mtime_ns
    eq("second run is a no-op", _imp.backfill_audit_identity(audit, records), 0)
    eq("and does not touch the file", audit.stat().st_mtime_ns, mtime)

if _failures:
    print("FAILED:")
    for f in _failures:
        print("  -", f)
    raise SystemExit(1)
print("ALL PASSED")
