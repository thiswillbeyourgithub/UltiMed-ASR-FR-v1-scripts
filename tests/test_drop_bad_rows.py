"""Stdlib test for 99_hf_release/04_drop_bad_rows.py.

Checks that `qc_status == "exhausted"` rows are dropped, that rows never synced by
03_sync_hotfix_results.py (no `qc_status` key) are kept but counted, and that an eval
clip whose text repeats a training text (up to punctuation and case) is dropped from
every manifest that references its audio, whatever directory that manifest lives in,
and that a label holding an ellipsis (truncated LLM output, anonymized date) is dropped.

Run: python tests/test_drop_bad_rows.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_sync_pass_guard import load_sync_module  # noqa: E402

SCRIPT = "04_drop_bad_rows.py"


def test_filter_exhausted() -> None:
    mod = load_sync_module(script=SCRIPT)
    rows = [
        {"audio_filepath": "a.flac", "qc_status": "original"},
        {"audio_filepath": "b.flac", "qc_status": "exhausted"},
        {"audio_filepath": "c.flac", "qc_status": "improved"},
        {"audio_filepath": "d.flac"},
        {"audio_filepath": "e.flac", "qc_status": "exhausted"},
    ]
    kept, dropped, n_unsynced = mod.filter_rows(rows, Path("/m"))
    assert [r["audio_filepath"] for r in kept] == ["a.flac", "c.flac", "d.flac"], kept
    assert [(r["audio_filepath"], why) for r, why in dropped] == [
        ("b.flac", mod.EXHAUSTED), ("e.flac", mod.EXHAUSTED)], dropped
    assert n_unsynced == 1, n_unsynced
    # Idempotent: filtering the kept rows again drops nothing.
    assert mod.filter_rows(kept, Path("/m"))[1] == []
    print("test_filter_exhausted: OK")


def test_eval_duplicates() -> None:
    mod = load_sync_module(script=SCRIPT)
    root = Path("/data/NeMO_files")
    train = [{"audio_filepath": "../PARHAF/t1.flac",
              "text": "Sur le plan cardiovasculaire, il n'y a pas de signe."}]
    evals = {
        root / "test.jsonl": [
            # Same sentence, punctuation and case differ: must be caught.
            {"audio_filepath": "../PARHAF/x1.flac",
             "text": "sur le plan cardiovasculaire il n'y a pas de signe"},
            {"audio_filepath": "../PARHAF/x2.flac", "text": "Une autre phrase."},
        ],
    }
    dup = mod.eval_duplicates(train, evals)
    assert list(dup) == ["/data/PARHAF/x1.flac"], dup

    # The same clip seen from a per-source manifest one level deeper is dropped too,
    # and the training twin (different audio) is kept.
    per_source = [
        {"audio_filepath": "../../PARHAF/x1.flac", "qc_status": "original"},
        {"audio_filepath": "../../PARHAF/t1.flac", "qc_status": "original"},
    ]
    kept, dropped, _ = mod.filter_rows(per_source, root / "PARHAF", dup)
    assert [r["audio_filepath"] for r in kept] == ["../../PARHAF/t1.flac"], kept
    assert [why for _, why in dropped] == [mod.DUPLICATE], dropped
    print("test_eval_duplicates: OK")


def test_ellipsis() -> None:
    mod = load_sync_module(script=SCRIPT)
    rows = [
        {"audio_filepath": "a.flac", "qc_status": "original",
         "text": "Les taux de rénine active à 12 milliu..."},
        {"audio_filepath": "b.flac", "qc_status": "original",
         "text": "La date de l'intervention est le …"},
        {"audio_filepath": "c.flac", "qc_status": "original", "text": "Tension à 12,8."},
    ]
    kept, dropped, _ = mod.filter_rows(rows, Path("/m"))
    assert [r["audio_filepath"] for r in kept] == ["c.flac"], kept
    assert [why for _, why in dropped] == [mod.ELLIPSIS, mod.ELLIPSIS], dropped
    print("test_ellipsis: OK")


if __name__ == "__main__":
    test_filter_exhausted()
    test_eval_duplicates()
    test_ellipsis()
