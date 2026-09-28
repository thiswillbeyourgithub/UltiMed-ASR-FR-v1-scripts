"""Stdlib test for 99_hf_release/04_drop_bad_rows.py.

Checks that `qc_status == "exhausted"` rows are dropped, that rows never synced by
03_sync_hotfix_results.py (no `qc_status` key) are kept but counted, and that an eval
clip whose text repeats a training text (up to punctuation and case) is dropped from
every manifest that references its audio, whatever directory that manifest lives in,
that a label holding an ellipsis (truncated LLM output, anonymized date) is dropped,
and that leaked LLM reasoning and a label too long for its clip are dropped.

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


def test_llm_leak_and_too_fast() -> None:
    mod = load_sync_module(script=SCRIPT)
    rows = [
        # The two real drug_sentence train rows found on 2026-09-28.
        {"audio_filepath": "leak.flac", "qc_status": "original", "duration": 9.6,
         "text": "tags Let me create varied contexts: - Sentence 1: 300 mg comprimé - initial prescription"},
        {"audio_filepath": "fast.flac", "qc_status": "original", "duration": 0.3,
         "text": "Renouvellement du traitement par nitrofurantoïne une gélule de 50 milligrammes "
                 "quatre prises quotidiennes pendant 7 jours."},
        # Legitimate English terms and French "sentence" stay.
        {"audio_filepath": "ok1.flac", "qc_status": "original", "duration": 15.5,
         "text": "Hypersignal FLAIR au niveau du bed nucleus of the accessory olfactory tract."},
        {"audio_filepath": "ok2.flac", "qc_status": "original", "duration": 8.0,
         "text": "Le tribunal a rendu une sentence arbitrale le 15 mars 2024."},
        {"audio_filepath": "nodur.flac", "qc_status": "original", "text": "Pas de durée connue."},
    ]
    kept, dropped, _ = mod.filter_rows(rows, Path("/m"))
    assert [r["audio_filepath"] for r in kept] == ["ok1.flac", "ok2.flac", "nodur.flac"], kept
    assert [why for _, why in dropped] == [mod.LLM_LEAK, mod.TOO_FAST], dropped
    print("test_llm_leak_and_too_fast: OK")


def test_asr_flagged() -> None:
    # Matched on <source>/<file> whatever directory the manifest resolves from.
    mod = load_sync_module(script=SCRIPT)
    rows = [{"audio_filepath": "../dictionary/a.flac", "text": "ok", "qc_status": "original"},
            {"audio_filepath": "../dictionary/b.flac", "text": "ok", "qc_status": "original"},
            {"audio_filepath": "../drugs/a.flac", "text": "ok", "qc_status": "original"}]
    kept, dropped, _ = mod.filter_rows(rows, Path("/x/NeMO_files"), asr_flagged={"dictionary/a.flac"})
    assert [r["audio_filepath"] for r in kept] == ["../dictionary/b.flac", "../drugs/a.flac"], kept
    assert [reason for _, reason in dropped] == [mod.ASR_FLAGGED]
    assert mod.load_asr_flagged(Path("/nonexistent.jsonl")) == set()
    print("test_asr_flagged: OK")


if __name__ == "__main__":
    test_filter_exhausted()
    test_eval_duplicates()
    test_ellipsis()
    test_llm_leak_and_too_fast()
    test_asr_flagged()
