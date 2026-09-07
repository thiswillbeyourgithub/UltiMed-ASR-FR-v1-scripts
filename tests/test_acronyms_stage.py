#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "click",
#   "loguru",
#   "tqdm",
#   "tenacity",
#   "litellm",
#   "rapidfuzz",
#   "tiktoken",
# ]
# ///
"""Tests for the 07_acronyms stage (07_acronyms/03_generate_texts.py).

Covers the CSV expansion (multi-pronunciation split, default pronunciation,
structural problem reporting, the index drift guard), the exact-case
occurrence regex shared by validation and source substitution, the
definition-presence check (direct, parenthetical gloss, scattered cognate
fallback), the pronunciation round-trip guard, the output row schema, and an
end-to-end engine run with the LLM stubbed out (asserting targets carry the
acronym and sources carry the pronunciation).

Run: python tests/test_acronyms_stage.py   (or)   uv run tests/test_acronyms_stage.py

This file was written with Claude Code.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path

import click

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "utils"))

from _pipeline_shared import LLMError, PricingTracker, TermMissingError  # noqa: E402

_ZERO_PRICE = {
    "prompt": 0.0,
    "completion": 0.0,
    "input_cache_read": 0.0,
    "input_cache_write": 0.0,
}


def _load_stage():
    """Import the digit-prefixed acronyms module by path."""
    spec = importlib.util.spec_from_file_location(
        "acronyms03", REPO / "07_acronyms" / "03_generate_texts.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load_stage()


def _write_csv(path: Path, rows: list[str]) -> Path:
    path.write_text("TERM,MEANING,PRONOUNCED_AS\n" + "".join(r + "\n" for r in rows),
                    encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# default_pronunciation
# ---------------------------------------------------------------------------


def test_default_pronunciation():
    cases = {
        "AAA": "A-A-A",
        "5-FU": "5-F-U",
        "gamma-GT": "gamma-G-T",
        "ST-plus": "S-T-plus",
        "aVf": "a-V-f",
        "ADAMTS-13": "A-D-A-M-T-S-13",
        "A": "A",
        "CH50": "C-H-50",
    }
    for term, want in cases.items():
        got = mod.default_pronunciation(term)
        assert got == want, f"{term!r}: expected {want!r}, got {got!r}"


# ---------------------------------------------------------------------------
# expand_csv
# ---------------------------------------------------------------------------


def test_expand_csv_multi_pron_and_default():
    with tempfile.TemporaryDirectory() as td:
        csv_path = _write_csv(Path(td) / "in.csv", [
            "ESAT,etablissement et service d'accompagnement par le travail,E-S-A-T;e-sath;e-sath",
            "AAA,anevrisme de l'aorte abdominale,",
        ])
        entries = mod.expand_csv(csv_path)
    # duplicate pron within the field ("e-sath" twice) deduplicates
    assert [e["pronunciation"] for e in entries] == ["E-S-A-T", "e-sath", "A-A-A"], entries
    assert [e["index"] for e in entries] == [0, 1, 2], "indices must be contiguous"
    assert [e["term"] for e in entries] == ["ESAT", "ESAT", "AAA"]
    assert [e["pron_index"] for e in entries] == [0, 1, 0]
    assert [e["n_prons"] for e in entries] == [2, 2, 1]
    assert all(e["category"] == "acronyms" for e in entries)
    assert entries[2]["definition"] == "anevrisme de l'aorte abdominale"


def test_expand_csv_reports_structural_problems_together():
    with tempfile.TemporaryDirectory() as td:
        csv_path = _write_csv(Path(td) / "in.csv", [
            "AVC,,",                     # empty MEANING
            "HTA,hypertension arterielle,",
            "HTA,doublon exact,",        # exact duplicate TERM
        ])
        raised = None
        try:
            mod.expand_csv(csv_path)
        except click.ClickException as e:
            raised = e
    assert raised is not None, "expected ClickException on structural problems"
    msg = str(raised.message)
    assert "empty MEANING" in msg, msg
    assert "duplicate TERM" in msg, msg
    assert "2 problem(s)" in msg, msg


def test_expand_csv_keeps_case_only_collisions():
    with tempfile.TemporaryDirectory() as td:
        csv_path = _write_csv(Path(td) / "in.csv", [
            "AVF,algie vasculaire de la face,",
            "aVf,derivation unipolaire des membres,",
        ])
        entries = mod.expand_csv(csv_path)
    assert [e["term"] for e in entries] == ["AVF", "aVf"], (
        "case-only collisions are legitimate (case IS the discriminator) and "
        "must keep both rows"
    )


def test_expand_csv_rejects_wrong_header():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "in.csv"
        p.write_text("TERM,DEFINITION,PRON\nAVC,accident,\n", encoding="utf-8")
        raised = None
        try:
            mod.expand_csv(p)
        except click.ClickException as e:
            raised = e
    assert raised is not None and "expected header" in str(raised.message)


# ---------------------------------------------------------------------------
# ensure_expanded drift guard
# ---------------------------------------------------------------------------


def test_ensure_expanded_writes_verifies_and_guards_drift():
    with tempfile.TemporaryDirectory() as td:
        csv_path = _write_csv(Path(td) / "in.csv", [
            "HTA,hypertension arterielle,",
        ])
        expanded = Path(td) / "expanded.jsonl"
        first = mod.ensure_expanded(csv_path, expanded)
        assert expanded.exists() and len(first) == 1
        again = mod.ensure_expanded(csv_path, expanded)
        assert again == first, "unchanged CSV must verify against the stored file"
        # Now shift indices by prepending a row: the stored expansion no longer
        # matches, which would re-key resume state + stage-05 file names.
        _write_csv(csv_path, [
            "AVC,accident vasculaire cerebral,",
            "HTA,hypertension arterielle,",
        ])
        raised = None
        try:
            mod.ensure_expanded(csv_path, expanded)
        except click.ClickException as e:
            raised = e
        assert raised is not None, "index drift must abort, not silently rewrite"
        assert "no longer matches" in str(raised.message)


# ---------------------------------------------------------------------------
# occurrence regex + source substitution
# ---------------------------------------------------------------------------


def test_occurrence_re_boundaries():
    rx = mod._occurrence_re("AVC")
    assert rx.search("un AVC constitue")
    assert rx.search("le patient post-AVC")          # hyphen adjacency allowed
    assert rx.search("l'AVC ischemique")             # apostrophe adjacency allowed
    assert rx.search("un AVC, recent")
    assert not rx.search("le service PAVCX")         # inside a longer code
    assert not rx.search("le code AVC2 note")        # digit glued right
    assert not rx.search("un avc en minuscules")     # exact case only
    rx2 = mod._occurrence_re("gamma-GT")
    assert rx2.search("la gamma-GT est elevee")
    assert not rx2.search("la gamma-GTT est elevee")


def test_substitute_pronunciation():
    entry = {"term": "ESAT", "pronunciation": "E-S-A-T"}
    out = mod.substitute_pronunciation(
        "Le patient travaille a l'ESAT et parle de l'ESAT positivement.", entry
    )
    assert out == "Le patient travaille a l'E-S-A-T et parle de l'E-S-A-T positivement.", out
    # No occurrence: returns the text unchanged (loud warning, not a crash).
    same = mod.substitute_pronunciation("Aucun sigle dans cette phrase la.", entry)
    assert same == "Aucun sigle dans cette phrase la."


# ---------------------------------------------------------------------------
# definition presence
# ---------------------------------------------------------------------------


def test_definition_presence_direct_and_paren_gloss():
    # Direct French meaning woven into one variant.
    mod._check_definition_presence(
        ["Une suspicion d'anevrisme de l'aorte abdominale motive le scanner."],
        {"index": 0, "term": "AAA", "definition": "anévrisme de l'aorte abdominale"},
    )
    # English meaning whose parenthetical French gloss appears in a variant.
    mod._check_definition_presence(
        ["L'echelle evalue les activites de la vie quotidienne du patient age."],
        {"index": 1, "term": "ADL",
         "definition": "activities of daily living (activités de la vie quotidienne)"},
    )


def test_definition_presence_scattered_cognate_fallback():
    # English expansion, no parens: enough distinctive words individually match
    # their French cognates inside ONE variant.
    mod._check_definition_presence(
        ["Un syndrome respiratoire aigu severe est evoque chez ce patient."],
        {"index": 2, "term": "SARS",
         "definition": "severe acute respiratory syndrome"},
    )


def test_definition_presence_raises_when_absent():
    raised = None
    try:
        mod._check_definition_presence(
            ["Le patient est convoque pour un controle biologique complet demain.",
             "Une surveillance rapprochee est organisee en ambulatoire des lundi."],
            {"index": 3, "term": "ICC",
             "definition": "insuffisance cardiaque chronique terminale"},
        )
    except LLMError as e:
        raised = e
    assert raised is not None, "absent meaning must raise (retryable) LLMError"
    assert "meaning" in str(raised)


# ---------------------------------------------------------------------------
# acronym_validate
# ---------------------------------------------------------------------------

_VALID_BATCH = [
    "Le patient a repris une activite encadree a l'ESAT depuis le mois de mars.",
    "Une orientation vers l'ESAT, c'est-a-dire un etablissement et service "
    "d'accompagnement par le travail, a ete validee en commission.",
]
_ESAT_ENTRY = {
    "index": 0,
    "term": "ESAT",
    "definition": "etablissement et service d'accompagnement par le travail",
    "pronunciation": "E-S-A-T",
    "pron_index": 0,
    "n_prons": 1,
    "category": "acronyms",
}


def test_acronym_validate_accepts_valid_batch():
    mod.acronym_validate(list(_VALID_BATCH), _ESAT_ENTRY)


def test_acronym_validate_rejects_wrong_case():
    batch = [_VALID_BATCH[0].replace("ESAT", "Esat"), _VALID_BATCH[1]]
    raised = None
    try:
        mod.acronym_validate(batch, _ESAT_ENTRY)
    except TermMissingError as e:
        raised = e
    assert raised is not None, (
        "a re-cased acronym must fail the verbatim presence check "
        "(the written form IS the training label)"
    )
    assert "verbatim" in str(raised)


def test_acronym_validate_rejects_missing_gloss():
    batch = [
        "Le patient a repris une activite encadree a l'ESAT depuis le mois de mars.",
        "Un rendez-vous de suivi a l'ESAT est programme pour la fin du trimestre.",
    ]
    raised = None
    try:
        mod.acronym_validate(batch, _ESAT_ENTRY)
    except LLMError as e:
        raised = e
    assert raised is not None and not isinstance(raised, TermMissingError), (
        "no-gloss batches must raise the retryable definition-presence LLMError"
    )


# ---------------------------------------------------------------------------
# round-trip guard + row schema
# ---------------------------------------------------------------------------


def test_roundtrip_check_pass_and_queue():
    target = "Le patient est suivi a l'ESAT depuis un an pour un accompagnement adapte."
    good_source = target.replace("ESAT", "E-S-A-T")
    with tempfile.TemporaryDirectory() as td:
        mod._roundtrip_path = Path(td) / "roundtrip.jsonl"
        mod._roundtrip_check(_ESAT_ENTRY, 0, target, good_source)
        assert not mod._roundtrip_path.exists(), (
            "a clean substitution must not be queued"
        )
        mod._roundtrip_check(
            _ESAT_ENTRY, 1, target, "Texte totalement different sans aucun rapport."
        )
        rows = [json.loads(x)
                for x in mod._roundtrip_path.read_text().splitlines() if x.strip()]
        assert len(rows) == 1, "a mangled source must be queued for review"
        row = rows[0]
        assert row["term"] == "ESAT" and row["variant_index"] == 1
        assert row["ratio"] < mod._ROUNDTRIP_MIN_RATIO
        assert "back_substituted" in row and "expected_source_without_pron" in row
    mod._roundtrip_path = mod.DEFAULT_ROUNDTRIP_QUEUE


def test_acronym_make_row_schema():
    target = "Le patient est suivi a l'ESAT depuis un an pour un accompagnement adapte."
    source = target.replace("ESAT", "E-S-A-T")
    with tempfile.TemporaryDirectory() as td:
        mod._roundtrip_path = Path(td) / "roundtrip.jsonl"
        row = mod.acronym_make_row(_ESAT_ENTRY, 1, target, source, "test-model")
    mod._roundtrip_path = mod.DEFAULT_ROUNDTRIP_QUEUE
    assert row == {
        "category": "acronyms",
        "term_index": 0,
        "term": "ESAT",
        "pronunciation": "E-S-A-T",
        "pron_index": 0,
        "n_prons": 1,
        "variant_index": 1,
        "asr_training_target": target,
        "asr_training_source": source,
        "source_definition": _ESAT_ENTRY["definition"],
        "model": "test-model",
    }, row


# ---------------------------------------------------------------------------
# user prompt shape (prompt-cache prefix across a term's pronunciations)
# ---------------------------------------------------------------------------


def test_user_prompt_shares_prefix_across_pronunciations():
    e1 = dict(_ESAT_ENTRY)
    e2 = dict(_ESAT_ENTRY, pronunciation="e-sath", pron_index=1)
    p1 = mod.build_acronym_user_prompt(e1, 3)
    p2 = mod.build_acronym_user_prompt(e2, 3)
    assert 'Pronounced as: "E-S-A-T"' in p1
    assert 'Pronounced as: "e-sath"' in p2
    head1, head2 = p1.split("Pronounced as:")[0], p2.split("Pronounced as:")[0]
    assert head1 == head2 and "Term: ESAT" in head1 and "Definition:" in head1, (
        "Term+Definition must precede the pronunciation line so DeepSeek "
        "prefix caching covers them across a term's pronunciations"
    )
    assert p1.index("Term:") < p1.index("Definition:") < p1.index("Pronounced as:")


# ---------------------------------------------------------------------------
# end-to-end engine run with the LLM stubbed out
# ---------------------------------------------------------------------------


def test_engine_run_end_to_end():
    calls = {"n": 0}

    def fake_call_llm(**kwargs):
        calls["n"] += 1
        up = kwargs["user_prompt"]
        term = re.search(r"Term:\s*(.+)", up).group(1).strip()
        definition = re.search(r"Definition:\s*(.+)", up).group(1).strip()
        k = calls["n"]
        return (
            f"<t>Une prise en charge en {term}, c'est-a-dire {definition}, est "
            f"proposee au patient numero {k} des cette semaine.</t>"
            f"<t>Le controle clinique de {term} est programme au prochain "
            f"rendez-vous, selon le compte rendu numero {k}.</t>"
        )

    # Patch BEFORE run(): build_acronym_adapter reads this module global.
    mod.call_llm = fake_call_llm
    PricingTracker._fetch_price = staticmethod(lambda model: dict(_ZERO_PRICE))

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        csv_path = _write_csv(td / "in.csv", [
            "ESAT,etablissement et service d'accompagnement par le travail,E-S-A-T;e-sath",
            "AAA,anevrisme de l'aorte abdominale,",
        ])
        out_path = td / "out.jsonl"
        stats = mod.run(
            input_path=csv_path,
            expanded_path=td / "expanded.jsonl",
            output_path=out_path,
            n_texts=2,
            model="test-model",
            n_jobs=1,
            timeout_s=5,
            provider=None,
            review_path=td / "review.jsonl",
            skip_path=td / "skips.jsonl",
            roundtrip_path=td / "roundtrip.jsonl",
            run_stats_path=td / "run_statistics.jsonl",
            run_log_path=td / "run.log",
        )
        rows = [json.loads(x) for x in out_path.read_text().splitlines() if x.strip()]
        assert len(rows) == 6, f"3 entries x 2 texts expected, got {len(rows)}"
        assert calls["n"] == 3, f"one LLM call per (term, pronunciation), got {calls['n']}"
        by_key = {(r["term_index"], r["variant_index"]) for r in rows}
        assert by_key == {(i, v) for i in range(3) for v in range(2)}
        for r in rows:
            term, pron = r["term"], r["pronunciation"]
            target, source = r["asr_training_target"], r["asr_training_source"]
            assert mod._occurrence_re(term).search(target), (
                f"target must carry the acronym verbatim: {target!r}"
            )
            assert pron in source, f"source must carry the pronunciation: {source!r}"
            assert not mod._occurrence_re(term).search(source), (
                f"the acronym must be fully substituted out of the source: {source!r}"
            )
        prons = {(r["term"], r["pronunciation"]) for r in rows}
        assert prons == {("ESAT", "E-S-A-T"), ("ESAT", "e-sath"), ("AAA", "A-A-A")}
        for q in ("review.jsonl", "skips.jsonl", "roundtrip.jsonl"):
            p = td / q
            content = p.read_text().strip() if p.exists() else ""
            assert content == "", f"{q} expected empty, got: {content[:200]}"
        assert stats.get("ok") == 3 and stats.get("rows") == 6, stats
        # run stats must exist and share one run_id
        recs = [json.loads(x)
                for x in (td / "run_statistics.jsonl").read_text().splitlines()
                if x.strip()]
        assert recs and recs[0]["type"] == "run_start" and recs[-1]["type"] == "run_end"
        assert len({r["run_id"] for r in recs}) == 1


def main() -> int:
    tests = [
        test_default_pronunciation,
        test_expand_csv_multi_pron_and_default,
        test_expand_csv_reports_structural_problems_together,
        test_expand_csv_keeps_case_only_collisions,
        test_expand_csv_rejects_wrong_header,
        test_ensure_expanded_writes_verifies_and_guards_drift,
        test_occurrence_re_boundaries,
        test_substitute_pronunciation,
        test_definition_presence_direct_and_paren_gloss,
        test_definition_presence_scattered_cognate_fallback,
        test_definition_presence_raises_when_absent,
        test_acronym_validate_accepts_valid_batch,
        test_acronym_validate_rejects_wrong_case,
        test_acronym_validate_rejects_missing_gloss,
        test_roundtrip_check_pass_and_queue,
        test_acronym_make_row_schema,
        test_user_prompt_shares_prefix_across_pronunciations,
        test_engine_run_end_to_end,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {t.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
