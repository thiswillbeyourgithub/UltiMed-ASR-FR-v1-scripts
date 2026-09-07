#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "click",
#     "loguru",
#     "tqdm",
#     "tenacity",
#     "litellm",
#     "rapidfuzz",
#     "tiktoken",
# ]
# ///
"""Smoke test for the append-only run_statistics.jsonl produced by the generator.

Runs `run()` end to end over a tiny synthetic input with the LLM stubbed out,
then asserts the stats file has the run_start (carrying every parameter, model
and provider), at least one progress record, and a run_end, all sharing one
run_id. Also checks the sampling defaults (temperature 0.7, top_p 0.95) are the
ones recorded, so a silent config drift would fail here.

Run:  python tests/test_run_statistics.py
(or)  uv run tests/test_run_statistics.py
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "utils"))

_ZERO_PRICE = {
    "prompt": 0.0,
    "completion": 0.0,
    "input_cache_read": 0.0,
    "input_cache_write": 0.0,
}


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "gen03", REPO / "01_dictionnary" / "03_generate_texts.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_call_llm(**kwargs):
    """Return one clean <t> block that echoes the term from the user prompt.

    Distinct terms -> distinct outputs, so the cross-term duplicate guard never
    trips and each row's term-presence check passes.
    """
    m = re.search(r"Term:\s*(.+)", kwargs["user_prompt"])
    term = m.group(1).strip() if m else "terme"
    return f"<t>Observation clinique concernant {term} ce jour precis.</t>"


def _write_input(path: Path, terms: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, term in enumerate(terms):
            row = {
                "term": term,
                "index": i,
                "page": 1,
                "definition": "",
                "examples": [],
                "score": 5,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_run_statistics_records():
    gen = _load_generator()
    gen.call_llm = _fake_call_llm
    gen.validate_asr_training_target = lambda variants, examples=None: None
    gen.PricingTracker._fetch_price = staticmethod(lambda model: dict(_ZERO_PRICE))

    # 10 unique tokenizable terms so processed hits _COST_LOG_EVERY (10) exactly
    # once, giving us a progress record between run_start and run_end.
    terms = [
        "aspirine", "metformine", "paracetamol", "amoxicilline", "ibuprofene",
        "insuline", "cortisone", "heparine", "digoxine", "furosemide",
    ]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        inp = td / "in.jsonl"
        _write_input(inp, terms)
        stats_path = td / "run_statistics.jsonl"

        counters = gen.run(
            input_path=inp,
            output_path=td / "out.jsonl",
            review_path=td / "review.jsonl",
            skip_path=td / "skips.jsonl",
            run_stats_path=stats_path,
            run_log_path=td / "run.log",
            n_variants=1,
            n_jobs=1,
            provider="deepseek",
        )
        assert counters["ok"] == 10, counters

        records = [json.loads(x) for x in stats_path.read_text().splitlines() if x.strip()]
        types = [r["type"] for r in records]
        assert types[0] == "run_start", types
        assert types[-1] == "run_end", types
        assert "progress" in types, types

        # Every record is self-identifying and shares one run_id.
        run_ids = {r["run_id"] for r in records}
        assert len(run_ids) == 1, run_ids
        for r in records:
            assert r["model"] == gen.DEFAULT_MODEL, r
            assert r["provider"] == "deepseek", r

        start = records[0]
        assert start["temperature"] == gen.DEFAULT_TARGET_TEMPERATURE == 0.7, start
        assert start["top_p"] == gen.DEFAULT_TOP_P == 0.95, start
        assert start["max_tokens"] == gen.DEFAULT_MAX_TOKENS, start
        assert start["n_variants_policy"] == 1, start
        assert len(start["system_prompt_sha256_12"]) == 12, start
        for key in ("input_path", "output_path", "run_stats_path", "run_log_path"):
            assert key in start, (key, start)

        end = records[-1]
        assert end["counters"]["ok"] == 10, end
        assert "validation_retries" in end and "transport_retries" in end, end


def test_run_statistics_appends_across_runs():
    gen = _load_generator()
    gen.call_llm = _fake_call_llm
    gen.validate_asr_training_target = lambda variants, examples=None: None
    gen.PricingTracker._fetch_price = staticmethod(lambda model: dict(_ZERO_PRICE))

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        stats_path = td / "run_statistics.jsonl"

        for run_no, term in enumerate(["aspirine", "metformine"]):
            inp = td / f"in_{run_no}.jsonl"
            _write_input(inp, [term])
            gen.run(
                input_path=inp,
                output_path=td / f"out_{run_no}.jsonl",
                review_path=td / "review.jsonl",
                skip_path=td / "skips.jsonl",
                run_stats_path=stats_path,
                run_log_path=td / "run.log",
                n_variants=1,
                n_jobs=1,
                provider="deepseek",
            )

        records = [json.loads(x) for x in stats_path.read_text().splitlines() if x.strip()]
        # Two runs, each writing run_start + run_end, appended (not overwritten).
        run_ids = {r["run_id"] for r in records}
        assert len(run_ids) == 2, f"expected 2 distinct runs appended, got {run_ids}"
        assert sum(r["type"] == "run_start" for r in records) == 2, records
        assert sum(r["type"] == "run_end" for r in records) == 2, records


def main() -> int:
    tests = [
        test_run_statistics_records,
        test_run_statistics_appends_across_runs,
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
