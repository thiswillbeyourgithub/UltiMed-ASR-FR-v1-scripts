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
"""Generate ASR training text pairs (asr_training_target + asr_training_source).

For each technical term in original_dictionnary.scored.normalized.jsonl, call an LLM (default:
openrouter/deepseek/deepseek-v4-pro) to produce N variants. Each variant is a
short French medical text that uses the term (`asr_training_target`, the written
ASR transcript / label). The paired `asr_training_source` (the text fed to the
local voxtral-tts engine) is then derived DETERMINISTICALLY from each target
line by `voxtral_normalize` (no second LLM call).

So there is ONE LLM call per term:

  Call 1 (asr_training_target): system prompt = PROMPT_GENERATE_ASR_TRAINING_TARGET.md
                        user prompt   = the term + definition + examples
                        output        = N <t>...</t> blocks

  Source pass (asr_training_source): voxtral_normalize.normalize_and_flag is
                        applied to each target line. voxtral reads written
                        medical text well, so this only fixes the small set it
                        mispronounces (staging Roman numerals, ARNm). Any unit
                        symbol that survives is logged to a review queue. See
                        01_dictionnary/VOXTRAL_QUIRKS.md.

Output uses a simple delimited XML-like format which avoids the cost of
structured-output enforcement.

Low-level machinery (validators, parsers, LLM call wrapper, retry loop,
OpenRouter helpers, pricing tracker) lives in `utils/_pipeline_shared.py`, and
the source-agnostic run loop (score-based variant-count policy, resumability,
dedup, run statistics, the deterministic voxtral source pass, review / skip
queues) lives in `utils/text_generation_engine.py`. Both are imported below.
This file keeps only the dictionary-specific bits: definition/examples
sanitization, the per-term user prompt for the target pass, the output row
schema, and a `StageAdapter` that wires them into the shared engine (with thin
`run` / `generate_variants_for_term` wrappers kept for a stable public API).

This module is designed to be imported and tweaked:

    from importlib import import_module
    m = import_module("01_dictionnary.03_generate_texts")
    m.run(input_path="parsed.jsonl", output_path="dataset.jsonl", limit=10)

Or as CLI:

    python 01_dictionnary/03_generate_texts.py --limit 10
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import click
from loguru import logger

# Shared pipeline lives in utils/ so both 01_dictionnary and 02_drugs can
# import from a single source of truth without duplication.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from _pipeline_shared import (  # noqa: E402
    # DEFAULT_MAX_TOKENS is re-exported (read through this module by the run
    # statistics test); the rest are the dictionary-specific validators and the
    # LLM call used by the adapter. The generic orchestration (parsers, retry
    # loop, dedup, error classes) now lives in text_generation_engine.
    DEFAULT_MAX_TOKENS,  # noqa: F401
    PricingTracker,
    call_llm,
    check_term_in_variants,
    sanitize_definition,
    validate_asr_training_target,
)
# Source-agnostic orchestration shared by every corpus stage. Imported (and
# re-exported) here so the dictionary module's public surface is unchanged while
# the generic pieces live in one place; the drugs / PARHAF / PARROT stages import
# the same engine instead of re-implementing it. See utils/text_generation_engine.py.
from text_generation_engine import (  # noqa: E402, F401
    NVariantsPolicy,
    StageAdapter,
    TermSkipped,
    build_target_user_prompt,
    iter_input,
    load_existing_state,
    parse_n_variants,
    resolve_n_variants,
    _normalize_ranges,
    _truncate,
)
from text_generation_engine import (  # noqa: E402
    generate_variants_for_term as _engine_generate_variants_for_term,
    run as _engine_run,
)


HERE = Path(__file__).resolve().parent

# Pipeline stage 3 input: the scored+normalized file produced by stage 2
# (02_token_check.py). It carries both a `score` (from 01_llm_scoring.py) and a
# tokenizable `term`, which is exactly what this generator requires: the
# n-variants policy reads the score and the run fails closed on any unscored row.
DEFAULT_INPUT = HERE / "original_dictionnary.scored.normalized.jsonl"
DEFAULT_OUTPUT = HERE / "generated_dataset.jsonl"
DEFAULT_ASR_TRAINING_TARGET_PROMPT = HERE / "PROMPT_GENERATE_ASR_TRAINING_TARGET.md"
# Review queue: any asr_training_source whose unit symbol survived deterministic
# normalization (an uncovered unit the TARGET LLM failed to spell out) is logged
# here for human review, while the row still ships (fail-open). See the residual
# backstop note in VOXTRAL_QUIRKS.md.
DEFAULT_REVIEW_QUEUE = HERE / "voxtral_review_queue.jsonl"
# Skip queue: a term whose variants keep dropping the term itself, even after
# the retry-with-feedback loop, is logged here and skipped (that one term only)
# rather than aborting the whole run. See TermSkipped and the run worker.
DEFAULT_SKIP_QUEUE = HERE / "term_missing_skips.jsonl"
# Run statistics: append-only. run_statistics.jsonl gets one machine-readable
# record per run boundary (run_start with every parameter used, periodic
# progress, run_end) so a run's config + cost + retry health can be diffed or
# handed to an LLM to spot misconfiguration. run_statistics.log mirrors the
# loguru stream to disk for the same run. Both are appended, never overwritten.
DEFAULT_RUN_STATS = HERE / "run_statistics.jsonl"
DEFAULT_RUN_LOG = HERE / "run_statistics.log"

DEFAULT_MODEL = "openrouter/deepseek/deepseek-v4-pro"
# OpenRouter provider slug to pin. Required for reliable prompt-cache hits:
# without pinning the router load-balances between upstreams and silently
# breaks the cache key. Empty string disables pinning. Match this to the
# author segment of DEFAULT_MODEL by default.
DEFAULT_PROVIDER = "deepseek"
# One variant per OpenAI tts voice (11 voices => 11 variants per term).
DEFAULT_N_VARIANTS = 11
DEFAULT_N_JOBS = 4
DEFAULT_TIMEOUT_S = 300  # per LLM call (overall, including network)

# Target generation must produce N *diverse* clinical contexts, but diversity is
# carried by the prompt's context-cycling list, not by cranking temperature. A
# moderate temperature plus a top_p that clips the pathological tail keeps the
# variety while cutting banned-char / bad-unit validation retries. (The source
# pass is deterministic and uses no LLM.) top_p is forwarded to call_llm; both
# are fixed here rather than exposed on the CLI.
DEFAULT_TARGET_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95

# DeepSeek v4 reasoning levels, escalated per attempt: the first (cheap) pass
# runs "non-think" for speed and cost during this testing phase, and a
# validation retry lets the provider default (thinking) kick in so the model
# only reasons harder when the cheap pass actually failed the validators. The
# system prompt also tells the model its intuition is usually right, so
# non-think should carry most terms.
# Confirmed against OpenRouter: {"enabled": False} disables reasoning, and the
# high/thinking level is the provider default, so the retry simply OMITS the
# argument (None -> call_llm sends no `reasoning`, provider default applies).
DEFAULT_REASONING = {"enabled": False}  # "non-think": first, cheap attempt
RETRY_REASONING = None                  # omit -> provider default (thinking)


# Per-term user prompt length caps. The system prompt stays cached across
# all terms, but the *user* message is per-term and uncached. An unbounded
# definition or reference example would balloon the per-term token cost
# and could push other content out of the model's attention window.
_MAX_DEFINITION_CHARS = 600
_MAX_EXAMPLE_CHARS = 300


# The `Definition` sanitizer (banned chars mapped to safe spoken equivalents)
# lives in utils/_pipeline_shared.sanitize_definition, shared with the acronyms
# stage rather than kept as a per-stage copy; imported above.


def _prepare_examples(term_entry: dict) -> list[str]:
    """Return non-empty reference examples truncated to _MAX_EXAMPLE_CHARS.

    Used by both the prompt builder and the example-leak validator so the
    LLM and the validator see the same truncated text.
    """
    raw = [e.strip() for e in (term_entry.get("examples") or []) if e.strip()]
    return [_truncate(e, _MAX_EXAMPLE_CHARS) for e in raw]


def build_asr_training_target_user_prompt(term_entry: dict, n_variants: int) -> str:
    """Build the user-message for the `asr_training_target` generation pass.

    `term_entry` is one row from parsed.jsonl with keys:
        term, definition, examples (list[str], may be empty), index, page,
        and optionally category (a provenance tag propagated to output rows).
    """
    term = term_entry["term"]
    definition = sanitize_definition((term_entry.get("definition") or "").strip())
    examples = _prepare_examples(term_entry)
    return build_target_user_prompt(
        term,
        n_variants,
        definition=definition,
        examples=examples,
        max_definition_chars=_MAX_DEFINITION_CHARS,
        max_example_chars=_MAX_EXAMPLE_CHARS,
    )


def _dict_make_row(term_entry: dict, variant_index: int, target: str, source: str, model: str) -> dict:
    """Build one output row for the dictionary stage."""
    return {
        # Provenance tag carried through from the input row (e.g. "dictionary").
        # Propagated, not hardcoded, so when the drugs / PARHAF / PARROT stages
        # reuse this schema the merged dataset can still tell which source
        # produced each pair. None if the input row carries no category.
        "category": term_entry.get("category"),
        "term_index": term_entry.get("index"),
        "term": term_entry.get("term"),
        "page": term_entry.get("page"),
        "variant_index": variant_index,
        "asr_training_target": target,
        "asr_training_source": source,
        "source_definition": term_entry.get("definition", ""),
        "model": model,
    }


def _dict_adapter(system_prompt: str, system_prompt_path=None) -> StageAdapter:
    """Build the dictionary StageAdapter for the shared engine.

    `call_llm` and `validate_asr_training_target` are read from THIS module's
    globals at the moment this function runs (not captured once at import), so a
    test that monkeypatches `<module>.call_llm` or
    `<module>.validate_asr_training_target` before calling `run` /
    `generate_variants_for_term` still takes effect through the engine.
    """
    def validate_fn(parsed, entry):
        # Stage-specific soft validators. The generic cross-term dedup check is
        # run by the engine separately, so it is intentionally not repeated here.
        validate_asr_training_target(parsed, examples=_prepare_examples(entry))
        check_term_in_variants(
            entry.get("term", ""), parsed, term_index=entry.get("index")
        )

    return StageAdapter(
        stage="dictionary",
        system_prompt=system_prompt,
        system_prompt_path=system_prompt_path,
        build_user_prompt=build_asr_training_target_user_prompt,
        validate_fn=validate_fn,
        make_row=_dict_make_row,
        call_llm_fn=call_llm,
        temperature=DEFAULT_TARGET_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        reasoning_first=DEFAULT_REASONING,
        reasoning_retry=RETRY_REASONING,
    )


def generate_variants_for_term(
    term_entry: dict,
    asr_training_target_system_prompt: str,
    n_variants: int = DEFAULT_N_VARIANTS,
    model: str = DEFAULT_MODEL,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    seen_target_hashes: set[str] | None = None,
    seen_source_hashes: set[str] | None = None,
    seen_lock: threading.Lock | None = None,
    provider: str | None = DEFAULT_PROVIDER,
    tracker: PricingTracker | None = None,
    review_path: str | Path = DEFAULT_REVIEW_QUEUE,
    skip_path: str | Path = DEFAULT_SKIP_QUEUE,
) -> list[dict]:
    """Dictionary wrapper over the shared engine (signature kept stable).

    Delegates to `text_generation_engine.generate_variants_for_term` with the
    dictionary adapter. Kept as a thin module-level function so existing callers
    and tests (which patch this module's `call_llm` / `validate_asr_training_target`)
    keep working unchanged.
    """
    return _engine_generate_variants_for_term(
        term_entry,
        _dict_adapter(asr_training_target_system_prompt),
        n_variants=n_variants,
        model=model,
        timeout_s=timeout_s,
        seen_target_hashes=seen_target_hashes,
        seen_source_hashes=seen_source_hashes,
        seen_lock=seen_lock,
        provider=provider,
        tracker=tracker,
        review_path=review_path,
        skip_path=skip_path,
    )


def run(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
    asr_training_target_prompt_path: str | Path = DEFAULT_ASR_TRAINING_TARGET_PROMPT,
    n_variants: int | str | dict = DEFAULT_N_VARIANTS,
    model: str = DEFAULT_MODEL,
    n_jobs: int = DEFAULT_N_JOBS,
    limit: int | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    skip_terms_filter=None,
    provider: str | None = DEFAULT_PROVIDER,
    review_path: str | Path = DEFAULT_REVIEW_QUEUE,
    skip_path: str | Path = DEFAULT_SKIP_QUEUE,
    run_stats_path: str | Path = DEFAULT_RUN_STATS,
    run_log_path: str | Path = DEFAULT_RUN_LOG,
) -> dict:
    """Dictionary entry-point: read the prompt, build the adapter, run the engine.

    Appends to output_path so the run is resumable. Returns a summary dict.

    `n_variants`: either an int (same count for every term) or a mapping of
    score ranges to counts, e.g. {"1-3": 2, "4-7": 4, "8-10": 11}. With a dict
    policy, the input must carry a "score" field (see 01_llm_scoring.py); terms
    whose score falls in no range are skipped.

    `skip_terms_filter`: optional callable(term_entry) -> bool. Return True to
    skip a term (e.g. filter by page range, length, etc.).
    """
    asr_training_target_system_prompt = Path(
        asr_training_target_prompt_path
    ).read_text(encoding="utf-8")
    adapter = _dict_adapter(
        asr_training_target_system_prompt,
        system_prompt_path=asr_training_target_prompt_path,
    )
    return _engine_run(
        adapter,
        input_path=input_path,
        output_path=output_path,
        n_variants=n_variants,
        model=model,
        n_jobs=n_jobs,
        limit=limit,
        timeout_s=timeout_s,
        skip_terms_filter=skip_terms_filter,
        provider=provider,
        review_path=review_path,
        skip_path=skip_path,
        run_stats_path=run_stats_path,
        run_log_path=run_log_path,
    )


@click.command(context_settings={"show_default": True})
@click.option(
    "--input-path", "-i", type=click.Path(exists=True, path_type=Path),
    default=str(DEFAULT_INPUT),
    help="scored+normalized input file (02_token_check.py output)"
)
@click.option(
    "--output-path", "-o", type=click.Path(path_type=Path),
    default=str(DEFAULT_OUTPUT), help="JSONL file to append generated rows to"
)
@click.option(
    "--review-path", type=click.Path(path_type=Path),
    default=str(DEFAULT_REVIEW_QUEUE),
    help=(
        "JSONL review queue: asr_training_source rows whose unit symbol survived "
        "deterministic normalization are appended here for human review"
    ),
)
@click.option(
    "--skip-path", type=click.Path(path_type=Path),
    default=str(DEFAULT_SKIP_QUEUE),
    help=(
        "JSONL skip queue: terms whose variants kept dropping the term itself "
        "past every retry are appended here and skipped, instead of aborting "
        "the run"
    ),
)
@click.option(
    "--asr-training-target-prompt-path", type=click.Path(exists=True, path_type=Path),
    default=str(DEFAULT_ASR_TRAINING_TARGET_PROMPT),
    help="System-prompt markdown file for the `asr_training_target` generation pass"
)
@click.option("--model", default=DEFAULT_MODEL, help="LLM model ID (litellm format, e.g. openrouter/...)")
@click.option(
    "--provider", default=DEFAULT_PROVIDER,
    help=(
        "OpenRouter provider slug to pin (e.g. 'deepseek'). Pinning is "
        "required for stable prompt-cache hits across calls. Pass an empty "
        "string to disable pinning."
    ),
)
@click.option(
    "--n-variants", "-n", default=str(DEFAULT_N_VARIANTS), type=str,
    help=(
        "Either an int (same count for every term) or a JSON dict mapping "
        "score ranges to counts, e.g. '{\"0\":0,\"1-3\":2,\"4-7\":4,\"8-10\":11}'. "
        "Dict keys must cover 0..10 with no overlaps; a count of 0 skips the "
        "range. Score comes from 01_llm_scoring.py."
    ),
)
@click.option("--n-jobs", "-j", default=DEFAULT_N_JOBS, type=int)
@click.option(
    "--limit", "-L", default=None, type=int,
    help="Only process the first N pending terms (good for smoke tests)"
)
@click.option("--timeout-s", default=DEFAULT_TIMEOUT_S, type=int)
@click.option("-v", "--verbose", count=True, help="-v for DEBUG logging")
def main(
    input_path: Path,
    output_path: Path,
    review_path: Path,
    skip_path: Path,
    asr_training_target_prompt_path: Path,
    model: str,
    provider: str,
    n_variants: str,
    n_jobs: int,
    limit: int | None,
    timeout_s: int,
    verbose: int,
) -> None:
    """CLI wrapper. For programmatic use, import `run` directly."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if verbose else "INFO",
        format="<green>{time:HH:mm:ss}</green> <level>{level: <7}</level> {message}",
    )
    run(
        input_path=input_path,
        output_path=output_path,
        review_path=review_path,
        skip_path=skip_path,
        asr_training_target_prompt_path=asr_training_target_prompt_path,
        model=model,
        provider=provider or None,
        n_variants=n_variants,
        n_jobs=n_jobs,
        limit=limit,
        timeout_s=timeout_s,
    )


if __name__ == "__main__":
    main()
