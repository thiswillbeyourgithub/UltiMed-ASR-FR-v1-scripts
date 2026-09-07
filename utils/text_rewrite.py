"""Shared pieces for the document-rewrite text stages (PARHAF, PARROT).

Stages 03 (PARHAF) and 04 (PARROT) do not turn a *term* into N dictation
sentences the way the dictionary and drugs stages do. Instead they take one raw
clinical document / radiology report and rewrite it into ONE faithful, flowing
French paragraph the ASR model can learn from: same medical jargon, but with the
colons, enumerations, headers and parenthetical asides removed. Both stages share
everything except which raw file they read, so the shared behaviour lives here and
each stage is a thin wrapper (mirroring how ``02_drugs/01_generate_drug_texts.py`` wraps
``text_generation_engine``).

This module provides the rewrite ``StageAdapter`` bits
(``build_rewrite_user_prompt``, ``rewrite_validate``, ``rewrite_make_row``,
``build_rewrite_adapter``) that plug into the same engine the scored stages use,
with ``require_score=False`` and a fixed one-variant-per-chunk policy.

The deterministic preprocessing (``strip_parens`` / ``chunk_text``) lives in the
stdlib-only ``text_chunking`` module so the per-stage ``03_chunk_for_rewrite.py``
preprocessors can import it without pulling in the LLM stack; they are re-exported
here for convenience.

Written with the help of Claude Code.
"""

from __future__ import annotations

import functools
from pathlib import Path

from _pipeline_shared import (  # noqa: E402
    LLMError,
    _FORBIDDEN_W_CHARS,
    _UNIT_SYMBOL_RE,
    _check_forbidden_chars,
    _check_no_refusal_or_english,
    _check_sentence_complete,
    call_llm,
)
from parakeet_tokenizer import ParakeetTokenizer  # noqa: E402
from text_chunking import DEFAULT_MAX_CHARS, chunk_text, strip_parens  # noqa: E402,F401
from text_generation_engine import StageAdapter, run as engine_run  # noqa: E402

# ---------------------------------------------------------------------------
# Rewrite StageAdapter: plugs the document->paragraph task into the shared engine
# ---------------------------------------------------------------------------

# Model / provider match the scored corpus stages so cache and cost behave the
# same across the whole dataset. Kept here so both stage wrappers share them.
DEFAULT_MODEL = "openrouter/deepseek/deepseek-v4-pro"
DEFAULT_PROVIDER = "deepseek"
DEFAULT_N_VARIANTS = 1  # exactly one rewritten paragraph per source chunk
DEFAULT_N_JOBS = 4
DEFAULT_TIMEOUT_S = 300
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95

# Reasoning escalation, matching the scored corpus stages: the first (cheap) pass
# runs "non-think", and a validation retry omits the argument so the provider
# default (thinking) only kicks in when the cheap pass failed the validators.
DEFAULT_REASONING = {"enabled": False}  # "non-think": first, cheap attempt
RETRY_REASONING = None                  # omit -> provider default (thinking)

# The rewrite must contain no colon (the whole point is to drop labelled /
# enumerated structure), on top of every character the base target prompt bans.
_REWRITE_FORBIDDEN_CHARS = _FORBIDDEN_W_CHARS | {":"}

# Length sanity vs the source chunk. Compression is expected (headers, boilerplate
# and repetition are dropped), so the lower bound is loose; the upper bound guards
# against runaway repetition or hallucinated content padding the paragraph out.
# The upper bound is generous because the prompt REQUIRES spelling every unit and
# number out in full French words, which legitimately inflates character length
# vs the shorthand source ("40 mg" -> "quarante milligrammes", "1-0-1" -> "un le
# matin, rien le midi, un le soir"). A 1.30 cap rejected exactly the densest,
# most term-rich medication / lab chunks (observed 1.32-1.67 on real skips), so
# it is set to 2.0: still catches genuine runaway repetition without punishing a
# faithful spell-out. Tiny chunks (which make the ratio noisy) are handled at the
# chunker with a minimum size, not here.
#
# Even 2.0 is not enough for every chunk: a lab panel is almost pure shorthand
# ("Na+ 142meq/L; K+ 3,6meq/L" -> "le sodium est a cent quarante-deux
# milliequivalents par litre, le potassium a trois virgule six milliequivalents
# par litre") and lands at 2.0x-3.2x however faithful the rewrite is. Those chunks
# are re-run with the upper bound turned off (``max_len_ratio=None``, the stage
# wrappers' ``--no-max-len-ratio``) rather than by raising the default, which would
# stop catching runaway repetition for the ~99.9% of chunks that are ordinary prose.
# The LOWER bound always applies: it guards against dropped findings, which the
# spell-out argument does not excuse.
_MIN_LEN_RATIO = 0.08
MAX_LEN_RATIO = 2.0

_TOK = ParakeetTokenizer()


def build_rewrite_user_prompt(entry: dict, n_variants: int) -> str:
    """Per-chunk user message: the raw source text to rewrite into one paragraph.

    ``n_variants`` is always 1 for this task (one paragraph per chunk); it is part
    of the engine's adapter signature, so it is accepted and ignored here.
    """
    source = entry.get("text", "")
    return (
        "<source>\n"
        f"{source}\n"
        "</source>\n"
        "Rewrite the text above as ONE flowing French paragraph inside a single "
        "<t>...</t> block, following every rule in the system prompt."
    )


def rewrite_validate(
    parsed: list[str], entry: dict, *, max_len_ratio: float | None = MAX_LEN_RATIO
) -> None:
    """Soft validators for a rewritten paragraph. Raise LLMError to trigger retry.

    Reuses the shared granular checks (forbidden chars, abbreviated units,
    refusal/English, sentence completeness) rather than the scored-stage bundle,
    because that bundle caps a variant at 600 chars, whereas a rewritten document
    paragraph is deliberately long. Adds the rewrite-specific rules: no colon
    (already folded into the forbidden set), tokenizable by Parakeet, and a length
    sanity check against the source chunk.

    ``max_len_ratio=None`` drops the upper bound only (the lower bound still
    applies), for the shorthand-dense chunks no faithful rewrite can fit under it.
    """
    _check_sentence_complete(parsed, label="rewrite")
    _check_no_refusal_or_english(parsed, label="rewrite")
    _check_forbidden_chars(parsed, _REWRITE_FORBIDDEN_CHARS, label="rewrite")
    for i, v in enumerate(parsed):
        unk = _TOK.offending_chars(v)
        if unk:
            chars = ", ".join(sorted(unk))
            raise LLMError(
                f"rewrite: paragraph {i} contains character(s) the Parakeet "
                f"tokenizer maps to <unk> (rewrite them in words): {chars}"
            )
        m = _UNIT_SYMBOL_RE.search(v)
        if m:
            raise LLMError(
                f"rewrite: paragraph {i} uses abbreviated unit {m.group(0)!r}; "
                f"spell every unit out in full French"
            )
    src_len = len(entry.get("text", "") or "")
    if src_len:
        for i, v in enumerate(parsed):
            ratio = len(v) / src_len
            if ratio < _MIN_LEN_RATIO:
                raise LLMError(
                    f"rewrite: paragraph {i} is only {ratio:.0%} of the source "
                    f"length; too much was dropped, keep all the findings"
                )
            if max_len_ratio is not None and ratio > max_len_ratio:
                raise LLMError(
                    f"rewrite: paragraph {i} is {ratio:.0%} of the source length; "
                    f"do not add or repeat content, stay faithful and concise"
                )


def rewrite_make_row(
    entry: dict, variant_index: int, target: str, source: str, model: str
) -> dict:
    """Output row for a rewrite stage (same target/source pair shape as the others).

    There is no scored term here, so ``term`` carries the chunk id purely so the
    engine's logs and skip queue stay readable; ``term_index`` / ``variant_index``
    keep resumability working exactly as for the scored stages.
    """
    return {
        "id": entry.get("id"),
        "category": entry.get("category"),
        "term_index": entry.get("index"),
        "variant_index": variant_index,
        "term": entry.get("id"),
        "asr_training_target": target,
        "asr_training_source": source,
        "source_id": entry.get("source_id"),
        "chunk_index": entry.get("chunk_index"),
        "model": model,
    }


def build_rewrite_adapter(
    stage: str,
    system_prompt: str,
    system_prompt_path: str | Path | None = None,
    call_llm_fn=call_llm,
    max_len_ratio: float | None = MAX_LEN_RATIO,
) -> StageAdapter:
    """Assemble the rewrite StageAdapter (require_score=False, one paragraph out).

    ``max_len_ratio=None`` turns the length upper bound off for this run; see
    ``rewrite_validate``.
    """
    return StageAdapter(
        stage=stage,
        system_prompt=system_prompt,
        system_prompt_path=system_prompt_path,
        build_user_prompt=build_rewrite_user_prompt,
        validate_fn=functools.partial(rewrite_validate, max_len_ratio=max_len_ratio),
        make_row=rewrite_make_row,
        call_llm_fn=call_llm_fn,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        # Cheap non-think first, thinking only on a validation retry (same policy
        # as the scored stages). See DEFAULT_REASONING / RETRY_REASONING.
        reasoning_first=DEFAULT_REASONING,
        reasoning_retry=RETRY_REASONING,
        require_score=False,
    )


def load_system_prompt(
    prompt_path: str | Path, example_path: str | Path | None = None
) -> str:
    """Read the shared rewrite base prompt, optionally with a per-stage example.

    The rules are identical for PARHAF and PARROT, so they live once in the base
    prompt; the only stage-specific part is the worked example (a clinical document
    for PARHAF, a radiology report for PARROT). Mirrors the drugs stage's
    ``base prompt + addendum`` composition to keep the shared rules DRY.
    """
    base = Path(prompt_path).read_text(encoding="utf-8")
    if example_path is None:
        return base
    example = Path(example_path).read_text(encoding="utf-8")
    return f"{base.rstrip()}\n\n{example.lstrip()}\n"


def run_rewrite(
    stage: str,
    input_path: str | Path,
    output_path: str | Path,
    system_prompt_path: str | Path,
    example_prompt_path: str | Path | None = None,
    *,
    model: str = DEFAULT_MODEL,
    provider: str | None = DEFAULT_PROVIDER,
    n_jobs: int = DEFAULT_N_JOBS,
    limit: int | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    review_path: str | Path = "voxtral_review_queue.jsonl",
    skip_path: str | Path = "term_missing_skips.jsonl",
    run_stats_path: str | Path = "run_statistics.jsonl",
    run_log_path: str | Path = "run_statistics.log",
    max_len_ratio: float | None = MAX_LEN_RATIO,
) -> dict:
    """Shared entry point for a rewrite stage: build the adapter and run the engine.

    A thin per-stage wrapper (03_PARHAF / 04_PARROT) just supplies its own paths.
    N is fixed at 1 (one paragraph per chunk) and the engine's score gate is off.
    ``max_len_ratio=None`` disables the length upper bound (see ``rewrite_validate``).
    """
    adapter = build_rewrite_adapter(
        stage,
        load_system_prompt(system_prompt_path, example_prompt_path),
        system_prompt_path,
        max_len_ratio=max_len_ratio,
    )
    return engine_run(
        adapter,
        input_path=input_path,
        output_path=output_path,
        n_variants=DEFAULT_N_VARIANTS,
        model=model,
        n_jobs=n_jobs,
        limit=limit,
        timeout_s=timeout_s,
        provider=provider,
        review_path=review_path,
        skip_path=skip_path,
        run_stats_path=run_stats_path,
        run_log_path=run_log_path,
    )
