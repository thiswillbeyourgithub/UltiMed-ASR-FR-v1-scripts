#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "click",
#     "loguru",
#     "litellm",
#     "tiktoken",
#     "tqdm",
#     "tenacity",
#     "rapidfuzz",
# ]
# ///
"""Rewrite each PARHAF chunk into one faithful French paragraph, via the shared engine.

Thin wrapper over ``utils/text_rewrite.run_rewrite`` (the same engine the
dictionary and drugs stages use). It reads the rewrite chunks produced by
``03_chunk_for_rewrite.py`` and, for each chunk, makes ONE LLM call that rewrites
the raw clinical text into a single flowing paragraph (``asr_training_target``),
then derives the ``asr_training_source`` deterministically via ``voxtral_normalize``.
All orchestration (retry-with-validation, dedup, resumability, run statistics,
review / skip queues, pricing pre-plan) lives in the engine; this file only
supplies the PARHAF paths. The score gate is off (``require_score=False``): PARHAF
chunks carry no score and the policy is a fixed one paragraph per chunk.

Written with the help of Claude Code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from text_rewrite import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_N_JOBS,
    DEFAULT_PROVIDER,
    DEFAULT_TIMEOUT_S,
    MAX_LEN_RATIO,
    run_rewrite,
)

HERE = Path(__file__).resolve().parent
_PROMPT = HERE.parent / "utils" / "PROMPT_REWRITE_TO_PARAGRAPH.md"
# Stage-specific worked example (a clinical document), appended to the shared base
# prompt. PARROT uses its own radiology example; the rules stay shared.
_EXAMPLE = HERE / "PROMPT_REWRITE_PARHAF_EXAMPLES.md"

DEFAULT_INPUT = HERE / "03_parhaf_rewrite_chunks.jsonl"
DEFAULT_OUTPUT = HERE / "generated_dataset.jsonl"
DEFAULT_REVIEW_QUEUE = HERE / "voxtral_review_queue.jsonl"
DEFAULT_SKIP_QUEUE = HERE / "term_missing_skips.jsonl"
DEFAULT_RUN_STATS = HERE / "run_statistics.jsonl"
DEFAULT_RUN_LOG = HERE / "run_statistics.log"


@click.command(context_settings={"show_default": True})
@click.option("--input-path", "-i", default=str(DEFAULT_INPUT), type=click.Path(path_type=Path))
@click.option("--output-path", "-o", default=str(DEFAULT_OUTPUT), type=click.Path(path_type=Path))
@click.option("--prompt-path", default=str(_PROMPT), type=click.Path(path_type=Path))
@click.option(
    "--example-prompt-path", default=str(_EXAMPLE), type=click.Path(path_type=Path),
    help="Stage-specific worked example appended to the shared rewrite prompt",
)
@click.option("--model", default=DEFAULT_MODEL, help="LLM model id")
@click.option(
    "--provider", default=DEFAULT_PROVIDER,
    help="OpenRouter provider slug to pin (empty string disables pinning)",
)
@click.option("--n-jobs", "-j", default=DEFAULT_N_JOBS, type=int)
@click.option(
    "--limit", "-L", default=None, type=int,
    help="Only process the first N pending chunks (smoke tests)",
)
@click.option("--timeout-s", default=DEFAULT_TIMEOUT_S, type=int)
@click.option(
    "--no-max-len-ratio", is_flag=True,
    help="Drop the length upper bound (paragraph vs source chunk). Shorthand-dense "
         "chunks (lab panels) spell out to 2x-3.2x however faithful the rewrite is, "
         "so they can only be generated with this on. The lower bound still applies.",
)
@click.option("-v", "--verbose", count=True, help="-v for DEBUG logging")
def main(
    input_path: Path,
    output_path: Path,
    prompt_path: Path,
    example_prompt_path: Path,
    model: str,
    provider: str,
    n_jobs: int,
    limit: int | None,
    timeout_s: int,
    no_max_len_ratio: bool,
    verbose: int,
) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if verbose else "INFO",
        format="<green>{time:HH:mm:ss}</green> <level>{level: <7}</level> {message}",
    )
    run_rewrite(
        stage="parhaf",
        input_path=input_path,
        output_path=output_path,
        system_prompt_path=prompt_path,
        example_prompt_path=example_prompt_path,
        model=model,
        provider=provider or None,
        n_jobs=n_jobs,
        limit=limit,
        timeout_s=timeout_s,
        max_len_ratio=None if no_max_len_ratio else MAX_LEN_RATIO,
        review_path=DEFAULT_REVIEW_QUEUE,
        skip_path=DEFAULT_SKIP_QUEUE,
        run_stats_path=DEFAULT_RUN_STATS,
        run_log_path=DEFAULT_RUN_LOG,
    )


if __name__ == "__main__":
    main()
