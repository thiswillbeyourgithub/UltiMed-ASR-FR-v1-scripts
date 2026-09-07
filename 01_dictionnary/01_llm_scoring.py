#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "click",
#     "litellm",
#     "tiktoken",
#     "loguru",
#     "tqdm",
#     "joblib",
#     "tenacity",
# ]
# ///
"""Score French medical dictionary terms for ASR fine-tuning usefulness.

Reads a JSONL dictionary file (fields: term, index, page, definition, examples)
and scores each term 0–10 for how useful it would be to include in a French
medical ASR fine-tuning dataset.

High scores (8-10): rare, technical medical vocabulary with non-trivial French
  pronunciation (drug names, syndrome eponyms, anatomical terms, disease names).
Low scores (0-3): pure abbreviations/acronyms, common everyday words, redirect-
  only entries with no substantive definition.

Output is a JSONL file (default: <input_stem>.scored.jsonl) with the original
fields plus a "score" (int 0-10) field. Entries whose scoring fails (an LLM
error after all retries, a truncated or censored completion, or a response with
no <score> tag) are omitted from the output rather than written with a null
score, so a later --resume retries them.

The run is resumable via --resume: already-scored indices are detected from the
output file and skipped.

Written with the help of aider.chat (https://github.com/Aider-AI/aider/).
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import joblib
import litellm
import tiktoken
from loguru import logger
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "openrouter/openai/gpt-oss-120b"
DEFAULT_N_JOBS = 4
# Pipeline stage 1 input. Kept next to the script so `uv run 01_llm_scoring.py`
# with no arguments reproduces the first stage (raw dictionary -> scored). The
# output defaults to <input_stem>.scored.jsonl alongside it, which is stage 2's
# (02_token_check.py) default input.
HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "original_dictionnary.jsonl"
# tiktoken doesn't have a Claude tokeniser; cl100k_base is a reasonable proxy
# for estimating token counts (GPT-4 tokeniser, similar subword granularity).
TIKTOKEN_ENCODING = "cl100k_base"

# Accepted finish_reason values from the completion API. Anything else means the
# response was cut short (length / max_tokens) or blocked (content_filter), so it
# must not be parsed into a score. KEEP IN SYNC with the same-named set in
# utils/_pipeline_shared.py: the two are deliberate copies because this script
# keeps a standalone joblib / tiktoken LLM stack rather than importing the shared
# call_llm (which would pull in the whole pipeline dependency chain).
_OK_FINISH_REASONS = {"stop", "end_turn", "completed", None}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

# The prompt is a module-level constant because it never changes between calls;
# injecting it once here avoids passing it through every function signature.
SYSTEM_PROMPT = """\
You are evaluating entries from a French medical dictionary for inclusion in an ASR (Automatic Speech Recognition) fine-tuning dataset.

Goal: improve a medical ASR model's ability to correctly transcribe technical French medical words and phrases when spoken aloud.

Given a dictionary entry (JSON with fields: term, definition, examples), assign a score from 0 to 10 reflecting how valuable adding this term to the ASR training data would be:

  8-10  Highly valuable: specific technical medical term — drug names, anatomical structures, rare disease names, syndrome eponyms, complex Latin-derived words — unlikely to appear in generic ASR training data and with a non-trivial French pronunciation.

  4-7   Moderately valuable: medical but relatively common, or only somewhat phonetically challenging.

  0-3   Low value:
        - Pure abbreviations or acronyms (e.g. "AAS", "A.O.") — no acoustic / pronunciation learning value.
        - Everyday words that happen to have a medical use.
        - Redirect / cross-reference entries with no substantive definition.
        - Very short Latin phrases rarely spoken in clinical French.

  0-1   Effectively zero value — obsolete / historical terms:
        - Terms explicitly marked as "vieilli", "désuet", "ancien", "historique", "obsolète" or equivalent in their definition.
        - Disease names, syndrome names or drug names that are no longer used in modern clinical practice and have been replaced by a different term.
        - Terms a clinician or patient would never say aloud in a contemporary French medical encounter.
        Score these 0 or 1 regardless of their phonetic complexity.

--- FEW-SHOT EXAMPLES ---

achondrogénèse : 10 (typical technical word)
syndrome d'Aarskog-Scott : 9 (rare syndrome eponym, complex foreign name, non-trivial pronunciation)
signe d'Abadie : 8 (named clinical sign eponym, neurological term, unlikely in generic ASR data)
paralysie a frigore : 8 (short Latin phrase, often spoken)
abaissement des bras en obstétrique : 5 (technical obstetric manoeuvre, specialised vocabulary, eponym references)
abaisse-langue : 5 (common clinical instrument, compound word, could appear in general corpora)
AAS : 2 (pure abbreviation, redirect-only, no pronunciation learning value)
érisiphaque : 0 (technical but obsolete)
traitement par l' œuf de caille: 0 (obsolete)


You may include a brief <thinking> block with your reasoning before the score if helpful. Reply using EXACTLY this XML format (no other text outside the tags):

<thinking>
(your reasoning here, including whether the term is obsolete/historical)
</thinking>
<score>N</score>

Where N is a single integer from 0 to 10. Nothing else inside <score>.
"""

# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

_SCORE_RE = re.compile(r"<score>\s*(\d+)\s*</score>", re.DOTALL)


def _parse_score(response_text: str) -> Optional[int]:
    """Extract and clamp the integer score from the first <score>…</score> tag.

    Returns None if no valid tag is found, so the caller can decide how to
    handle the failure (log a warning, write None, retry later).
    """
    match = _SCORE_RE.search(response_text)
    if match is None:
        return None
    val = int(match.group(1))
    # Clamp defensively in case the model drifts outside the 0-10 range
    return max(0, min(10, val))


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


class TokenCounter:
    """Accumulates estimated input/output token counts via tiktoken.

    Thread-safe: internal lock protects the running totals so multiple
    joblib threads can call add_input / add_output concurrently.
    """

    def __init__(self, encoding_name: str = TIKTOKEN_ENCODING) -> None:
        self._enc = tiktoken.get_encoding(encoding_name)
        self._lock = threading.Lock()
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    def count(self, text: str) -> int:
        """Return the number of tokens in *text*."""
        return len(self._enc.encode(text))

    def add_input(self, text: str) -> None:
        n = self.count(text)
        with self._lock:
            self.input_tokens += n

    def add_output(self, text: str) -> None:
        n = self.count(text)
        with self._lock:
            self.output_tokens += n

    def summary(self) -> str:
        total = self.input_tokens + self.output_tokens
        return (
            f"Token estimate (tiktoken {TIKTOKEN_ENCODING}): "
            f"input={self.input_tokens:,}  output={self.output_tokens:,}  "
            f"total={total:,}"
        )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

# Max retries and backoff bounds for transient LLM API errors (rate limits,
# server errors, timeouts). Exponential wait: 2^attempt seconds, capped at 60s.
_LLM_MAX_RETRIES = 6
_LLM_BACKOFF_MIN_SECONDS = 2
_LLM_BACKOFF_MAX_SECONDS = 60


@retry(
    stop=stop_after_attempt(_LLM_MAX_RETRIES),
    wait=wait_exponential(
        min=_LLM_BACKOFF_MIN_SECONDS,
        max=_LLM_BACKOFF_MAX_SECONDS,
    ),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, "WARNING"),  # type: ignore[arg-type]
    reraise=True,
)
def _call_llm(
    *,
    user_message: str,
    model: str,
    base_url: Optional[str],
    counter: TokenCounter,
) -> str:
    """Send a single request to the LLM and return the raw response text.

    Retries up to ``_LLM_MAX_RETRIES`` times with exponential backoff on any
    exception (covers rate limits, transient server errors, timeouts).
    Updates *counter* with estimated token usage.
    """
    counter.add_input(SYSTEM_PROMPT + user_message)

    # Mark the system prompt as a cache breakpoint so OpenRouter caches it
    # across requests. Required for Anthropic/Gemini; harmless no-op for
    # providers where caching is automatic (OpenAI, DeepSeek, Grok, Moonshot,
    # gpt-oss family). See https://openrouter.ai/docs/features/prompt-caching
    kwargs: dict = dict(
        model=model,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
    )
    if base_url is not None:
        kwargs["api_base"] = base_url

    response = litellm.completion(**kwargs)
    choice = response.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    # Finish-reason guard: refuse a truncated ("length" / "max_tokens") or
    # censored ("content_filter") completion instead of silently parsing a
    # half-written response into a null score. Raising here triggers the @retry
    # wrapper above, which is what makes a transient truncation recoverable.
    # Mirrors the guard in utils/_pipeline_shared.call_llm.
    if finish_reason not in _OK_FINISH_REASONS:
        raise RuntimeError(
            f"llm call ended with finish_reason={finish_reason!r} "
            f"(length=truncated at the provider max, content_filter=censored)"
        )
    text: str = choice.message.content  # type: ignore[union-attr]
    if text is None:
        raise RuntimeError(
            f"litellm returned None content (finish_reason={finish_reason!r})"
        )

    counter.add_output(text)
    return text


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------


def _collect_processed_indices(output_path: Path) -> set[int]:
    """Read *output_path* and return the set of already-scored term indices.

    Uses the "index" field (unique per dictionary entry) as the resume key,
    because dictionary entries have no audio_filepath like NeMo manifests do.
    """
    seen: set[int] = set()
    if not output_path.exists():
        return seen
    for raw_line in output_path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        idx = obj.get("index")
        if idx is not None:
            seen.add(int(idx))
    return seen


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def _score_terms(
    *,
    entries: list[dict],
    model: str,
    base_url: Optional[str],
    output_path: Path,
    counter: TokenCounter,
    resume: bool,
    n_jobs: int,
) -> None:
    """Score each dictionary entry and append results to *output_path*.

    Each successfully scored entry gets an int "score" field (0 to 10) appended
    and is written as a JSONL line. Entries whose scoring fails are logged and
    skipped (not written), so a later --resume retries them.

    In resume mode we append to the existing file and skip already-scored
    indices; otherwise we rotate the existing file and start fresh.
    """
    already_done: set[int] = set()
    if resume:
        already_done = _collect_processed_indices(output_path)
        logger.info(f"Resuming: {len(already_done)} entries already scored")
    elif output_path.exists():
        # Rotate rather than silently overwrite previous results
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = output_path.with_suffix(f".{ts}{output_path.suffix}")
        logger.info(f"Output exists, rotating: {output_path} -> {backup}")
        output_path.rename(backup)

    file_mode = "a" if resume else "w"

    # Build work list, skipping already-done indices
    work_items: list[dict] = [
        e for e in entries if int(e.get("index", -1)) not in already_done
    ]
    logger.info(f"{len(work_items)} entries to score ({len(already_done)} skipped)")

    write_lock = threading.Lock()
    stats: dict[str, int] = {"scored": 0, "failed": 0}

    def _score_one(entry: dict, f_out, pbar: tqdm) -> None:  # noqa: ANN001
        """Score a single entry: call LLM, parse score, write JSON line."""
        # Send only the fields relevant to the scoring decision; omitting
        # "page" keeps the payload focused and slightly more token-efficient.
        payload = {
            k: entry[k] for k in ("term", "definition", "examples") if k in entry
        }
        user_message = json.dumps(payload, ensure_ascii=False)

        score: Optional[int] = None
        try:
            raw = _call_llm(
                user_message=user_message,
                model=model,
                base_url=base_url,
                counter=counter,
            )
            score = _parse_score(raw)
            if score is None:
                logger.warning(
                    f"[{entry.get('index')}] {entry.get('term')!r}: "
                    f"no <score> tag in response; skipping (a later --resume "
                    f"will retry it)"
                )
        except Exception as exc:
            logger.error(
                f"[{entry.get('index')}] {entry.get('term')!r}: LLM call failed "
                f"after retries: {exc}"
            )

        with write_lock:
            if score is not None:
                result = dict(entry)
                result["score"] = score
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()
                stats["scored"] += 1
                logger.info(
                    f"[{entry.get('index')}] {entry.get('term')!r} -> score={score}"
                )
            else:
                # Do not persist a failed row. Leaving the index absent from the
                # output is exactly what lets a later --resume pick it up again,
                # instead of writing a null score that resume would treat as done.
                stats["failed"] += 1
            pbar.update(1)

    with open(output_path, file_mode, encoding="utf-8") as f_out:
        with tqdm(
            total=len(work_items), desc="Scoring", unit="term", smoothing=0.05
        ) as pbar:
            joblib.Parallel(n_jobs=n_jobs, backend="threading")(
                joblib.delayed(_score_one)(entry, f_out, pbar) for entry in work_items
            )

    logger.info(f"Done — scored={stats['scored']}  failed={stats['failed']}")
    logger.info(f"Output: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.argument(
    "input_path",
    required=False,
    default=str(DEFAULT_INPUT),
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output JSONL path. Defaults to <input_stem>.scored.jsonl alongside the input.",
)
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    show_default=True,
    help="litellm model identifier.",
)
@click.option(
    "--base-url",
    default=None,
    help="Optional base URL override for the LLM API (passed as api_base to litellm).",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Append to an existing output file, skipping already-scored indices.",
)
@click.option(
    "--n-jobs",
    default=DEFAULT_N_JOBS,
    show_default=True,
    help="Number of parallel threads for LLM calls (joblib threading backend).",
)
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Process only the first N entries (useful for a quick smoke-test).",
)
@click.option(
    "--no-confirm",
    is_flag=True,
    default=False,
    help="Skip the token-estimate confirmation prompt (useful for automated runs).",
)
def main(
    input_path: Path,
    output_path: Optional[Path],
    model: str,
    base_url: Optional[str],
    resume: bool,
    n_jobs: int,
    limit: Optional[int],
    no_confirm: bool,
) -> None:
    """Score French medical dictionary terms for ASR fine-tuning usefulness.

    INPUT_PATH is a JSONL file where each line has at least the fields:
    term, index, definition, examples (i.e. original_dictionnary.jsonl).

    Each entry is sent to an LLM which returns a score 0–10. The output is a
    JSONL file with the original fields plus a "score" field.  Entries where
    the LLM call failed completely are written with score=null so they can be
    retried with --resume.

    Examples
    --------
    Score all terms::

        uv run llm_filtering.py 01_dictionnary/original_dictionnary.jsonl

    Resume an interrupted run::

        uv run llm_filtering.py 01_dictionnary/original_dictionnary.jsonl --resume

    Quick smoke-test on 20 entries without a confirmation prompt::

        uv run llm_filtering.py 01_dictionnary/original_dictionnary.jsonl \\
            --limit 20 --no-confirm
    """
    if output_path is None:
        # Place output alongside input: original_dictionnary.scored.jsonl
        output_path = input_path.with_suffix(".scored.jsonl")

    logger.info(f"Model:  {model}")
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")

    # Load all entries up-front so we know the total count for progress display
    entries: list[dict] = []
    for raw_line in input_path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            entries.append(json.loads(raw_line))
        except json.JSONDecodeError as exc:
            logger.warning(f"Skipping non-JSON line: {exc}")

    if limit is not None:
        entries = entries[:limit]
        logger.info(f"--limit {limit}: processing {len(entries)} entries")
    else:
        logger.info(f"Loaded {len(entries)} entries")

    counter = TokenCounter()

    # --- Pre-flight token estimate ---
    # Respect --resume so the estimate only covers entries that will actually
    # be sent to the LLM, giving an accurate cost preview.
    already_done: set[int] = set()
    if resume:
        already_done = _collect_processed_indices(output_path)

    system_tokens = counter.count(SYSTEM_PROMPT)
    sample_tokens = 0
    n_to_score = 0
    for entry in entries:
        if int(entry.get("index", -1)) in already_done:
            continue
        payload = {
            k: entry[k] for k in ("term", "definition", "examples") if k in entry
        }
        sample_tokens += counter.count(json.dumps(payload, ensure_ascii=False))
        n_to_score += 1

    estimated_input = (system_tokens * n_to_score) + sample_tokens
    logger.info(
        f"Estimated input tokens: ~{estimated_input:,} "
        f"({n_to_score} entries × ~{system_tokens} system tokens + entry text)"
    )

    if not no_confirm and not click.confirm("Proceed?", default=True):
        logger.info("Aborted by user.")
        raise SystemExit(0)

    _score_terms(
        entries=entries,
        model=model,
        base_url=base_url,
        output_path=output_path,
        counter=counter,
        resume=resume,
        n_jobs=n_jobs,
    )

    logger.info(counter.summary())


if __name__ == "__main__":
    main()
