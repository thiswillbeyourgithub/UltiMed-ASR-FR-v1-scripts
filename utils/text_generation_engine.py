# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "loguru",
#     "tqdm",
#     "tenacity",
#     "litellm",
#     "rapidfuzz",
#     "tiktoken",
# ]
# ///
"""Source-agnostic engine for the text-generation stages.

Every corpus stage (dictionary, drugs, and the upcoming PARHAF / PARROT stages)
turns scored input rows into `asr_training_target` / `asr_training_source` pairs
through the same orchestration: a score-based variant-count policy, a
retry-with-validation LLM call, a deterministic voxtral source pass, cross-run
resumability, target/source dedup, run statistics, and review / skip queues.

That orchestration is the same for every source; only a few things differ per
stage (the input's term/score fields, the system prompt, the per-item user
prompt, and which validators apply). So the orchestration lives here and each
stage supplies the stage-specific bits as an adapter, instead of every stage
re-implementing (and drifting on) the engine.

This first extraction moves the genuinely source-agnostic, dependency-free
pieces: the n-variants score policy, the JSONL input iterator, the resume-state
scanner, and the per-term skip signal. The dictionary stage imports and
re-exports them so its public surface is unchanged; later steps move the run
loop and the per-term generation function here too.
"""

from __future__ import annotations

import datetime
import json
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from loguru import logger
from tqdm import tqdm

from _pipeline_shared import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT_S,
    BlockCountError,
    CountMismatchError,
    LLMError,
    PricingTracker,
    TermMissingError,
    ValidationError,
    _check_no_cross_term_dup,
    _retry_with_validation,
    _text_hash,
    parse_asr_training_target,
)
from voxtral_normalize import (  # noqa: E402
    append_jsonl_row,
    normalize_and_flag,
    record_review,
)

# Default variant count when a caller does not resolve one from the score
# policy. Overridden per call by run() (which resolves it per term) and by the
# stage wrappers, so this is only a floor for direct callers / tests.
DEFAULT_N_VARIANTS = 11


class TermSkipped(RuntimeError):
    """Signal: a single term was skipped after a persistent recoverable error.

    Not a failure and not fatal. Raised by generate_variants_for_term when
    the retry-with-feedback loop still could not get a clean result for one
    term: either the model kept dropping the term (term-missing) or kept
    emitting the wrong `<t>` block count (block-count). The run worker turns
    it into a skip (already logged to the skip queue) so one stubborn term
    does not abort the run. `reason` records which case it was.
    """

    def __init__(self, term_index, term, reason="persistent term-missing"):
        super().__init__(f"term {term_index} {term!r} skipped after {reason}")
        self.term_index = term_index
        self.term = term
        self.reason = reason


# Type alias: n_variants policy can be either a single int (same count for all
# terms) or a dict mapping inclusive (lo, hi) score ranges to variant counts.
NVariantsPolicy = int | dict[tuple[int, int], int]


def parse_n_variants(value) -> NVariantsPolicy:
    """Parse an --n-variants argument into either an int or a range->count dict.

    Accepts:
      * int (e.g. 11): same count for every term.
      * dict (already parsed) like {"1-3": 2, "4-7": 4, "8-10": 11}.
      * str: either an int literal ("11") or a JSON dict literal
        ('{"1-3": 2, "4-7": 4, "8-10": 11}'). Keys may be ranges "lo-hi"
        (inclusive) or single integers.

    Scores not matched by any range produce 0 variants and the term is skipped,
    which is intentional: ranges let callers exclude unscored / zero-score
    obsolete entries by simply omitting them from the mapping.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        if value <= 0:
            raise ValueError(
                f"--n-variants as int must be > 0 (got {value}); a value of 0 "
                "would silently skip every term"
            )
        return value
    if isinstance(value, dict):
        return _normalize_ranges(value)
    s = str(value).strip()
    try:
        n = int(s)
    except ValueError:
        pass
    else:
        if n <= 0:
            raise ValueError(
                f"--n-variants as int must be > 0 (got {n}); a value of 0 "
                "would silently skip every term"
            )
        return n
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError(
            f"--n-variants must be an int or a JSON dict, got {type(obj).__name__}"
        )
    return _normalize_ranges(obj)


def _normalize_ranges(d: dict) -> dict[tuple[int, int], int]:
    """Convert {'1-3': 2, '8': 11} into {(1, 3): 2, (8, 8): 11}.

    Validates that the ranges fully cover the integers 0..10 (the scoring
    domain produced by 01_llm_scoring.py) with no overlaps. A count of 0 is
    allowed and means "skip terms in this range".
    """
    result: dict[tuple[int, int], int] = {}
    for k, v in d.items():
        key = str(k).strip()
        if "-" in key:
            lo_s, hi_s = key.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
        else:
            lo = hi = int(key)
        if lo > hi:
            raise ValueError(f"invalid range {key!r}: lo > hi")
        result[(lo, hi)] = int(v)

    seen: dict[int, tuple[int, int]] = {}
    for (lo, hi) in result:
        for s in range(lo, hi + 1):
            if s in seen:
                raise ValueError(
                    f"score {s} is covered by both {seen[s]} and {(lo, hi)}; "
                    "ranges must not overlap"
                )
            seen[s] = (lo, hi)
    missing = [s for s in range(0, 11) if s not in seen]
    extra = [s for s in seen if s < 0 or s > 10]
    if missing:
        raise ValueError(
            f"score range mapping must cover 0..10; missing scores: {missing}"
        )
    if extra:
        raise ValueError(
            f"score range mapping must stay within 0..10; out-of-range: {sorted(set(extra))}"
        )
    return result


def resolve_n_variants(policy: NVariantsPolicy, score) -> int:
    """Resolve the variant count for one term based on its score.

    Returns 0 when the score does not match any range in a dict policy (the
    caller should skip the term). An int policy ignores the score entirely.
    """
    if isinstance(policy, int):
        return policy
    if score is None:
        return 0
    for (lo, hi), n in policy.items():
        if lo <= score <= hi:
            return n
    return 0


def iter_input(input_path: Path) -> Iterator[dict]:
    n_yielded = 0
    with input_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    f"skipping malformed line {lineno} in {input_path}: {exc}"
                )
                continue
            # Stage inputs that do not carry their own "index" (e.g. the drugs
            # file) get a stable position-based one so resumability (keyed on
            # term_index + variant_index) and the output term_index work.
            # Dictionary rows already carry index, so setdefault is a no-op for
            # them. Stable only while the input file is append-only / unreordered.
            if isinstance(row, dict):
                row.setdefault("index", n_yielded)
            n_yielded += 1
            yield row


def load_existing_state(
    output_path: Path,
) -> tuple[dict[int, set[int]], set[str], set[str]]:
    """Scan an existing output JSONL and return:
      * a map of term_index -> set of variant_indexes already written
      * a set of sha256 hashes of every asr_training_target text on disk
      * a set of sha256 hashes of every asr_training_source text on disk

    Hashes are kept per-column so a target string colliding with a source
    string on a different row is not treated as a duplicate.
    """
    done: dict[int, set[int]] = {}
    target_hashes: set[str] = set()
    source_hashes: set[str] = set()
    if not output_path.exists():
        return done, target_hashes, source_hashes
    with output_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"skipping unparseable line in {output_path}")
                continue
            idx = row.get("term_index")
            vidx = row.get("variant_index")
            if idx is not None and vidx is not None:
                done.setdefault(idx, set()).add(vidx)
            t = row.get("asr_training_target")
            if t:
                target_hashes.add(_text_hash(t))
            s = row.get("asr_training_source")
            if s:
                source_hashes.add(_text_hash(s))
    return done, target_hashes, source_hashes


def _truncate(text: str, max_chars: int) -> str:
    """Cut `text` to `max_chars` chars, appending `...` if it was longer."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def build_target_user_prompt(
    term: str,
    n_variants: int,
    definition: str = "",
    examples=(),
    max_definition_chars: int = 600,
    max_example_chars: int = 300,
    extra_lines=(),
) -> str:
    """Assemble the per-item user message for the target pass.

    This is the format the shared base system prompt
    (PROMPT_GENERATE_ASR_TRAINING_TARGET.md) expects: a `Term:` line, an optional
    `Definition:` hint, optional reference examples, the count, and a reminder to
    emit exactly N `<t>...</t>` blocks with the Term in each. Every stage uses
    it: the dictionary passes a sanitized definition + examples; the drugs stage
    passes sampled pharmaceutical presentations as the definition hint (the drugs
    prompt addendum treats a Definition line as a factual presentation cue).

    `extra_lines` are stage-specific data lines inserted AFTER the Definition
    (e.g. the acronyms stage's `Pronounced as: "..."` hint, whose semantics its
    prompt addendum explains). Putting them after Term+Definition keeps a common
    user-message prefix across calls that differ only in those lines, which is
    what provider-side prompt caching can reuse.

    Callers are expected to have already sanitized `definition`, `extra_lines`
    and stripped / prepared `examples`; truncation here is a defensive length
    cap so one oversized field cannot balloon the per-item (uncached) token
    cost.
    """
    parts: list[str] = [f"Term: {term}"]
    if definition:
        parts.append(f"Definition: {_truncate(definition, max_definition_chars)}")
    for line in extra_lines:
        parts.append(line)
    if examples:
        parts.append(
            "Reference examples (style and register cues only, do not "
            "paraphrase, do not reuse phrasing):"
        )
        for i, e in enumerate(examples, start=1):
            parts.append(f"{i}. {_truncate(e, max_example_chars)}")
    parts.append(f"Produce {n_variants} variants.")
    parts.append(
        f"Reminder: emit exactly {n_variants} <t>...</t> blocks. "
        f"Put the Term in each one. Spell out every unit."
    )
    return "\n".join(parts)


@dataclass
class StageAdapter:
    """The small set of stage-specific behaviours the engine needs injected.

    Everything else (retry-with-validation, the deterministic voxtral source
    pass, dedup, resumability, run statistics, review / skip queues, pricing
    pre-plan, the thread pool) is identical across corpus stages and lives in
    the engine. A stage supplies:

    * ``build_user_prompt(entry, n_variants) -> str``: the per-item user message
      for the target pass (dictionary: term + definition + examples; drugs:
      substance + sampled presentations; PARHAF / PARROT: the source document).
    * ``validate_fn(parsed, entry) -> None``: the stage's soft validators. It
      must raise ``LLMError`` subclasses (``TermMissingError`` / ``BlockCountError``)
      or ``ValidationError`` on failure so the retry loop can feed them back.
      The engine runs the generic cross-term dedup check separately, so this
      hook only carries stage-specific checks (banned chars, example leak,
      term presence). Pass a no-op to opt out (e.g. a stage that rephrases a
      whole document has no single term to require).
    * ``make_row(entry, variant_index, target, source, model) -> dict``: the
      output row schema for this stage.
    * ``call_llm_fn``: the LLM call. Injected (rather than imported) so a stage
      wrapper can pass its own module-global ``call_llm``, which keeps test
      monkeypatching working through the wrapper.

    The sampling knobs (``temperature`` / ``top_p`` / ``reasoning_first`` /
    ``reasoning_retry``) are per-stage but engine-recorded in run statistics.
    ``system_prompt`` is the already-read prompt text; ``system_prompt_path`` is
    kept only for the run-config snapshot.
    """

    stage: str
    system_prompt: str
    build_user_prompt: Callable[[dict, int], str]
    validate_fn: Callable[[list[str], dict], None]
    make_row: Callable[[dict, int, str, str, str], dict]
    call_llm_fn: Callable[..., str]
    temperature: float = 0.7
    top_p: float = 0.95
    reasoning_first: object = None
    reasoning_retry: object = None
    system_prompt_path: str | Path | None = None
    # Whether the input rows must carry an integer score in 0..10. True for the
    # scored corpus stages (dictionary, drugs): a missing score is a scorer gap
    # and aborts the run so the dataset never silently shrinks. False for the
    # rewrite stages (PARHAF, PARROT), whose input is raw document/report text
    # with no per-term score and a fixed N=1 variant policy, so there is nothing
    # to score against.
    require_score: bool = True
    # Optional deterministic transform applied to each validated target BEFORE
    # voxtral normalization derives the TTS source: (target, entry) -> the text
    # handed to normalize_and_flag. None keeps the default (the source derives
    # from the target itself). The acronyms stage uses it to substitute the
    # written acronym with its quoted pronunciation; it must stay LLM-free so
    # target/source alignment remains exact by construction.
    source_transform: Callable[[str, dict], str] | None = None


def generate_variants_for_term(
    term_entry: dict,
    adapter: StageAdapter,
    n_variants: int = DEFAULT_N_VARIANTS,
    model: str = "",
    timeout_s: int = DEFAULT_TIMEOUT_S,
    seen_target_hashes: set[str] | None = None,
    seen_source_hashes: set[str] | None = None,
    seen_lock: threading.Lock | None = None,
    provider: str | None = None,
    tracker: PricingTracker | None = None,
    review_path: str | Path = "voxtral_review_queue.jsonl",
    skip_path: str | Path = "term_missing_skips.jsonl",
) -> list[dict]:
    """Produce a list of N row dicts for one input entry, ready to write to JSONL.

    Runs ONE LLM call to generate N `asr_training_target` transcripts, then
    derives each `asr_training_source` deterministically from its target via
    `voxtral_normalize` (no second LLM call). The system prompt is small and
    stable so prompt caching kicks in across entries. All stage-specific
    behaviour comes from `adapter`.
    """
    term_index = term_entry.get("index")
    term_str = term_entry.get("term", "")
    user_prompt = adapter.build_user_prompt(term_entry, n_variants)

    # Counts this closure's own invocations (one per validation attempt) so the
    # first pass uses reasoning_first and every retry uses reasoning_retry.
    # Transport-level retries inside call_llm reuse the level.
    attempt_state = {"n": 0}

    def produce_asr_training_target(retry_context=None) -> list[str]:
        attempt_state["n"] += 1
        reasoning = (
            adapter.reasoning_first
            if attempt_state["n"] == 1
            else adapter.reasoning_retry
        )
        raw = adapter.call_llm_fn(
            user_prompt=user_prompt,
            system_prompt=adapter.system_prompt,
            model=model,
            timeout_s=timeout_s,
            retry_context=retry_context,
            temperature=adapter.temperature,
            top_p=adapter.top_p,
            reasoning=reasoning,
            provider=provider,
            tracker=tracker,
        )
        try:
            # Inside the try so LLMError subclasses (BlockCountError on a wrong
            # <t> count, TermMissingError on a dropped term) get raw_output
            # attached and are retried with feedback rather than aborting the run.
            parsed = parse_asr_training_target(raw, n_variants)
            adapter.validate_fn(parsed, term_entry)
            _check_no_cross_term_dup(
                parsed, seen_target_hashes, seen_lock, label="asr_training_target"
            )
        except LLMError as e:
            e.raw_output = raw
            raise
        return parsed

    try:
        asr_training_target_texts = _retry_with_validation(
            produce_asr_training_target,
            label=f"{adapter.stage} term {term_index} asr_training_target",
            tracker=tracker,
        )
    except ValidationError as exc:
        # A per-term soft-validation exhaustion: the model could not produce a
        # clean batch for THIS term within the retry budget (the ValidationError's
        # __cause__ is the last underlying LLMError). Skip the single term (log it
        # to the skip queue) and let the run continue: one stubborn term out of
        # ~62k must never abort a multi-hour, ~$20 run. Which validator fires on
        # the final attempt is arbitrary: a malformed source term (e.g. the
        # garbled "NKCC2 gene sigle angl. pour") flips between term-missing and
        # forbidden-char (a genetics "c.1964G>A" the model won't verbalise)
        # failures across retries, so EVERY cause is skip-not-fatal, not just
        # term-missing / wrong <t> count. A systemic problem shows up as a high
        # skipped count in the run stats, not as one aborted run. CountMismatchError
        # (the internal target/source length invariant) is not an LLMError, never
        # reaches here, and stays fatal.
        cause = exc.__cause__
        _REASON_BY_CAUSE = {
            TermMissingError: ("term_missing_after_retries", "persistent term-missing"),
            BlockCountError: ("block_count_after_retries", "persistent wrong <t> count"),
        }
        reason_code, reason_text = next(
            (v for cls, v in _REASON_BY_CAUSE.items() if isinstance(cause, cls)),
            ("validation_after_retries", "persistent soft-validation failure"),
        )
        skip_row = {
            "term": term_str,
            "term_index": term_index,
            "reason": reason_code,
            "detail": str(cause),
        }
        if seen_lock is not None:
            with seen_lock:
                append_jsonl_row(skip_path, skip_row)
        else:
            append_jsonl_row(skip_path, skip_row)
        raise TermSkipped(term_index, term_str, reason=reason_text) from exc

    # Deterministic SOURCE pass for local voxtral-tts: NO LLM. The written
    # label (asr_training_target) already spells units out, so voxtral_normalize
    # only applies the small FIX set voxtral demonstrably needs (Roman numerals
    # after a staging/anatomy word, ARNm). See voxtral_normalize.py and
    # 01_dictionnary/VOXTRAL_QUIRKS.md. Because source is derived 1:1 from each
    # target line (through the stage's optional deterministic source_transform,
    # e.g. acronym -> quoted pronunciation), alignment is exact by construction
    # and drift is bounded to those known transforms.
    asr_training_source_texts: list[str] = []
    for target in asr_training_target_texts:
        spoken = (
            adapter.source_transform(target, term_entry)
            if adapter.source_transform is not None
            else target
        )
        source, residuals = normalize_and_flag(spoken)
        if residuals:
            # A unit symbol survived: an uncovered unit the TARGET LLM failed to
            # spell out. Fail-open (ship the normalized text) but log it to the
            # review queue so bad audio never ships silently.
            review_entry = dict(
                term=term_str,
                original=spoken,
                normalized=source,
                residuals=residuals,
            )
            if seen_lock is not None:
                with seen_lock:
                    record_review(review_path, **review_entry)
            else:
                record_review(review_path, **review_entry)
        asr_training_source_texts.append(source)

    if len(asr_training_target_texts) != len(asr_training_source_texts):
        raise CountMismatchError(
            f"asr_training_target/asr_training_source length mismatch for term "
            f"{term_entry.get('index')} ({term_entry.get('term')!r}): "
            f"{len(asr_training_target_texts)} vs {len(asr_training_source_texts)}"
        )

    rows: list[dict] = []
    for i, (target, source) in enumerate(
        zip(asr_training_target_texts, asr_training_source_texts)
    ):
        rows.append(adapter.make_row(term_entry, i, target, source, model))
    return rows


def run(
    adapter: StageAdapter,
    input_path: str | Path,
    output_path: str | Path,
    n_variants: int | str | dict = DEFAULT_N_VARIANTS,
    model: str = "",
    n_jobs: int = 4,
    limit: int | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    skip_terms_filter=None,
    provider: str | None = None,
    review_path: str | Path = "voxtral_review_queue.jsonl",
    skip_path: str | Path = "term_missing_skips.jsonl",
    run_stats_path: str | Path = "run_statistics.jsonl",
    run_log_path: str | Path = "run_statistics.log",
) -> dict:
    """Iterate scored input rows and generate N variants per row via `adapter`.

    Appends to output_path so the run is resumable. Returns a summary dict.
    Source-agnostic: every corpus stage calls this with its own StageAdapter.

    `n_variants`: either an int (same count for every term) or a mapping of
    score ranges to counts, e.g. {"1-3": 2, "4-7": 4, "8-10": 11}. With a
    dict policy, the input must carry a "score" field (see 01_llm_scoring.py);
    terms whose score falls in no range are skipped.

    `skip_terms_filter`: optional callable(term_entry) -> bool. Return True
    to skip a term (e.g. filter by page range, length, etc.).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    review_path = Path(review_path)
    skip_path = Path(skip_path)
    run_stats_path = Path(run_stats_path)
    run_log_path = Path(run_log_path)

    # run_id ties every stats record and log line back to one invocation. Mirror
    # the loguru stream to a per-run log file (append) so the cost / validation
    # lines are stored on disk, not only printed.
    run_id = f"{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_stats_path.parent.mkdir(parents=True, exist_ok=True)
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        run_log_path,
        level="INFO",
        mode="a",
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss} {level: <7} [run "
        + run_id
        + "] {message}",
    )

    def emit_stat(record_type: str, **fields) -> dict:
        """Append one machine-readable record to run_statistics.jsonl.

        Every record is self-identifying (run_id, model, provider) so a change
        between runs is visible line by line. Append-only: never overwrites.
        """
        record = {
            "type": record_type,
            "run_id": run_id,
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "provider": provider,
            **fields,
        }
        append_jsonl_row(run_stats_path, record)
        return record

    policy = parse_n_variants(n_variants)
    if isinstance(policy, int):
        logger.info(f"n-variants policy: fixed {policy} per term")
    else:
        logger.info(f"n-variants policy: per-score ranges {dict(policy)}")
    if provider:
        logger.info(f"openrouter provider pinned to {provider!r} (allow_fallbacks=False)")
    else:
        logger.warning(
            "no openrouter provider pinned: prompt-cache hits will be best-effort only"
        )
    logger.info(f"voxtral review queue (residual units): {review_path}")
    logger.info(f"term-missing skip queue: {skip_path}")

    done_variants, seen_target_hashes, seen_source_hashes = load_existing_state(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()

    pending: list[tuple[dict, int]] = []
    excluded_by_limit: list[tuple[dict, int]] = []
    skipped_no_score_match = 0
    total_terms_in_scope = 0
    total_rows_in_scope = 0
    initial_done_terms = 0
    initial_done_rows = 0
    full_total_terms_in_scope = 0
    full_total_rows_in_scope = 0
    limit_reached = False
    for entry in iter_input(input_path):
        score = entry.get("score")
        # Fail closed on unscored input for the SCORED stages only. A missing or
        # null score used to be silently skipped (resolve_n_variants -> 0), so
        # any term the scorer failed on vanished from the dataset with no
        # warning. Scored corpus stages (dictionary, drugs) require an integer
        # score from 01_llm_scoring.py and abort otherwise, so gaps surface
        # loudly instead of shrinking the dataset. The rewrite stages (PARHAF,
        # PARROT) set adapter.require_score=False: their raw-text input carries
        # no score and uses a fixed N=1 policy, so the score gate is skipped.
        if adapter.require_score:
            if score is None:
                raise ValueError(
                    f"entry index={entry.get('index')} term={entry.get('term')!r} "
                    f"has no score (missing or null). Re-score the input with "
                    f"01_llm_scoring.py (a --resume run picks up unscored rows) "
                    f"before generating."
                )
            if (
                not isinstance(score, int) or isinstance(score, bool)
                or score < 0 or score > 10
            ):
                raise ValueError(
                    f"invalid score {score!r} for entry index={entry.get('index')} "
                    f"term={entry.get('term')!r}: must be int in 0..10"
                )
        idx = entry.get("index")
        n = resolve_n_variants(policy, score)
        if n <= 0:
            skipped_no_score_match += 1
            continue
        if skip_terms_filter is not None and skip_terms_filter(entry):
            continue
        already_done = (
            len(done_variants.get(idx, set())) if idx is not None else 0
        )
        already_done = min(already_done, n)
        full_total_terms_in_scope += 1
        full_total_rows_in_scope += n
        if limit_reached:
            if already_done < n:
                excluded_by_limit.append((entry, n))
            continue
        total_terms_in_scope += 1
        total_rows_in_scope += n
        if already_done >= n:
            initial_done_terms += 1
            initial_done_rows += n
            continue
        initial_done_rows += already_done
        pending.append((entry, n))
        if limit is not None and len(pending) >= limit:
            limit_reached = True

    logger.info(
        f"resumability: {len(done_variants)} terms have prior rows on disk"
    )
    if not isinstance(policy, int):
        logger.info(
            f"skipped (no matching score range or unscored): {skipped_no_score_match}"
        )

    total_rows = sum(n for _, n in pending)
    scope_str = (
        f"scope total: {total_terms_in_scope} terms / {total_rows_in_scope} rows"
    )
    if limit is not None:
        scope_str += (
            f"; full dataset (no --limit): "
            f"{full_total_terms_in_scope} terms / {full_total_rows_in_scope} rows"
        )
    logger.info(
        f"pending: {len(pending)} terms, {total_rows} rows to generate "
        f"({scope_str}, "
        f"already done: {initial_done_terms} terms / {initial_done_rows} rows)"
    )

    tracker = PricingTracker(model)
    logger.info(
        f"pricing: {model} prompt=${tracker.price['prompt']*1e6:.3f}/M "
        f"completion=${tracker.price['completion']*1e6:.3f}/M "
        f"cache_read=${tracker.price['input_cache_read']*1e6:.3f}/M"
    )
    # Pre-plan: tokenize each upcoming target prompt so estimate_remaining()
    # can project tokens-left to generate. There is now only ONE LLM call per
    # term (the source pass is deterministic and costs no tokens).
    sys_target_tt = tracker.count(adapter.system_prompt)
    for entry, n in pending:
        target_user_tt = tracker.count(adapter.build_user_prompt(entry, n))
        tracker.add_planned(sys_target_tt + target_user_tt)
    for entry, n in excluded_by_limit:
        target_user_tt = tracker.count(adapter.build_user_prompt(entry, n))
        tracker.add_planned(sys_target_tt + target_user_tt, full_only=True)
    plan_str = (
        f"pricing pre-plan: {tracker.planned_calls} calls, "
        f"{tracker.planned_prompt_tiktoken} tiktoken prompt tokens, "
        f"projected=${tracker.projected_total():.4f}"
    )
    if tracker.is_limited:
        plan_str += (
            f"; full dataset (no --limit): {tracker.full_planned_calls} calls, "
            f"{tracker.full_planned_prompt_tiktoken} tiktoken prompt tokens, "
            f"projected=${tracker.projected_total_full():.4f}"
        )
    logger.info(plan_str)

    # Snapshot every parameter this run used (sampling knobs, paths, the policy,
    # a prompt fingerprint, pricing and the plan) so run_statistics.jsonl alone
    # is enough to reconstruct or diagnose the run.
    config = {
        "stage": adapter.stage,
        "temperature": adapter.temperature,
        "top_p": adapter.top_p,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "reasoning_first_attempt": adapter.reasoning_first,
        "reasoning_retry": adapter.reasoning_retry,
        "n_variants_raw": (
            n_variants if isinstance(n_variants, (int, str, dict)) else str(n_variants)
        ),
        "n_variants_policy": (
            policy
            if isinstance(policy, int)
            else {f"{lo}-{hi}": n for (lo, hi), n in policy.items()}
        ),
        "n_jobs": n_jobs,
        "limit": limit,
        "timeout_s": timeout_s,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "review_path": str(review_path),
        "skip_path": str(skip_path),
        "system_prompt_path": str(adapter.system_prompt_path),
        "run_stats_path": str(run_stats_path),
        "run_log_path": str(run_log_path),
        "system_prompt_sha256_12": _text_hash(adapter.system_prompt)[:12],
        "pricing_usd_per_million": {
            k: round(v * 1e6, 4) for k, v in tracker.price.items()
        },
        "pending_terms": len(pending),
        "pending_rows": total_rows,
        "resume_prior_terms": len(done_variants),
        "planned_calls": tracker.planned_calls,
        "planned_prompt_tiktoken": tracker.planned_prompt_tiktoken,
        "projected_total_usd": round(tracker.projected_total(), 6),
    }
    logger.info(f"run params: {json.dumps(config, ensure_ascii=False, default=str)}")
    emit_stat("run_start", **config)

    counters = {"ok": 0, "failed": 0, "rows": 0, "dup": 0, "skipped": 0}
    logger.info(
        f"dup-check baseline: {len(seen_target_hashes)} unique asr_training_target / "
        f"{len(seen_source_hashes)} unique asr_training_source strings already on disk"
    )

    def worker(
        entry: dict, n: int
    ) -> tuple[dict, list[dict] | None, str | None, bool]:
        try:
            rows = generate_variants_for_term(
                entry,
                adapter,
                n_variants=n,
                model=model,
                timeout_s=timeout_s,
                seen_target_hashes=seen_target_hashes,
                seen_source_hashes=seen_source_hashes,
                seen_lock=write_lock,
                provider=provider,
                tracker=tracker,
                review_path=review_path,
                skip_path=skip_path,
            )
            return entry, rows, None, False
        except TermSkipped as exc:
            # One term skipped after a persistent soft-validation failure (term
            # missing, wrong <t> count, or a forbidden char the model would not
            # verbalise), already logged to the skip queue. Not fatal: the run
            # continues.
            logger.warning(str(exc))
            return entry, None, None, True
        except (CountMismatchError, ValidationError):
            # Defensive backstop: generate_variants_for_term now converts every
            # per-term ValidationError into a TermSkipped, so only CountMismatchError
            # (a real internal invariant violation) should reach here and abort.
            raise
        except Exception:  # noqa: BLE001
            return entry, None, traceback.format_exc(), False

    _COST_LOG_EVERY = 10
    processed = 0

    with output_path.open("a", encoding="utf-8") as out_fh:
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            futures = [pool.submit(worker, e, n) for e, n in pending]
            terms_pbar = tqdm(
                total=total_terms_in_scope,
                initial=initial_done_terms,
                desc="terms",
                unit="term",
                position=0,
            )
            rows_pbar = tqdm(
                total=total_rows_in_scope,
                initial=initial_done_rows,
                desc="rows ",
                unit="row",
                position=1,
            )
            try:
                for fut in as_completed(futures):
                    entry, rows, err, skipped = fut.result()
                    if skipped:
                        counters["skipped"] += 1
                        terms_pbar.update(1)
                        continue
                    if err is not None:
                        counters["failed"] += 1
                        logger.error(
                            f"term {entry.get('index')} ({entry.get('term')!r}) failed: {err}"
                        )
                        terms_pbar.update(1)
                        continue
                    assert rows is not None
                    already = done_variants.get(entry.get("index"), set())
                    fresh_rows = [r for r in rows if r["variant_index"] not in already]
                    with write_lock:
                        for row in fresh_rows:
                            for k, bucket in (
                                ("asr_training_target", seen_target_hashes),
                                ("asr_training_source", seen_source_hashes),
                            ):
                                h = _text_hash(row[k])
                                if h in bucket:
                                    counters["dup"] += 1
                                    logger.warning(
                                        f"duplicate {k}: term "
                                        f"{row['term_index']} ({row['term']!r}) "
                                        f"variant {row['variant_index']} hashes to "
                                        f"an existing {k} string on disk: "
                                        f"{row[k]!r}"
                                    )
                                bucket.add(h)
                            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out_fh.flush()
                    counters["ok"] += 1
                    counters["rows"] += len(fresh_rows)
                    processed += 1
                    terms_pbar.update(1)
                    rows_pbar.update(len(fresh_rows))
                    terms_pbar.set_postfix(tracker.tqdm_postfix(), refresh=False)
                    if processed % _COST_LOG_EVERY == 0:
                        logger.info(tracker.summary_line())
                        emit_stat(
                            "progress",
                            processed=processed,
                            counters=dict(counters),
                            **tracker.stats_dict(),
                        )
            except (CountMismatchError, ValidationError) as exc:
                logger.error(
                    f"{type(exc).__name__}: cancelling pending tasks and aborting run"
                )
                emit_stat(
                    "run_error",
                    error_type=type(exc).__name__,
                    error=str(exc),
                    counters=dict(counters),
                    **tracker.stats_dict(),
                )
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            finally:
                rows_pbar.close()
                terms_pbar.close()

    logger.info(f"done: {counters}")
    logger.info(f"final {tracker.summary_line()}")
    emit_stat("run_end", counters=dict(counters), **tracker.stats_dict())
    return counters
