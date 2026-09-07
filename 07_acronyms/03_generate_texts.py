# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "click",
#   "loguru",
#   "litellm",
#   "tiktoken",
#   "tqdm",
#   "tenacity",
#   "rapidfuzz",
# ]
# ///

"""Generate ASR training text pairs for medical acronyms, via the shared engine.

This is the acronyms adapter for `utils/text_generation_engine.py` (the same
engine the dictionary and drugs stages use). Input is the manually filtered
Wikipedia acronym CSV (`TERM,MEANING,PRONOUNCED_AS`):

* `TERM` is the written acronym exactly as the ASR label must carry it (ESAT,
  aVf, gamma-GT). It must be Parakeet-tokenizable; a startup gate aborts on any
  term the tokenizer would map to `<unk>`.
* `MEANING` is the expansion, always handed to the LLM as the `Definition:`
  hint; exactly one variant per call must gloss it in French (validated).
* `PRONOUNCED_AS` is how the local TTS engine should SPEAK the acronym, quoted
  to the LLM as `Pronounced as: "..."` so it picks articles and elision to
  match. A `;` separates several valid pronunciations, each generating its own
  N texts. When empty, a default is derived per hyphen segment: lowercase word
  segments stay whole, everything else splits into dash-joined characters with
  digit runs kept together ("AAA" -> "A-A-A", "5-FU" -> "5-F-U",
  "ST-plus" -> "S-T-plus", "ADAMTS-13" -> "A-D-A-M-T-S-13", "aVf" -> "a-V-f").

The CSV is first EXPANDED to one JSONL entry per (term, pronunciation) pair
with a stable `index` (stage 05 derives audio file names from it and the
engine resumes by it). On re-runs the stored expansion is compared against a
fresh one and the run aborts on drift, so editing the CSV can never silently
re-key already-generated rows.

Target/source split: the LLM writes ONLY `asr_training_target` variants, each
containing the acronym verbatim (exact case, validated with the same
word-boundary matcher the source pass uses). `asr_training_source` is derived
deterministically: the acronym is substituted with the pronunciation
(`StageAdapter.source_transform`), then `voxtral_normalize` applies the usual
TTS fixes. No second LLM call, so the pair can only differ in the known
transforms. A round-trip guard re-substitutes the pronunciation back with the
acronym in the final source and fuzzy-compares against the voxtral-normalized
target; misses go to `roundtrip_review_queue.jsonl` fail-open.

Prompt-cache posture (DeepSeek block prefix caching): one identical system
prompt (shared base + this stage's addendum) across every call, the OpenRouter
provider pinned, and the per-call user message ordered Term + Definition first
with the `Pronounced as:` line after, so the calls of a multi-pronunciation
term share their user-message prefix too.

Output rows feed `05_generate_audio/01_generate_audio.py` (which reads
`term_index` / `variant_index` / `term` / `asr_training_source`) and then the
stage 06 QC loop.

Created with assistance from Claude Code.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import threading
from pathlib import Path

import click
from loguru import logger
from rapidfuzz import fuzz

# Shared pipeline + engine live in utils/ so every corpus stage calls one
# implementation instead of a private copy. See utils/text_generation_engine.py
# and utils/_pipeline_shared.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from _pipeline_shared import (  # noqa: E402
    LLMError,
    TermMissingError,
    _content_words,
    _strip_accents,
    call_llm,
    sanitize_definition,
    validate_asr_training_target,
)
from parakeet_tokenizer import ParakeetTokenizer  # noqa: E402
from text_generation_engine import (  # noqa: E402
    StageAdapter,
    build_target_user_prompt,
    run as engine_run,
)
# Same base-prompt + per-stage-addendum composition as the rewrite stages.
from text_rewrite import load_system_prompt  # noqa: E402
from voxtral_normalize import append_jsonl_row, normalize_and_flag  # noqa: E402


HERE = Path(__file__).resolve().parent
_PROMPT_DIR = HERE.parent / "01_dictionnary"

# System prompt = the shared base target prompt (unchanged, same cache key
# family as the other stages) + the acronyms addendum next to this script.
DEFAULT_BASE_PROMPT = _PROMPT_DIR / "PROMPT_GENERATE_ASR_TRAINING_TARGET.md"
DEFAULT_ADDENDUM = HERE / "PROMPT_GENERATE_ASR_TRAINING_TARGET_ACRONYMS.md"

DEFAULT_INPUT = HERE / "wikipedia_acronyms.filtered.authorfiltered.csv"
# The (term, pronunciation)-expanded engine input derived from the CSV; kept on
# disk so the stable per-entry `index` is inspectable and drift-checkable.
DEFAULT_EXPANDED = HERE / "wikipedia_acronyms.expanded.jsonl"
DEFAULT_OUTPUT = HERE / "generated_dataset.jsonl"
DEFAULT_REVIEW_QUEUE = HERE / "voxtral_review_queue.jsonl"
DEFAULT_SKIP_QUEUE = HERE / "term_missing_skips.jsonl"
# Round-trip guard queue: rows whose source, with the pronunciation substituted
# back to the acronym, no longer reads like the target (fail-open, see
# _roundtrip_check).
DEFAULT_ROUNDTRIP_QUEUE = HERE / "roundtrip_review_queue.jsonl"
DEFAULT_RUN_STATS = HERE / "run_statistics.jsonl"
DEFAULT_RUN_LOG = HERE / "run_statistics.log"

# Match the dictionary/drugs stages so prompt caching and provider pinning
# behave the same across corpus stages.
DEFAULT_MODEL = "openrouter/deepseek/deepseek-v4-pro"
DEFAULT_PROVIDER = "deepseek"
# Texts per (acronym, pronunciation) pair; a multi-pronunciation acronym gets
# N texts PER pronunciation (one engine entry each).
DEFAULT_N_TEXTS = 3
DEFAULT_N_JOBS = 4
DEFAULT_TIMEOUT_S = 300
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95

# Reasoning escalation, matching the other scored stages: cheap non-think
# first, provider-default thinking only on a validation retry.
DEFAULT_REASONING = {"enabled": False}
RETRY_REASONING = None

_MAX_DEFINITION_CHARS = 600

EXPECTED_HEADER = ["TERM", "MEANING", "PRONOUNCED_AS"]

# Definition-presence thresholds (accent-stripped rapidfuzz scores): a variant
# carries the gloss when the full meaning (or a parenthetical chunk of it)
# partial-matches, or, fallback, when at least half of its distinctive words do
# individually (English expansions glossed in French keep cognate words close
# even when the whole span diverges).
_DEF_PRESENCE_MIN = 75
_DEF_WORD_MIN = 80
# Round-trip acceptance: source with pronunciation->acronym undone vs the
# voxtral-normalized target. Below this the row is queued for human review.
_ROUNDTRIP_MIN_RATIO = 90


# ---------------------------------------------------------------------------
# CSV -> expanded engine input
# ---------------------------------------------------------------------------


def default_pronunciation(term: str) -> str:
    """Derive the spoken form of an acronym with no PRONOUNCED_AS.

    Per hyphen segment: a lowercase alphabetic word of >= 2 letters is kept
    whole (it is already speakable: "plus", "gamma"); anything else splits into
    single characters with digit runs kept together. Segments and characters
    rejoin with dashes: "AAA" -> "A-A-A", "5-FU" -> "5-F-U",
    "gamma-GT" -> "gamma-G-T", "ST-plus" -> "S-T-plus", "aVf" -> "a-V-f",
    "ADAMTS-13" -> "A-D-A-M-T-S-13".
    """
    parts: list[str] = []
    for seg in term.split("-"):
        if not seg:
            continue
        if len(seg) >= 2 and seg.isalpha() and seg.islower():
            parts.append(seg)
        else:
            parts.extend(re.findall(r"\d+|.", seg))
    return "-".join(parts)


def expand_csv(csv_path: Path) -> list[dict]:
    """Parse + validate the acronym CSV and expand to one entry per (term, pron).

    Each entry carries a stable sequential `index` (the engine's resume key and
    stage 05's filename key), the acronym, its MEANING as `definition` (required
    non-empty: one variant per call must gloss it), one pronunciation, and its
    position among the term's pronunciations. All structural problems are
    collected and reported together so the CSV can be fixed in one pass.
    """
    problems: list[str] = []
    entries: list[dict] = []
    seen_terms: set[str] = set()
    with Path(csv_path).open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = [h.strip() for h in next(reader, [])]
        if header != EXPECTED_HEADER:
            raise click.ClickException(
                f"{csv_path}: expected header {EXPECTED_HEADER}, got {header}"
            )
        for line_no, row in enumerate(reader, start=2):
            if not row or not any(c.strip() for c in row):
                continue
            if len(row) != 3:
                problems.append(f"line {line_no}: {len(row)} columns, expected 3")
                continue
            term, meaning, pron_field = (c.strip() for c in row)
            if not term:
                problems.append(f"line {line_no}: empty TERM")
                continue
            if not meaning:
                problems.append(
                    f"line {line_no} ({term!r}): empty MEANING (the definition "
                    f"is always sent to the LLM and one variant must gloss it)"
                )
            if term in seen_terms:
                problems.append(f"line {line_no}: duplicate TERM {term!r}")
            elif term.casefold() in {t.casefold() for t in seen_terms}:
                # Legitimate when case IS the discriminator (AVF the algie
                # vasculaire vs aVf the ECG lead); presence validation and the
                # source substitution are exact-case, so they cannot cross.
                logger.warning(
                    f"line {line_no}: TERM {term!r} collides with another term "
                    f"up to case; keeping both (case-sensitive labels)"
                )
            seen_terms.add(term)
            prons: list[str] = []
            for p in pron_field.split(";"):
                p = p.strip()
                if p and p not in prons:
                    prons.append(p)
            if not prons:
                prons = [default_pronunciation(term)]
            for pron_index, pron in enumerate(prons):
                entries.append(
                    {
                        "index": len(entries),
                        "term": term,
                        "definition": meaning,
                        "pronunciation": pron,
                        "pron_index": pron_index,
                        "n_prons": len(prons),
                        "category": "acronyms",
                    }
                )
    if problems:
        raise click.ClickException(
            f"{csv_path}: {len(problems)} problem(s):\n  " + "\n  ".join(problems)
        )
    return entries


_EXPANSION_KEYS = (
    "index", "term", "definition", "pronunciation", "pron_index", "n_prons",
    "category",
)


def ensure_expanded(csv_path: Path, expanded_path: Path) -> list[dict]:
    """Write the expanded engine input, or verify the stored one still matches.

    The engine resumes by `index` and stage 05 names audio files by it, so a
    CSV edit that shifts indices would silently re-key already-generated rows.
    A fresh expansion is therefore compared against the stored file and any
    drift aborts the run with instructions, instead of proceeding.
    """
    fresh = expand_csv(csv_path)
    expanded_path = Path(expanded_path)
    if expanded_path.exists():
        with expanded_path.open(encoding="utf-8") as fh:
            existing = [json.loads(line) for line in fh if line.strip()]
        strip = lambda rows: [{k: r.get(k) for k in _EXPANSION_KEYS} for r in rows]  # noqa: E731
        if strip(existing) != strip(fresh):
            raise click.ClickException(
                f"{expanded_path.name} no longer matches the expansion of "
                f"{csv_path.name}. Entry indices key the engine resume state and "
                f"the stage 05 audio file names, so continuing would mismatch "
                f"already-generated rows. If the CSV change is intentional, "
                f"delete {expanded_path.name} (and review/remove the already "
                f"generated outputs) then re-run."
            )
        logger.info(f"expanded input verified: {len(existing)} entries (unchanged)")
        return existing
    with expanded_path.open("w", encoding="utf-8") as fh:
        for e in fresh:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    logger.info(f"expanded {csv_path.name} -> {expanded_path.name}: {len(fresh)} entries")
    return fresh


def gate_terms_tokenizable(entries: list[dict]) -> None:
    """Abort before any LLM spend if a TERM would tokenize to <unk>.

    The per-variant whitelist backstop in validate_asr_training_target would
    reject every batch for such a term anyway (the acronym must appear verbatim),
    so failing here, listing every offender at once, is strictly cheaper.
    """
    tok = ParakeetTokenizer()
    bad: dict[str, str] = {}
    for e in entries:
        term = e.get("term", "")
        if term in bad:
            continue
        offending = tok.offending_chars(term)
        if offending:
            bad[term] = "".join(sorted(offending))
    if bad:
        listing = ", ".join(f"{t!r} ({chars!r})" for t, chars in bad.items())
        raise click.ClickException(
            f"{len(bad)} term(s) contain characters the Parakeet tokenizer maps "
            f"to <unk>; fix the CSV first: {listing}"
        )


# ---------------------------------------------------------------------------
# Validation + deterministic source substitution
# ---------------------------------------------------------------------------


def _occurrence_re(text: str) -> re.Pattern:
    """Exact-case matcher used BOTH to validate the acronym's presence in a
    target and to substitute it in the source pass, so any validated target is
    guaranteed to substitute. Letters/digits must not touch either side (an
    acronym inside a longer code does not count); hyphens and apostrophes may
    ("post-AVC", "l'ESAT" are real occurrences, and hyphen-adjacent
    substitution still reads fine for the TTS)."""
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(text)}(?![A-Za-z0-9])")


def substitute_pronunciation(target: str, entry: dict) -> str:
    """The stage's `source_transform`: swap the written acronym for its quoted
    pronunciation before voxtral normalization derives the TTS text."""
    rx = _occurrence_re(entry.get("term", ""))
    pron = entry.get("pronunciation", "")
    out, n_subs = rx.subn(lambda _m: pron, target)
    if n_subs == 0:
        # Unreachable for validated targets (acronym_validate uses the same
        # regex); kept as a loud trace if the two ever diverge.
        logger.warning(
            f"source substitution found no occurrence of {entry.get('term')!r} "
            f"in validated target: {target!r}"
        )
    return out


def _definition_candidates(raw: str) -> list[str]:
    """The acceptable gloss anchors for a MEANING: the full sanitized text plus
    every parenthetical chunk (English expansions usually carry their French
    gloss in parens; a leading "ou"/"or" is dropped)."""
    cands: list[str] = []
    full = sanitize_definition(raw.strip())
    if full:
        cands.append(full)
    for chunk in re.findall(r"\(([^()]*)\)", raw):
        c = re.sub(r"^\s*(ou|or)\s+", "", chunk.strip(), flags=re.IGNORECASE)
        c = sanitize_definition(c)
        if len(c) >= 4 and c not in cands:
            cands.append(c)
    return cands


def _check_definition_presence(parsed: list[str], entry: dict) -> None:
    """Require >= 1 variant to state the acronym's meaning (fuzzy containment).

    A variant passes when the full meaning or any parenthetical chunk
    partial-matches it (accent-stripped), or, as a fallback for glosses the
    model reworded or translated into French, when at least half of a
    candidate's distinctive words individually match within that one variant
    (medical French/English cognates keep single words close even when the
    span score collapses). Raising LLMError keeps this retryable."""
    cands = _definition_candidates(entry.get("definition") or "")
    if not cands:
        return
    best = 0.0
    for v in parsed:
        v_lc = _strip_accents(v.lower())
        for c in cands:
            c_lc = _strip_accents(c.lower())
            score = fuzz.partial_ratio(c_lc, v_lc)
            best = max(best, score)
            if score >= _DEF_PRESENCE_MIN:
                return
            words = [w for w in _content_words(c_lc) if len(w) >= 4]
            if len(words) >= 2:
                hits = sum(
                    1 for w in words if fuzz.partial_ratio(w, v_lc) >= _DEF_WORD_MIN
                )
                if hits * 2 >= len(words):
                    return
    raise LLMError(
        f"term {entry.get('index')} {entry.get('term')!r}: no variant states the "
        f"meaning (best fuzzy score {best:.0f} < {_DEF_PRESENCE_MIN}); exactly one "
        f"variant must weave in a French gloss of: {cands[0]!r}"
    )


def acronym_validate(parsed: list[str], entry: dict) -> None:
    """Acronym-stage soft validators: shared target checks + verbatim presence
    + the definition-bearing variant.

    Presence is exact-case and word-boundary (via `_occurrence_re`, the same
    regex the source substitution uses) rather than the shared fuzzy check: the
    written form of an acronym IS the training label, so "Avc" or "A.V.C." for
    "AVC" must fail and retry, not warn."""
    validate_asr_training_target(parsed, examples=[])
    term = entry.get("term", "")
    rx = _occurrence_re(term)
    for i, v in enumerate(parsed):
        if not rx.search(v):
            raise TermMissingError(
                f"term {entry.get('index')} {term!r}: the acronym must appear "
                f"verbatim (same case, same hyphens/digits) in every variant; "
                f"not found in variant {i}: {v!r}"
            )
    _check_definition_presence(parsed, entry)


# ---------------------------------------------------------------------------
# Output rows + round-trip guard
# ---------------------------------------------------------------------------

_roundtrip_lock = threading.Lock()
# Set by run(); module-level so make_row (whose signature the engine fixes)
# can reach it.
_roundtrip_path: Path = DEFAULT_ROUNDTRIP_QUEUE


def _roundtrip_check(entry: dict, variant_index: int, target: str, source: str) -> None:
    """The requested invariant: substituting the pronunciation BACK to the
    acronym in the final TTS source must yield (almost) the voxtral-normalized
    target; only the known deterministic transforms may differ. A miss means
    normalization mangled the pronunciation or the substitution interacted with
    its surroundings, so the row is queued for human review, fail-open (the row
    still ships; one odd pronunciation must not abort a run)."""
    term = entry.get("term", "")
    pron = entry.get("pronunciation", "")
    back = _occurrence_re(pron).sub(lambda _m: term, source)
    expected, _residuals = normalize_and_flag(target)
    ratio = fuzz.ratio(back, expected)
    if ratio >= _ROUNDTRIP_MIN_RATIO:
        return
    logger.warning(
        f"roundtrip: term {entry.get('index')} {term!r} pron {pron!r} variant "
        f"{variant_index} ratio {ratio:.0f} < {_ROUNDTRIP_MIN_RATIO}; queued for review"
    )
    row = {
        "term": term,
        "term_index": entry.get("index"),
        "pronunciation": pron,
        "variant_index": variant_index,
        "asr_training_target": target,
        "asr_training_source": source,
        "back_substituted": back,
        "expected_source_without_pron": expected,
        "ratio": round(float(ratio), 1),
    }
    with _roundtrip_lock:
        append_jsonl_row(_roundtrip_path, row)


def acronym_make_row(
    entry: dict, variant_index: int, target: str, source: str, model: str
) -> dict:
    """Output row for the acronyms stage (core pair fields shared with the other
    stages; stage 05 reads term_index/variant_index/term/asr_training_source)."""
    _roundtrip_check(entry, variant_index, target, source)
    return {
        "category": entry.get("category", "acronyms"),
        "term_index": entry.get("index"),
        "term": entry.get("term"),
        "pronunciation": entry.get("pronunciation"),
        "pron_index": entry.get("pron_index"),
        "n_prons": entry.get("n_prons"),
        "variant_index": variant_index,
        "asr_training_target": target,
        "asr_training_source": source,
        "source_definition": entry.get("definition", ""),
        "model": model,
    }


# ---------------------------------------------------------------------------
# Adapter + entry points
# ---------------------------------------------------------------------------


def build_acronym_user_prompt(entry: dict, n_variants: int) -> str:
    """Per-entry user message: Term + Definition first (shared across a term's
    pronunciations, prompt-cache friendly), then the pronunciation line whose
    semantics the stage addendum defines."""
    definition = sanitize_definition((entry.get("definition") or "").strip())
    pron = entry.get("pronunciation", "")
    return build_target_user_prompt(
        entry.get("term", ""),
        n_variants,
        definition=definition,
        max_definition_chars=_MAX_DEFINITION_CHARS,
        extra_lines=(f'Pronounced as: "{pron}"',),
    )


def build_acronym_adapter(system_prompt: str, system_prompt_path=None) -> StageAdapter:
    """Assemble the acronyms StageAdapter.

    `call_llm` is read from this module's globals when this runs, so it can be
    monkeypatched in tests the same way the other stage wrappers allow.
    `require_score=False`: the CSV carries no usefulness score; every
    (term, pronunciation) entry gets the same fixed N texts.
    """
    return StageAdapter(
        stage="acronyms",
        system_prompt=system_prompt,
        system_prompt_path=system_prompt_path,
        build_user_prompt=build_acronym_user_prompt,
        validate_fn=acronym_validate,
        make_row=acronym_make_row,
        call_llm_fn=call_llm,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        reasoning_first=DEFAULT_REASONING,
        reasoning_retry=RETRY_REASONING,
        require_score=False,
        source_transform=substitute_pronunciation,
    )


def run(
    input_path: str | Path = DEFAULT_INPUT,
    expanded_path: str | Path = DEFAULT_EXPANDED,
    output_path: str | Path = DEFAULT_OUTPUT,
    base_prompt_path: str | Path = DEFAULT_BASE_PROMPT,
    addendum_path: str | Path = DEFAULT_ADDENDUM,
    n_texts: int = DEFAULT_N_TEXTS,
    model: str = DEFAULT_MODEL,
    n_jobs: int = DEFAULT_N_JOBS,
    limit: int | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    provider: str | None = DEFAULT_PROVIDER,
    review_path: str | Path = DEFAULT_REVIEW_QUEUE,
    skip_path: str | Path = DEFAULT_SKIP_QUEUE,
    roundtrip_path: str | Path = DEFAULT_ROUNDTRIP_QUEUE,
    run_stats_path: str | Path = DEFAULT_RUN_STATS,
    run_log_path: str | Path = DEFAULT_RUN_LOG,
    expand_only: bool = False,
) -> dict:
    """Acronyms entry-point: expand + gate the CSV, then run the engine."""
    global _roundtrip_path
    _roundtrip_path = Path(roundtrip_path)
    entries = ensure_expanded(Path(input_path), Path(expanded_path))
    gate_terms_tokenizable(entries)
    if expand_only:
        logger.info("--expand-only: expansion written/verified and terms gated, stopping")
        return {"entries": len(entries), "expand_only": True}
    system_prompt = load_system_prompt(base_prompt_path, addendum_path)
    adapter = build_acronym_adapter(system_prompt, system_prompt_path=base_prompt_path)
    return engine_run(
        adapter,
        input_path=expanded_path,
        output_path=output_path,
        n_variants=int(n_texts),
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


@click.command(context_settings={"show_default": True})
@click.option(
    "--input-path", "-i", type=click.Path(exists=True, path_type=Path),
    default=str(DEFAULT_INPUT),
    help="Acronym CSV (TERM,MEANING,PRONOUNCED_AS; ';' separates pronunciations)",
)
@click.option(
    "--expanded-path", type=click.Path(path_type=Path),
    default=str(DEFAULT_EXPANDED),
    help="Expanded one-entry-per-(term,pronunciation) JSONL fed to the engine",
)
@click.option(
    "--output-path", "-o", type=click.Path(path_type=Path),
    default=str(DEFAULT_OUTPUT), help="JSONL file to append generated rows to",
)
@click.option(
    "--base-prompt-path", type=click.Path(exists=True, path_type=Path),
    default=str(DEFAULT_BASE_PROMPT),
    help="Shared base target system prompt (01_dictionnary/)",
)
@click.option(
    "--addendum-path", type=click.Path(exists=True, path_type=Path),
    default=str(DEFAULT_ADDENDUM),
    help="Acronyms system-prompt addendum appended to the base prompt",
)
@click.option("--model", default=DEFAULT_MODEL, help="LLM model ID (litellm format)")
@click.option(
    "--provider", default=DEFAULT_PROVIDER,
    help="OpenRouter provider slug to pin (empty string to disable pinning)",
)
@click.option(
    "--n-texts", "-n", default=DEFAULT_N_TEXTS, type=int,
    help="Texts per (acronym, pronunciation) pair",
)
@click.option("--n-jobs", "-j", default=DEFAULT_N_JOBS, type=int)
@click.option(
    "--limit", "-L", default=None, type=int,
    help="Only process the first N pending entries (smoke tests)",
)
@click.option("--timeout-s", default=DEFAULT_TIMEOUT_S, type=int)
@click.option(
    "--review-path", type=click.Path(path_type=Path),
    default=str(DEFAULT_REVIEW_QUEUE),
    help="JSONL review queue for asr_training_source residual units",
)
@click.option(
    "--skip-path", type=click.Path(path_type=Path),
    default=str(DEFAULT_SKIP_QUEUE),
    help="JSONL skip queue for entries whose variants kept failing validation",
)
@click.option(
    "--roundtrip-path", type=click.Path(path_type=Path),
    default=str(DEFAULT_ROUNDTRIP_QUEUE),
    help="JSONL review queue for failed pronunciation round-trip checks",
)
@click.option(
    "--expand-only", is_flag=True, default=False,
    help="Only write/verify the expanded JSONL and run the tokenizer gate, no LLM",
)
@click.option("-v", "--verbose", count=True, help="-v for DEBUG logging")
def main(
    input_path: Path,
    expanded_path: Path,
    output_path: Path,
    base_prompt_path: Path,
    addendum_path: Path,
    model: str,
    provider: str,
    n_texts: int,
    n_jobs: int,
    limit: int | None,
    timeout_s: int,
    review_path: Path,
    skip_path: Path,
    roundtrip_path: Path,
    expand_only: bool,
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
        expanded_path=expanded_path,
        output_path=output_path,
        base_prompt_path=base_prompt_path,
        addendum_path=addendum_path,
        model=model,
        provider=provider or None,
        n_texts=n_texts,
        n_jobs=n_jobs,
        limit=limit,
        timeout_s=timeout_s,
        review_path=review_path,
        skip_path=skip_path,
        roundtrip_path=roundtrip_path,
        expand_only=expand_only,
    )


if __name__ == "__main__":
    main()
