"""Shared pipeline infrastructure for the ASR text-pair generators.

Hosts code reused by the corpus generators (currently
`01_dictionnary/03_generate_texts.py` and `02_drugs/01_generate_drug_texts.py`; the
PARHAF/PARROT stages will import the same helpers as they are wired up):

  * error classes (LLMError, CountMismatchError, BlockCountError, TermMissingError, ValidationError)
  * the `<t>` target parser and `validate_asr_training_target`
  * `check_term_in_variants` and its English-shorthand expansion table
  * the retry-with-validation loop and per-attempt feedback context
  * litellm wrapper `call_llm` with prompt-cache markers and finish-reason guard
  * OpenRouter helpers (`_fetch_openrouter_endpoints`,
    `_format_endpoints_table`, `_log_available_endpoints`)
  * `PricingTracker` for live cost projection

The `asr_training_source` (TTS) text is derived deterministically from each
target line by `voxtral_normalize.py`, so this module no longer hosts an LLM
source pass (its parser, validators, drift/alignment checks and prompt builder
were removed with that switch).

Per-corpus parts (definition/examples handling for the dictionary;
substance-context handling for drugs) stay in their respective callers.

The module is consumed via the dictionary and drug scripts adding the
utils/ directory to sys.path and importing names from here, e.g.

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
    from _pipeline_shared import call_llm, PricingTracker, ...

Created with assistance from Claude Code.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

import litellm
import tiktoken
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    Timeout as APITimeoutError,
)
from loguru import logger
from rapidfuzz import fuzz
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


# ---------------------------------------------------------------------------
# Defaults reused by callers (kept here so the dictionary and drug scripts
# pick the same timeout / retry budget unless they explicitly override).
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUT_S = 300
# Completion cap per call. Generous on purpose: you only pay for tokens actually
# generated, so a high cap just bounds a runaway rather than costing money. Named
# here (rather than a literal in call_llm) so a run can record the exact value.
DEFAULT_MAX_TOKENS = 16000


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMError(RuntimeError):
    """Raised when the LLM returns an error or unparseable output."""


class CountMismatchError(RuntimeError):
    """Fatal: an internal target/source length invariant was violated.

    Reserved for the code-bug case where the deterministic source pass did
    not yield exactly one asr_training_source per asr_training_target (the
    loop is 1:1, so this is unreachable short of a bug). It stays fatal
    because a length mismatch means the row's pair-by-index alignment is
    meaningless and shipping it would quietly corrupt the dataset.

    A wrong `<t>` block count FROM THE LLM is a different, recoverable case:
    see BlockCountError, which is retried with feedback and, if it persists,
    skips the single term instead of aborting the run.
    """


class TermMissingError(LLMError):
    """A generated variant does not contain the input term.

    Subclass of LLMError, so _retry_with_validation retries it with
    feedback (the model is told which variant dropped the term); a single
    over-paraphrased variant usually fixes itself on the next attempt.
    Only if it persists past _VALIDATION_MAX_ATTEMPTS does it surface, as a
    ValidationError whose __cause__ is this error. The dictionary generator
    then skips that one term (logging it) instead of aborting the run. See
    01_dictionnary/03_generate_texts.py (TermSkipped).
    """


class BlockCountError(LLMError):
    """The LLM returned the wrong number of `<t>` blocks (or an empty one).

    Subclass of LLMError, so _retry_with_validation retries it with feedback
    (the model is told how many blocks it emitted versus how many were
    expected); a transient over- or under-count usually self-corrects on the
    next attempt. Only if it persists past _VALIDATION_MAX_ATTEMPTS does it
    surface, as a ValidationError whose __cause__ is this error, and the
    dictionary generator then skips that one term (logging it) instead of
    aborting the run. Contrast CountMismatchError, which is a fatal internal
    invariant rather than a recoverable LLM-output slip.
    """


class ValidationError(RuntimeError):
    """Fatal: a soft validator (banned chars, units, drift, dup, ...) kept
    failing after the configured number of retries.

    Raised by _retry_with_validation on exhaustion with the last LLMError
    as __cause__. A persistent term-missing (TermMissingError, itself an
    LLMError) surfaces this way too; the dictionary generator special-cases
    that __cause__ to skip the single term rather than abort. Every other
    ValidationError fast-fails the run at the worker / executor level, on
    the expectation that persistent failure means the prompt or model is
    misbehaving and continuing would silently corrupt the dataset.
    """


# ---------------------------------------------------------------------------
# Tiny text utilities
# ---------------------------------------------------------------------------


def _strip_accents(text: str) -> str:
    """Drop combining diacritics so accented and bare forms compare equal.

    Used by the fuzzy term-presence check so that the French rendering the
    model writes ("gène ABCG5", "sténose") matches an unaccented term
    ("gene ABCG5", "stenose") without a spurious partial_ratio penalty.
    """
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def _output_preview(text: str, limit: int = 500) -> str:
    """Compact preview of LLM output for warning messages."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [{len(text) - limit} more chars]"


def _preview_head_tail(text: str, limit: int) -> str:
    """Head+tail truncation: keep the first and last half of `limit` chars.

    Used when feeding a failed generation back to the model. A validation
    error can sit at either end (a missing opening block, a truncated closing
    one), so unlike `_output_preview` (head-only) this keeps both ends and
    elides the middle, capping how much a very long previous answer can inflate
    the next request.
    """
    if len(text) <= limit:
        return text
    half = limit // 2
    return (
        f"{text[:half]}\n... [{len(text) - limit} chars elided] ...\n{text[-half:]}"
    )


# Reasoning models can surface a chain-of-thought block inline. DeepSeek hides
# it, but a provider or model swap could expose it, and the `<t>` parser would
# then wrongly extract any example variant written mid-thought. Strip it first.
_REASONING_BLOCK_RE = re.compile(
    r"<(think|thinking|reasoning)\b[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_REASONING_OPEN_RE = re.compile(
    r"<(?:think|thinking|reasoning)\b[^>]*>", re.IGNORECASE
)


def _strip_reasoning(text: str) -> str:
    """Remove <think>/<thinking>/<reasoning> blocks from raw model output.

    Closed blocks are deleted. An unclosed trailing open tag (a reasoning
    stream cut off at max_tokens) drops everything after it, since no valid
    `<t>` output can follow a reasoning block that never closed. Defensive: on
    the current DeepSeek path the model hides reasoning and this is a no-op.
    """
    if not text:
        return text
    text = _REASONING_BLOCK_RE.sub("", text)
    m = _REASONING_OPEN_RE.search(text)
    if m:
        text = text[: m.start()]
    return text


def _text_hash(text: str) -> str:
    """sha256 of the stripped text, used for duplicate detection."""
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Parakeet tokenizability: deterministic folds + the authoritative <unk> gate
# ---------------------------------------------------------------------------

# parakeet_tokenizer.py + parakeet_vocab.txt live in utils/ next to this module,
# so its own directory is the right import path regardless of the caller's CWD
# (the dictionary / drug scripts run from subfolders).
_UTILS_DIR = Path(__file__).resolve().parent
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))
from parakeet_tokenizer import ParakeetTokenizer, describe  # noqa: E402

# One shared coverage detector (loads the vocab once). nfkc=True mirrors what
# Parakeet's SentencePiece tokenizer does at train time, so offending_chars()
# answers exactly "would this character land on <unk> when training?".
_PARAKEET_TOK = ParakeetTokenizer(nfkc=True)

# Typography-only glyphs we fold to their covered equivalent at parse time, so
# the stored text is clean and the model does not burn a validation retry (or
# abort the whole run) over a pure typography reflex (a non-breaking space, a
# "…", a superscript "²", a curly apostrophe): there is nothing semantic for the
# LLM to "fix". Most entries are glyphs NFKC (which the tokenizer applies)
# already folds to a covered character. The curly single quotes are instead a
# lossless straight-apostrophe normalization: NFKC leaves U+2019 alone, so an
# un-folded "l'os" written with the typographic quote would hit the forbidden-
# char / <unk> gate on every retry even though "'" is the correct written form.
# This is the same fold 01_dictionnary/02_token_check.py applies to input terms.
# Unicode space separators are handled generically in _normalize_tokenizable_text.
# Meaning-bearing symbols (+, <, >, =, ...) are deliberately absent here:
# silently replacing them would corrupt clinical meaning, so they are left for
# the forbidden-char / tokenizer gate to reject and re-prompt.
_TOKENIZABLE_FOLDS = {
    "…": "...",  # horizontal ellipsis -> three periods
    "¹": "1",    # superscript one
    "²": "2",    # superscript two ("mètre carré")
    "³": "3",    # superscript three
    "’": "'",    # right single quote U+2019 -> straight apostrophe (French elision)
    "‘": "'",    # left single quote U+2018 -> straight apostrophe
    # Typographic hyphen / minus variants that are the plain hyphen-minus in
    # meaning but map to <unk> (Parakeet lacks them). Folding is lossless: the
    # written form is the same word ("cortico-resistant"). En dash U+2013 and em
    # dash U+2014 are deliberately NOT here: they stay in _FORBIDDEN_W_CHARS so a
    # real dash aside is re-prompted, not silently turned into a hyphen.
    "‐": "-",    # U+2010 hyphen
    "‑": "-",    # U+2011 non-breaking hyphen
    "−": "-",    # U+2212 minus sign
}


def _normalize_tokenizable_text(text: str) -> str:
    """Fold typography-only glyphs to their covered equivalents.

    Replaces every Unicode space separator (NBSP, narrow NBSP, thin space, ...)
    with a plain ASCII space and applies :data:`_TOKENIZABLE_FOLDS`. Leaves
    every meaning-bearing character untouched.
    """
    out: list[str] = []
    for ch in text:
        if ch in _TOKENIZABLE_FOLDS:
            out.append(_TOKENIZABLE_FOLDS[ch])
        elif ch != " " and unicodedata.category(ch) == "Zs":
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


# ---------------------------------------------------------------------------
# Block parsers
# ---------------------------------------------------------------------------

_ASR_TRAINING_TARGET_RE = re.compile(r"<t>(?P<t>.*?)</t>", re.DOTALL)


def parse_asr_training_target(text: str, expected: int) -> list[str]:
    """Extract `<t>...</t>` blocks. Fatal on count mismatch or empty block."""
    matches = list(_ASR_TRAINING_TARGET_RE.finditer(text))
    if len(matches) != expected:
        logger.error(
            f"parse_asr_training_target: expected {expected} <t> blocks, parsed {len(matches)} "
            f"from {len(text)} chars. Output preview:\n{_output_preview(text)}"
        )
        raise BlockCountError(
            f"expected exactly {expected} <t> blocks, parsed {len(matches)} from "
            f"{len(text)} chars of output. Emit exactly {expected} blocks."
        )
    parsed = [_normalize_tokenizable_text(m.group("t")).strip() for m in matches]
    empty = [i for i, v in enumerate(parsed) if not v]
    if empty:
        raise BlockCountError(
            f"empty <t> block(s) at index {empty} (after stripping). The "
            f"pair-by-index alignment with source would be meaningless."
        )
    return parsed


# ---------------------------------------------------------------------------
# Term-presence check (with English-shorthand expansion support)
# ---------------------------------------------------------------------------

# Fuzzy term-presence thresholds (rapidfuzz partial_ratio, 0..100).
_TERM_PRESENCE_EXACT = 90
_TERM_PRESENCE_CLOSE = 75

_WORD_PRONOUNCED_ACRONYMS = {"SIDA", "RAS", "NASA", "CIM"}
_INITIALISM_RE = re.compile(r"^[A-Z]{2,}[a-z]?$")

# French function words dropped when checking a multi-word term word-by-word:
# they carry no distinctive medical signal and would match almost any text.
_TERM_STOPWORDS = {
    "de", "du", "des", "la", "le", "les", "un", "une", "et", "en",
    "au", "aux", "ou", "d", "l", "a", "sur", "par", "avec", "sans",
}


def _content_words(text: str) -> list[str]:
    """Distinctive words of a multi-word term (stopwords/short tokens dropped).

    Splits on non-letter/digit runs, keeps tokens of length >= 3 that are not
    French function words. Used only as a scattered-match fallback: a long
    descriptive term ("absence de corps calleux, polydactylie postaxiale ...")
    is usually woven into a sentence rather than quoted verbatim, so its words
    end up spread apart and partial_ratio scores the whole span low even though
    every distinctive word is present.
    """
    toks = re.findall(r"[^\W_]+", text, flags=re.UNICODE)
    return [t for t in toks if len(t) >= 3 and t not in _TERM_STOPWORDS]


def _component_present(component: str, haystack_lc: str) -> bool:
    """Is one component of a combination drug term present in the variant?

    Order- and position-independent: a combination like "codeine + paracetamol"
    is spoken as "de la codeine et ensuite un peu de paracetamol", so each
    component is matched on its own anywhere in the sentence. The component
    scores as its own span first (partial_ratio, so inflected forms still hit),
    and if that is inconclusive it falls back to requiring each distinctive word
    of the component to appear. Inputs are already lowercased and accent-stripped.
    """
    component = component.strip()
    if not component:
        return True
    if fuzz.partial_ratio(component, haystack_lc) >= _TERM_PRESENCE_CLOSE:
        return True
    words = _content_words(component)
    if words:
        return min(fuzz.partial_ratio(w, haystack_lc) for w in words) >= _TERM_PRESENCE_CLOSE
    return False


def _maybe_spaced_form(term: str) -> str | None:
    """Return the space-separated form of a letter-by-letter initialism.

    "ADN" -> "A D N", "ARNm" -> "A R N m". Returns None when the term is not
    a letter-by-letter initialism (mixed case, word-pronounced acronym,
    contains digits or hyphens, etc.).
    """
    if not term:
        return None
    if not _INITIALISM_RE.fullmatch(term):
        return None
    if term in _WORD_PRONOUNCED_ACRONYMS:
        return None
    return " ".join(term)


# Mirrors the <input_term_expansions> table in the dictionary prompt.
# Drug scripts get an empty match for these (drug names are not English
# shorthand) so the validator behaves identically.
_ENGLISH_SHORTHAND_EXPANSIONS: dict[str, list[str]] = {
    "mrsa": ["SARM", "staphylocoque doré méthi-résistant"],
    "hfref": ["insuffisance cardiaque à fraction d'éjection altérée"],
    "hfpef": ["insuffisance cardiaque à fraction d'éjection préservée"],
    "ef": ["fraction d'éjection"],
    "nstemi": ["infarctus du myocarde sans sus-décalage ST", "non-STEMI"],
    "stemi": ["infarctus du myocarde avec sus-décalage ST"],
    "bp": ["tension artérielle"],
    "sbp": ["TA systolique", "tension artérielle systolique"],
    "dbp": ["TA diastolique", "tension artérielle diastolique"],
    "egfr": ["DFG estimé"],
    "c. diff": ["Clostridium difficile"],
}


def check_term_in_variants(
    term: str,
    variants: list[str],
    term_index=None,
    extra_candidates: list[str] | None = None,
) -> None:
    """Validate that every variant contains the input term (fuzzily).

    Uses rapidfuzz.fuzz.partial_ratio so inflected forms still score high
    (e.g. "diabétiques" vs "diabétique"). For each variant:
      * score >= 90 -> silent pass.
      * 75 <= score < 90 -> warn (likely inflection or case difference).
      * score < 75 -> raise TermMissingError (retryable via
        _retry_with_validation; only fatal if it persists past every retry).

    English-shorthand terms (HFrEF, MRSA, ...) score against their French
    expansion as well. `extra_candidates` lets callers add domain-specific
    aliases (e.g. for drugs, the active substance name for a brand-keyed
    row); the variant matches if any candidate scores high enough.

    Extra candidates are derived for two shapes the raw needle mishandles:
    a trailing combining-form hyphen ("acantho-" -> stem "acantho", which the
    text completes as "acanthose") and a slash acronym ("AC/A" -> "AC par A",
    the spoken form the model and voxtral_normalize both use).

    Scattered-match fallback: if a multi-word term scores below the close
    threshold as a single span, it still passes when every distinctive word
    (stopwords/short tokens dropped) is individually present, since long
    descriptive terms get woven into a sentence rather than quoted verbatim.

    Combination drugs ("codeine + paracetamol"): a "+"-joined term is treated as
    independent components, each matched on its own anywhere in the variant, so
    word order and the distance between the two names do not matter (the sentence
    can read "de la codeine et ensuite un peu de paracetamol"). Every component
    must be present. Empty term is a silent no-op.
    """
    spec = _term_match_spec(term, extra_candidates)
    if spec is None:
        logger.warning(
            f"term-presence: term {term_index} {term!r} reduces to a degenerate "
            f"needle after blacklist stripping; skipping check"
        )
        return
    cands, fallback_words, combo_components = spec
    for i, v in enumerate(variants):
        v_lc = _strip_accents(v.lower())
        tag, info = _eval_term_in_variant(spec, v_lc)
        if tag in ("exact", "combo_ok"):
            continue
        if tag == "close":
            logger.warning(
                f"term-presence: term {term_index} {term!r} only fuzzy-matched "
                f"variant {i} (score={info:.0f}): {v!r}"
            )
            continue
        if tag == "scattered":
            span_score, worst_word = info
            logger.warning(
                f"term-presence: term {term_index} {term!r} matched variant "
                f"{i} only word-by-word (span score={span_score:.0f}, worst "
                f"word={worst_word:.0f}): {v!r}"
            )
            continue
        if tag == "combo_missing":
            raise TermMissingError(
                f"term {term_index} {term!r}: combination component(s) {info!r} "
                f"not found in variant {i} (order-independent match): {v!r}"
            )
        raise TermMissingError(
            f"term {term_index} {term!r} not found in variant {i} "
            f"(partial_ratio={info:.0f} < {_TERM_PRESENCE_CLOSE}): {v!r}"
        )


def term_present_in_variant(term: str, variant: str) -> bool:
    """Boolean core of :func:`check_term_in_variants` for one (term, variant).

    Returns True when the term is present (exact, close-fuzzy, scattered-word, or
    all combination components matched), False otherwise. A degenerate / empty
    term is treated as *not an anchor* (returns False) so callers can OR several
    candidate anchors together without a blank one matching everything. No
    logging: :func:`check_term_in_variants` wraps the same decision with warnings
    and retryable errors, while the drugs stage uses this for OR-across-anchors
    (raw brand term OR its ``substances`` active ingredients).
    """
    spec = _term_match_spec(term, None)
    if spec is None:
        return False
    tag, _info = _eval_term_in_variant(spec, _strip_accents(variant.lower()))
    return tag in ("exact", "close", "scattered", "combo_ok")


def _eval_term_in_variant(spec, v_lc):
    """Decide how a prepared term ``spec`` matches one accent-stripped variant.

    Single source of the presence decision shared by
    :func:`check_term_in_variants` and :func:`term_present_in_variant`. Returns a
    ``(tag, info)`` pair: ``exact``/``close`` (info=span score), ``scattered``
    (info=(span, worst-word)), ``combo_ok``/``combo_missing`` (info=missing list),
    or ``missing`` (info=span score).
    """
    cands, fallback_words, combo_components = spec
    if combo_components is not None:
        missing = [c for c in combo_components if not _component_present(c, v_lc)]
        return ("combo_ok", missing) if not missing else ("combo_missing", missing)
    score = max(fuzz.partial_ratio(c, v_lc) for c in cands)
    if score >= _TERM_PRESENCE_EXACT:
        return ("exact", score)
    if score >= _TERM_PRESENCE_CLOSE:
        return ("close", score)
    if len(fallback_words) >= 2:
        word_scores = [fuzz.partial_ratio(w, v_lc) for w in fallback_words]
        if min(word_scores) >= _TERM_PRESENCE_CLOSE:
            return ("scattered", (score, min(word_scores)))
    return ("missing", score)


def _term_match_spec(term: str, extra_candidates: list[str] | None):
    """Prepare a term for presence matching: build its candidate spellings.

    Returns ``(cands, fallback_words, combo_components)`` or None when the term is
    empty / reduces to a single character after blacklist stripping (presence is a
    no-op). Extracted so :func:`check_term_in_variants` and
    :func:`term_present_in_variant` share the same candidate-building logic.
    """
    needle = (term or "").strip()
    if not needle:
        return None
    stripped = "".join(c for c in needle if c not in _FORBIDDEN_W_CHARS and c != ".")
    if len(stripped.strip()) <= 1:
        return None
    candidates = [needle.lower()]
    if stripped.lower() != needle.lower():
        candidates.append(stripped.lower())
    # Trailing combining-form hyphen ("acantho-"): the model writes the
    # completed word ("acanthose"), so match the stem, not the morpheme.
    if needle.endswith("-"):
        stem = needle.rstrip("-").strip().lower()
        if len(stem) > 1 and stem not in candidates:
            candidates.append(stem)
    # Slash acronyms ("AC/A"): the written label spells the slash as " par "
    # (the model does this and voxtral_normalize mirrors it), so add that
    # spoken form; the bare "/" is otherwise stripped to a run-on ("ACA").
    if "/" in needle:
        spoken_slash = " ".join(needle.replace("/", " par ").split()).lower()
        if spoken_slash and spoken_slash not in candidates:
            candidates.append(spoken_slash)
    expansions = _ENGLISH_SHORTHAND_EXPANSIONS.get(needle.lower())
    if expansions:
        candidates.extend(e.lower() for e in expansions)
    spaced = _maybe_spaced_form(needle)
    if spaced is not None:
        candidates.append(spaced.lower())
    spaced_stripped = _maybe_spaced_form(stripped)
    if spaced_stripped is not None and spaced_stripped.lower() not in candidates:
        candidates.append(spaced_stripped.lower())
    if extra_candidates:
        for c in extra_candidates:
            c_lc = (c or "").strip().lower()
            if c_lc and c_lc not in candidates:
                candidates.append(c_lc)
    # Compare accent-insensitively: the model writes the accented French form
    # ("gène", "sténose") while the dictionary term may be unaccented, and that
    # difference should not count against presence.
    cands = [_strip_accents(c) for c in candidates]
    # Distinctive words of the (cleaned) term, for the scattered-match fallback.
    fallback_words = _content_words(_strip_accents(stripped.lower()))
    # Combination drugs ("codeine + paracetamol"): the "+" joins independent
    # components. Match each on its own, in any order and position, rather than as
    # one span. Split the raw needle on "+" (it is stripped from `stripped`); keep
    # this path only when at least two real components remain.
    combo_components = None
    if "+" in needle:
        parts = [_strip_accents(p.strip().lower()) for p in needle.split("+")]
        parts = [p for p in parts if p]
        if len(parts) >= 2:
            combo_components = parts
    return cands, fallback_words, combo_components


# ---------------------------------------------------------------------------
# Validator constants (forbidden chars, unit symbols, refusals, thresholds)
# ---------------------------------------------------------------------------

# Banned in every <t> block by PROMPT_GENERATE_ASR_TRAINING_TARGET.md rule 4.
# Two groups: (1) non-spoken / badly-read punctuation (parentheses, dashes,
# guillemets, slash, degree/micro/euro, curly apostrophe U+2019; ASCII U+0027
# is the only allowed apostrophe); (2) math/comparison glyphs that must be
# verbalised ("plus", "supérieur à", "fois", ...) rather than left as a symbol
# the Parakeet tokenizer cannot represent. Group (2) recurs often enough in
# clinical text ("O+", "TA > 140") to ban explicitly; the tokenizer gate in
# validate_asr_training_target() is the catch-all for the rarer long tail.
_FORBIDDEN_W_CHARS = set("()—–;\"«»/°µ€’") | set("+<>=*&#~→×")

# Substitutions applied to a per-item `Definition` hint before it lands in the
# user prompt. Definitions are shown to the model, never emitted verbatim, but
# characters the target validators ban (or that the Parakeet tokenizer maps to
# <unk>, like Greek letters) are mapped to safe spoken equivalents so the model
# is not tempted to mimic them in its <t> blocks (which would trip the
# validators and burn retries). Shared by the dictionary and acronyms stages.
DEFINITION_SUBSTITUTIONS: list[tuple[str, str]] = [
    ("(", ", "),
    (")", ", "),
    ("—", ", "),
    ("–", ", "),
    (";", ","),
    ("«", ""),
    ("»", ""),
    ('"', ""),
    ("/", " par "),
    ("°", " degrés "),
    ("µ", "micro"),
    ("€", " euros"),
    ("’", "'"),
    # Greek letters appear in expansions ("α-1-antitrypsine", "acide
    # γ-aminobutyrique"); Parakeet maps them to <unk>, so spell them out.
    ("α", "alpha"),
    ("β", "bêta"),
    ("γ", "gamma"),
    ("δ", "delta"),
    # A definition cell may carry embedded newlines / tabs (e.g. a multiline
    # CSV field); the prompt format wants a single `Definition:` line.
    ("\n", " "),
    ("\t", " "),
]


def sanitize_definition(text: str) -> str:
    """Replace banned characters with safe equivalents and tidy whitespace."""
    if not text:
        return text
    out = text
    for src, dst in DEFINITION_SUBSTITUTIONS:
        out = out.replace(src, dst)
    while ",," in out or ", ," in out:
        out = out.replace(",,", ",").replace(", ,", ",")
    while "  " in out:
        out = out.replace("  ", " ")
    out = re.sub(r"\s+,", ",", out)
    out = re.sub(r",\s*(?=[.!?:])", "", out)
    return out.strip(" ,")

# Digit-prefixed abbreviated units the prompt requires spelled out. The single
# letters m, s, h, g, L are too common in normal French to flag safely; this
# regex deliberately catches only multi-character symbols.
_UNIT_SYMBOL_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*"
    r"(mg|kg|mL|mmHg|mmol|mEq|mOsm|kDa|ng|mm|cm|UI)\b"
)

_REFUSAL_OPENINGS = (
    "je ne peux pas",
    "je suis désolé",
    "désolé, je",
    "désolée, je",
    "i cannot",
    "i can't",
    "i'm sorry",
    "i am sorry",
    "sorry, i",
    "i apologize",
    "as an ai",
    "en tant qu'ia",
    "en tant qu'assistant",
)

_ENGLISH_STOPWORDS = (
    " the ", " and ", " is ", " of ", " with ", " for ", " that ",
    " this ", " from ", " has ", " was ", " were ", " are ", " have ",
)
_ENGLISH_STOPWORD_LIMIT = 3

# NBSP, tab, or two consecutive spaces; target prompt's <unicode> block bans them.
_BAD_WHITESPACE_RE = re.compile("[\xa0\t]|  ")

# Diversification: consecutive variants must not share near-identical openings.
_DIVERSIFICATION_HEAD = 80
_DIVERSIFICATION_MAX_RATIO = 85
# Pairwise near-clone check on full text.
_NEAR_CLONE_MAX_RATIO = 90
# Reference-example leakage (used only when callers pass examples). Scored on
# the AVERAGE of partial_ratio (containment: is the example present at all?) and
# ratio (length-aware full-string similarity). Averaging encodes coverage: a
# short example merely appearing in a long variant scores low (the term just
# appears, which the term-presence check in fact requires), while a whole
# reference sentence reproduced scores high (most of the variant IS the
# example). At 85 only near-complete reproduction is flagged; ~80 would also
# catch a single reference sentence copied verbatim amid original text.
_EXAMPLE_LEAK_MAX_RATIO = 85

# Per-variant length bounds (chars).
_MIN_VARIANT_LEN = 40
_MAX_VARIANT_LEN = 600

_SENTENCE_END_CHARS = {".", "!", "?", "…"}


# ---------------------------------------------------------------------------
# Validator helpers
# ---------------------------------------------------------------------------


def _check_sentence_complete(variants: list[str], label: str) -> None:
    """Raise LLMError if a variant doesn't end in sentence punctuation."""
    for i, v in enumerate(variants):
        tail = v.rstrip().rstrip('"').rstrip()
        if not tail or tail[-1] not in _SENTENCE_END_CHARS:
            raise LLMError(
                f"{label}: variant {i} does not end in sentence-terminating "
                f"punctuation (looks truncated): {_output_preview(v, 200)!r}"
            )


def _check_length_bounds(variants: list[str], label: str) -> None:
    """Raise LLMError if any variant is outside [_MIN_VARIANT_LEN, _MAX_VARIANT_LEN]."""
    for i, v in enumerate(variants):
        n = len(v)
        if n < _MIN_VARIANT_LEN or n > _MAX_VARIANT_LEN:
            raise LLMError(
                f"{label}: variant {i} length {n} outside bounds "
                f"[{_MIN_VARIANT_LEN}, {_MAX_VARIANT_LEN}]: "
                f"{_output_preview(v, 200)!r}"
            )


def _check_forbidden_chars(
    variants: list[str], banned: set[str], label: str
) -> None:
    """Raise LLMError if any variant contains a banned character."""
    for i, v in enumerate(variants):
        bad = sorted({c for c in v if c in banned})
        if bad:
            raise LLMError(
                f"{label}: variant {i} contains forbidden char(s) {bad}: "
                f"{_output_preview(v, 200)!r}"
            )


def _check_no_cross_term_dup(
    variants: list[str],
    seen_hashes: set[str] | None,
    seen_lock: threading.Lock | None,
    label: str,
) -> None:
    """Raise LLMError if any variant duplicates a string already on disk.

    Caller threads in the run-level per-column hash set (populated from
    the output JSONL at startup and grown at write time). The set must
    be column-scoped: mixing target/source would flag a legitimate
    same-row source==target pair when no source-transformation rule
    applies.
    """
    if seen_hashes is None or seen_lock is None:
        return
    for i, v in enumerate(variants):
        h = _text_hash(v)
        with seen_lock:
            hit = h in seen_hashes
        if hit:
            raise LLMError(
                f"{label}: variant {i} duplicates existing text on disk: "
                f"{_output_preview(v, 200)!r}"
            )


def _check_no_intra_batch_dup(variants: list[str], label: str) -> None:
    """Raise LLMError if two variants in the batch are identical (post-strip)."""
    seen: dict[str, int] = {}
    for i, v in enumerate(variants):
        prev = seen.get(v)
        if prev is not None:
            raise LLMError(
                f"intra-batch {label} duplicate: variant {i} == variant {prev}: "
                f"{_output_preview(v, 200)!r}"
            )
        seen[v] = i


def _check_no_near_clones(variants: list[str], label: str) -> None:
    """Raise LLMError when any two variants are near-identical on full text."""
    lc = [v.lower() for v in variants]
    for i in range(len(lc)):
        for j in range(i + 1, len(lc)):
            ratio = fuzz.ratio(lc[i], lc[j])
            if ratio >= _NEAR_CLONE_MAX_RATIO:
                raise LLMError(
                    f"{label}: variants {i} and {j} are near-clones "
                    f"(full-text ratio={ratio:.0f} >= {_NEAR_CLONE_MAX_RATIO}): "
                    f"{_output_preview(variants[i], 150)!r} vs "
                    f"{_output_preview(variants[j], 150)!r}"
                )


def _check_no_refusal_or_english(variants: list[str], label: str) -> None:
    """Raise LLMError when a variant looks like a refusal or English text."""
    for i, v in enumerate(variants):
        v_lc = v.lower()
        for pat in _REFUSAL_OPENINGS:
            if pat in v_lc:
                raise LLMError(
                    f"{label}: variant {i} looks like a refusal "
                    f"(contains {pat!r}): {_output_preview(v, 200)!r}"
                )
        padded = f" {v_lc} "
        hits = sum(1 for w in _ENGLISH_STOPWORDS if w in padded)
        if hits >= _ENGLISH_STOPWORD_LIMIT:
            raise LLMError(
                f"{label}: variant {i} looks like English "
                f"({hits} stopwords matched, output must be French): "
                f"{_output_preview(v, 200)!r}"
            )


def _check_no_example_leak(variants: list[str], examples: list[str]) -> None:
    """Raise LLMError when a variant reproduces a reference example.

    Scores each (example, variant) pair by the AVERAGE of partial_ratio
    (containment) and ratio (length-aware similarity). This encodes coverage:
    a short example buried in a long variant scores low (the term merely
    appears, which the term-presence check requires), while a whole reference
    sentence reproduced scores high (most of the variant IS the example). It
    replaces a crude example-length cutoff and flags real reuse without firing
    on shared terminology. See _EXAMPLE_LEAK_MAX_RATIO.
    """
    if not examples:
        return
    for i, v in enumerate(variants):
        v_lc = v.lower()
        for j, ex in enumerate(examples):
            ex_lc = ex.lower()
            score = (fuzz.partial_ratio(ex_lc, v_lc) + fuzz.ratio(ex_lc, v_lc)) / 2
            if score >= _EXAMPLE_LEAK_MAX_RATIO:
                raise LLMError(
                    f"asr_training_target: variant {i} reproduces reference "
                    f"example {j} (avg(partial_ratio, ratio)={score:.0f} >= "
                    f"{_EXAMPLE_LEAK_MAX_RATIO}): "
                    f"variant={_output_preview(v, 150)!r} "
                    f"example={_output_preview(examples[j], 150)!r}"
                )


# ---------------------------------------------------------------------------
# Aggregate validators
# ---------------------------------------------------------------------------


def validate_asr_training_target(
    variants: list[str], examples: list[str] | None = None
) -> None:
    """Post-parse soft validators for <t> blocks.

    Raises LLMError on any rule violation so the caller can retry. The
    checks are length bounds, sentence-completeness, intra-batch dups,
    near-clones, refusals/English, example leakage (when `examples` is
    provided), forbidden chars, abbreviated units, disallowed
    whitespace, and consecutive-opening similarity.
    """
    _check_length_bounds(variants, label="asr_training_target")
    _check_sentence_complete(variants, label="asr_training_target")
    _check_no_intra_batch_dup(variants, label="asr_training_target")
    _check_no_near_clones(variants, label="asr_training_target")
    _check_no_refusal_or_english(variants, label="asr_training_target")
    _check_no_example_leak(variants, examples or [])
    _check_forbidden_chars(variants, _FORBIDDEN_W_CHARS, label="asr_training_target")
    # Whitelist backstop. The blacklist above and the prompt cover the common
    # offenders, but only the tokenizer's own coverage set is authoritative on
    # what Parakeet can emit without <unk>. This catches the long tail of
    # math/comparison glyphs a blacklist would miss (≥, ≤, ‰, ·, ...). Runs
    # after the deterministic fold in parse_asr_training_target(), so a hit here
    # is a genuinely meaning-bearing symbol the model must rewrite in words.
    # (The TTS source text is derived downstream by voxtral_normalize and is
    # never tokenized by Parakeet, so it needs no such gate.)
    for i, v in enumerate(variants):
        unk = _PARAKEET_TOK.offending_chars(v)
        if unk:
            chars = ", ".join(describe(c) for c in sorted(unk))
            raise LLMError(
                f"asr_training_target: variant {i} contains character(s) the "
                f"Parakeet tokenizer maps to <unk> (rewrite them in words): "
                f"{chars}: {_output_preview(v, 200)!r}"
            )
    for i, v in enumerate(variants):
        m = _UNIT_SYMBOL_RE.search(v)
        if m:
            raise LLMError(
                f"variant {i} uses abbreviated unit {m.group(0)!r} "
                f"(spell it out in full French): {_output_preview(v, 200)!r}"
            )
        ws = _BAD_WHITESPACE_RE.search(v)
        if ws:
            kind = {"\xa0": "NBSP", "\t": "tab"}.get(ws.group(0), "double space")
            raise LLMError(
                f"variant {i} contains disallowed whitespace ({kind}): "
                f"{_output_preview(v, 200)!r}"
            )
    heads = [v[:_DIVERSIFICATION_HEAD].lower() for v in variants]
    for i in range(len(heads)):
        for j in range(i + 1, len(heads)):
            ratio = fuzz.ratio(heads[i], heads[j])
            if ratio >= _DIVERSIFICATION_MAX_RATIO:
                raise LLMError(
                    f"variants {i} and {j} share near-identical openings "
                    f"(ratio={ratio:.0f} >= {_DIVERSIFICATION_MAX_RATIO}, "
                    f"clinical contexts must differ): "
                    f"{_output_preview(variants[i], 150)!r} vs "
                    f"{_output_preview(variants[j], 150)!r}"
                )


# ---------------------------------------------------------------------------
# Retry-with-validation loop
# ---------------------------------------------------------------------------


_VALIDATION_MAX_ATTEMPTS = 5


def _build_retry_context(
    attempt: int, last_error: "LLMError | None", last_raw: str | None
) -> dict | None:
    """Decide what extra context (if any) to pass to the next produce_fn call.

    Alternates between fresh (no context) and feedback (previous raw output
    + error message) attempts so a stuck model gets unstuck. Attempts >= 4
    add an "extra focus" warning. Length-truncated errors always retry
    fresh: feeding back the truncated output only inflates the next user
    message and makes truncation more likely.
    """
    if attempt == 1 or last_error is None:
        return None
    if getattr(last_error, "truncated", False):
        return None
    feedback_attempt = (attempt % 2 == 0)
    extra_focus = (attempt >= 4)
    if feedback_attempt and last_raw is not None:
        return {
            "previous_output": last_raw,
            "error_message": str(last_error),
            "extra_focus": extra_focus,
        }
    if extra_focus:
        return {
            "previous_output": None,
            "error_message": None,
            "extra_focus": True,
        }
    return None


def _retry_with_validation(produce_fn, label: str, tracker=None):
    """Run produce_fn, retrying LLMError up to _VALIDATION_MAX_ATTEMPTS.

    LLMError (including its TermMissingError subclass) triggers another
    attempt, alternating fresh / feedback context; exhaustion raises
    ValidationError with the last error attached as __cause__. A wrong or
    empty `<t>` block count is a BlockCountError (an LLMError), so it retries
    here too and, on exhaustion, surfaces as a skippable ValidationError.
    Only CountMismatchError, the fatal internal target/source length
    invariant (not an LLMError), bubbles through untouched.

    When `tracker` is given, each re-ask increments its validation-retry
    counter so the run's retry rate is visible alongside cost.
    """
    last_error: LLMError | None = None
    last_raw: str | None = None
    for attempt in range(1, _VALIDATION_MAX_ATTEMPTS + 1):
        retry_context = _build_retry_context(attempt, last_error, last_raw)
        try:
            return produce_fn(retry_context=retry_context)
        except LLMError as e:
            last_error = e
            last_raw = getattr(e, "raw_output", None)
            # Count as a retry only when another attempt follows; the final
            # failing attempt is an exhaustion, not a retry.
            if tracker is not None and attempt < _VALIDATION_MAX_ATTEMPTS:
                tracker.note_validation_retry()
            mode = (
                "feedback" if (retry_context and retry_context.get("previous_output"))
                else ("fresh+focus" if (retry_context and retry_context.get("extra_focus"))
                      else "fresh")
            )
            logger.warning(
                f"{label}: soft validation failed on attempt "
                f"{attempt}/{_VALIDATION_MAX_ATTEMPTS} (mode={mode}): {e}"
            )
    raise ValidationError(
        f"{label}: soft validation failed after "
        f"{_VALIDATION_MAX_ATTEMPTS} attempts. Last error: {last_error}"
    ) from last_error


# ---------------------------------------------------------------------------
# OpenRouter helpers and pricing tracker
# ---------------------------------------------------------------------------

_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_OPENROUTER_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{slug}/endpoints"


def _strip_openrouter_prefix(model: str) -> str:
    """litellm wants 'openrouter/<author>/<slug>'; OR's /models lists '<author>/<slug>'."""
    return model[len("openrouter/"):] if model.startswith("openrouter/") else model


def _fetch_openrouter_endpoints(model: str) -> list[dict] | None:
    """Return the list of provider endpoints OpenRouter exposes for `model`.

    Returns None when the request fails or the model is unknown so the
    caller can fall back to a plain re-raise without masking the original
    error.
    """
    slug = _strip_openrouter_prefix(model)
    url = _OPENROUTER_ENDPOINTS_URL.format(slug=slug)
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        logger.warning(f"could not fetch {url}: {exc}")
        return None
    payload = data.get("data") if isinstance(data, dict) else None
    if not isinstance(payload, dict):
        return None
    endpoints = payload.get("endpoints")
    if not isinstance(endpoints, list):
        return None
    return endpoints


def _format_endpoints_table(endpoints: list[dict]) -> str:
    """Build a human-readable list of providers with prompt/completion prices.

    Sorted cheapest-first by (prompt + completion) so the most attractive
    candidate to retry with appears at the top.
    """
    rows: list[tuple[float, str]] = []
    for ep in endpoints:
        pricing = ep.get("pricing") or {}
        try:
            prompt_per_m = float(pricing.get("prompt", 0) or 0) * 1e6
            completion_per_m = float(pricing.get("completion", 0) or 0) * 1e6
            cache_read_per_m = float(pricing.get("input_cache_read", 0) or 0) * 1e6
        except (TypeError, ValueError):
            prompt_per_m = completion_per_m = cache_read_per_m = 0.0
        slug = (
            ep.get("provider_slug")
            or ep.get("tag")
            or ep.get("provider_name")
            or ep.get("name")
            or "?"
        )
        name = ep.get("name") or slug
        ctx = ep.get("context_length")
        ctx_str = f" ctx={ctx}" if ctx else ""
        quant = ep.get("quantization")
        quant_str = f" quant={quant}" if quant else ""
        total_per_m = prompt_per_m + completion_per_m
        rows.append((
            total_per_m,
            f"  - slug={slug!r} name={name!r}{ctx_str}{quant_str} "
            f"prompt=${prompt_per_m:.3f}/M completion=${completion_per_m:.3f}/M "
            f"cache_read=${cache_read_per_m:.3f}/M total=${total_per_m:.3f}/M"
        ))
    rows.sort(key=lambda r: r[0])
    lines = [r[1] for r in rows]
    return "\n".join(lines) if lines else "  (no endpoints listed)"


_NO_ENDPOINTS_MARKERS = (
    "no endpoints",
    "no allowed providers",
    "no providers available",
)


def _looks_like_no_endpoints_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _NO_ENDPOINTS_MARKERS)


def _log_available_endpoints(model: str, provider: str | None, exc: BaseException) -> None:
    """On a 'no endpoints' error, dump the providers OpenRouter does have."""
    endpoints = _fetch_openrouter_endpoints(model)
    if endpoints is None:
        logger.error(
            f"litellm error for model {model!r} (pinned provider={provider!r}): "
            f"{exc}\nCould not fetch OpenRouter endpoints list to suggest "
            f"alternatives. Check that the model id is correct."
        )
        return
    table = _format_endpoints_table(endpoints)
    hint = ""
    pinned_present = provider and any(
        provider == (ep.get("provider_slug") or ep.get("tag") or "").split("/", 1)[0]
        for ep in endpoints
    )
    if pinned_present:
        hint = (
            f"\nHint: provider {provider!r} IS listed for this model, so the "
            f"404 is account-wide filtering, not a model/provider mismatch. "
            f"Check https://openrouter.ai/settings/privacy for: "
            f"'Allow training providers', Zero Data Retention enforcement, "
            f"and any ignored or allowed-providers lists. Per-request routing "
            f"cannot loosen those. Workaround: pick a non-filtered provider "
            f"from the table below."
        )
    logger.error(
        f"litellm error for model {model!r} (pinned provider={provider!r}): {exc}\n"
        f"OpenRouter exposes {len(endpoints)} endpoint(s) for this model. "
        f"Re-run with --provider <slug> picked from the list below "
        f"(or --provider '' to disable pinning):\n{table}{hint}"
    )


class PricingTracker:
    """Track running spend on an OpenRouter run and project total cost.

    Prices come from a single GET /api/v1/models at startup. Per-call cost
    uses the usage block returned by the API. The projected total uses
    tiktoken (gpt-4o, o200k_base) to count what we will send, then a
    rolling actual/tiktoken ratio to convert that estimate into the
    model's own tokens. Completion tokens are projected from a rolling
    mean of observed completions.

    Thread-safe.
    """

    _COMPLETION_FALLBACK = 1500.0
    _RATIO_FALLBACK = 1.15
    _RATIO_WARMUP_CALLS = 3

    def __init__(self, model: str) -> None:
        self.model = model
        self._lock = threading.Lock()
        self._enc = tiktoken.get_encoding("o200k_base")

        self.price = self._fetch_price(model)
        self.actual_prompt = 0
        self.actual_completion = 0
        self.actual_cached = 0
        self.calls = 0
        self.cost_prompt = 0.0
        self.cost_cached = 0.0
        self.cost_completion = 0.0
        self.tiktoken_prompt_sent = 0
        self.planned_prompt_tiktoken = 0
        self.planned_calls = 0
        self.full_planned_prompt_tiktoken = 0
        self.full_planned_calls = 0
        # Retry accounting, split by which layer retried: transport = the
        # call_llm tenacity loop (network errors + malformed/truncated/censored
        # responses), validation = the soft-validator loop in
        # _retry_with_validation. Surfaced live so a run's health is visible.
        self.transport_retries = 0
        self.validation_retries = 0

    @staticmethod
    def _fetch_price(model: str) -> dict[str, float]:
        slug = _strip_openrouter_prefix(model)
        try:
            with urllib.request.urlopen(_OPENROUTER_MODELS_URL, timeout=15) as resp:
                data = json.load(resp)
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            logger.warning(
                f"PricingTracker: could not fetch {_OPENROUTER_MODELS_URL}: {exc}. "
                f"Cost tracking disabled (all zeros)."
            )
            return {"prompt": 0.0, "completion": 0.0,
                    "input_cache_read": 0.0, "input_cache_write": 0.0}
        rows = data.get("data", data) if isinstance(data, dict) else data
        for row in rows:
            if row.get("id") == slug:
                p = row.get("pricing") or {}
                return {
                    "prompt": float(p.get("prompt", 0) or 0),
                    "completion": float(p.get("completion", 0) or 0),
                    "input_cache_read": float(p.get("input_cache_read", 0) or 0),
                    "input_cache_write": float(p.get("input_cache_write", 0) or 0),
                }
        logger.warning(
            f"PricingTracker: model {slug!r} not found in OpenRouter /models. "
            f"Cost tracking disabled."
        )
        return {"prompt": 0.0, "completion": 0.0,
                "input_cache_read": 0.0, "input_cache_write": 0.0}

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def add_planned(self, prompt_text_tiktoken: int, *, full_only: bool = False) -> None:
        with self._lock:
            if not full_only:
                self.planned_prompt_tiktoken += prompt_text_tiktoken
                self.planned_calls += 1
            self.full_planned_prompt_tiktoken += prompt_text_tiktoken
            self.full_planned_calls += 1

    def record(
        self,
        *,
        prompt_tiktoken: int,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        cached_tokens: int | None,
    ) -> float:
        pt = prompt_tokens or 0
        ct = completion_tokens or 0
        cached = cached_tokens or 0
        non_cached = max(0, pt - cached)
        c_prompt = non_cached * self.price["prompt"]
        c_cached = cached * self.price["input_cache_read"]
        c_completion = ct * self.price["completion"]
        cost = c_prompt + c_cached + c_completion
        with self._lock:
            self.tiktoken_prompt_sent += prompt_tiktoken
            self.actual_prompt += pt
            self.actual_completion += ct
            self.actual_cached += cached
            self.calls += 1
            self.cost_prompt += c_prompt
            self.cost_cached += c_cached
            self.cost_completion += c_completion
        return cost

    def note_transport_retry(self) -> None:
        """Count one call_llm-layer retry (network or bad-response failure)."""
        with self._lock:
            self.transport_retries += 1

    def note_validation_retry(self) -> None:
        """Count one soft-validation retry (a failed attempt that will re-ask)."""
        with self._lock:
            self.validation_retries += 1

    @property
    def cost(self) -> float:
        return self.cost_prompt + self.cost_cached + self.cost_completion

    @property
    def prompt_ratio(self) -> float:
        if self.calls < self._RATIO_WARMUP_CALLS or self.tiktoken_prompt_sent == 0:
            return self._RATIO_FALLBACK
        return self.actual_prompt / self.tiktoken_prompt_sent

    @property
    def mean_completion_tokens(self) -> float:
        if self.calls == 0:
            return self._COMPLETION_FALLBACK
        return self.actual_completion / self.calls

    @property
    def cache_hit_rate(self) -> float:
        if self.actual_prompt == 0:
            return 0.0
        return self.actual_cached / self.actual_prompt

    def _estimate_remaining(self, planned_calls: int, planned_tt: int) -> float:
        with self._lock:
            remaining_calls = max(0, planned_calls - self.calls)
            remaining_tt = max(0, planned_tt - self.tiktoken_prompt_sent)
            ratio = self.prompt_ratio
            cache_rate = self.cache_hit_rate
            mean_comp = self.mean_completion_tokens
        est_prompt_tok = remaining_tt * ratio
        est_cached_tok = est_prompt_tok * cache_rate
        est_uncached_tok = est_prompt_tok - est_cached_tok
        est_completion_tok = remaining_calls * mean_comp
        return (
            est_uncached_tok * self.price["prompt"]
            + est_cached_tok * self.price["input_cache_read"]
            + est_completion_tok * self.price["completion"]
        )

    def estimate_remaining(self) -> float:
        return self._estimate_remaining(
            self.planned_calls, self.planned_prompt_tiktoken
        )

    def estimate_remaining_full(self) -> float:
        return self._estimate_remaining(
            self.full_planned_calls, self.full_planned_prompt_tiktoken
        )

    def projected_total(self) -> float:
        return self.cost + self.estimate_remaining()

    def projected_total_full(self) -> float:
        return self.cost + self.estimate_remaining_full()

    @property
    def is_limited(self) -> bool:
        return self.full_planned_calls > self.planned_calls

    def stats_dict(self) -> dict:
        """Thread-safe snapshot of running cost + retry counters for logging.

        The structured counterpart to `summary_line`, meant to be embedded in a
        run_statistics.jsonl record so a run's cost and retry health can be
        machine-read (or handed to an LLM) after the fact.
        """
        with self._lock:
            calls = self.calls
            planned = self.planned_calls
            c_prompt = self.cost_prompt
            c_cached = self.cost_cached
            c_completion = self.cost_completion
            actual_prompt = self.actual_prompt
            actual_completion = self.actual_completion
            actual_cached = self.actual_cached
            tx = self.transport_retries
            vl = self.validation_retries
            cache_pct = self.cache_hit_rate
            ratio = self.prompt_ratio
        per = max(1, calls)
        return {
            "calls": calls,
            "planned_calls": planned,
            "cost_usd": round(c_prompt + c_cached + c_completion, 6),
            "cost_prompt_usd": round(c_prompt, 6),
            "cost_cached_usd": round(c_cached, 6),
            "cost_completion_usd": round(c_completion, 6),
            "projected_total_usd": round(self.projected_total(), 6),
            "cache_hit_rate": round(cache_pct, 4),
            "prompt_tok_ratio": round(ratio, 4),
            "prompt_tokens": actual_prompt,
            "completion_tokens": actual_completion,
            "cached_tokens": actual_cached,
            "transport_retries": tx,
            "validation_retries": vl,
            "transport_retries_per_call": round(tx / per, 4),
            "validation_retries_per_call": round(vl / per, 4),
        }

    def summary_line(self) -> str:
        with self._lock:
            calls = self.calls
            planned = self.planned_calls
            c_prompt = self.cost_prompt
            c_cached = self.cost_cached
            c_completion = self.cost_completion
            cache_pct = 100 * self.cache_hit_rate
            ratio = self.prompt_ratio
            tx = self.transport_retries
            vl = self.validation_retries
        cost = c_prompt + c_cached + c_completion
        projected = self.projected_total()
        full_str = ""
        if self.is_limited:
            full_str = f" (full_dataset=${self.projected_total_full():.4f})"
        per = max(1, calls)
        return (
            f"cost: calls={calls}/{planned} "
            f"spent=${cost:.4f} "
            f"(prompt=${c_prompt:.4f} cache_read=${c_cached:.4f} "
            f"completion=${c_completion:.4f}) "
            f"projected=${projected:.4f}{full_str} "
            f"cache_hit={cache_pct:.1f}% tok_ratio={ratio:.2f} "
            f"retries: transport={tx} ({tx / per:.3f}/call) "
            f"validation={vl} ({vl / per:.3f}/call)"
        )

    def tqdm_postfix(self) -> dict[str, str]:
        with self._lock:
            c_prompt = self.cost_prompt
            c_cached = self.cost_cached
            c_completion = self.cost_completion
            per = max(1, self.calls)
            tx_per = self.transport_retries / per
            vl_per = self.validation_retries / per
        postfix = {
            "$": f"{c_prompt + c_cached + c_completion:.3f}",
            "$p": f"{c_prompt:.3f}",
            "$c": f"{c_cached:.3f}",
            "$o": f"{c_completion:.3f}",
            "est": f"{self.projected_total():.3f}",
            "cache": f"{100 * self.cache_hit_rate:.0f}%",
            "txR": f"{tx_per:.2f}",
            "vlR": f"{vl_per:.2f}",
        }
        if self.is_limited:
            postfix["est_full"] = f"{self.projected_total_full():.3f}"
        return postfix


# ---------------------------------------------------------------------------
# litellm call wrapper
# ---------------------------------------------------------------------------


def _tracker_from_retry_state(retry_state) -> "PricingTracker | None":
    """Find the PricingTracker among the retried call's args/kwargs, if any.

    call_llm receives `tracker` (by keyword in the generators), so the tenacity
    before_sleep hook can reach it to count transport-layer retries without a
    global or a contextvar.
    """
    kwargs = getattr(retry_state, "kwargs", None) or {}
    candidate = kwargs.get("tracker")
    if isinstance(candidate, PricingTracker):
        return candidate
    for arg in getattr(retry_state, "args", ()) or ():
        if isinstance(arg, PricingTracker):
            return arg
    return None


def _log_retry(retry_state) -> None:
    """tenacity before_sleep callback that routes through loguru.

    Fires once per retry (before each backoff sleep), so it is also where the
    transport-retry counter is bumped on the shared PricingTracker.
    """
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc is None:
        return
    tracker = _tracker_from_retry_state(retry_state)
    if tracker is not None:
        tracker.note_transport_retry()
    logger.warning(
        f"tenacity retry #{retry_state.attempt_number} after "
        f"{type(exc).__name__}: {exc}"
    )


_RETRYABLE_EXCEPTIONS = (APIError, APIConnectionError, APITimeoutError, LLMError)

_OK_FINISH_REASONS = {"stop", "end_turn", "completed", None}


# Cap on how much of a failed generation is echoed back in feedback. A long
# answer would otherwise inflate every retry (and, near max_tokens, make the
# retry more likely to truncate too). The validator error already quotes the
# specific offending variant, so a head+tail excerpt is enough to orient the
# model without shipping the whole thing.
_FEEDBACK_PREVIEW_LIMIT = 1200


def _apply_retry_context(user_prompt: str, retry_context: dict | None) -> str:
    """Tack a retry feedback suffix onto the user prompt.

    Keeps the system prompt AND the original user prompt as an untouched
    prefix (feedback is appended after both), so prefix-caching providers keep
    hitting on system + user across retries and only the appended feedback is
    uncached. The echoed previous output is bounded by `_FEEDBACK_PREVIEW_LIMIT`
    and wrapped in short `<verr>` / `<prev>` tags so the model can tell the
    validator error from its own prior text.
    """
    if not retry_context:
        return user_prompt
    parts: list[str] = [user_prompt]
    if retry_context.get("previous_output") is not None:
        prev = _preview_head_tail(
            retry_context["previous_output"], _FEEDBACK_PREVIEW_LIMIT
        )
        parts.append(
            "\n\n---\n"
            "PREVIOUS ATTEMPT FAILED VALIDATION. Fix the specific issue below; "
            "do NOT repeat the same mistake.\n"
            f"<verr>{retry_context['error_message']}</verr>\n"
            f"<prev>{prev}</prev>\n"
            "---"
        )
    if retry_context.get("extra_focus"):
        parts.append(
            "\n\nIMPORTANT: Multiple previous attempts have failed validation "
            "on this same input. Re-read the system prompt rules carefully "
            "before generating, and double-check the output against them."
        )
    parts.append("\n\nRegenerate the full output now.")
    return "".join(parts)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
    before_sleep=_log_retry,
    reraise=True,
)
def call_llm(
    user_prompt: str,
    system_prompt: str,
    model: str,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    retry_context: dict | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    provider: str | None = None,
    reasoning: dict | None = None,
    tracker: PricingTracker | None = None,
    api_base: str | None = None,
) -> str:
    """Call litellm once and return the raw assistant text.

    Adds cache_control to the system prompt so providers that support
    prompt caching (Anthropic, OpenRouter proxied models) can reuse the
    cached system prompt across calls. `retry_context` (optional) appends
    feedback from the previous failed soft-validation attempt to the user
    message. `temperature` / `top_p` (optional) are forwarded only when set,
    so the provider's own defaults apply otherwise. `reasoning` (optional) is
    forwarded verbatim as the request body's `reasoning` object (OpenRouter's
    unified reasoning control, e.g. `{"enabled": False}` or `{"effort":
    "high"}`); merged alongside provider pinning, not overwriting it. `api_base`
    (optional) overrides the endpoint URL for self-hosted / custom gateways.
    Retries up to 5 times with exponential backoff on transient API failures
    and on LLMError.
    """
    final_user_prompt = _apply_retry_context(user_prompt, retry_context)
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {"role": "user", "content": final_user_prompt},
    ]

    max_tokens = DEFAULT_MAX_TOKENS
    completion_kwargs: dict = {
        "model": model,
        "messages": messages,
        "timeout": timeout_s,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        completion_kwargs["temperature"] = temperature
    if top_p is not None:
        completion_kwargs["top_p"] = top_p
    # Build extra_body incrementally so provider pinning and reasoning control
    # coexist (both live in the OpenRouter request body, not the OpenAI schema).
    extra_body: dict = {}
    if provider:
        extra_body["provider"] = {"order": [provider], "allow_fallbacks": False}
    if reasoning is not None:
        extra_body["reasoning"] = reasoning
    if extra_body:
        completion_kwargs["extra_body"] = extra_body
    if api_base:
        completion_kwargs["api_base"] = api_base
    try:
        response = litellm.completion(**completion_kwargs)
    except Exception as exc:
        if _looks_like_no_endpoints_error(exc):
            _log_available_endpoints(model, provider, exc)
        raise

    choice = response.choices[0]
    content = choice.message.content
    finish_reason = getattr(choice, "finish_reason", None)

    if finish_reason not in _OK_FINISH_REASONS:
        exc = LLMError(
            f"llm call ended with finish_reason={finish_reason!r} "
            f"(content_filter=censored, length=truncated at max_tokens={max_tokens})"
        )
        if finish_reason in ("length", "max_tokens"):
            exc.truncated = True
        raise exc
    if content is None:
        raise LLMError(
            f"litellm returned None content (finish_reason={finish_reason!r})"
        )
    content = _strip_reasoning(content)
    if finish_reason is None:
        logger.warning(
            "llm call returned None finish_reason; treating as success "
            "but this is provider-dependent"
        )

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    cached_tokens = None
    if usage is not None:
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached_tokens = getattr(details, "cached_tokens", None)

    pct = (
        f" ({100 * completion_tokens / max_tokens:.0f}% of max)"
        if isinstance(completion_tokens, int) else ""
    )
    cached_str = f", cached={cached_tokens}" if cached_tokens else ""
    cost_str = ""
    if tracker is not None:
        prompt_tiktoken = tracker.count(system_prompt) + tracker.count(final_user_prompt)
        call_cost = tracker.record(
            prompt_tiktoken=prompt_tiktoken,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
        )
        cost_str = f", cost=${call_cost:.5f}"
    logger.info(
        f"llm call ok: completion={completion_tokens}{pct}, "
        f"prompt={prompt_tokens}{cached_str}, finish={finish_reason}{cost_str}"
    )
    return content
