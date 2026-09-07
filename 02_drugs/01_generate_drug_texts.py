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

"""Generate ASR training text pairs for drugs, via the shared engine.

This is the drugs adapter for `utils/text_generation_engine.py`: the same engine
the dictionary stage uses. For each drug row it produces N
`asr_training_target` variants (short French medication-dictation sentences that
name the drug) and derives each `asr_training_source` deterministically with
`voxtral_normalize`. All the orchestration (score-based variant counts, the
retry-with-validation loop, dedup, resumability, run statistics, review / skip
queues, pricing pre-plan) lives in the engine, so this file only supplies the
drug-specific bits:

* the input schema (`drugs_freq_dosages.jsonl`: term / type / category / score /
  substances / dosages);
* the system prompt = the shared base prompt + the drugs addendum (both under
  01_dictionnary/), which steer the model to emit the SAME `<t>...</t>` blocks
  with the SAME validators as the dictionary stage;
* the per-item user prompt: the drug name plus a Definition hint describing its
  real pharmaceutical presentations (forms + dosages), which the drugs addendum
  treats as a factual presentation cue;
* the output row schema and the term-presence validator: the concrete drug
  anchor(s) of the label must appear in each variant. ATC combination labels are
  reduced by `presence_needle` first, so filler like "EN ASSOCIATION" or "ET
  DIURETIQUES" is not required and a real "+"/"ET" combination requires both
  active drugs (order-independent).

Reuses `utils/_pipeline_shared` (`call_llm`, validators) and the engine rather
than a private LLM stack, so there is one implementation of every shared concern.

Created with assistance from Claude Code.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import click
from loguru import logger

# Shared pipeline + engine live in utils/ so every corpus stage calls one
# implementation instead of a private copy. See utils/text_generation_engine.py
# and utils/_pipeline_shared.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from _pipeline_shared import (  # noqa: E402
    TermMissingError,
    _content_words,
    _strip_accents,
    call_llm,
    check_term_in_variants,
    term_present_in_variant,
    validate_asr_training_target,
)
from text_generation_engine import (  # noqa: E402
    StageAdapter,
    build_target_user_prompt,
    run as engine_run,
)


HERE = Path(__file__).resolve().parent
_PROMPT_DIR = HERE.parent / "01_dictionnary"

# The drugs system prompt is the shared base target prompt plus the drugs
# addendum, kept next to it under 01_dictionnary/. The addendum re-steers only
# the role and clinical contexts; the base rules, banned chars, spelled-out
# units and the exact <t>...</t> output format still apply, which is why the
# engine's parser and validators are reused unchanged.
DEFAULT_BASE_PROMPT = _PROMPT_DIR / "PROMPT_GENERATE_ASR_TRAINING_TARGET.md"
DEFAULT_DRUGS_ADDENDUM = _PROMPT_DIR / "PROMPT_GENERATE_ASR_TRAINING_TARGET_DRUGS.md"

DEFAULT_INPUT = HERE / "drugs_freq_dosages.jsonl"
DEFAULT_OUTPUT = HERE / "generated_dataset.jsonl"
DEFAULT_REVIEW_QUEUE = HERE / "voxtral_review_queue.jsonl"
DEFAULT_SKIP_QUEUE = HERE / "term_missing_skips.jsonl"
DEFAULT_RUN_STATS = HERE / "run_statistics.jsonl"
DEFAULT_RUN_LOG = HERE / "run_statistics.log"

# Match the dictionary stage so prompt caching and provider pinning behave the
# same across corpus stages.
DEFAULT_MODEL = "openrouter/deepseek/deepseek-v4-pro"
DEFAULT_PROVIDER = "deepseek"
DEFAULT_N_VARIANTS = 11
DEFAULT_N_JOBS = 4
DEFAULT_TIMEOUT_S = 300
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95

# Reasoning escalation, matching the dictionary stage: the first (cheap) pass
# runs "non-think" for speed and cost, and a validation retry omits the argument
# so the provider default (thinking) kicks in only when the cheap pass failed the
# validators. The shared base prompt tells the model its first instinct is
# usually right, so non-think should carry most drugs.
DEFAULT_REASONING = {"enabled": False}  # "non-think": first, cheap attempt
RETRY_REASONING = None                  # omit -> provider default (thinking)

# A per-item Definition hint that lists every form+dosage would balloon the
# uncached per-drug prompt; cap it (the model only needs a representative
# palette to vary presentations across variants).
_MAX_DEFINITION_CHARS = 600


def describe_presentations(dosages: dict) -> str:
    """Turn a drug's `dosages` map into a compact French presentation hint.

    `dosages` is {substance: {form: [[article_form, dose1, dose2, ...], ...]}}.
    Produces e.g. "comprimé dosé à 250 mg, 500 mg, 1 g ; suppositoire dosé à
    100 mg, 200 mg", which the drugs addendum reads as a factual cue for
    realistic presentations (the model still spells units out in its output).
    Doses are de-duplicated per form, preserving first-seen order.
    """
    chunks: list[str] = []
    for _substance, forms in (dosages or {}).items():
        if not isinstance(forms, dict):
            continue
        for form_name, variants in forms.items():
            doses: list[str] = []
            for variant in variants or []:
                if not isinstance(variant, list):
                    continue
                for dose in variant[1:]:
                    if dose and dose not in doses:
                        doses.append(dose)
            if doses:
                chunks.append(f"{form_name} dosé à {', '.join(doses)}")
            else:
                chunks.append(str(form_name))
    return " ; ".join(chunks)


def build_drug_user_prompt(entry: dict, n_variants: int) -> str:
    """Per-drug user message: the drug name + its presentation hint as Definition.

    Reuses the shared `build_target_user_prompt` so the drugs and dictionary
    stages send the exact same user-message format to the shared base system
    prompt. Case of the drug name is left as given; the base prompt's
    capitalization rule (INN lowercase, brand capitalized) fixes it, and the
    term-presence check is accent- and case-insensitive.
    """
    term = entry.get("term", "")
    definition = describe_presentations(entry.get("dosages", {}))
    return build_target_user_prompt(
        term,
        n_variants,
        definition=definition,
        max_definition_chars=_MAX_DEFINITION_CHARS,
    )


# Many drug labels are ATC combination names that pad a real, nameable drug with
# a generic therapeutic-class descriptor the model will never speak verbatim:
# "IRBESARTAN ET DIURETIQUES", "LIDOCAINE EN ASSOCIATION", "AMOXICILLINE ET
# INHIBITEUR D'ENZYME". The `substances` field does not help here (for these
# substance-type rows it just echoes the label), so we strip the descriptor from
# the label to recover the concrete anchor(s) that must actually appear. This set
# is drug-domain (ATC) vocabulary, hence it lives in this stage rather than the
# generic shared validator. Accent-stripped, lowercase.
_GENERIC_DRUG_CLASS_WORDS = {
    "association", "associations", "associes", "associe", "associee", "associees",
    "sequentiel",
    "diuretiques", "diuretique", "thiazidiques", "thiazidique",
    "epargneurs", "epargneur", "potassiques", "potassique",
    "antiinfectieux", "anti", "infectieux", "antibacteriens", "antibacterien",
    "antihypertenseurs", "antihypertenseur", "psycholeptiques", "psycholeptique",
    "corticosteroides", "corticosteroide", "antiacides", "antiacide",
    "antitussives", "antitussifs", "analgesiques", "anesthesiques",
    "inhibiteur", "inhibiteurs", "enzyme", "decarboxylase", "comt",
    "transcriptase", "inverse", "nucleosidiques", "nucleotidiques",
    "preparations", "preparation", "medicaments", "medicament",
    "autres", "autre", "divers", "diverse", "diverses",
    "produits", "produit", "gras", "sels", "mineraux", "mineral", "minerals",
    "electrolytes", "hydrates", "carbone", "substances", "substance",
    "nasales", "nasale", "tractus", "alimentaire", "metabolisme", "vitamines",
}

# Splits a label into its atomic components: the combination connectors ("+",
# ",", " et ", " et/ou ", " avec "), and everything from " en association "
# onward (a pure descriptor tail like "en association avec des psycholeptiques").
# The "," alternative deliberately excludes a French decimal comma inside a dosage
# (e.g. "TREPROSTIN.TLO2,5MG/ML"): a comma flanked by digits on both sides is a
# decimal point, not a combination separator, so splitting it would fabricate a
# spurious two-component anchor that no variant can satisfy.
_PRESENCE_SEP = re.compile(
    r"\s+et/ou\s+|\s+et\s+|\s+avec\s+|\ben association\b.*$|(?<!\d),(?!\d)|[+]",
    re.IGNORECASE,
)


def presence_needle(term: str) -> str:
    """Reduce a drug label to the concrete anchor(s) that must appear in a variant.

    Drops the generic ATC combination-class descriptors, keeping only the real,
    nameable drug words. Real multi-drug combinations are rejoined with "+", so
    the shared term check requires every real component (order-independent). A
    fully generic label ("ASSOCIATIONS", "SELS MINERAUX EN ASSOCIATION") reduces
    to "", signalling the caller to skip the presence check (there is no specific
    name to require).
    """
    low = _strip_accents(term.lower())
    components: list[str] = []
    for part in _PRESENCE_SEP.split(low):
        if not part:
            continue
        real = [w for w in _content_words(part) if w not in _GENERIC_DRUG_CLASS_WORDS]
        if real:
            components.append(" ".join(real))
    return " + ".join(components)


def _acceptable_needles(entry: dict) -> list[str]:
    """The presence anchors that satisfy a drug row: its term plus its substances.

    A drug is validly named by its label OR by its active ingredient(s). Some rows
    carry a truncated brand presentation code the model cannot speak verbatim
    (e.g. "EMTRICIT/TENOF.MYL200/245"), so the model names the real molecules
    instead; the `substances` field ("TENOFOVIR DISOPROXIL ET EMTRICITABINE")
    supplies those as fallback anchors. Each raw label is reduced by
    `presence_needle` (filler dropped, real combinations kept), empties skipped,
    order preserved and deduped.
    """
    needles: list[str] = []

    def _add(raw: str) -> None:
        n = presence_needle(raw or "")
        if n and n not in needles:
            needles.append(n)

    _add(entry.get("term", ""))
    subs = entry.get("substances") or []
    if isinstance(subs, str):
        subs = [subs]
    for s in subs:
        _add(s)
    return needles


def drug_validate(parsed: list[str], entry: dict) -> None:
    """Drug-stage soft validators: the shared target checks + term presence.

    No per-item reference examples exist for drugs (the worked examples live in
    the system prompt), so example-leak is a no-op (examples=[]). The generic
    cross-term dedup check is run by the engine separately.

    Presence is checked against `_acceptable_needles(entry)`, not the raw label:
    each label is reduced by `presence_needle` (an ATC combination only requires
    its concrete drug anchor(s); a purely generic class like "ASSOCIATIONS"
    reduces to "" and is skipped). When a row also has `substances`, those active
    ingredients are added as alternative anchors, so a variant is valid if it
    names EITHER the (reduced) label OR the molecule(s) - which rescues truncated
    brand codes the model can only speak by their real ingredient names.
    """
    validate_asr_training_target(parsed, examples=[])
    needles = _acceptable_needles(entry)
    if not needles:
        return
    idx = entry.get("index")
    # Single anchor (the common case): keep the shared check so its close /
    # scattered fuzzy-match warnings and rich errors are preserved.
    if len(needles) == 1:
        check_term_in_variants(needles[0], parsed, term_index=idx)
        return
    # Several acceptable anchors (label plus its `substances`): a variant is valid
    # if ANY anchor is present, so the model may name the brand OR the molecule(s).
    for i, v in enumerate(parsed):
        if not any(term_present_in_variant(n, v) for n in needles):
            raise TermMissingError(
                f"term {idx} {entry.get('term')!r}: none of the acceptable drug "
                f"anchors {needles!r} found in variant {i}: {v!r}"
            )


def drug_make_row(
    entry: dict, variant_index: int, target: str, source: str, model: str
) -> dict:
    """Output row for the drugs stage (shares the core pair fields with dict)."""
    return {
        # Provenance so the merged dataset can tell which source produced each
        # pair; "drugs" already tagged on the input rows, propagated not hardcoded.
        "category": entry.get("category", "drugs"),
        "term_index": entry.get("index"),
        "term": entry.get("term"),
        # Drug-specific provenance: substance vs brand, and the active substances.
        "type": entry.get("type"),
        "substances": entry.get("substances"),
        "variant_index": variant_index,
        "asr_training_target": target,
        "asr_training_source": source,
        "model": model,
    }


def build_drug_adapter(system_prompt: str, system_prompt_path=None) -> StageAdapter:
    """Assemble the drugs StageAdapter.

    `call_llm` is read from this module's globals when this runs, so it can be
    monkeypatched in tests the same way the dictionary stage allows.
    """
    return StageAdapter(
        stage="drugs",
        system_prompt=system_prompt,
        system_prompt_path=system_prompt_path,
        build_user_prompt=build_drug_user_prompt,
        validate_fn=drug_validate,
        make_row=drug_make_row,
        call_llm_fn=call_llm,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        # Cheap non-think first, thinking only on a validation retry (same policy
        # as the dictionary stage). See DEFAULT_REASONING / RETRY_REASONING.
        reasoning_first=DEFAULT_REASONING,
        reasoning_retry=RETRY_REASONING,
    )


def load_system_prompt(base_path: Path, addendum_path: Path) -> str:
    """Concatenate the shared base target prompt and the drugs addendum."""
    base = Path(base_path).read_text(encoding="utf-8")
    addendum = Path(addendum_path).read_text(encoding="utf-8")
    return f"{base.rstrip()}\n\n{addendum.lstrip()}\n"


def run(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
    base_prompt_path: str | Path = DEFAULT_BASE_PROMPT,
    drugs_addendum_path: str | Path = DEFAULT_DRUGS_ADDENDUM,
    n_variants: int | str | dict = DEFAULT_N_VARIANTS,
    model: str = DEFAULT_MODEL,
    n_jobs: int = DEFAULT_N_JOBS,
    limit: int | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    provider: str | None = DEFAULT_PROVIDER,
    review_path: str | Path = DEFAULT_REVIEW_QUEUE,
    skip_path: str | Path = DEFAULT_SKIP_QUEUE,
    run_stats_path: str | Path = DEFAULT_RUN_STATS,
    run_log_path: str | Path = DEFAULT_RUN_LOG,
) -> dict:
    """Drugs entry-point: build the system prompt + adapter and run the engine."""
    system_prompt = load_system_prompt(base_prompt_path, drugs_addendum_path)
    adapter = build_drug_adapter(system_prompt, system_prompt_path=base_prompt_path)
    return engine_run(
        adapter,
        input_path=input_path,
        output_path=output_path,
        n_variants=n_variants,
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
    help="Scored drug input JSONL (term/type/category/score/substances/dosages)",
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
    "--drugs-addendum-path", type=click.Path(exists=True, path_type=Path),
    default=str(DEFAULT_DRUGS_ADDENDUM),
    help="Drugs system-prompt addendum appended to the base prompt",
)
@click.option("--model", default=DEFAULT_MODEL, help="LLM model ID (litellm format)")
@click.option(
    "--provider", default=DEFAULT_PROVIDER,
    help="OpenRouter provider slug to pin (empty string to disable pinning)",
)
@click.option(
    "--n-variants", "-n", default=str(DEFAULT_N_VARIANTS), type=str,
    help=(
        "Either an int (same count per drug) or a JSON dict mapping score ranges "
        "to counts, e.g. '{\"0\":0,\"1-3\":2,\"4-7\":4,\"8-10\":11}'. Score comes "
        "from 01_llm_scoring.py."
    ),
)
@click.option("--n-jobs", "-j", default=DEFAULT_N_JOBS, type=int)
@click.option(
    "--limit", "-L", default=None, type=int,
    help="Only process the first N pending drugs (smoke tests)",
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
    help="JSONL skip queue for drugs whose variants kept dropping the term",
)
@click.option("-v", "--verbose", count=True, help="-v for DEBUG logging")
def main(
    input_path: Path,
    output_path: Path,
    base_prompt_path: Path,
    drugs_addendum_path: Path,
    model: str,
    provider: str,
    n_variants: str,
    n_jobs: int,
    limit: int | None,
    timeout_s: int,
    review_path: Path,
    skip_path: Path,
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
        base_prompt_path=base_prompt_path,
        drugs_addendum_path=drugs_addendum_path,
        model=model,
        provider=provider or None,
        n_variants=n_variants,
        n_jobs=n_jobs,
        limit=limit,
        timeout_s=timeout_s,
        review_path=review_path,
        skip_path=skip_path,
    )


if __name__ == "__main__":
    main()
