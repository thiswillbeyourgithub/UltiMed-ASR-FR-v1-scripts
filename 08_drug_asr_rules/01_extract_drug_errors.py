#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "litellm", "tiktoken", "tqdm", "tenacity", "rapidfuzz", "wordfreq"]
# ///
"""Collect how an ASR model misspells each drug name, from a hypotheses file.

Input: a hypotheses file ``{"audio": <path>, "hyp": <text>}`` per line, written by the
NeMo repo's ``perso/transcribe_manifests.py`` (see this folder's README for the command),
plus the manifests whose labels to compare against. Clips are joined on
``utils/nemo_manifest.clip_key`` (last two path components), like
``06_hotfixes/04_flag_asr_defects.py``, so the hypotheses may come from another checkout.

Drug words are spotted in the LABEL with a lexicon of accent-folded lowercase words:

- every spelling in ``utils/drug_casing.json`` (the lexicon ``label_conventions.DrugCaser``
  cases drug names with; it only lists drugs whose casing needed fixing, so it misses
  most plain lowercase DCIs like ``rivaroxaban``);
- every word of every stage-02 drug's presence anchors (``_acceptable_needles`` from
  ``02_drugs/01_generate_drug_texts.py``, imported rather than copied: it already strips
  ATC filler like "ET DIURETIQUES" or "EN ASSOCIATION"). That import pulls the text
  generators' dependencies (litellm, ...) into this script's header, which is the price
  of not duplicating the filler list.

Words shorter than ``MIN_DRUG_LEN`` or holding a digit are left out: they are sigles,
dosage codes or ordinary words ("fer", "b12") that a spelling rule must not touch.

Every label token found in the lexicon is one drug occurrence, so any manifest works
(drug names inside PARHAF or dictionary sentences count too) and a sentence naming two
drugs yields two. An occurrence is keyed by its accent-folded form and displayed with
its ``drug_casing.json`` canonical spelling when there is one, else the most frequent
label spelling (``géfitinib``).

For each occurrence the label and hypothesis are aligned word by word (difflib):

- drug token inside an ``equal`` block, or replaced by a form that only differs in
  accents or case: correct;
- inside a ``delete`` block: dropped, counted but never a variant (nothing to rewrite);
- inside a ``replace`` block: the hypothesis words that stand for it are the variant.
  When the replaced label stretch holds more than the drug (``de mirtazapine`` ->
  ``demi tazapine``), a character-level alignment inside the stretch maps the drug's
  characters onto the hypothesis and the variant is widened to whole hypothesis words.

When the variant swallows a neighbouring label word (an elided article, label
``l'étoposide``, hypothesis ``létoposide``; or a preposition, ``sous venlafaxine`` ->
``souvenent la vaccine``), that word belongs to the correction TARGET (``l'étoposide``,
``sous venlafaxine``), or the fix would eat it. A neighbour counts as swallowed when at
least half of its letters align inside the variant. Such variants are listed under
``targets``.

Tokenization: lowercase, punctuation dropped, French elisions split off
(``l'alfacalcidol`` -> ``l'`` ``alfacalcidol``) so the drug is its own token; a hyphen
inside a word is kept. Variants are rendered with the elision re-attached
(``d'hexaméthasone``). Casing is irrelevant here: the rules built by
``02_build_fix_rules.py`` match case-insensitively and write the canonical casing.

Output (default ``08_drug_asr_rules/drug_asr_errors.json``), most errors first::

    {"mirtazapine": {"n_seen": 212, "n_correct": 190, "n_deleted": 1,
                     "errors": {"myrtazapine": 15, "mire tazapine": 6}, "targets": {}},
     "étoposide": {..., "errors": {"létoposide": 2}, "targets": {"létoposide": "l'étoposide"}}}

    uv run 08_drug_asr_rules/01_extract_drug_errors.py <hyps.jsonl> <manifest> [<manifest> ...]

This file was written by Claude Code.
"""
from __future__ import annotations

import difflib
import importlib.util
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "utils"))
from nemo_manifest import clip_key, read_jsonl, resolve_audio  # noqa: E402

DEFAULT_OUT = _HERE / "drug_asr_errors.json"
DRUG_CASING = _ROOT / "utils" / "drug_casing.json"
DRUG_ENTRIES = _ROOT / "02_drugs" / "drugs_freq_dosages.jsonl"
DRUG_STAGE = _ROOT / "02_drugs" / "01_generate_drug_texts.py"
MIN_DRUG_LEN = 5
MAX_TERM_ZIPF = 3.0

_ELIDED = r"(?:[cdjlmnst]|qu|jusqu|lorsqu|puisqu)"
# Only the forms French actually elides, so an apostrophe inside a name is not cut.
_ELISION = re.compile(rf"\b({_ELIDED})'(?=\w)", re.IGNORECASE)
_ELIDED_HEAD = re.compile(rf"{_ELIDED}'")
# Anything that is not a letter, digit, apostrophe or hyphen separates words.
_SEP = re.compile(r"[^\w'-]+")
_HYPHEN_OR_SPACE = re.compile(r"[\s-]+")


def fold(word: str) -> str:
    """Lowercase and strip accents: ``Paroxétine`` -> ``paroxetine``."""
    nfkd = unicodedata.normalize("NFKD", word.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def tokens(text: str) -> list[str]:
    """Lowercased words, elisions split off, edge hyphens and stray quotes trimmed.

    >>> tokens("Le patient prend de l'Alfacalcidol, 0,25 µg.")
    ['le', 'patient', 'prend', 'de', "l'", 'alfacalcidol', '0', '25', 'µg']

    The typographic apostrophe counts as the straight one, or ``l’antivirus`` in a correct
    text would not block the variant ``l'antivirus``:

    >>> tokens("l’antivirus") == tokens("l'antivirus")
    True
    """
    text = _ELISION.sub(r"\1' ", text.lower().replace("’", "'"))
    out = []
    for w in _SEP.split(text):
        w = w.strip("-").lstrip("'")
        # Keep an elided head's apostrophe ("l'"), drop a stray closing quote ("mot'").
        if w.endswith("'") and not _ELIDED_HEAD.fullmatch(w):
            w = w.rstrip("'")
        if w:
            out.append(w)
    return out


def join_tokens(toks: list[str]) -> str:
    """Inverse of the elision split: ``["d'", "hexaméthasone"]`` -> ``d'hexaméthasone``."""
    return re.sub(r"' ", "'", " ".join(toks))


def _stage02_anchor_words() -> set[str]:
    """Folded words of every stage-02 drug's presence anchors."""
    spec = importlib.util.spec_from_file_location("generate_drug_texts", DRUG_STAGE)
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    words = set()
    for line in DRUG_ENTRIES.open(encoding="utf-8"):
        for needle in stage._acceptable_needles(json.loads(line)):
            words.update(fold(w) for w in tokens(needle))
    return words


def term_words(path: Path, max_zipf: float = MAX_TERM_ZIPF) -> set[str]:
    """Folded words of the ``term`` field of a JSONL lexicon (e.g. ``01_dictionnary/original_dictionnary.jsonl``),
    kept only when rarer in French than ``max_zipf`` (wordfreq Zipf scale), so a term like "syndrome de
    Brugada" adds ``brugada`` but not ``syndrome``: errors on everyday words are not term errors.

    >>> import tempfile; f = Path(tempfile.mkstemp(suffix=".jsonl")[1])
    >>> _ = f.write_text('{"term": "syndrome de Brugada"}\\n{"term": "l’acardiacie"}\\n'.replace("’", "'"), encoding="utf-8")
    >>> sorted(term_words(f))
    ['acardiacie', 'brugada']
    """
    from wordfreq import zipf_frequency
    words = set()
    for line in path.open(encoding="utf-8"):
        for w in tokens(json.loads(line)["term"]):
            if not _ELIDED_HEAD.fullmatch(w) and zipf_frequency(w, "fr") < max_zipf:
                words.add(fold(w))
    return words


def load_lexicon(extra: tuple[Path, ...] = (), max_term_zipf: float = MAX_TERM_ZIPF) -> dict[str, str | None]:
    """Folded drug word -> canonical spelling (``None`` when only stage 02 knows it, the
    label's own spelling is then used), plus the ``term_words`` of each ``extra`` lexicon (``None`` too)."""
    data = json.loads(DRUG_CASING.read_text(encoding="utf-8"))
    lex: dict[str, str | None] = {}
    for canon in data["caps"].values():
        lex.setdefault(fold(canon), canon)
    for spelling, canon in data["variants"].items():
        lex.setdefault(fold(spelling), canon)
    for w in _stage02_anchor_words():
        lex.setdefault(w, None)
    for path in extra:
        for w in term_words(path, max_term_zipf):
            lex.setdefault(w, None)
    return {w: c for w, c in lex.items() if len(w) >= MIN_DRUG_LEN and not re.search(r"\d", w)}


def _hyp_span(ref: list[str], i1: int, i2: int, lo_i: int, hi_i: int,
              hyp: list[str], j1: int, j2: int) -> tuple[list[str], int, int]:
    """Hypothesis words standing for ``ref[lo_i:hi_i]`` inside a replaced stretch, plus
    the label range ``[lo, hi)`` those words actually cover.

    A stretch whose label side is exactly that span maps wholesale. Otherwise the
    stretch is aligned character by character and the hypothesis words overlapping the
    span's characters are returned. Those words can swallow a neighbouring label word
    (``sous fluoxétine`` -> ``soufflue oxétine``): a neighbour with at least half of
    its letters inside them joins the range, so the fix restores it instead of eating
    it (``[lo, hi)`` then reaches past ``[lo_i, hi_i)``).

    >>> _hyp_span(["de", "mirtazapine"], 0, 2, 1, 2, ["demi", "tazapine"], 0, 2)
    (['demi', 'tazapine'], 0, 2)
    >>> _hyp_span(["prend", "mirtazapine", "le"], 0, 3, 1, 2, ["prend", "myrtazapine", "la"], 0, 3)
    (['myrtazapine'], 1, 2)
    >>> _hyp_span(["sous", "fluoxétine"], 0, 2, 1, 2, ["soufflue", "oxétine"], 0, 2)
    (['soufflue', 'oxétine'], 0, 2)
    """
    if (lo_i, hi_i) == (i1, i2):
        return hyp[j1:j2], lo_i, hi_i
    a = " ".join(ref[i1:i2])
    b_words = hyp[j1:j2]
    b = " ".join(b_words)
    starts, pos = [], 0
    for w in ref[i1:i2]:
        starts.append(pos)
        pos += len(w) + 1
    start, end = starts[lo_i - i1], starts[hi_i - 1 - i1] + len(ref[hi_i - 1])
    to_b = {}
    for blk in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks():
        to_b.update((blk.a + k, blk.b + k) for k in range(blk.size))
    mapped = [to_b[k] for k in range(start, end) if k in to_b]
    if not mapped:
        return b_words, lo_i, hi_i
    lo, hi = min(mapped), max(mapped)
    picked, pos = [], 0
    for w in b_words:
        if pos <= hi and pos + len(w) > lo:
            picked.append(w)
            lo, hi = min(lo, pos), max(hi, pos + len(w) - 1)
        pos += len(w) + 1

    def swallowed(k: int) -> bool:
        letters = [starts[k - i1] + c for c, ch in enumerate(ref[k]) if ch not in "'-"]
        inside = sum(lo <= to_b.get(c, -1) <= hi for c in letters)
        return 2 * inside >= len(letters)

    while lo_i > i1 and swallowed(lo_i - 1):
        lo_i -= 1
    while hi_i < i2 and swallowed(hi_i):
        hi_i += 1
    return picked, lo_i, hi_i


def drug_outcomes(label: str, hyp: str, lex: dict[str, str | None]
                  ) -> list[tuple[str, str, str | None, str]]:
    """``(folded key, label spelling, variant, target)`` per drug occurrence in the label.
    ``variant`` is ``None`` when the model got it right and ``""`` when it dropped it;
    ``target`` is what the variant should be rewritten to.

    >>> lex = {"mirtazapine": None, "paroxetine": "paroxétine", "etoposide": None}
    >>> drug_outcomes("Arrêt de la mirtazapine.", "Arrêt de la myrtazapine.", lex)
    [('mirtazapine', 'mirtazapine', 'myrtazapine', 'mirtazapine')]
    >>> [o[2] for o in drug_outcomes("Paroxétine 20 mg puis mirtazapine", "paroxetine 20 mg puis mire tazapine", lex)]
    [None, 'mire tazapine']
    >>> drug_outcomes("prendre mirtazapine le soir", "prendre le soir", lex)[0][2]
    ''
    >>> drug_outcomes("arrêt de l'étoposide hier", "arrêt de létoposide hier", lex)
    [('etoposide', 'étoposide', 'létoposide', "l'étoposide")]
    >>> drug_outcomes("traité sous venlafaxine depuis", "traité souvenent la vaccine depuis", {"venlafaxine": None})
    [('venlafaxine', 'venlafaxine', 'souvenent la vaccine', 'sous venlafaxine')]
    >>> drug_outcomes("un anti-TNF alpha", "un anti-TNF-alpha", {"alpha": None})[0][2] is None
    True

    A variant holding the decoder's ``<unk>`` token (which tokenizing turns into the word
    ``unk``) is dropped: no runtime output ever matches it (onnx-asr glues ``<unk>`` to
    the next word, the web decoder deletes it), so a rule learned from it is dead.

    >>> drug_outcomes("Arrêt de la mirtazapine.", "Arrêt de la <unk>tazapine.", lex)
    []
    >>> drug_outcomes("Arrêt de la mirtazapine.", "Arrêt de la myrta <unk>", lex)
    []
    """
    ref, h = tokens(label), tokens(hyp)
    drugs = {i: fold(w) for i, w in enumerate(ref) if fold(w) in lex}
    if not drugs:
        return []
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ref, b=h, autojunk=False).get_opcodes():
        for i in range(i1, i2):
            if i not in drugs:
                continue
            key, spelling = drugs[i], ref[i]
            if tag == "equal":
                out.append((key, spelling, None, spelling))
                continue
            if tag == "delete":
                out.append((key, spelling, "", spelling))
                continue
            span, lo, hi = _hyp_span(ref, i1, i2, i, i + 1, h, j1, j2)
            # A label word the variant swallowed belongs to the target ("l'étoposide" ->
            # "létoposide", "sous venlafaxine" -> "souvenent la vaccine").
            variant, target = join_tokens(span), join_tokens(ref[lo:hi])
            if "unk" in span:
                continue
            # An accent, case or hyphen-for-space slip is not a spelling error (the drug
            # casing pass of label_conventions fixes the first two, and a rule for the
            # third would match the label itself: its patterns accept either separator).
            slip = _HYPHEN_OR_SPACE.sub(" ", fold(variant)) == _HYPHEN_OR_SPACE.sub(" ", fold(target))
            out.append((key, spelling, None if slip else variant, target))
    return out


def collect(pairs, lex: dict[str, str | None]) -> dict[str, dict]:
    """Aggregate ``(label, hyp)`` pairs into the per-drug report, most errors first."""
    seen, correct, deleted, spellings = Counter(), Counter(), Counter(), defaultdict(Counter)
    errors: dict[str, Counter] = defaultdict(Counter)
    targets: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for label, hyp in pairs:
        for key, spelling, variant, target in drug_outcomes(label, hyp, lex):
            seen[key] += 1
            spellings[key][spelling] += 1
            if variant is None:
                correct[key] += 1
            elif variant == "":
                deleted[key] += 1
            else:
                errors[key][variant] += 1
                targets[key][variant][target] += 1
    report = {}
    for key in sorted(seen, key=lambda k: (-sum(errors[k].values()), k)):
        name = lex[key] or spellings[key].most_common(1)[0][0]
        tg = {v: c.most_common(1)[0][0] for v, c in targets[key].items()}
        report[name] = {
            "n_seen": seen[key], "n_correct": correct[key], "n_deleted": deleted[key],
            "errors": dict(errors[key].most_common()),
            # Only the variants whose fix is not the plain drug name (glued elisions).
            "targets": {v: t for v, t in tg.items() if fold(t) != key},
        }
    return report


def lexicon_options(f):
    """``--lexicon`` / ``--max-term-zipf``, shared with 02 so both scripts see the same lexicon."""
    f = click.option("--max-term-zipf", default=MAX_TERM_ZIPF, show_default=True,
                     help="a --lexicon word at least this frequent in French (wordfreq Zipf) is left out")(f)
    return click.option("--lexicon", "lexicons", multiple=True, type=click.Path(exists=True, path_type=Path),
                        help="extra JSONL term list (field 'term'), e.g. 01_dictionnary/original_dictionnary.jsonl: "
                             "its rare words are tracked like drug names (repeatable)")(f)


@click.command()
@click.argument("hyps", type=click.Path(exists=True, path_type=Path))
@click.argument("manifests", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out", default=str(DEFAULT_OUT), show_default="08_drug_asr_rules/drug_asr_errors.json",
              type=click.Path(path_type=Path))
@lexicon_options
def main(hyps: Path, manifests: tuple[Path, ...], out: Path, lexicons: tuple[Path, ...], max_term_zipf: float) -> None:
    """Write the per-drug misspelling report from a hypotheses file."""
    hyp_by_key = {}
    for line in hyps.open():
        r = json.loads(line)
        hyp_by_key[clip_key(r["audio"])] = r["hyp"]
    lex = load_lexicon(lexicons, max_term_zipf)
    logger.info(f"lexicon: {len(lex)} words")
    pairs, seen_keys = [], set()
    for m in manifests:
        for row in read_jsonl(m):
            key = clip_key(resolve_audio(m.parent, row["audio_filepath"]))
            if key in hyp_by_key and key not in seen_keys:
                seen_keys.add(key)
                pairs.append((row["text"], hyp_by_key[key]))
    report = collect(pairs, lex)
    n_err = sum(sum(v["errors"].values()) for v in report.values())
    n_seen = sum(v["n_seen"] for v in report.values())
    logger.info(f"{len(pairs)} clips ({len(hyp_by_key)} hypotheses): {n_seen} drug occurrences of "
                f"{len(report)} drugs, {n_err} misspelled, "
                f"{sum(v['n_deleted'] for v in report.values())} dropped")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
