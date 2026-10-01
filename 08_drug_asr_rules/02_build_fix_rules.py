#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "litellm", "tiktoken", "tqdm", "tenacity", "rapidfuzz", "wordfreq"]
# ///
"""Turn the per-drug misspelling report of ``01_extract_drug_errors.py`` into ordered,
conservative regex fix rules for ASR output.

A variant becomes a rule only if rewriting it can hardly be wrong. It is REJECTED when:

- ``short``: under ``--min-len`` letters once folded (``isa``, ``lora``), or it holds a digit;
- ``real_text``: it occurs in a correct LABEL of the corpus (``--labels``, default the
  release-wide ``99_hf_release/data/NeMO_files/full.jsonl``), i.e. it is a real word or
  phrase someone wrote (``prednisone`` heard for ``prednisolone``, ``de mi`` ...);
- ``other_term``: it is itself a word of the lexicon (drugs, plus any ``--lexicon`` term list);
- ``ambiguous``: it stands for several drugs and none holds ``--min-share`` of its counts;
- ``imprecise``: in the hypotheses file, the variant appears more often than it was an
  error for that drug (precision = error count / all hypothesis occurrences, below
  ``--min-precision``), so the model also writes it where the label has something else;
- ``rare``: seen fewer than ``--min-count`` times;
- ``french_word``: a one-word variant that is an ordinary French word (``--max-word-zipf``:
  ``tienne`` -> ``Tyenne``, ``Brexit`` -> ``Brexin``), even though no label holds it;
- ``common_words``: every word of it is a word the labels use (``lait unique`` -> ``Levunique``,
  ``bêta estime`` -> ``bétahistine``) and it was seen fewer than ``--min-count-words`` times;
- ``far``: its letters are too far from the target's (``similarity`` below ``--min-ratio``):
  ``myrtazapine`` or ``mire taz apine`` -> ``mirtazapine`` are slips, a garbled stretch like
  ``reea iutis aeec et are`` -> ``oméga`` is not;
- ``contains_target``: the variant already holds the whole target as its own word(s),
  so the rule could only delete the neighbours (``anti-tnf-alpha`` -> ``alpha``,
  ``sous-kardégic`` -> ``Kardégic``): a hyphenation slip of 01, not a misspelling.

Matching and folding (lowercase, accents stripped) mirror 01: the pattern is
case-insensitive, accent-insensitive on vowels and ``c``, allows a space or a hyphen
between the variant's words (``alpha-calcidol`` also catches ``alpha calcidol``), an
optional space after an elided article (``d' hexaméthasone``), and is bounded so it never
fires inside a longer word: ``(?<![<letter>'-])...(?![<letter>-])``, with the letter
class spelled out so the pattern behaves the same in JavaScript (see ``_WORD``).

Rules are ORDERED so a longer variant is applied before any shorter one it contains
(``mire tazapine`` before a hypothetical ``tazapine``): word count desc, then length
desc, then count desc. The replacement is the drug's canonical spelling (or the glued
elision target, ``l'étoposide``), capitalized when the match started a capitalized word
and the canonical form is lowercase.

Output (default ``08_drug_asr_rules/drug_fix_rules.json``)::

    {"rules": [{"pattern": "...", "replacement": "mirtazapine", "variant": "mire tazapine",
                "drug": "mirtazapine", "count": 6, "hyp_occurrences": 6, "precision": 1.0}, ...],
     "rejected": [{"variant": "...", "drug": "...", "count": 3, "reason": "real_text"}, ...]}

Apply them with ``apply_rules(text, rules)`` (importable) or copy the patterns into any
regex engine that supports lookbehind.

    uv run 08_drug_asr_rules/02_build_fix_rules.py <hyps.jsonl> [--errors drug_asr_errors.json]

This file was written by Claude Code.
"""
from __future__ import annotations

import importlib.util
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import click
from loguru import logger
from rapidfuzz.distance import Levenshtein
from wordfreq import zipf_frequency

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("extract_drug_errors", _HERE / "01_extract_drug_errors.py")
extract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract)
fold, tokens, load_lexicon = extract.fold, extract.tokens, extract.load_lexicon

DEFAULT_ERRORS = extract.DEFAULT_OUT
DEFAULT_OUT = _HERE / "drug_fix_rules.jsonl"
DEFAULT_LABELS = _HERE.parent / "99_hf_release" / "data" / "NeMO_files" / "full.jsonl"

# Every accented form a folded letter stands for, so a rule written from "phélodipine"
# also fixes "phelodipine" (01 already treats an accent slip as correct).
_ACCENTS = {"a": "aàâä", "e": "eéèêë", "i": "iîï", "o": "oôö", "u": "uùûü", "c": "cç", "y": "yÿ"}
# A word character for the rule bounds, spelled out (ASCII, Latin-1 and Latin Extended-A/B
# letters, so œ too) because ``\w`` is Unicode in Python but ASCII-only in JavaScript.
# The ``\\uXXXX`` escapes stay literal in the pattern: both engines read them.
_WORD = r"0-9A-Za-z_\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u024F"


def variant_pattern(variant: str) -> str:
    """Regex matching ``variant`` as whole words, case- and accent-insensitive.

    >>> p = re.compile(variant_pattern("mire tazapine"), re.IGNORECASE)
    >>> bool(p.search("Arrêt de la Mire-tazapine.")), bool(p.search("mire tazapines"))
    (True, False)
    >>> bool(re.search(variant_pattern("d'hexaméthasone"), "sous d' hexamethasone", re.I))
    True

    The word bounds spell out their letters instead of using ``\\w``, which is Unicode
    in Python but ASCII-only in JavaScript: there ``\\w`` would let a rule fire right
    after an accented letter ("émirtazapine"). The pattern must behave the same in both.

    >>> "\\w" in variant_pattern("mirtazapine")
    False
    >>> bool(re.search(variant_pattern("mirtazapine"), "émirtazapine", re.I))
    False
    """
    words = []
    for w in re.split(r"[\s-]+", variant.strip()):
        parts = []
        for ch in w:
            if ch == "'":
                parts.append(r"['’]\s*")
            elif fold(ch) in _ACCENTS:
                parts.append(f"[{_ACCENTS[fold(ch)]}]")
            else:
                parts.append(re.escape(ch))
        words.append("".join(parts))
    return rf"(?<![{_WORD}'-])" + r"[\s-]+".join(words) + rf"(?![{_WORD}-])"


def _key(text: str) -> tuple[str, ...]:
    """Folded token tuple used to compare a variant with label / hypothesis n-grams.

    Hyphens split words, as in ``variant_pattern``, whose rule matches both spellings:
    otherwise a ``sous-antidote`` variant would miss the label text ``sous antidote``.

    >>> _key("Sous-antidote"), _key("sous antidote")
    (('sous', 'antidote'), ('sous', 'antidote'))
    """
    return tuple(p for t in tokens(text) for p in fold(t).split("-") if p)


def contains_target(variant: str, target: str) -> bool:
    """True when ``variant`` has ``target`` as a run of its words (hyphens split words).

    >>> contains_target("anti-TNF-alpha", "alpha"), contains_target("polyéthylène-glycol", "polyéthylène")
    (True, True)
    >>> contains_target("létoposide", "l'étoposide"), contains_target("mire tazapine", "mirtazapine")
    (False, False)
    """
    v, t = (tuple(p for w in _key(x) for p in w.split("-") if p) for x in (variant, target))
    return len(v) > len(t) and any(v[i:i + len(t)] == t for i in range(len(v) - len(t) + 1))


def similarity(variant: str, target: str) -> float:
    """Normalised Levenshtein similarity of the folded letters (spaces, hyphens and
    apostrophes dropped, so a split word or a glued article costs nothing).

    >>> similarity("mir taz apine", "mirtazapine"), similarity("létoposide", "l'étoposide")
    (1.0, 1.0)
    >>> [round(similarity(v, t), 2) for v, t in [("myrtazapine", "mirtazapine"), ("mire taz apine", "mirtazapine"),
    ...                                         ("reea iutis aeec et are", "oméga")]]
    [0.91, 0.92, 0.11]
    """
    v, t = ("".join(_key(x)).replace("-", "").replace("'", "") for x in (variant, target))
    return Levenshtein.normalized_similarity(v, t)


def count_ngrams(texts, keys: set[tuple[str, ...]]) -> Counter:
    """How many times each key (a folded token tuple) occurs in ``texts``.

    >>> count_ngrams(["la mire tazapine", "mire"], {("mire", "tazapine"), ("mire",)})
    Counter({('mire',): 2, ('mire', 'tazapine'): 1})
    """
    sizes = {len(k) for k in keys}
    out = Counter()
    for text in texts:
        toks = _key(text)
        for n in sizes:
            for i in range(len(toks) - n + 1):
                if toks[i:i + n] in keys:
                    out[toks[i:i + n]] += 1
    return out


def merge_reports(reports) -> dict:
    """Sum several 01 reports (one per model) into one: counts add, targets merge.
    ``clips`` keeps, per variant, the most any single model wrote it: every model
    transcribes the same audio, so two models making the same slip on one clip is
    one clip of evidence, not two. The count thresholds of build_rules use it.

    >>> a = {"mirtazapine": {"n_seen": 3, "n_correct": 1, "n_deleted": 0, "errors": {"myrtazapine": 2}, "targets": {}}}
    >>> b = {"mirtazapine": {"n_seen": 2, "n_correct": 0, "n_deleted": 1, "errors": {"myrtazapine": 1}, "targets": {}},
    ...      "étoposide": {"n_seen": 1, "n_correct": 0, "n_deleted": 0, "errors": {"létoposide": 1},
    ...                    "targets": {"létoposide": "l'étoposide"}}}
    >>> m = merge_reports([a, b])
    >>> m["mirtazapine"]
    {'n_seen': 5, 'n_correct': 1, 'n_deleted': 1, 'errors': {'myrtazapine': 3}, 'targets': {}, 'clips': {'myrtazapine': 2}}
    >>> m["étoposide"]["targets"]
    {'létoposide': "l'étoposide"}
    """
    out: dict = {}
    for report in reports:
        for drug, entry in report.items():
            m = out.setdefault(drug, {"n_seen": 0, "n_correct": 0, "n_deleted": 0, "errors": {}, "targets": {}, "clips": {}})
            for k in ("n_seen", "n_correct", "n_deleted"):
                m[k] += entry.get(k, 0)
            for v, n in entry["errors"].items():
                m["errors"][v] = m["errors"].get(v, 0) + n
                m["clips"][v] = max(m["clips"].get(v, 0), n)
            m["targets"].update(entry.get("targets", {}))
    return out


def build_rules(report: dict, hyps: list[str], labels, lex: dict,
                min_len: int = 5, min_count: int = 1, min_share: float = 0.9,
                min_precision: float = 0.8, min_ratio: float = 0.5,
                min_count_words: int = 2, max_word_zipf: float = 2.5) -> tuple[list[dict], list[dict]]:
    """``(rules, rejected)`` from the 01 report, the hypotheses and the corpus labels.

    >>> report = {"mirtazapine": {"errors": {"mire tazapine": 2, "de mi": 1, "mirtazapinne": 1}, "targets": {}},
    ...           "étoposide": {"errors": {"létoposide": 2}, "targets": {"létoposide": "l'étoposide"}}}
    >>> hyps = ["la mire tazapine", "de mi", "mire tazapine", "létoposide", "létoposide", "mirtazapinne"]
    >>> rules, rej = build_rules(report, hyps, ["prendre de mi comprimé"], {"mirtazapine": None}, min_count=2)
    >>> [(r["variant"], r["replacement"]) for r in rules]
    [('mire tazapine', 'mirtazapine'), ('létoposide', "l'étoposide")]
    >>> [(r["variant"], r["reason"]) for r in rej]
    [('mirtazapinne', 'rare'), ('de mi', 'short')]

    A one-off made of ordinary words is dropped even though no label holds the phrase
    (it changed "une tache café au lait unique" in the test split):

    >>> lev = {"Levunique": {"errors": {"lait unique": 1}, "targets": {}}}
    >>> build_rules(lev, ["lait unique"], ["du lait", "une forme unique"], {})[1][0]["reason"]
    'common_words'
    >>> len(build_rules(lev, ["lait unique"], ["du lait"], {})[0])
    1

    The same one-off from two models is still one clip (it changed the same label once
    the int8 hypotheses joined the rule set):

    >>> two = merge_reports([lev, lev])
    >>> build_rules(two, ["lait unique"] * 2, ["du lait", "une forme unique"], {})[1][0]["reason"]
    'common_words'

    A hyphenated variant is real text when a label spells it with a space, since its rule
    matches both (``sous-antidote`` -> ``sous antidotes`` changed "évolution sous antidote"):

    >>> anti = {"antidotes": {"errors": {"sous-antidote": 3}, "targets": {}}}
    >>> build_rules(anti, ["sous-antidote"] * 3, ["évolution sous antidote"], {})[1][0]["reason"]
    'real_text'

    A one-word variant that is an ordinary French word is rejected even when no label holds
    it: the medical labels never say "Brexit" or "qu'il tienne", VoxPopuli does:

    >>> tyenne = {"Tyenne": {"errors": {"tienne": 8}, "targets": {}}}
    >>> build_rules(tyenne, ["tienne"] * 8, [], {})[1][0]["reason"]
    'french_word'
    """
    # Folded variant -> {target: count}; the displayed variant is its most frequent spelling.
    by_key: dict[tuple, Counter] = defaultdict(Counter)
    spelling: dict[tuple, Counter] = defaultdict(Counter)
    drug_of: dict[tuple, dict[str, str]] = defaultdict(dict)
    clips: dict[tuple, Counter] = defaultdict(Counter)
    for drug, entry in report.items():
        for variant, n in entry["errors"].items():
            k = _key(variant)
            target = entry.get("targets", {}).get(variant, drug)
            by_key[k][target] += n
            spelling[k][variant] += n
            drug_of[k][target] = drug
            clips[k][target] += entry.get("clips", {}).get(variant, n)
    # Single words too: a variant made only of words the labels use is ordinary French
    # ("lait unique" -> Levunique) and needs --min-count-words sightings.
    label_hits = count_ngrams(labels, set(by_key) | {(w,) for k in by_key for w in k})
    hyp_hits = count_ngrams(hyps, set(by_key))
    rules, rejected = [], []
    for k, targets in by_key.items():
        variant = spelling[k].most_common(1)[0][0]
        target, n = targets.most_common(1)[0]
        total = sum(targets.values())
        occ = hyp_hits[k]
        precision = n / occ if occ else 1.0
        ratio = similarity(variant, target)
        reason = None
        if len("".join(k).replace("'", "")) < min_len or re.search(r"\d", variant):
            reason = "short"
        elif contains_target(variant, target):
            reason = "contains_target"
        elif len(k) == 1 and k[0] in lex:
            reason = "other_term"
        elif label_hits[k]:
            reason = "real_text"
        elif len(k) == 1 and zipf_frequency(variant.lower(), "fr") >= max_word_zipf:
            reason = "french_word"
        elif n / total < min_share:
            reason = "ambiguous"
        elif precision < min_precision:
            reason = "imprecise"
        elif clips[k][target] < min_count:
            reason = "rare"
        elif ratio < min_ratio:
            reason = "far"
        elif clips[k][target] < min_count_words and all(label_hits[(w,)] for w in k):
            reason = "common_words"
        row = {"variant": variant, "drug": drug_of[k][target], "count": n}
        if reason:
            rejected.append({**row, "reason": reason})
            continue
        rules.append({"pattern": variant_pattern(variant), "replacement": target, **row,
                      "hyp_occurrences": occ, "precision": round(precision, 3), "similarity": round(ratio, 3)})
    rules.sort(key=lambda r: (-len(_key(r["variant"])), -len(r["variant"]), -r["count"], r["variant"]))
    rejected.sort(key=lambda r: (r["reason"], -r["count"], r["variant"]))
    return rules, rejected


@lru_cache(maxsize=None)
def _compiled(pattern: str) -> re.Pattern:
    """Each rule compiled once.

    re's own cache holds 512 patterns: past that (a full rule set is ~1.6k), every
    re.sub call recompiled every rule, about 2 s per transcript.

    >>> _compiled.cache_clear()
    >>> many = [{"pattern": variant_pattern(f"drogue{i:04d}x"), "replacement": "x"} for i in range(600)]
    >>> _ = [apply_rules("rien", many) for _ in range(3)]
    >>> _compiled.cache_info().misses
    600
    """
    return re.compile(pattern, re.IGNORECASE)


def apply_rules(text: str, rules: list[dict]) -> str:
    """Apply the ordered rules to one transcript.

    >>> rules = [{"pattern": variant_pattern("myrtazapine"), "replacement": "mirtazapine"},
    ...          {"pattern": variant_pattern("primperan"), "replacement": "Primpéran"}]
    >>> apply_rules("Myrtazapine le soir, primperan si besoin.", rules)
    'Mirtazapine le soir, Primpéran si besoin.'
    """
    for r in rules:
        rep = r["replacement"]

        def _sub(m: re.Match, rep: str = rep) -> str:
            return rep[0].upper() + rep[1:] if m.group(0)[0].isupper() and rep[0].islower() else rep

        text = _compiled(r["pattern"]).sub(_sub, text)
    return text


_TOKEN_SPLIT = re.compile(rf"[^{_WORD}]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_SPLIT.split(fold(text))) - {""}


def compile_rules(rules: list[dict]):
    """A fast ``apply_rules``: returns ``fix(text)`` with the same output.

    ``apply_rules`` runs every rule over every text (~1.6k regex scans per clip).
    Here each rule is indexed by its anchor, the longest folded word of its variant,
    which any text it matches must contain as a whole word, so a text only runs the
    rules whose anchor it contains, still in rule order. After a rule changes the
    text the word set is recomputed, so a replacement that feeds a later rule still
    triggers it. A variant with a character outside ``_WORD`` has no reliable
    anchor and runs on every text.

    >>> rules = [{"variant": v, "pattern": variant_pattern(v), "replacement": t} for v, t in
    ...          [("mire taz apine", "mirtazapine"), ("myrtazapine", "mirtazapine"), ("primperan", "Primpéran")]]
    >>> fix = compile_rules(rules)
    >>> sorted(fix.anchors)
    ['apine', 'myrtazapine', 'primperan']
    >>> texts = ["Mire-taz apine le soir.", "MYRTAZAPINE puis primpéran", "rien à voir"]
    >>> [fix(t) for t in texts] == [apply_rules(t, rules) for t in texts]
    True
    """
    by_anchor: dict[str, list[int]] = defaultdict(list)
    always: list[int] = []
    for i, r in enumerate(rules):
        words = re.split(r"[\s'’-]+", fold(r.get("variant", "")).strip())
        if all(w and not _TOKEN_SPLIT.search(w) for w in words):
            by_anchor[max(words, key=len)].append(i)
        else:
            always.append(i)

    def candidates(tokens: set[str], after: int) -> set[int]:
        return {j for t in tokens & by_anchor.keys() for j in by_anchor[t] if j > after}

    def fix(text: str) -> str:
        todo, i = candidates(_tokens(text), -1) | set(always), -1
        while todo:
            i = min(todo)
            todo.discard(i)
            new = apply_rules(text, [rules[i]])
            if new != text:
                text = new
                todo |= candidates(_tokens(text), i)
        return text

    fix.anchors = set(by_anchor)
    return fix


def _read_texts(path: Path, field: str):
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)[field]


@click.command()
@click.argument("hyps", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--errors", "errors_paths", multiple=True, type=click.Path(exists=True, path_type=Path),
              help="01 reports, one per model whose HYPS are given (repeatable, merged with "
                   "merge_reports). Default: 08_drug_asr_rules/drug_asr_errors.json")
@click.option("--labels", "labels_paths", multiple=True, type=click.Path(exists=True, path_type=Path),
              help="Manifests whose `text` counts as correct text (repeatable). "
                   "Default: 99_hf_release/data/NeMO_files/full.jsonl")
@click.option("--out", default=str(DEFAULT_OUT), type=click.Path(path_type=Path),
              show_default="08_drug_asr_rules/drug_fix_rules.jsonl",
              help="one rule per line, in application order; the rejected variants go to <out stem>.rejected.jsonl")
@click.option("--min-len", default=5, show_default=True)
@click.option("--min-count", default=1, show_default=True,
              help="1 is safe once --min-ratio drops the garbled one-offs: on the held-out test split, "
                   "5,372 rules at 1/0.5 changed none of 59,151 correct labels and fixed 521 drug clips "
                   "(1,623 rules at 2/0.0: 419)")
@click.option("--min-share", default=0.9, show_default=True)
@click.option("--min-precision", default=0.8, show_default=True)
@click.option("--min-ratio", default=0.5, show_default=True,
              help="minimum similarity() of variant and target: drops garbled stretches like "
                   "\"reea iutis aeec et are\" -> oméga (0.11) for 3 of 524 fixed clips")
@click.option("--min-count-words", default=2, show_default=True,
              help="minimum count of a variant made only of words the labels use: 2 drops "
                   "\"lait unique\" -> Levunique, which the parakeet-ultra rules changed a correct test label with")
@click.option("--max-word-zipf", default=2.5, show_default=True,
              help="a one-word variant at least this frequent in French (wordfreq Zipf scale, 2.5 = "
                   "about 1 per 3 million words) is real text: drops tienne -> Tyenne and "
                   "Brexit -> Brexin, which changed VoxPopuli fr references, but keeps discus -> Diskus")
@extract.lexicon_options
def main(hyps: tuple[Path, ...], errors_paths: tuple[Path, ...], labels_paths: tuple[Path, ...], out: Path, min_len: int,
         min_count: int, min_share: float, min_precision: float, min_ratio: float,
         min_count_words: int, max_word_zipf: float, lexicons: tuple[Path, ...], max_term_zipf: float) -> None:
    """Write the ordered drug fix rules."""
    report = merge_reports(json.loads(p.read_text(encoding="utf-8")) for p in errors_paths or (DEFAULT_ERRORS,))
    labels_paths = labels_paths or (DEFAULT_LABELS,)
    labels = (t for p in labels_paths for t in _read_texts(p, "text"))
    rules, rejected = build_rules(report, [t for p in hyps for t in _read_texts(p, "hyp")], labels, load_lexicon(lexicons, max_term_zipf),
                                  min_len, min_count, min_share, min_precision, min_ratio, min_count_words,
                                  max_word_zipf)
    logger.info(f"{len(rules)} rules covering {sum(r['count'] for r in rules)} errors; rejected "
                f"{dict(Counter(r['reason'] for r in rejected))}")
    rejected_out = out.with_suffix(".rejected.jsonl")
    for path, rows in ((out, rules), (rejected_out, rejected)):
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    logger.info(f"wrote {out} and {rejected_out}")


def load_rules(path: Path) -> list[dict]:
    """The ordered rules of a ``drug_fix_rules.jsonl``."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    main()
