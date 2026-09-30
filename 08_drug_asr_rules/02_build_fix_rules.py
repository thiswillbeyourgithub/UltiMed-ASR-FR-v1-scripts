#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "litellm", "tiktoken", "tqdm", "tenacity", "rapidfuzz"]
# ///
"""Turn the per-drug misspelling report of ``01_extract_drug_errors.py`` into ordered,
conservative regex fix rules for ASR output.

A variant becomes a rule only if rewriting it can hardly be wrong. It is REJECTED when:

- ``short``: under ``--min-len`` letters once folded (``isa``, ``lora``), or it holds a digit;
- ``real_text``: it occurs in a correct LABEL of the corpus (``--labels``, default the
  release-wide ``99_hf_release/data/NeMO_files/full.jsonl``), i.e. it is a real word or
  phrase someone wrote (``prednisone`` heard for ``prednisolone``, ``de mi`` ...);
- ``other_drug``: it is itself a word of the drug lexicon;
- ``ambiguous``: it stands for several drugs and none holds ``--min-share`` of its counts;
- ``imprecise``: in the hypotheses file, the variant appears more often than it was an
  error for that drug (precision = error count / all hypothesis occurrences, below
  ``--min-precision``), so the model also writes it where the label has something else;
- ``rare``: seen fewer than ``--min-count`` times;
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

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("extract_drug_errors", _HERE / "01_extract_drug_errors.py")
extract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract)
fold, tokens, load_lexicon = extract.fold, extract.tokens, extract.load_lexicon

DEFAULT_ERRORS = extract.DEFAULT_OUT
DEFAULT_OUT = _HERE / "drug_fix_rules.json"
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
    """Folded token tuple used to compare a variant with label / hypothesis n-grams."""
    return tuple(fold(t) for t in tokens(text))


def contains_target(variant: str, target: str) -> bool:
    """True when ``variant`` has ``target`` as a run of its words (hyphens split words).

    >>> contains_target("anti-TNF-alpha", "alpha"), contains_target("polyéthylène-glycol", "polyéthylène")
    (True, True)
    >>> contains_target("létoposide", "l'étoposide"), contains_target("mire tazapine", "mirtazapine")
    (False, False)
    """
    v, t = (tuple(p for w in _key(x) for p in w.split("-") if p) for x in (variant, target))
    return len(v) > len(t) and any(v[i:i + len(t)] == t for i in range(len(v) - len(t) + 1))


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


def build_rules(report: dict, hyps: list[str], labels, lex: dict,
                min_len: int = 5, min_count: int = 2, min_share: float = 0.9,
                min_precision: float = 0.8) -> tuple[list[dict], list[dict]]:
    """``(rules, rejected)`` from the 01 report, the hypotheses and the corpus labels.

    >>> report = {"mirtazapine": {"errors": {"mire tazapine": 2, "de mi": 1, "mirtazapinne": 1}, "targets": {}},
    ...           "étoposide": {"errors": {"létoposide": 2}, "targets": {"létoposide": "l'étoposide"}}}
    >>> hyps = ["la mire tazapine", "de mi", "mire tazapine", "létoposide", "létoposide", "mirtazapinne"]
    >>> rules, rej = build_rules(report, hyps, ["prendre de mi comprimé"], {"mirtazapine": None})
    >>> [(r["variant"], r["replacement"]) for r in rules]
    [('mire tazapine', 'mirtazapine'), ('létoposide', "l'étoposide")]
    >>> [(r["variant"], r["reason"]) for r in rej]
    [('mirtazapinne', 'rare'), ('de mi', 'short')]
    """
    # Folded variant -> {target: count}; the displayed variant is its most frequent spelling.
    by_key: dict[tuple, Counter] = defaultdict(Counter)
    spelling: dict[tuple, Counter] = defaultdict(Counter)
    drug_of: dict[tuple, dict[str, str]] = defaultdict(dict)
    for drug, entry in report.items():
        for variant, n in entry["errors"].items():
            k = _key(variant)
            target = entry.get("targets", {}).get(variant, drug)
            by_key[k][target] += n
            spelling[k][variant] += n
            drug_of[k][target] = drug
    label_hits = count_ngrams(labels, set(by_key))
    hyp_hits = count_ngrams(hyps, set(by_key))
    rules, rejected = [], []
    for k, targets in by_key.items():
        variant = spelling[k].most_common(1)[0][0]
        target, n = targets.most_common(1)[0]
        total = sum(targets.values())
        occ = hyp_hits[k]
        precision = n / occ if occ else 1.0
        reason = None
        if len("".join(k).replace("'", "")) < min_len or re.search(r"\d", variant):
            reason = "short"
        elif contains_target(variant, target):
            reason = "contains_target"
        elif len(k) == 1 and k[0] in lex:
            reason = "other_drug"
        elif label_hits[k]:
            reason = "real_text"
        elif n / total < min_share:
            reason = "ambiguous"
        elif precision < min_precision:
            reason = "imprecise"
        elif n < min_count:
            reason = "rare"
        row = {"variant": variant, "drug": drug_of[k][target], "count": n}
        if reason:
            rejected.append({**row, "reason": reason})
            continue
        rules.append({"pattern": variant_pattern(variant), "replacement": target, **row,
                      "hyp_occurrences": occ, "precision": round(precision, 3)})
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


def _read_texts(path: Path, field: str):
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)[field]


@click.command()
@click.argument("hyps", type=click.Path(exists=True, path_type=Path))
@click.option("--errors", default=str(DEFAULT_ERRORS), type=click.Path(exists=True, path_type=Path),
              show_default="08_drug_asr_rules/drug_asr_errors.json")
@click.option("--labels", "labels_paths", multiple=True, type=click.Path(exists=True, path_type=Path),
              help="Manifests whose `text` counts as correct text (repeatable). "
                   "Default: 99_hf_release/data/NeMO_files/full.jsonl")
@click.option("--out", default=str(DEFAULT_OUT), type=click.Path(path_type=Path),
              show_default="08_drug_asr_rules/drug_fix_rules.json")
@click.option("--min-len", default=5, show_default=True)
@click.option("--min-count", default=2, show_default=True,
              help="a variant seen once is as likely noise as a pattern (e.g. \"reea iutis aeec et are\" -> oméga)")
@click.option("--min-share", default=0.9, show_default=True)
@click.option("--min-precision", default=0.8, show_default=True)
def main(hyps: Path, errors: Path, labels_paths: tuple[Path, ...], out: Path, min_len: int,
         min_count: int, min_share: float, min_precision: float) -> None:
    """Write the ordered drug fix rules."""
    report = json.loads(errors.read_text(encoding="utf-8"))
    labels_paths = labels_paths or (DEFAULT_LABELS,)
    labels = (t for p in labels_paths for t in _read_texts(p, "text"))
    rules, rejected = build_rules(report, list(_read_texts(hyps, "hyp")), labels, load_lexicon(),
                                  min_len, min_count, min_share, min_precision)
    logger.info(f"{len(rules)} rules covering {sum(r['count'] for r in rules)} errors; rejected "
                f"{dict(Counter(r['reason'] for r in rejected))}")
    out.write_text(json.dumps({"rules": rules, "rejected": rejected}, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
