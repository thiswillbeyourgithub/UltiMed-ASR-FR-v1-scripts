#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "litellm", "tiktoken", "tqdm", "tenacity", "rapidfuzz", "wordfreq"]
# ///
# (dependencies: same list as 02_build_fix_rules.py, which this script imports; keep them in sync)
"""Merge the fix rules that write the same replacement into one alternation pattern.

A rule file lists one rule per variant, so ``drug_fix_rules.jsonl`` repeats the whole
pattern scaffolding for each of the variants of one drug. Here the rules of one replacement
(and one name guard) become a single rule, ``"variants": [...]`` with the pattern
``bounds(?:variant 1|variant 2|...)``, at the place of its first member, variants in file
order (longest first, so at one position the longest variant wins, like before).

Sequential and merged rules are not always equivalent: a short member now fires before the
rules that sat between it and its group's first member. So the merged file is checked on
every text given (hyps, labels, any outside text): a group whose merged rule fired on a text
that came out different is split back into its original rules, and the check repeats until
every text comes out exactly as with the original file.

    uv run 08_drug_asr_rules/03_merge_rules.py drug_fix_rules.jsonl merged.jsonl \\
        --hyps drugs_hyps.UltiMed.jsonl ... --labels ../99_hf_release/data/NeMO_files/full.jsonl ...
"""
import importlib.util
import json
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("b", _HERE / "02_build_fix_rules.py")
b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b)


def _prefix(r: dict) -> str:
    return (b._AFTER_TITLE if r.get("name") else "") + b._START


def merge_rules(rules: list[dict], keep: set[int] = frozenset()) -> list[tuple[list[int], dict]]:
    """``(member indices, rule)`` per merged group, in order; groups whose first index is in
    ``keep`` stay as their original rules.

    >>> rules = [{"variant": v, "pattern": b.variant_pattern(v), "replacement": t} for v, t in
    ...          [("mire taz apine", "mirtazapine"), ("primperan", "Primpéran"), ("myrtazapine", "mirtazapine")]]
    >>> merged = merge_rules(rules)
    >>> [(ix, r.get("variants")) for ix, r in merged]
    [([0, 2], ['mire taz apine', 'myrtazapine']), ([1], None)]
    >>> fix = b.compile_rules([r for _, r in merged])
    >>> fix("Mire-taz apine puis myrtazapine, primperan")
    'Mirtazapine puis mirtazapine, Primpéran'
    >>> [ix for ix, _ in merge_rules(rules, keep={0})]
    [[0], [1], [2]]
    """
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(rules):
        groups.setdefault((r["replacement"], bool(r.get("name"))), []).append(i)
    out = []
    for ix in groups.values():
        if len(ix) == 1 or ix[0] in keep:
            out += [([i], rules[i]) for i in ix]
            continue
        first, bodies = rules[ix[0]], []
        for i in ix:
            p, pre = rules[i]["pattern"], _prefix(rules[i])
            assert p.startswith(pre) and p.endswith(b._END), rules[i]
            bodies.append(p[len(pre):-len(b._END)])
        rule = {"pattern": _prefix(first) + "(?:" + "|".join(bodies) + ")" + b._END,
                "replacement": first["replacement"], "variants": [rules[i]["variant"] for i in ix],
                "count": sum(rules[i].get("count", 0) for i in ix)}
        if first.get("name"):
            rule["name"] = True
        out.append((ix, rule))
    out.sort(key=lambda g: g[0][0])
    return out


def merge_checked(rules: list[dict], texts: list[str]) -> list[dict]:
    """Merged rules giving exactly the outputs of ``rules`` on every text in ``texts``.

    Merged, ``bb`` would fire before ``bb cc`` and turn "bb cc" into "X cc", so X stays split:

    >>> rules = [{"variant": v, "pattern": b.variant_pattern(v), "replacement": t} for v, t in
    ...          [("aa bb", "X"), ("bb cc", "Y"), ("bb", "X"), ("zz", "Z"), ("zzz", "Z")]]
    >>> [r.get("variants", r.get("variant")) for r in merge_checked(rules, ["bb cc", "aa bb zz"])]
    ['aa bb', 'bb cc', 'bb', ['zz', 'zzz']]
    """
    fix = b.compile_rules(rules)
    want = {t: o for t in texts if (o := fix(t)) != t}  # untouched texts must stay untouched too
    keep: set[int] = set()
    while True:
        merged = merge_rules(rules, keep)
        mfix, bad = b.compile_rules([r for _, r in merged]), set()
        for t in texts:
            tr = []
            if mfix(t, tr) != want.get(t, t):
                bad |= {merged[j][0][0] for j in tr if len(merged[j][0]) > 1}
        if not bad:
            break
        logger.info(f"{len(bad)} merged groups change some text: split back")
        keep |= bad
    return [r for _, r in merged]


@click.command()
@click.argument("rules_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("out", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--hyps", "hyps_paths", multiple=True, type=click.Path(exists=True, path_type=Path), help="hyps jsonl (field hyp)")
@click.option("--labels", "labels_paths", multiple=True, type=click.Path(exists=True, path_type=Path),
              help="manifest jsonl (field text) or .txt (one sentence per line)")
def main(rules_path: Path, out: Path, hyps_paths: tuple[Path, ...], labels_paths: tuple[Path, ...]):
    rules = b.load_rules(rules_path)
    texts = sorted({t for p in hyps_paths for t in b._read_texts(p, "hyp")} | {t for p in labels_paths for t in b._read_texts(p, "text")})
    logger.info(f"{len(rules)} rules, checking on {len(texts)} unique texts")
    merged = merge_checked(rules, texts)
    with open(out, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"wrote {len(merged)} rules to {out} ({out.stat().st_size / 1e6:.1f} MB vs {rules_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
