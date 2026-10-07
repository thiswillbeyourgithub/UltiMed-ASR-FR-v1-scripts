#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "litellm", "tiktoken", "tqdm", "tenacity", "rapidfuzz", "wordfreq"]
# ///
"""Drop the variants holding the word ``unk`` from a (merged or not) rules file.

Before 01_extract_drug_errors.py refused them, the decoder's ``<unk>`` token reached the
rules as the word ``unk`` ("unk sophagienne" -> oesophagienne). No runtime output matches
such a variant (onnx-asr glues ``<unk>`` to the next word, the web decoder deletes it), so
removing it changes no output. A merged rule keeps its other variants, its pattern rebuilt
from them exactly as 03_merge_rules.py builds it; a rule left with no variant goes.

    uv run 08_drug_asr_rules/04_drop_unk_variants.py IN.jsonl OUT.jsonl [--labels L.jsonl ...]

``--labels`` also drops the variants the fixed ``real_text`` guard of 02 (2026-10-07) would
have rejected: a last word French elides ("part d") that a label holds with its apostrophe
("fait part d'un"); the end bound lets such a rule fire right before the apostrophe.

This file was written by Claude Code.
"""
import importlib.util
import json
import re
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("b", _HERE / "02_build_fix_rules.py")
b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b)
_mspec = importlib.util.spec_from_file_location("m", _HERE / "03_merge_rules.py")
m = importlib.util.module_from_spec(_mspec)
_mspec.loader.exec_module(m)


def has_unk(variant: str) -> bool:
    """True when ``unk`` is one of the variant's words.

    >>> [has_unk(v) for v in ("unk sophagienne", "d unk orsum", "funkster", "Unk")]
    [True, True, False, True]
    """
    return "unk" in re.split(r"[\s-]+", variant.lower())


def drop_unk_variants(rules: list[dict], bad=has_unk) -> tuple[list[dict], int]:
    """``(rules without the variants ``bad`` flags (default: unk ones), variants dropped)``.

    >>> mk = lambda vs, t: m.merge_rules([{"variant": v, "pattern": b.variant_pattern(v), "replacement": t} for v in vs])[0][1]
    >>> rules = [mk(["unk sophagienne", "esophagienne"], "oesophagienne"), mk(["cephazoline unk"], "céfazoline")]
    >>> out, n = drop_unk_variants(rules)
    >>> n, [(r["variants"], r["replacement"]) for r in out]
    (2, [(['esophagienne'], 'oesophagienne')])
    >>> out[0]["pattern"] == b.variant_pattern("esophagienne")
    True
    """
    out, dropped = [], 0
    for r in rules:
        vs = r.get("variants") or [r.get("variant")]
        keep = [v for v in vs if v is None or not bad(v)]
        dropped += len(vs) - len(keep)
        if len(keep) == len(vs):
            out.append(r)
            continue
        if not keep:
            continue
        pre = m._prefix(r)
        body = lambda v: b.variant_pattern(v)[len(b._START):-len(b._END)]
        build = lambda vs: pre + "(?:" + "|".join(body(v) for v in vs) + ")" + b._END
        # Only rebuild what the same code would have built; anything else stays untouched.
        assert build(vs) == r["pattern"] or (len(vs) == 1 and pre + body(vs[0]) + b._END == r["pattern"]), r
        r = dict(r, pattern=build(keep) if len(keep) > 1 else pre + body(keep[0]) + b._END)
        if "variants" in r:
            r["variants"] = keep
        out.append(r)
    return out, dropped


def elided_in_labels(rules: list[dict], labels) -> set[str]:
    """The variants ending in an elidable letter that a label holds with its apostrophe, which
    ``02_build_fix_rules.py`` rejects as ``real_text`` since 2026-10-07 but older files kept.

    >>> rules = [{"variants": ["part d", "pardet"]}, {"variant": "etoposie d"}]
    >>> sorted(elided_in_labels(rules, ["il nous fait part d'un stress", "etoposie orale"]))
    ['part d']
    """
    vs = {v for r in rules for v in (r.get("variants") or [r.get("variant")]) if v}
    keys = {v: b._elided(b._key(v)) for v in vs}
    hits = b.count_ngrams(labels, {k for v, k in keys.items() if k != b._key(v)})
    return {v for v, k in keys.items() if hits[k]}


@click.command()
@click.argument("src", type=click.Path(exists=True, path_type=Path))
@click.argument("dst", type=click.Path(path_type=Path))
@click.option("--labels", "labels_paths", multiple=True, type=click.Path(exists=True, path_type=Path),
              help="also drop the variants ending in an elided letter these labels hold (manifest jsonl or .txt)")
def main(src: Path, dst: Path, labels_paths: tuple[Path, ...]) -> None:
    rules = b.load_rules(src)
    real = elided_in_labels(rules, [t for p in labels_paths for t in b._read_texts(p, "text")]) if labels_paths else set()
    if real:
        logger.info(f"{len(real)} variants with an elided last letter occur in a label: {sorted(real)[:20]}")
    out, dropped = drop_unk_variants(rules, lambda v: has_unk(v) or v in real)
    with dst.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    logger.info(f"{src}: dropped {dropped} variants, {len(rules)} -> {len(out)} rules")


if __name__ == "__main__":
    main()
