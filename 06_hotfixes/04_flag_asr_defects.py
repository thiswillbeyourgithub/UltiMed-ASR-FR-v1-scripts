#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru"]
# ///
"""Flag clips whose audio does not say their label, from a fine-tuned Parakeet sweep.

Stage 06's Whisper QC misses one TTS failure: voxtral sometimes babbles a few words
BEFORE the sentence ("Jackie Orbad décide sa voix de tension à T. Courrier de suivi...",
"Pardon, Madame Petit. Prescription de..."), and Whisper silently skips that preamble,
so the clip scored clean. A Parakeet fine-tuned on UltiMed transcribes it faithfully.
Found on 2026-09-28 by transcribing val/test/PARROT with run 1.6.0 step 8000 (~14 such
clips in 116k), then the whole corpus.

Input: a hypotheses file ``{"audio": <path>, "hyp": <text>}`` per line, written by the
NeMo repo's ``perso/transcribe_manifests.py``, plus the manifests whose labels to check.
Only the last two path components (``dictionary/012095_0006_sonde_de_churet.flac``) are
compared, so the hypotheses may have been made from another checkout or symlink.

A clip is flagged when, aligning label and hypothesis word by word:

- ``preamble``: the hypothesis STARTS with >= 2 words the label does not have (a pure
  insertion at position 0, not a replacement of the label's first word). The label always starts at the start of the audio, so words
  before it are TTS babble, never a model error on a correct clip.
- ``skipped``: >= 3 real words (>= 3 letters, not a spelled number or code letter, which
  recognizers glue into one token) of one label stretch are missing from the hypothesis,
  with less than half of the stretch's letters left in its place (so a compound written
  as one word is not a skip), AND at
  least half of them are missing from the stage-06 Whisper transcript too
  (``stt_transcript``). Two independent recognizers agreeing that the audio lacks a
  phrase means the TTS skipped it; one alone can be a recognition error.

Mid-sentence misreads (``2 grammes 10 grands cents`` for ``2,10 grammes``) are NOT
flagged: they cannot be told apart from a Parakeet error without listening.

Both texts are compared after lowercasing and dropping punctuation, like the NeMo
normalizer, which is enough since only whole missing or extra words matter here.

Output (default ``99_hf_release/asr_flagged_clips.jsonl``, committed): one row per
flagged clip with its key, reason and the offending words, sorted by key so a re-run
diffs cleanly. ``99_hf_release/04_drop_bad_rows.py`` drops the listed clips.

    uv run 06_hotfixes/04_flag_asr_defects.py <hyps.jsonl> <manifest> [<manifest> ...]

This file was written by Claude Code.
"""
from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "utils"))
from nemo_manifest import clip_key, read_jsonl, resolve_audio  # noqa: E402

DEFAULT_OUT = _HERE.parent / "99_hf_release" / "asr_flagged_clips.jsonl"
PREAMBLE = "preamble"
SKIPPED = "skipped"
MIN_PREAMBLE_WORDS = 2
MIN_SKIPPED_WORDS = 3
_PUNCT = re.compile(r"[^\w\s']")


def words(text: str) -> list[str]:
    return _PUNCT.sub(" ", text.lower()).split()


# Spelled numbers, which a recognizer writes as digits (one token for several words).
_NUMBER_WORDS = frozenset("""zéro un une deux trois quatre cinq six sept huit neuf dix onze douze
treize quatorze quinze seize vingt vingts trente quarante cinquante soixante cent cents mille
virgule et""".split())


def _is_content(word: str) -> bool:
    """A word of >= 3 letters that is not a spelled number."""
    return len(word) >= 3 and word.isalpha() and word not in _NUMBER_WORDS


def _merged_diffs(ops: list[tuple]) -> list[tuple[int, int, int, int]]:
    """Adjacent non-equal opcodes merged into one ``(i1, i2, j1, j2)`` span: difflib may
    split one reworded stretch into an insert, one matching word, then a replace ("et
    typage hpa maternel et paternel" -> "étypage hpa maternelle" + "et" + "paternelle"),
    and the replace alone looks like 4 missing words. Spans at most one matching word
    apart are therefore merged."""
    spans: list[list[int]] = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        if spans and i1 - spans[-1][1] <= 1 and j1 - spans[-1][3] <= 1:
            spans[-1][1], spans[-1][3] = i2, j2
        else:
            spans.append([i1, i2, j1, j2])
    return [tuple(sp) for sp in spans]


def find_defect(label: str, hyp: str, stt: str | None) -> tuple[str, str] | None:
    """Return ``(reason, offending words)`` for a defective clip, else ``None``.

    >>> find_defect("Courrier de suivi.", "Jackie Orbad dit. Courrier de suivi.", None)
    ('preamble', 'jackie orbad dit')
    >>> find_defect("le patient est sorti hier soir", "le patient", "le patient")[0]
    'skipped'
    >>> find_defect("le patient est sorti hier soir", "le patient", "le patient est sorti hier soir") is None
    True
    >>> find_defect("marqueurs cd 11 cd 18 positifs", "marqueurs cd11 cd18 positifs", "marqueurs positifs") is None
    True
    >>> find_defect("une uvulo palato pharyngo plastie", "une uvulopalatopharyngoplastie", "une") is None
    True
    >>> find_defect("anticorps et typage hpa maternel et paternel ok", "anticorps étypage hpa maternelle et paternelle ok", "anticorps ok") is None
    True
    """
    ref, h = words(label), words(hyp)
    ops = difflib.SequenceMatcher(a=ref, b=h, autojunk=False).get_opcodes()
    tag, i1, i2, j1, j2 = ops[0]
    # A pure insertion only: a "replace" at the start is usually the model splitting the
    # label's first word ("Suvreza" -> "Essus UV réza"), not extra audio.
    if tag == "insert" and j2 - j1 >= MIN_PREAMBLE_WORDS:
        return PREAMBLE, " ".join(h[j1:j2])
    if not stt:
        return None
    # Label word indices Whisper's transcript does match.
    s_ops = difflib.SequenceMatcher(a=ref, b=words(stt), autojunk=False).get_opcodes()
    heard_by_whisper = {i for t, a1, a2, _, _ in s_ops if t == "equal" for i in range(a1, a2)}
    for i1, i2, j1, j2 in _merged_diffs(ops):
        if (i2 - i1) - (j2 - j1) < MIN_SKIPPED_WORDS:
            continue
        # Count only real words: a spelled code ("cd 11 cd 18", "p t 4 n 0") or number
        # ("deux mille vingt quatre") loses words just because a recognizer glues it
        # into one token ("CD11", "pT4N0", "2024"), with nothing missing from the audio.
        span = [i for i in range(i1, i2) if _is_content(ref[i])]
        # And most of the stretch's LETTERS must be gone too: "uvulo palato pharyngo
        # plastie" -> "uvulopalatopharyngoplastie" drops 3 words but no sound.
        glued = len("".join(h[j1:j2])) > len("".join(ref[i1:i2])) / 2
        if len(span) >= MIN_SKIPPED_WORDS and not glued:
            if sum(i in heard_by_whisper for i in span) < len(span) / 2:
                return SKIPPED, " ".join(ref[i1:i2])
    return None


@click.command()
@click.argument("hyps", type=click.Path(exists=True, path_type=Path))
@click.argument("manifests", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out", default=str(DEFAULT_OUT), show_default="99_hf_release/asr_flagged_clips.jsonl",
              type=click.Path(path_type=Path))
def main(hyps: Path, manifests: tuple[Path, ...], out: Path) -> None:
    """Flag TTS preambles and confirmed skips from a Parakeet hypotheses file."""
    hyp_by_key = {}
    for line in hyps.open():
        r = json.loads(line)
        hyp_by_key[clip_key(r["audio"])] = r["hyp"]
    flagged, n_checked = {}, 0
    for m in manifests:
        for row in read_jsonl(m):
            key = clip_key(resolve_audio(m.parent, row["audio_filepath"]))
            if key in flagged or key not in hyp_by_key:
                continue
            n_checked += 1
            defect = find_defect(row["text"], hyp_by_key[key], row.get("stt_transcript"))
            if defect:
                flagged[key] = {"clip": key, "reason": defect[0], "words": defect[1],
                                "text": row["text"], "hyp": hyp_by_key[key]}
    logger.info(f"checked {n_checked} clips ({len(hyp_by_key)} hypotheses), flagged {len(flagged)}: "
                f"{sum(f['reason'] == PREAMBLE for f in flagged.values())} {PREAMBLE}, "
                f"{sum(f['reason'] == SKIPPED for f in flagged.values())} {SKIPPED}")
    with out.open("w") as f:
        for key in sorted(flagged):
            f.write(json.dumps(flagged[key], ensure_ascii=False) + "\n")
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
