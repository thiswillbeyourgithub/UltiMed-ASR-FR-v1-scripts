"""Stdlib tests for utils/text_chunking: paren strip, table strip, chunking.

Run: python tests/test_text_chunking.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from text_chunking import (  # noqa: E402
    DEFAULT_MAX_CHARS,
    DEFAULT_MIN_CHARS,
    chunk_text,
    strip_parens,
    strip_table_markup,
)


def test_strip_parens_nested():
    assert strip_parens("le foie (hypertrophie (modérée) diffuse) est normal") == (
        "le foie est normal"
    )


def test_strip_parens_keeps_newlines():
    out = strip_parens("ligne un (aparte)\n\nligne deux")
    assert out == "ligne un\n\nligne deux"


def test_strip_parens_unbalanced():
    assert strip_parens("valeur 12 mm ) suite") == "valeur 12 mm suite"


def test_chunk_short_stays_one():
    text = "Petite phrase medicale.\n\nAutre phrase courte."
    chunks = chunk_text(text, max_chars=4000)
    assert len(chunks) == 1
    assert "\n" not in chunks[0]
    assert chunks[0] == "Petite phrase medicale. Autre phrase courte."


def test_chunk_packs_paragraphs_under_cap():
    paras = "\n\n".join(f"Paragraphe numero {i} de taille moyenne." for i in range(10))
    # min_chars=0 to test packing in isolation (the tiny cap is far below the
    # production 200-char merge floor, which would otherwise fold every chunk).
    chunks = chunk_text(paras, max_chars=80, min_chars=0)
    assert len(chunks) > 1
    assert all(len(c) <= 80 for c in chunks)
    # nothing dropped: every paragraph index survives somewhere
    joined = " ".join(chunks)
    assert all(f"numero {i} " in joined for i in range(10))


def test_chunk_splits_long_paragraph_on_sentences():
    long_para = " ".join(f"Phrase {i} complete." for i in range(50))
    chunks = chunk_text(long_para, max_chars=100, min_chars=0)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_strip_table_markup_drops_ruler_and_border_lines():
    text = (
        "Panel genetique\n"
        "+----------+--------+\n"
        "| Gene     | Statut |\n"
        "==========\n"
        "Conclusion normale.\n"
    )
    out = strip_table_markup(text)
    lines = [ln.strip() for ln in out.split("\n") if ln.strip()]
    assert len(lines) == 3  # the two border lines and the === ruler are dropped
    assert lines[0] == "Panel genetique"
    assert lines[-1] == "Conclusion normale."
    assert "Gene" in lines[1] and "Statut" in lines[1]
    assert "|" not in out
    assert "+---" not in out and "====" not in out


def test_strip_table_markup_collapses_inline_ruler_keeps_prose():
    # A 4+ dash run inside a textual line is an inline ruler -> space; a lone
    # underscore / ">" inside content is left untouched (handled downstream).
    text = "Technique CSTS_B_v1 : ----------- transcript NM c.2573T>G analyse."
    out = strip_table_markup(text)
    assert "-----" not in out
    assert "CSTS_B_v1" in out  # single underscore preserved
    assert "c.2573T>G" in out  # genetics notation preserved


def test_strip_table_markup_keeps_normal_hyphenated_prose():
    text = "Angle cervico-femoral gauche a 91 degres.\n- Micro-saignement note."
    assert strip_table_markup(text) == text  # single hyphens are not rulers


def _words(n: int) -> str:
    """A whitespace-safe filler paragraph of length >= n (grows by whole words)."""
    s = "mot"
    while len(s) < n:
        s += " mot"
    return s


def test_chunk_rebalances_instead_of_breaching_the_cap():
    # The real case: a body paragraph packed so close to the cap that the 24-char
    # tail after it cannot be folded in. Note the packer already joins whatever
    # fits, so a surviving fragment is by definition one the cap forbids merging.
    # One chunk is one spoken clip and the ASR trainer silently drops clips over
    # its max_duration, so the cap wins: the body donates text to the fragment
    # instead of swallowing it.
    body = _words(3994)
    tail = "Pertes sanguines 300 cc."
    assert len(tail) < DEFAULT_MIN_CHARS
    text = body + "\n\n" + tail
    # min_chars=0 leaves the tail as its own stray clip; the floor rebalances it.
    unmerged = chunk_text(text, max_chars=4000, min_chars=0)
    assert len(unmerged) == 2 and unmerged[-1] == tail
    chunks = chunk_text(text, max_chars=4000, min_chars=DEFAULT_MIN_CHARS)
    assert len(chunks) == 2
    assert all(len(c) <= 4000 for c in chunks)
    assert all(len(c) >= DEFAULT_MIN_CHARS for c in chunks)  # no stray fragment
    assert chunks[-1].endswith(tail)
    # Rebalanced, not merely re-packed: the two are within a word of even.
    assert abs(len(chunks[0]) - len(chunks[1])) < 30
    assert " ".join(chunks).count("mot") == body.count("mot")  # nothing lost


def test_chunk_cap_never_exceeded_on_a_run_of_fragments():
    # A long run of below-floor fragments must not snowball past the cap.
    text = "\n\n".join(["Fin." for _ in range(200)])
    chunks = chunk_text(text, max_chars=60, min_chars=50)
    assert chunks, "fragments must not be dropped"
    assert all(len(c) <= 60 for c in chunks)
    assert "".join(chunks).count("Fin.") == 200  # nothing lost in the merge


def test_default_cap_fits_the_trainer_duration_ceiling():
    # The cap is a CLIP-LENGTH cap: NeMo's train_ds.max_duration is 45.0 s and
    # measured stage-05 audio speaks a raw chunk at ~0.065 s per character, so a
    # cap much above ~600 chars starts producing clips the trainer throws away.
    assert DEFAULT_MAX_CHARS * 0.065 <= 45.0
    assert DEFAULT_MIN_CHARS < DEFAULT_MAX_CHARS


def test_chunk_single_short_doc_stays_one():
    # A whole document below the floor has no neighbour: it is returned as-is,
    # not dropped.
    chunks = chunk_text("Court compte rendu.", max_chars=4000)
    assert chunks == ["Court compte rendu."]


def test_chunk_drops_table_end_to_end():
    text = (
        "Compte rendu.\n"
        "+------+------+\n"
        "| A    | B    |\n"
        "+------+------+\n"
        "Fin du document."
    )
    chunks = chunk_text(text, max_chars=4000, min_chars=0)
    joined = " ".join(chunks)
    assert "|" not in joined and "+---" not in joined
    assert "Compte rendu." in joined and "Fin du document." in joined


def test_chunk_empty():
    assert chunk_text("   \n\n  ") == []


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run()
