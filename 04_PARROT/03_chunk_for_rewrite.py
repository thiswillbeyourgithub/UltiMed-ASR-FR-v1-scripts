#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "click",
# ]
# ///
"""Turn the French PARROT reports into rewrite-ready chunks for the LLM stage.

PARROT twin of ``03_PARHAF/03_chunk_for_rewrite.py``: the input preprocessor for
the PARROT rewrite stage (``04_generate_texts.py``). Unlike
``02_clean_split_texts.py`` (now audit-only), it keeps the raw report text intact,
only stripping parenthesised spans and splitting each report into coherent chunks
capped at ``--max-chars``. The LLM then rewrites each chunk into one faithful
paragraph.

Reads ``PARROT_v1_0_french.jsonl`` (``no`` + ``report``) and writes
``03_parrot_rewrite_chunks.jsonl`` with one row per chunk::

    {"id": "parrot-{no}-c{chunk_index}", "category": "parrot",
     "source_id": no, "chunk_index": k, "text": "<raw chunk>"}

The deterministic chunking lives in ``utils/text_chunking.py`` (shared with the
PARHAF preprocessor); this file only wires the PARROT input shape to it.

Written with the help of Claude Code.
"""
import json
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from text_chunking import DEFAULT_MAX_CHARS, chunk_text, strip_parens  # noqa: E402

SRC = Path("PARROT_v1_0_french.jsonl")
DST = Path("03_parrot_rewrite_chunks.jsonl")


@click.command(context_settings={"show_default": True})
@click.option("--input-path", "-i", default=str(SRC), type=click.Path(path_type=Path))
@click.option("--output-path", "-o", default=str(DST), type=click.Path(path_type=Path))
@click.option("--max-chars", default=DEFAULT_MAX_CHARS, type=int, help="Chunk size cap")
def main(input_path: Path, output_path: Path, max_chars: int) -> None:
    if not input_path.exists():
        sys.exit(f"missing {input_path}")
    n_reports = 0
    n_chunks = 0
    with input_path.open(encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for lineno, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            report = row.get("report")
            if not isinstance(report, str):
                continue
            n_reports += 1
            no = row.get("no", lineno)
            for chunk_index, chunk in enumerate(
                chunk_text(strip_parens(report), max_chars=max_chars)
            ):
                entry = {
                    "id": f"parrot-{no}-c{chunk_index}",
                    "category": "parrot",
                    "source_id": no,
                    "chunk_index": chunk_index,
                    "text": chunk,
                }
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                n_chunks += 1
    print(f"read {n_reports} reports, wrote {n_chunks} rewrite chunks to {output_path}")


if __name__ == "__main__":
    main()
