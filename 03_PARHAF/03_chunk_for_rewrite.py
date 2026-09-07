#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "click",
# ]
# ///
"""Turn the raw PARHAF documents into rewrite-ready chunks for the LLM stage.

This is the input preprocessor for the PARHAF rewrite stage (``04_generate_texts.py``).
Unlike ``02_clean_split_texts.py`` (now an audit-only line cleaner), this keeps
the raw document text intact: it only strips parenthesised spans (per the agreed
policy) and splits each document into coherent chunks capped at ``--max-chars`` so
no single chunk overruns the model's useful context. The LLM then rewrites each
chunk into one faithful paragraph.

Reads ``01_parhaf_documents.jsonl`` (``id`` + ``documents.text`` list) and writes
``03_parhaf_rewrite_chunks.jsonl`` with one row per chunk::

    {"id": "{orig_id}-t{text_index}-c{chunk_index}", "category": "parhaf",
     "source_id": orig_id, "chunk_index": k, "text": "<raw chunk>"}

The deterministic chunking lives in ``utils/text_chunking.py`` (shared with the
PARROT preprocessor); this file only wires the PARHAF input shape to it.

Written with the help of Claude Code.
"""
import json
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from text_chunking import DEFAULT_MAX_CHARS, chunk_text, strip_parens  # noqa: E402

SRC = Path("01_parhaf_documents.jsonl")
DST = Path("03_parhaf_rewrite_chunks.jsonl")


@click.command(context_settings={"show_default": True})
@click.option("--input-path", "-i", default=str(SRC), type=click.Path(path_type=Path))
@click.option("--output-path", "-o", default=str(DST), type=click.Path(path_type=Path))
@click.option("--max-chars", default=DEFAULT_MAX_CHARS, type=int, help="Chunk size cap")
def main(input_path: Path, output_path: Path, max_chars: int) -> None:
    if not input_path.exists():
        sys.exit(f"missing {input_path}")
    n_docs = 0
    n_chunks = 0
    with input_path.open(encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_docs += 1
            orig_id = row["id"]
            texts = row.get("documents", {}).get("text", []) or []
            for text_index, text in enumerate(texts):
                if not isinstance(text, str):
                    continue
                for chunk_index, chunk in enumerate(
                    chunk_text(strip_parens(text), max_chars=max_chars)
                ):
                    entry = {
                        "id": f"{orig_id}-t{text_index}-c{chunk_index}",
                        "category": "parhaf",
                        "source_id": orig_id,
                        "chunk_index": chunk_index,
                        "text": chunk,
                    }
                    fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    n_chunks += 1
    print(f"read {n_docs} documents, wrote {n_chunks} rewrite chunks to {output_path}")


if __name__ == "__main__":
    main()
