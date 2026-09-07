#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyarrow"]
# ///
"""Extract id, local_id and documents from the PARHAF parquet into a jsonl.

Written with the help of Claude Code.
"""
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

SRC = Path("train-00000-of-00001.parquet")
DST = Path("01_parhaf_documents.jsonl")

KEEP = ("id", "local_id", "documents")


def main() -> None:
    if not SRC.exists():
        sys.exit(f"missing {SRC}")
    table = pq.read_table(SRC, columns=list(KEEP))
    with DST.open("w", encoding="utf-8") as fh:
        for row in table.to_pylist():
            fh.write(json.dumps({k: row[k] for k in KEEP}, ensure_ascii=False))
            fh.write("\n")
    print(f"wrote {table.num_rows} rows to {DST}")


if __name__ == "__main__":
    main()
