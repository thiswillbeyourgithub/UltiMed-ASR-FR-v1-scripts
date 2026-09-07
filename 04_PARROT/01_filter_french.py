#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: 01_filter_french.py <input.jsonl> <output.jsonl>", file=sys.stderr)
        sys.exit(2)

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])

    kept = 0
    total = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            total += 1
            obj = json.loads(line)
            if str(obj.get("language", "")).strip().lower() == "french":
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1

    print(f"kept {kept}/{total} lines -> {dst}", file=sys.stderr)


if __name__ == "__main__":
    main()
