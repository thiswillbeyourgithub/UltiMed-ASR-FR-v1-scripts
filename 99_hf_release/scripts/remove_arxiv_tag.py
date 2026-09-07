#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub>=1.0",
# ]
# ///
"""
Drop a lingering auto-derived ``arxiv:<id>`` tag from the HF dataset repo.

The Hub scans the dataset card for ``arxiv.org/abs/<id>`` links, derives an
``arxiv:<id>`` tag from them and caches it in the repo's tag list. Once the
link is removed from README.md the tag can still linger until the card is
re-parsed. Re-pushing the corrected README.md re-triggers that extraction and
drops the stale tag. This script does exactly that and prints the tags before
and after so you can confirm it worked.

It only touches README.md (metadata), never the audio/Parquet. Written with
Claude Code.

Auth: reads the token from the ``HF_TOKEN`` env var; if unset, falls back to a
prior ``hf auth login``. Nothing is written to disk.

Usage:
    export HF_TOKEN=hf_...
    uv run scripts/remove_arxiv_tag.py            # re-push README, show tags
    uv run scripts/remove_arxiv_tag.py --dry-run  # just show current tags
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

# Same repo the release scripts target. This is the public HF handle, not a
# machine/user path, so it is safe to hardcode (see sibling upload_to_hf.py).
REPO_ID = "Olicorne/UltiMed-ASR-FR-v1"
REPO_TYPE = "dataset"
README = Path(__file__).resolve().parent.parent / "README.md"


def arxiv_tags(tags):
    """Return only the ``arxiv:...`` entries from a repo tag list."""
    return sorted(t for t in (tags or []) if t.lower().startswith("arxiv:"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Re-push README.md to drop a stale auto-derived arxiv tag.",
    )
    ap.add_argument("--repo", default=REPO_ID, help=f"dataset repo id (default: {REPO_ID})")
    ap.add_argument("--dry-run", action="store_true", help="only print current tags, push nothing")
    ap.add_argument(
        "--message",
        default="Re-parse card metadata (drop stale arxiv tag)",
        help="commit message for the README re-push",
    )
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")  # None => huggingface_hub uses the cached login
    api = HfApi(token=token)

    info = api.dataset_info(args.repo)
    before = arxiv_tags(info.tags)
    print(f"current arxiv tags on {args.repo}: {before or '(none)'}")

    if not before:
        print("nothing to remove: no arxiv tag is currently attached.")
        # still fall through to the re-push if the user forces it, but by
        # default there is nothing to do.
        if not args.dry_run:
            print("skipping re-push (add the link back and rerun only if you expected a tag).")
        return 0

    if args.dry_run:
        print("dry-run: not pushing README.md.")
        return 0

    if not README.is_file():
        sys.exit(f"error: no README.md at {README}")

    # Guard: HF re-derives the tag from ANY arxiv-bearing link in the card
    # (bibtex blocks included), so if the local README still holds one, pushing
    # it would just re-add the very tag we are trying to drop. The plain `doi`
    # / `eprint` bibtex fields are NOT parsed by HF, only clickable URLs are.
    lowered = README.read_text(encoding="utf-8").lower()
    triggers = [
        m
        for m in ("arxiv.org/abs/", "arxiv.org/pdf/", "arxiv.org/", "doi.org/10.48550/arxiv")
        if m in lowered
    ]
    if triggers:
        sys.exit(
            "error: local README.md still contains an arxiv link "
            f"({', '.join(triggers)}); remove it first or HF re-adds the tag on parse."
        )

    api.upload_file(
        path_or_fileobj=str(README),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type=REPO_TYPE,
        commit_message=args.message,
    )
    print("re-pushed: README.md")

    # HF re-derives arxiv tags on an ASYNC re-index job, so an immediate re-read
    # here almost always still shows the old tag. This is expected; it is not a
    # failure. Recheck later rather than trusting this line.
    after = arxiv_tags(api.dataset_info(args.repo).tags)
    print(f"arxiv tags immediately after push (re-index is async): {after or '(none)'}")
    print(
        "the tag clears on HF's next re-index of the card, which can take from a\n"
        "few minutes up to a while. Recheck the dataset page later. If it still\n"
        "will not clear once EVERY arxiv reference is gone from the card, open a\n"
        "discussion on the HF Hub forum (no client API deletes a derived tag)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
