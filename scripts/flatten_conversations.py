#!/usr/bin/env python3
"""
Flatten conversation files so that each message's full content is on the same
line as its marker. Markers supported:
  - "### Human:", "### User:", "### Assistant:"
  - "HUMAN:", "ASSISTANT:", "USER:" (rare)

The script recursively processes conversation .txt files under
  - data/dataset/gpt5_*
  - data/dataset/llama2_*

Usage:
  python scripts/flatten_conversations.py --in-place

Dry run (default) prints a summary without writing changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


# Marker detection configuration
MARKER_PREFIXES = (
    "### Human:",
    "### User:",
    "### Assistant:",
    "HUMAN:",
    "USER:",
    "ASSISTANT:",
)


def is_marker_line(line: str) -> bool:
    stripped = line.lstrip(" ")
    return any(stripped.startswith(pfx) for pfx in MARKER_PREFIXES)


def split_marker_and_content(line: str) -> Tuple[str, str]:
    """Return (marker_with_colon_and_any_leading_spaces, content_after_colon).

    Preserves the exact marker substring up to and including the first colon.
    If no colon exists, treats entire line as marker and empty content.
    """
    if ":" in line:
        idx = line.find(":")
        marker = line[: idx + 1]
        content = line[idx + 1 :].rstrip("\n")
        # Normalize any single leading space after colon (content will be trimmed later)
        return marker, content
    return line.rstrip("\n"), ""


def flatten_conversation_text(text: str) -> str:
    """Collapse multi-line messages so each marker has its full content on one line.

    Algorithm:
    - Scan lines; when a marker line is found, start a new block and collect all
      subsequent lines until the next marker or EOF.
    - Join the block content with single spaces and place it on the same line as
      the marker.
    - Preserve the exact marker substring (including "### " and capitalization).
    - Leave non-marker lines between messages treated as part of the preceding
      message.
    """
    lines = text.splitlines()

    out_lines: List[str] = []

    current_marker: str | None = None
    current_content_parts: List[str] = []

    def flush_current():
        if current_marker is None:
            return
        # Normalize whitespace within content: collapse newlines to single spaces
        content = " ".join(part.strip() for part in current_content_parts if part is not None)
        content = " ".join(content.split())  # collapse runs of whitespace
        # Ensure exactly one space after marker when content exists
        if content:
            out_lines.append(f"{current_marker} {content}".rstrip())
        else:
            out_lines.append(current_marker.rstrip())

    for raw_line in lines:
        if is_marker_line(raw_line):
            # Starting a new block; flush previous one
            flush_current()
            # Start new message
            marker, initial_content = split_marker_and_content(raw_line)
            current_marker = marker.rstrip()  # keep exact marker text (incl. any leading spaces)
            current_content_parts = [initial_content]
        else:
            # Continuation of current message or stray text before first marker
            if current_marker is None:
                # If text precedes any marker, keep it as-is (conservative)
                out_lines.append(raw_line.rstrip("\n"))
            else:
                current_content_parts.append(raw_line)

    # Flush last message
    flush_current()

    return "\n".join(out_lines) + ("\n" if text.endswith("\n") else "")


def iter_target_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        # Only .txt files
        yield from root.rglob("*.txt")


def process_file(path: Path, write: bool) -> Tuple[bool, int, int]:
    original = path.read_text(encoding="utf-8")
    flattened = flatten_conversation_text(original)
    changed = flattened != original
    if changed and write:
        path.write_text(flattened, encoding="utf-8")
    # Simple metric: line counts
    return changed, original.count("\n"), flattened.count("\n")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Flatten conversation files so messages are single-line per marker")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write changes back to files (default is dry run)",
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        type=Path,
        default=[
            Path("data/dataset/gpt5_rigidity_1").parent,  # data/dataset
        ],
        help=(
            "Optional root directories to scan. If omitted, scans data/dataset/ for gpt5_* and llama2_* subdirs."
        ),
    )

    args = parser.parse_args(argv)

    # Derive specific target subdirectories under provided roots matching gpt5_* and llama2_*
    target_dirs: List[Path] = []
    for root in args.roots:
        if not root.exists():
            continue
        for sub in root.iterdir():
            name = sub.name
            if sub.is_dir() and (name.startswith("gpt5_") or name.startswith("llama2_")):
                target_dirs.append(sub)

    if not target_dirs:
        print("No target directories found. Ensure you have data/dataset/gpt5_* or llama2_*.")
        return 1

    total_files = 0
    changed_files = 0
    for path in iter_target_files(target_dirs):
        # Only process conversation files; still keep generic *.txt in case naming varies
        if not path.name.endswith(".txt"):
            continue
        total_files += 1
        changed, before_lines, after_lines = process_file(path, write=args.in_place)
        if changed:
            changed_files += 1
            action = "UPDATED" if args.in_place else "WOULD UPDATE"
            print(f"{action}: {path} (lines {before_lines} -> {after_lines})")

    print(
        f"Done. Examined {total_files} files. "
        f"{'Updated' if args.in_place else 'Would update'} {changed_files} files."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


