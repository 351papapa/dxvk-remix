#!/usr/bin/env python3
"""Apply the host renderer edits to a pinned NUMOS source file."""

import argparse
from hashlib import sha256
from pathlib import Path
import re


HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@\n")


def apply_overlay(source, patch):
    source_lines = source.splitlines(keepends=True)
    metadata, _, _, *patch_lines = patch.splitlines(keepends=True)
    if metadata.strip() != f"base-sha256 {sha256(source.encode('utf-8')).hexdigest()}":
        raise ValueError("NUMOS renderer source differs from the pinned overlay base")
    output = []
    cursor = 0
    consumed = 0
    end = 0
    for line in patch_lines:
        hunk = HUNK.fullmatch(line)
        if hunk is not None:
            count = int(hunk[2] or 1)
            start = int(hunk[1]) - int(count != 0)
            if cursor != end or start < cursor or start + count > len(source_lines):
                raise ValueError(f"Invalid overlay range: {line.rstrip()}")
            output.extend(source_lines[cursor:start])
            cursor, end = start, start + count
            consumed += 1
        elif consumed and line[:1] in (" ", "+", "-"):
            if line[0] != "+":
                if cursor >= end or source_lines[cursor] != line[1:]:
                    raise ValueError(f"NUMOS source no longer matches the overlay at line {cursor + 1}")
                cursor += 1
            if line[0] != "-":
                output.append(line[1:])
        else:
            raise ValueError(f"Invalid overlay line: {line.rstrip()}")
    if not consumed or cursor != end:
        raise ValueError("Incomplete NUMOS renderer overlay")
    output.extend(source_lines[cursor:])
    return "".join(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("patch", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        result = apply_overlay(args.source.read_text(encoding="utf-8"), args.patch.read_text(encoding="utf-8"))
    except ValueError as error:
        parser.exit(1, f"{args.source}: {error}\n")
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(result)


if __name__ == "__main__":
    main()
