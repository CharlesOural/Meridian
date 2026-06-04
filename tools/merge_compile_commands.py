#!/usr/bin/env python3
"""Merge colcon's per-package compile_commands.json files into one at the
workspace root, so clangd indexes the whole workspace uniformly.

Run from anywhere after a `colcon build` that exported compile commands:
    python3 tools/merge_compile_commands.py
"""
import glob
import json
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
entries = []
seen = set()
for path in sorted(glob.glob(os.path.join(root, "build", "*", "compile_commands.json"))):
    try:
        for e in json.load(open(path)):
            key = (e.get("directory"), e.get("file"))
            if key not in seen:
                seen.add(key)
                entries.append(e)
    except (OSError, ValueError) as exc:
        print(f"skip {path}: {exc}", file=sys.stderr)

out = os.path.join(root, "compile_commands.json")
with open(out, "w") as f:
    json.dump(entries, f, indent=1)
print(f"merged {len(entries)} entries from build/*/ -> {out}")
