"""Diff the Rust and Python positions at a divergent decision point.

Usage::

    .venv\\Scripts\\python tools\\diff_state.py <rust-dump-file> <python-dump-file>

Both files are JSON objects (the state fingerprint). Prints every path whose
value differs, so a divergence can be attributed to one rule.
"""

from __future__ import annotations

import io
import json
import sys


def walk(left, right, path: str, out: list[str]) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left:
                out.append(f"{path}.{key}: missing in rust (python {right[key]!r})")
            elif key not in right:
                out.append(f"{path}.{key}: missing in python (rust {left[key]!r})")
            else:
                walk(left[key], right[key], f"{path}.{key}", out)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            out.append(f"{path}: length {len(left)} (rust) vs {len(right)} (python)")
        for index, (a, b) in enumerate(zip(left, right)):
            walk(a, b, f"{path}[{index}]", out)
        return
    if left != right:
        out.append(f"{path}: rust {left!r} != python {right!r}")


def main() -> int:
    rust_text = io.open(sys.argv[1], encoding="utf-8-sig").read()
    lines = rust_text.splitlines()
    payload = None
    for index, line in enumerate(lines):
        if line.strip() == "=== RUST FINGERPRINT BEGIN ===":
            payload = lines[index + 1]
            break
    if payload is None:
        print("no fingerprint dump found in the Rust output")
        return 2
    rust = json.loads(payload)
    python = json.load(io.open(sys.argv[2], encoding="utf-8-sig"))

    differences: list[str] = []
    walk(rust, python, "", differences)
    if not differences:
        print("no differences in the compared object")
        return 0
    print(f"{len(differences)} differing field(s):")
    for line in differences[:120]:
        print(" ", line.lstrip("."))
    return 1


if __name__ == "__main__":
    sys.exit(main())
