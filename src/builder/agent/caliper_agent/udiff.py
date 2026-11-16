"""Applies a unified diff to a text, to turn a command's preview into an ACP diff.

The command layer returns a spec change as a unified diff of the canonical
TOML, in the format of `diff -u` (docs/plans/builder-session.md, Diffs). An ACP
diff carries the old and the new text instead, so the sidecar applies the
command's diff to the open spec's text from `spec.get`. Every context and
removed line must match the old text where the hunk says; on any mismatch, or
on a diff it cannot read (the line diff's own "could not be computed" form
included), `apply_unified_diff()` returns None and the caller shows the
unified diff as text instead.
"""

from __future__ import annotations

import re

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_NO_NEWLINE = "\\ No newline at end of file"
MAX_DIFF_CHARS = 4 << 20


def apply_unified_diff(old: str, diff: str) -> str | None:
    if diff == "":
        return old
    if len(diff) > MAX_DIFF_CHARS:
        return None
    old_lines = old.splitlines(keepends=True)
    lines = diff.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    i = 0
    while i < len(lines) and (lines[i].startswith("--- ") or lines[i].startswith("+++ ")):
        i += 1
    out: list[str] = []
    pos = 0  # the next line of old_lines to copy
    hunks = 0
    while i < len(lines):
        m = _HUNK.match(lines[i])
        if not m:
            return None
        hunks += 1
        old_start, old_count = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
        new_count = int(m.group(4) if m.group(4) is not None else 1)
        # A hunk that removes nothing names the line after which it adds.
        first = old_start - 1 if old_count > 0 else old_start
        if first < pos or first > len(old_lines):
            return None
        out.extend(old_lines[pos:first])
        pos = first
        i += 1
        seen_old = seen_new = 0
        while i < len(lines) and not lines[i].startswith("@@"):
            line = lines[i]
            i += 1
            if line == _NO_NEWLINE:
                continue
            body = line[1:] + ("" if i < len(lines) and lines[i] == _NO_NEWLINE else "\n")
            tag = line[:1]
            if tag in (" ", "-"):
                if pos >= len(old_lines) or old_lines[pos] != body:
                    return None
                pos += 1
                seen_old += 1
                if tag == " ":
                    out.append(body)
                    seen_new += 1
            elif tag == "+":
                out.append(body)
                seen_new += 1
            else:
                return None
        if seen_old != old_count or seen_new != new_count:
            return None
    if hunks == 0:
        return None
    out.extend(old_lines[pos:])
    return "".join(out)
