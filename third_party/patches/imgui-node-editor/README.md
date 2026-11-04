# Local imgui-node-editor patches

Caliper carries a small delta on top of the pinned upstream imgui-node-editor **without forking it**, the same
way as llama.cpp (see `../llama.cpp/README.md`): the submodule pin stays on a fetchable upstream commit, the patch
lives in caliper's history, and `cmake/Dependencies.cmake` applies it at configure time (idempotent and atomic).

## The patch

Base commit: **`55a7dbf`** (the submodule pin).

- `0001-crude_json-read-UTF-8-and-unicode-escapes.patch` — `crude_json`, which the live inspector uses to read
  the request channel's replies, read each byte as a signed `char`. On arm64 macOS `char` is signed, so any
  byte above 127 failed the parse, and a reply carrying UTF-8 text (a generated word in another script, an
  emoji, a vocabulary entry) was rejected whole and closed the channel. The patch reads bytes as unsigned so
  UTF-8 passes through unchanged, and writes `\uXXXX` escapes (including surrogate pairs) as UTF-8 instead of
  truncating them to one byte.

Upstream: the `#todo` comments in `crude_json.cpp` acknowledge both gaps; this is a candidate for an upstream PR.
