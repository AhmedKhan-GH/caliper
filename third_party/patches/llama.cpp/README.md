# Local llama.cpp patches

Caliper carries a small set of deltas on top of the pinned upstream llama.cpp
**without forking it**. The submodule pin stays on a fetchable upstream commit;
these patch files live in caliper's own history; and `cmake/Dependencies.cmake`
applies them at configure time (idempotent + atomic — safe to re-run).

## Why not a fork or an upstream PR

These are **compatibility shims for ollama's GGUF encoding**, not upstream bugs.
Verified against the actual `qwen3.6:27b` ollama blob: it names the SSM delta
tensor `blk.N.ssm_dt` (plain) and writes 3-element rope sections, whereas
upstream's own `convert_hf_to_gguf.py` emits `ssm_dt.bias` and 4 sections — and
upstream's loader reads those. A canonically-converted model loads on stock
upstream; ollama's blobs do not. So the fix belongs here (to consume the ~51 GB
of GGUFs ollama already downloaded, unmodified), not in a loader PR that would
mask a converter difference. If you ever want it upstream, the genuine report is
against ollama's converter, or regenerate the GGUFs with upstream tooling and
drop these patches entirely.

## The patches

Base commit: **`1a68ec937`** (the submodule pin — an upstream commit, fetchable
from ggerganov/llama.cpp). Applying both patches reproduces the working tree of
`52f3747` exactly (verified: `git -C third_party/llama.cpp diff 52f3747da` is
empty after apply).

- `0001-add-GPT-OSS-*.patch` — GPT-OSS (OpenAI MoE) architecture support.
- `0002-fix-Qwen3.5-3.6-loading-*.patch` — Qwen3.5/3.6 hybrid loading: tolerate
  3-element rope sections (zero-pad), read `ssm_dt` without the `.bias` suffix,
  use per-layer KV head counts, allow partial load of sibling VLM/MTP tensors.

## Notes

- `.gitmodules` sets `ignore = dirty` for llama.cpp so the applied-patch working
  tree does not show as modified in `git status`. Consequence: hand-edits to the
  submodule source are also hidden — **edit via these patch files, not in place.**
- Only used when the opengllama applet is built (`-DCALIPER_BUILD_OPENGLLAMA=ON`).

## Regenerating (after editing the delta, or bumping the pin)

```sh
cd third_party/llama.cpp
# ... make your commits on top of the pinned base ...
rm ../patches/llama.cpp/*.patch
git format-patch <pinned-base>..HEAD -o ../patches/llama.cpp
```
