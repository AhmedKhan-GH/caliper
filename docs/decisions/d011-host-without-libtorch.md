# D11. The host ships without libtorch

| Date | Status |
|---|---|
| 2026-06-10 | Accepted |

## Decision

The host binary does not link libtorch. The tensor bridge talks to CUDA and Metal directly, and the metrics store uses an embedded DuckDB. Torch is an applet concern.

## Reason

The host stays small, about 50 MB, and torch arrives only for applets that need it.

## Notes

The code implements this decision: the CMake option `CALIPER_WITH_TORCH=OFF` builds the host without torch, the device query uses Metal directly, and DuckDB is linked statically into the host. Loading torch on demand as a pack is tied to [D5](d005-runtime-packs.md).
