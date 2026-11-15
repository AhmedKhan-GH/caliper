# D33. A missing capability falls back to a working slower path and says so

| Date | Status |
|---|---|
| 2026-07-05 | Accepted |

## Decision

When a capability is missing, the host falls back to a working, slower path and reports that it did. It never produces a wrong image and never prints a status line that is not true. Zero-copy is claimed only when the zero-copy path actually drew.

## Reason

Applets then run on every machine, and users can trust what the instrument reports. The cost is that each capability needs a second path that is tested.

## Notes

Examples: the import of an allocation returns `false` on any miss and the caller falls back ([D34](d034-opt-in-imported-allocations.md)), the CUDA handoff falls back to synchronous ([D28](d028-synchronous-interop-first.md)), and interop turns off on a device mismatch ([D27](d027-uuid-matched-device-pairing.md)).
