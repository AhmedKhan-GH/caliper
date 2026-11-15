# D5. Host-managed runtime packs, one libtorch per process

| Date | Status |
|---|---|
| 2026-06-10 | Proposed |

## Decision

Large runtimes such as libtorch are installed as packs that the host manages, not bundled by each applet. A process loads one libtorch for the whole session.

## Reason

Without packs every applet that uses torch would carry its own multi-gigabyte copy. The trade is a stated limit: two applets that need different torch versions cannot share a session, and only separate processes would remove that limit.

## Notes

No pack manager exists yet; the build vendors torch directly. [D11](d011-host-without-libtorch.md) depends on this decision for loading torch on demand.
