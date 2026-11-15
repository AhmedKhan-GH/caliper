# D8. Applet repositories keep their history

| Date | Status |
|---|---|
| 2026-06-10 | Proposed |

## Decision

When an applet moves into its own repository, it moves with its git history, extracted with `git filter-repo`, rather than being copied in as a fresh start.

## Reason

The history of an applet is part of its life, and a clean copy would throw away the record of how it was built.

## Notes

No applet has moved yet. The decision applies to the move described in [Applets outside the core](../plans/applets-outside-the-core.md).
