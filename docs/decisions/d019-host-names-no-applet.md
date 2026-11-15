# D19. The host names no applet

| Date | Status |
|---|---|
| 2026-11-09 | Proposed |

## Decision

The host names no applet. Applets declare a window role (`dock_hint`) and the host lays out roles. The core build compiles no applets, and CI checks both rules.

## Reason

Adding an applet should never edit the host. Today the default dock layout in `src/main.cpp` lists applet window titles, and the root CMake file builds everything under `applets/`.

## Notes

See [Applets outside the core](../plans/applets-outside-the-core.md). Neither rule is built yet: no `dock_hint` exists in the source or the SDK.
