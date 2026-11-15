# D20. Applet locations and search order

| Date | Status |
|---|---|
| 2026-11-09 | Proposed |

## Decision

The host looks for applets in development paths first, then the user applets folder, then the applets shipped with the app. When the same id appears twice, the higher version wins. Applet data lives in `data/<id>`, and caches that can be rebuilt live in the operating system's cache folder.

## Reason

Developers can override a shipped applet, and a user-installed update replaces a shipped copy without touching the app.

## Notes

The host currently scans only the data folder's applets directory and the one next to the executable. See [D21](d021-shipped-applets-in-the-app-bundle.md) for the shipped location.
