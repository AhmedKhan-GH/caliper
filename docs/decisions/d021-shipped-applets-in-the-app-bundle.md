# D21. Shipped applets live in Contents/PlugIns

| Date | Status |
|---|---|
| 2026-11-09 | Proposed |

## Decision

Applets shipped with the app live in `Caliper.app/Contents/PlugIns/` and are signed with the app. Nothing writes into the app bundle after it ships.

## Reason

Apple expects loadable code in `Contents/PlugIns/`, and writing into a signed bundle breaks its signature.

## Notes

This replaces the executable's `applets/` folder as the location for shipped applets, which is where they still resolve today. It is the shipped tier in [D20](d020-applet-search-order.md).
