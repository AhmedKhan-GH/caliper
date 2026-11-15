# D22. Signing policy for third-party applets

| Date | Status |
|---|---|
| 2026-11-09 | Proposed |

## Decision

On macOS the host either carries the `disable-library-validation` entitlement or accepts only applets signed by the same team. `caliper install` handles the quarantine attribute for downloaded applets. The choice must be made before the first third-party applet.

## Reason

A notarized host with the hardened runtime refuses to load libraries signed by another team, so third-party applets cannot load without one of these two policies.

## Notes

Nothing in the repository signs code yet. Compare the [trust model](../explanations/trust-model.md).
