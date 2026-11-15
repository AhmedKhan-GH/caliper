# D41. The applet registry at caliperapp.ai

| Date | Status |
|---|---|
| 2026-11-15 | Proposed |

## Decision

The applet registry is served at caliperapp.ai and backed by a registry repository that takes publications by pull request. The host gets a Browse tab and installs applets by id and version.

## Reason

A served registry lets users find and install applets without knowing about git. It keeps what worked in [D6](d006-registry-as-a-git-repository.md), review of publications by pull request, and the rule that sideloading always works.

## Notes

This supersedes D6. See the [registry plan](../plans/registry.md).
