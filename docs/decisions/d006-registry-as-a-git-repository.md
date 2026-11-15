# D6. Registry as a git repository

| Date | Status |
|---|---|
| 2026-06-10 | Superseded |

## Decision

Applets are published through a registry that is a git repository, in the style of a Homebrew tap. Curation is review of pull requests, and installing an applet from a local file always works.

## Reason

A repository needs no hosted infrastructure, and pull-request review is a workable curation process.

## Notes

Superseded by [D41](d041-registry-at-caliperapp.md), which serves the registry at caliperapp.ai but keeps pull-request curation and the rule that sideloading always works. The loader already implements the sideloading rule.
