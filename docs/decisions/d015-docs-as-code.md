# D15. Documentation as code

| Date | Status |
|---|---|
| 2026-07-01 | Accepted |

## Decision

The documentation is a MkDocs Material site in the repository, laid out by the Diátaxis types. A change to behaviour updates its pages in the same commit, and the strict MkDocs build is a gate. Reference pages embed the real headers and manifests with snippets, so a moved file fails the build.

## Reason

Documentation written afterwards goes stale. Updating pages with the change, embedding code instead of pasting it, and checking links strictly keep the site true by mechanism. A generated API reference (`mkdocs-cxxdox`) and versioned publishing (`mike`) were planned as later additions to the same site; cxxdox is alpha software, and the embedded reference pages remain the fallback.

## Notes

The documentation now lives under `docs/` in the Diátaxis folders.
