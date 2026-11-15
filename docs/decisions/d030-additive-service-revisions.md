# D30. Additive revisions are new ids with prefix-identical tables

| Date | Status |
|---|---|
| 2026-07-05 | Accepted |

## Decision

A service grows by publishing a new id, `caliper.<name>.v1_N`, whose function table begins with the unchanged earlier table and appends new entries. The old header, table and id stay frozen. Features inside a revision are announced through a `caps()` bit.

## Reason

Old applets keep running against new hosts, and new applets can check for the new id and fall back. No epoch bump is needed. The cost is that hosts and applets of different ages must be tested against each other.

## Notes

Examples are `tensor_bridge` v1_1 and v1_2, `geometry` v1_1 to v1_3, `metrics` v1_1 and `export` v1_1. See [D24](d024-stream-ordered-handoff-through-caps.md) and [D31](d031-geometry-grows-by-additive-records.md).
