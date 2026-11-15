# D37. Metrics reads go through metrics.v1_1 query

| Date | Status |
|---|---|
| 2026-07-12 | Accepted |

## Decision

A consumer reads metrics across the ABI through `metrics.v1_1.query`, a read-only call that runs on the live writer and returns an Arrow stream. It does not attach to the DuckDB file.

## Reason

A read-only DuckDB attach sees only rows that the writer has checkpointed to the file, and a live metrics writer does not checkpoint after each scalar, so the table would never show live rows. The same barrier applies between processes. The cost is that one more revision of the service must be kept.

## Notes

See [D30](d030-additive-service-revisions.md) and [D3](d003-no-framework-types-across-the-abi.md).
