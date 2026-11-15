# D39. No I/O on the frame thread, live tables are TEMP

| Date | Status |
|---|---|
| 2026-07-05 | Accepted |

## Decision

The frame thread does no I/O at all: no SQL, no file writes, no store calls. Tables that are rebuilt at high frequency for live display are DuckDB `TEMP` tables, not tables in the persistent store.

## Reason

Rebuilding an on-disk table on the frame thread made EmbedScope freeze on Windows, because its cost grew with the age of the session until it ran every frame. A fast path on one machine is a slow path on another. The persistent store is meant for data that is deliberately saved and read back.

## Notes

Work that takes longer than about 5 ms belongs on a background thread. `jobs.v1` is for work a user watches or cancels, and internal maintenance loops get threads owned by the applet.
