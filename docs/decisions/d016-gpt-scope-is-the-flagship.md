# D16. GPTScope is the flagship applet

| Date | Status |
|---|---|
| 2026-07-02 | Accepted |

## Decision

The reference applet and the proof that the platform is general is `gpt_scope`, a small GPT trained on TinyShakespeare, built on the full service stack from the start. The earlier flagship is archived and will not move onto the services. `data.v1` and `artifacts.v1` are frozen only when a real applet needs them.

## Reason

A new applet written the way an outside author would write it tests the platform better than retrofitting old code. Freezing headers that nothing consumes would break the rule of extracting designs from working code rather than inventing them.

## Notes

The code agrees: `gpt_scope` exists, and the earlier applet sits under `applets/legacy/`. [D18](d018-data-and-artifacts-built-for-embedscope.md) records the demand that later froze the two services.
