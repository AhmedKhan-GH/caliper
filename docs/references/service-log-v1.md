# caliper.log.v1

Service id `caliper.log.v1` — timestamped log lines from an applet into the host. This page embeds the header verbatim; the docs build fails if the file moves.

```c
--8<-- "sdk/include/caliper/services/log_v1.h"
```

## Semantics

- **Pre-formatted lines.** `message_utf8` is a single, already-formatted UTF-8
  line — no `printf` format string is interpreted, and no trailing newline is
  required (the host adds one). Do the formatting on the applet side and pass the
  finished string.
- **Where it goes.** The `caliper` executable stamps each line with a local
  `HH:MM:SS` timestamp and its level tag and writes it to **stderr**. A program that
  embeds the core through Caliper Embed and sets `log_fn` in its `CaliperCoreDesc` receives the
  applet lines through that callback instead, tagged `[applet] `, and nothing
  goes to stderr; see [Embedding](embedding.md).
- **Levels.** `level` is one of `CALIPER_LOG_DEBUG`/`INFO`/`WARN`/`ERROR`; any
  value outside `0..3` is treated as `INFO`.
- **Threading.** Unlike `caliper.ui.v1`, `log()` may be called from any thread
  the applet owns (e.g. a background worker). **But** the `CaliperLogV1*` table
  pointer itself must be obtained via `get_service(CALIPER_LOG_V1)` during
  `initialize()` and cached — resolve it on the UI thread, then call it from
  wherever you need to log.
