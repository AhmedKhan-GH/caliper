# caliper.log.v1

Service id `caliper.log.v1` — structured logs into the host console (PLATFORM.md §7.1); fuller semantics arrive at Task 13. This page embeds the header verbatim; the docs build fails if the file moves.

```c
--8<-- "sdk/include/caliper/services/log_v1.h"
```

## Semantics

- **Pre-formatted lines.** `message_utf8` is a single, already-formatted UTF-8
  line — no `printf` format string is interpreted, and no trailing newline is
  required (the host adds one). Do the formatting on the applet side and pass the
  finished string.
- **Where it goes today.** The host stamps each line with a local `HH:MM:SS`
  timestamp and its level tag and writes it to **stderr**. A dedicated console
  panel is planned later; the ABI does not change when that lands — the same
  table keeps working.
- **Levels.** `level` is one of `CALIPER_LOG_DEBUG`/`INFO`/`WARN`/`ERROR`; any
  value outside `0..3` is treated as `INFO`.
- **Threading.** Unlike `caliper.ui.v1`, `log()` may be called from any thread
  the applet owns (e.g. a background worker). **But** the `CaliperLogV1*` table
  pointer itself must be obtained via `get_service(CALIPER_LOG_V1)` during
  `initialize()` and cached — resolve it on the UI thread, then call it from
  wherever you need to log.
