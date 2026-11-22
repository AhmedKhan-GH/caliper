# caliper.log.v1

Service id `caliper.log.v1` sends timestamped log lines from an applet to the host.

## Types

<!-- c-api types sdk/include/caliper/services/log_v1.h CaliperLogLevel level=0 -->

## Calls

<!-- c-api calls sdk/include/caliper/services/log_v1.h -->

## Ids

<!-- c-api ids sdk/include/caliper/services/log_v1.h -->

## Semantics

The service has these rules:

- **Pre-formatted lines.** The host does not interpret a `printf` format string, so format the text in the applet and pass the result.
- **Where it goes.** The `caliper` executable adds a local `HH:MM:SS` timestamp and the level tag to each line. It then writes the line to **stderr**. A program that embeds the core through Caliper Embed can set `log_fn` in its `CaliperCoreDesc`. That program receives the applet lines through the callback, tagged `[applet] `, and nothing goes to stderr. See [Embedding](../../web/embedding-api.md).
- **Threading.** Unlike `caliper.ui.v1`, `log()` can run on any thread the applet owns, for example a background worker. The applet must get the `CaliperLogV1*` table pointer with `get_service(CALIPER_LOG_V1)` during `initialize()` on the UI thread. It then caches the pointer and uses it from any thread.
