# D9. A CLI as subcommands of the host binary

| Date | Status |
|---|---|
| 2026-06-10 | Proposed |

## Decision

The command-line tool is a set of subcommands of the host binary (`caliper new`, `dev`, `package`, `install`, `publish`), not a separate program.

## Reason

There is one artifact to install, as with the `code` command of Visual Studio Code.

## Notes

The host `main` takes no arguments yet, so no CLI exists. Whether `new` is still needed depends on the [applet builder](../plans/applet-builder.md).
