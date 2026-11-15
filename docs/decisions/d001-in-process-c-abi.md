# D1. In-process C ABI plus C++ sugar

| Date | Status |
|---|---|
| 2026-06-10 | Accepted |

## Decision

Applets are native libraries loaded into the host process. They talk to the host through a plain C ABI, with a thin C++ header layer (the sugar) on top. The ABI is not a set of C++ interfaces and not an inter-process protocol.

## Reason

An applet that runs in the host's own frame loop can hand over tensors without copying them, which is the central property of the instrument, and a C ABI stays stable across compiler and standard library versions. The trade is that an applet crash takes the host down with it. Running applets in separate processes remains an option for trust later; it is not needed as a foundation.
