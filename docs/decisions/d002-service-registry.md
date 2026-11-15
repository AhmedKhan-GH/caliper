# D2. Host context is a service registry

| Date | Status |
|---|---|
| 2026-06-10 | Accepted |

## Decision

The host hands each applet a context whose `get_service` function returns a table of functions for a named, versioned service id. New capabilities arrive as new service ids rather than as changes to the base ABI.

## Reason

The base ABI can then stay small and stable for years while the host grows. The cost is that every applet must handle a missing service, which the [honest fallbacks](d033-honest-fallbacks.md) rule already requires.

## Notes

The code implements this decision: `CaliperHost` in `abi.h` carries `get_service`, and the host serves all of its services through it. Revisions follow [D30](d030-additive-service-revisions.md).
