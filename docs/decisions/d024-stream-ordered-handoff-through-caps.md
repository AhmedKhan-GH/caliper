# D24. CaliperTensor.stream gains meaning through bridge v1_1 caps

| Date | Status |
|---|---|
| 2026-07-05 | Accepted |

## Decision

A non-NULL `CaliperTensor.stream` means that the host should order the device update on the producer's stream or queue instead of waiting for the device to drain. `tensor_bridge.v1_1` adds a `caps()` call whose first bit says the host honours this. A NULL stream keeps the v1 behaviour, and adapters check the bit before they skip the drain.

## Reason

It removes the largest measured stall on both platforms without touching the frozen struct or the epoch. The cost is one more registered service id and a compatibility matrix of old host with new applet and the reverse, which the negotiation tests must pin.

## Notes

This is an instance of [D30](d030-additive-service-revisions.md).
