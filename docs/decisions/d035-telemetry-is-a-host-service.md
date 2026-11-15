# D35. Telemetry is a host service with host-owned providers

| Date | Status |
|---|---|
| 2026-07-12 | Accepted |

## Decision

Live telemetry is a host service, `feed.v1`. The host owns the providers, buffers samples in per-channel ring buffers with timestamps and sequence numbers, and offers a pull-based read that works from any thread and never blocks. Applets copy samples into their own tensors.

## Reason

One service avoids every applet inventing its own ingestion. Pulling with caller-held cursors needs no callbacks across the ABI and no state per reader. Data loss shows up as gaps in the sequence numbers and is never silent, and a host with no provider reports no capability instead of fake data.

## Notes

Providers are owned by the host. Letting applets register sources is not built.
