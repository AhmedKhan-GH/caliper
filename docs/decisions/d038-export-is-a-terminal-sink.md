# D38. Export is a terminal sink

| Date | Status |
|---|---|
| 2026-07-12 | Accepted |

## Decision

Export produces pixel-exact images of views, and numbered frame sequences, together with a JSON sidecar. It does not compose figures. Axes, captions, typography and layout belong to the paper toolchain, and video assembly happens outside, with one documented ffmpeg command.

## Reason

The claim of the instrument is that the pixels are the tensors, and export keeps that claim bit for bit. A typography engine would add scope without strengthening the claim. The sidecar records the Caliper version, applet id, time, camera, colour map and range, view size and the state the applet supplies, so that an exported image is traceable. The same tensors, camera and colour map give an identical PNG on the same backend.
