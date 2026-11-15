# D40. The separate-process viewer is the primary live mode

| Date | Status |
|---|---|
| 2026-10-24 | Accepted |

## Decision

Live viewing from a Python script has two modes behind one view layer in libcaliper. The primary mode attaches a separate viewer process to the script. The second mode opens a Caliper window inside the script's own process, with its limits documented.

## Reason

Only the separate process keeps the window responsive whatever the training loop does. In-process, window input needs the main thread, which is the training thread. The signal-timer workaround hangs with some data loaders, and running the training on a worker thread changes how scripts are written. The separate viewer also outlives the script. Both modes are zero-copy and ran at 94 to 99 percent of the throughput of a control that did no snapshots.
