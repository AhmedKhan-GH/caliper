# Live-view benchmarks

`bench_live.py` and `bench_show.py` measure what `caliper.show` and `caliper.points` cost a training loop and how
fast their snapshots reach the screen, through the shared-memory live session and the in-process window. The
network benchmark has its own page, [README-net.md](README-net.md). The renderer core's migration measures every
step with these two scripts, against the thresholds in the design page of the renderer core.

## Running them

Build the native modules and the viewer, stage the package from the build folder, and run from the repository root
with a project-local environment that has torch:

```sh
uv venv && uv pip install torch numpy
scripts/py_stage.sh cmake-build-notorch
export PYTHONPATH=cmake-build-notorch/py
```

Neither script opens a window by default. `bench_live.py` starts the viewer with `--headless`, an offscreen canvas
that polls and draws about 50 times a second, and `--window` brings back the windowed viewer. `bench_show.py`
renders window mode on the in-process offscreen canvas and starts the attach-mode viewer headless, and `--visible`
brings back both windows. Each run is bounded: a row lasts its `--seconds` after its warm-up, and the viewer exits
on its own after the row.

## bench_live.py

One size per invocation; the modes of a table run interleaved within each repeat:

```sh
for s in 256 1024 2048; do
  .venv/bin/python python/bench/bench_live.py table --size $s --repeat 3
done
for n in 100000 1000000 4000000; do
  .venv/bin/python python/bench/bench_live.py table --kind points --size $n --repeat 3
done
```

The heatmap rows show a `size` x `size` weight; the points rows (`--kind points`) show an (N, 3) cloud with
`caliper.points`, N = `--size`. Both write `results/S-table-<device><size>.txt` (points:
`S-table-points-<device><N>.txt`), or the name given with `--out`, with one Markdown row per mode and repeat, the
1-minute load average at its start and end, and every row's raw JSON below the table. The viewer's log and latency
samples of each row stay in the run's private data folder; `CALIPER_BENCH_KEEP_RAW=1` writes them to
`results/S-raw-<tag>-viewer.txt` and `-latency.txt` instead, where a row's tag can repeat the name of a file already
there.

Every row that has a viewer reports steps per second against the baseline and against the commit-matched
control, the cost of `caliper.show` or `caliper.points` (median microseconds), the step-to-screen latency (median
and p95) and the viewer's frames per second and path label, such as `W:compute-imported`. The viewer runs with
`--stats`, so the same rows also report, over the 2 s stats windows that lie inside the measured window:

| Column | What it is |
|---|---|
| frame ms p50 / p99 | The interval between the ends of consecutive frames, the median over the windows of each window's percentile |
| GPU ms update p50 / p99 | The GPU time of a texture update (a colormap dispatch or a blit); for a points row, of the view's draw (`draw`) |
| GPU ms frame p50 | The GPU time of the frame's own pass |
| viewer copies per frame (bytes) | The viewer's copies of tensor data on the way to pixels, per frame, with the bytes; 0 on a zero-copy route |

The raw JSON keeps the p99 of every GPU time, the worst frame and the copies of the shown view.

## bench_show.py

```sh
.venv/bin/python python/bench/bench_show.py --mode window --device mps --size 1024
.venv/bin/python python/bench/bench_show.py --mode attach --device mps --size 1024
```

It writes `results/P-<mode>-<device>-<size><tag>.txt`. Next to its table it prints, for each caliper row, the
frame time p50 and p99, the GPU time of an update and the copies per frame of the process that draws: in window
mode from the core's `frame_stats()` (the script sets `CALIPER_METAL_STATS`), in attach mode from the viewer's
stats windows.

## Results of the renderer core's steps

The files of a migration step carry the step's name, such as `S-table-mps1024-step0.txt`. The step's summary,
with the machine, the build, the counters of each route and the load at each measurement, is
`results/step<N>-baseline.txt` for the baseline and `results/step<N>-<before|after>.txt` for a step that changes a
path.
