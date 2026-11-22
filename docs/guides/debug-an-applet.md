# Debug an applet

An applet is a shared library the host loads at runtime, so you don't debug it as
its own program — you attach a debugger to the running **`caliper`** host. The
dylib carries your own symbols (built `Debug`, they're right there), so once
attached you get full source-level stepping, breakpoints, and variable
inspection inside `on_init` / `on_frame` / `on_cleanup`.

## Attach a debugger to the running host

Build both the host and your applet with debug info, then launch and attach.
The host builds with the `debug` preset in your Caliper checkout, and the
applet builds in its own folder against the installed SDK, as in
[Your first applet](../tutorials/first-applet.md#4-build-it), with
`-DCMAKE_BUILD_TYPE=Debug`. Start the host with `caliper dev` and the applet's
build folder, so that it loads your debug build first.

**LLDB (command line)**

```bash
cmake --preset debug && cmake --build --preset debug      # in the Caliper checkout
./cmake-build-debug/caliper dev <applet folder>/build/applets &
lldb -p "$(pgrep -f 'cmake-build-debug/caliper')"
```

At the `(lldb)` prompt set a breakpoint by symbol or file:line and continue —
your applet's translation unit is visible even though it lives in a separate
dylib:

```
(lldb) breakpoint set --name HelloApplet::on_frame
(lldb) breakpoint set --file hello.cpp --line 39
(lldb) continue
```

**CLion**

Use **Run ▸ Attach to Process…**, filter for `caliper`, and pick the running
host. Set breakpoints in your applet source as usual. Because the applet dylib is
loaded lazily when you launch the applet from the landing screen, a breakpoint in
`on_frame` won't bind until the applet is opened in the app — CLion resolves it
the moment the dylib is mapped in, which is expected. Breakpoints in `on_init`
bind at open time; to catch very early setup, break in `on_init` and step from
there.

Because the host loads the dylib after startup, prefer *attaching to the running
process* over launching the host under the debugger — the symbols resolve when
the applet is opened, and you avoid stepping through the host's whole boot.

## Log through the host, not `printf`

Do **not** use `printf`/`std::cout` for diagnostics. Log through the
`caliper.log.v1` service, which the sugar layer exposes on your `Host`:

```cpp
bool on_init(caliper::Host& host) override {
    host.log_info("hello.on_init");                 // caliper.log.v1
    // host.log_error("...") for failures
    ...
}
```

Host logging is unified: messages go to the same sink as the host's own
diagnostics (the console the host is launched from, and the dev-mode tail), they
carry the applet's identity, and they keep working when your dylib's stdio is not
wired to a terminal — which it often isn't once the app is bundled. The raw
service is `caliper.log.v1` (declare it in your manifest's `required` list, as the
hello applet does); the sugar `host.log_info` / `host.log_error` are thin
wrappers over it.

When the loader turns an applet away, the diagnostic is on the **landing
screen**, not in the console. The applet's row in the Instruments pane shows its
status (`REFUSED`, `FAILED` or `QUARANTINED`), and with the row selected the
Inspector shows the contractual reason under Status and beside the disabled
`⏎ Launch` button (see [refusals](../references/refusals.md)) — a missing or
malformed manifest, an `id`/`version` that disagrees with the descriptor, an
unmet required service, or an ABI-epoch mismatch. The same rows name the fault
for applets that crashed and were quarantined, and the Activity pane logs each
`launch failed` and `quarantined` event with the applet's name. If an applet
doesn't appear at all, run `caliper check` on the folder you loaded it from: it
lists every manifest it finds there, with the status and reason the host would
give, without opening a window ([Caliper CLI](../references/cli.md#caliper-check)).

A background job that faults or throws prints one line on the host's stderr,
`[jobs] job #3 "train" failed: SIGSEGV (invalid memory access)`, and the jobs
tray marks it `FAILED` with the reason until you press `clear finished`, even
after the host has gone Home. Activity logs it as a `job failed` row. If the job
was submitted from one of the applet's entry points, the applet is quarantined
as well: Activity's next row is `quarantined`, and the status reads
`crashed in background job "train": <reason>` (see
[`caliper.jobs.v1`](../references/service-jobs-v1.md)).

When a fault interrupts the applet's UI code, the host unwinds whatever the
applet left open and prints what it closed, e.g.
`[ui] unwound after an applet fault: Missing End() x2, ImPlot: missing EndPlot()`.
That line tells you how far the frame got before the fault.

**Trying again.** If the fault was a C++ exception, select the applet and press
`⌘R Restart` in the Inspector: the host releases the old instance and launches a
fresh one without restarting Caliper. Restart reuses the library already loaded,
so it re-runs the same code; it is for checking that the applet recovers, not
for loading a fix. To run a rebuilt dylib, and after any signal (the Inspector
then says `restart Caliper to reload`), quit and start Caliper again.

## Where per-applet data lives

The host hands you a writable directory via `host.data_dir()`, and that is the
**only** place an applet should read or write files — never build your own paths
or drop dotfiles in `$HOME`. It is `data/<id>/` under the app-data root, named
by the applet's manifest id, so two applets never collide:

- **macOS:** `~/Library/Application Support/Caliper/`
- **Linux:** `$XDG_DATA_HOME/caliper/` (or `~/.local/share/caliper/`; the folder name is lowercase)
- **Windows:** `%APPDATA%/Caliper/`

`CALIPER_DATA_DIR` moves the whole root, and every applet's data folder with
it: with `CALIPER_DATA_DIR=/tmp/caliper-try`, an applet with the id
`com.example.viewer` gets `/tmp/caliper-try/data/com.example.viewer/`. That is
a quick way to run an applet against an empty data folder without touching the
real one. `caliper shot` uses a temporary root of its own unless
`CALIPER_DATA_DIR` is set; see [Caliper CLI](../references/cli.md#caliper-shot).

The host also keeps `.caliper-data.toml` there for an applet that declares a data
schema; leave that file alone. For example, an applet can build `data_dir() + "/settings.txt"` and
persist a single value across relaunches. (The `signal_scope` applet in
`applets/legacy-dev/` does this, but it is archived and the default build does
not compile it.)

When a stored file looks wrong, inspect it directly on disk under that root
(it's plain files, not a database), delete it to reset to defaults, and relaunch.

## Find what stalls the frame

A window that runs at 2 fps while the applet "does nothing" is a frame-thread
stall. Start with what the host already tells you, then narrow the cause with
timers, working from the outside in.

**The frame watchdog.** The host times the applet's `on_frame` call. When it
exceeds about 250 ms for three consecutive frames, the menu bar shows
`slow: long work belongs in background jobs` and the flag stays until the applet
is relaunched (see [the trust model](../explanations/trust-model.md#frame-watchdog)).
The watchdog times only `on_frame`. A stall in rendering, in presenting or in a
different thread's contention with the frame thread does not raise it, so a
frame that feels slow with no flag raised points away from your frame function.

**The probe ladder.** Add temporary timers and log the result through
`caliper.log.v1`. Go down one rung at a time, because a timer on a lower rung
cannot see time spent on a higher one:

1. **The frame gap around rendering.** Compare the time between two consecutive
   frames with the time spent inside your `on_frame`. If the gap is long and
   `on_frame` is short, the time is spent outside your code, in the host's
   render or present step or waiting on a lock your frame thread shares with a
   worker.
2. **The loop phases.** Time each phase of the host's frame loop separately:
   event polling, the applet frame, the UI draw, the render and the present. This
   names the phase that owns the gap.
3. **The applet's draw phases.** If the applet frame owns the time, time each
   window's drawing block, each texture update and each service call inside
   `on_frame`.

Timers inside the applet alone mislead. They show the applet as innocent when the
cost is in the host, and they show a long `on_frame` without saying which call
inside it blocked.

**The usual cause is I/O on the frame thread.** The rule is that the frame thread
does no I/O: no file access, no network call, no SQL query and no store call
([The frame loop](../introduction/architecture.md#the-frame-loop)). The cost of that work
differs by platform and by session age, so it can look fine on one machine and
freeze the window on another. EmbedScope froze at about 2 fps on Windows because
it rebuilt a table in the on-disk data store on the frame thread twice a second,
and each rebuild took longer than the last. The fixes that apply to any applet:

- Run the work on a thread the applet owns, or in a job for work the user cares
  about, and publish the result to the frame thread under a lock with a
  generation counter.
- Build tables for live visualization as `TEMP` tables, which live in memory and
  do not write to the store's log. See
  [`caliper.data.v1`](../references/service-data-v1.md#persistence-and-high-frequency-rebuilds).
- Create a texture once and call `update_texture` afterwards. On Vulkan with CUDA
  a create costs several times an update, so releasing and re-creating a texture
  every refresh is a frame-thread cost.
- Run expensive probes only while the window that shows them is open.
