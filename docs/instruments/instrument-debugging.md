# Instrument Debugging

An instrument with a library is a shared library that the host loads at runtime. Thus you attach a debugger to the running **`caliper`** host. A `Debug` build of the dylib carries your symbols, so you get source-level stepping and breakpoints inside `on_init`, `on_frame` and `on_cleanup`.

```mermaid
flowchart LR
    build["Debug build"] -->|"caliper dev"| host["Running host"]
    host -->|"attach"| dbg["Debugger"]
    dbg -->|"open instrument"| bp["Breakpoints bind"]
```

Breakpoints in the instrument bind only after you open the instrument, because the host loads it after startup.

## Debugger Attachment

Build both the host and your instrument with debug info. The host builds with the `debug` preset in your Caliper checkout. The instrument builds in its own folder against the installed SDK with `-DCMAKE_BUILD_TYPE=Debug`, as in [First Instrument](first-instrument.md#build). Start the host with `caliper dev` and the instrument's build folder, so that it loads your debug build first.

Attach to the running process and do not launch the host under the debugger. Then you skip the host's boot.

**LLDB (command line)**

```bash
cmake --preset debug && cmake --build --preset debug      # in the Caliper checkout
./cmake-build-debug/caliper dev <instrument folder>/build/instruments &
lldb -p "$(pgrep -f 'cmake-build-debug/caliper')"
```

At the `(lldb)` prompt, set a breakpoint by symbol or by file and line, then continue.

```
(lldb) breakpoint set --name HelloWorldInstrument::on_frame
(lldb) breakpoint set --file hello_world.cpp --line 39
(lldb) continue
```

**CLion**

Use **Run ▸ Attach to Process…**, filter for `caliper`, and pick the running host. Set breakpoints in your instrument source as usual. A breakpoint in `on_frame` does not bind until you open the instrument from the landing screen. CLion then resolves it when the dylib is mapped in. Breakpoints in `on_init` bind at open time. To catch very early setup, break in `on_init` and step from there.

## Host Logging

Do **not** use `printf` or `std::cout` for diagnostics. Log through the `caliper.log.v1` service, which the sugar layer exposes on your `Host`:

```cpp
bool on_init(caliper::Host& host) override {
    host.log_info("hello_world.on_init");           // caliper.log.v1
    // host.log_error("...") for failures
    ...
}
```

Host logging goes to the same sink as the host's own diagnostics, which are the launch console and the dev-mode tail. Each message carries the instrument's identity. Logging keeps working when the dylib's stdio has no terminal, as in a bundled app. Declare `caliper.log.v1` in the `required` list of your manifest, as the Hello World instrument does.

When the loader turns an instrument away, the diagnostic is on the **landing screen** and not in the console. The instrument's row in the Instruments pane shows its status: `REFUSED`, `FAILED` or `QUARANTINED`. When you select the row, the Inspector shows the reason under Status and beside the disabled `⏎ Launch` button (see [Refusal Messages](../reference/refusal-messages.md)). The reason is one of these:

- A missing or malformed manifest
- An `id` or `version` that disagrees with the descriptor
- An unmet required service
- An ABI-epoch mismatch

The same rows name the fault for instruments that crashed and were quarantined. The Activity pane logs each `launch failed` and `quarantined` event with the instrument's name. If an instrument does not appear at all, run `caliper check` on its folder. It lists every manifest there, with the status and reason the host would give, and opens no window ([Instrument Commands](../reference/instrument-commands.md#caliper-check)).

A background job that faults or throws prints one line on the host's stderr, for example `[jobs] job #3 "train" failed: SIGSEGV (invalid memory access)`. The jobs tray marks the job `FAILED` with the reason until you press `clear finished`. Activity logs a `job failed` row. If an entry point of the instrument submitted the job, the host also quarantines the instrument, and the status reads `crashed in background job "train": <reason>` (see [`caliper.jobs.v1`](services/caliper-jobs-v1.md)).

When a fault interrupts the instrument's UI code, the host unwinds whatever the instrument left open and prints what it closed, for example `[ui] unwound after an instrument fault: Missing End() x2, ImPlot: missing EndPlot()`. That line shows how far the frame got before the fault.

**Trying again.** If the fault was a C++ exception, select the instrument and press `⌘R Restart` in the Inspector. The host releases the old instance and launches a fresh one. Restart reuses the loaded library, so it runs the same code. Use it to check that the instrument recovers, not to load a fix. To run a rebuilt dylib, quit and start Caliper again. Do the same after any signal, when the Inspector says `restart Caliper to reload`.

## Data Location

The host gives you a writable directory through `host.data_dir()`. It is the **only** place an instrument reads or writes files. Never build your own paths or put dotfiles in `$HOME`. The directory is `data/<id>/` under the app-data root, named by the manifest id, so two instruments never collide. The app-data root is:

- **macOS:** `~/Library/Application Support/Caliper/`
- **Linux:** `$XDG_DATA_HOME/caliper/`, or `~/.local/share/caliper/` (the folder name is lowercase)
- **Windows:** `%APPDATA%/Caliper/`

`CALIPER_DATA_DIR` moves the whole root. With `CALIPER_DATA_DIR=/tmp/caliper-try`, an instrument with the id `com.example.viewer` gets `/tmp/caliper-try/data/com.example.viewer/`. Use this to run an instrument against an empty data folder. `caliper shot` uses a temporary root of its own unless `CALIPER_DATA_DIR` is set (see [Shot and Serve Commands](../reference/shot-and-serve-commands.md#caliper-shot)).

An instrument can build `data_dir() + "/settings.txt"` and keep a value across relaunches. The host also keeps `.caliper-data.toml` there for an instrument that declares a data schema. Leave that file alone.

The stored files are plain files and not a database. When one looks wrong, inspect it on disk. To reset to defaults, delete it and relaunch.

## Frame Stall Diagnosis

A window that runs at 2 fps while the instrument does nothing has a frame-thread stall. Start with what the host tells you. Then narrow the cause with timers, from the outside in.

**The frame watchdog.** The host times the `on_frame` call of the instrument. When it exceeds about 250 ms for three consecutive frames, the menu bar shows `slow: long work belongs in background jobs`. The flag stays until you relaunch the instrument (see [Trust Model](trust-model.md#frame-watchdog)). The watchdog times only `on_frame`. A stall in rendering, in presenting, or in contention with another thread does not raise it. Thus a slow frame with no flag points away from your frame function.

**The probe ladder.** Add temporary timers and log the result through `caliper.log.v1`. Go down one rung at a time, because a timer on a lower rung cannot see time spent on a higher one:

1. **The frame gap around rendering.** Compare the time between two consecutive frames with the time inside your `on_frame`. If the gap is long and `on_frame` is short, the time goes to the render or present step of the host, or to a wait on a lock that a worker shares with your frame thread.
2. **The loop phases.** Time each phase of the host's frame loop: event polling, the instrument frame, the UI draw, the render and the present. This names the phase that owns the gap.
3. **The instrument's draw phases.** If the instrument frame owns the time, time each window's drawing block, each texture update and each service call inside `on_frame`.

Timers inside the instrument alone mislead. They hide a cost in the host and do not name the call that blocked.

**The usual cause is I/O on the frame thread.** The frame thread does no I/O: no file access, no network call, no SQL query and no store call ([Frame Loop](../introduction/architecture.md#frame-loop)). The cost of that work differs by platform and by session age, so it can freeze the window on one machine only. These fixes apply to any instrument:

- Run the work on a thread the instrument owns, or in a job for work the user cares about. Publish the result to the frame thread under a lock with a generation counter.
- Build tables for live visualization as `TEMP` tables. They live in memory and do not write to the store's log (see [`caliper.data.v1`](services/caliper-data-v1.md#persistence-and-rebuilds)).
- Create a texture once and call `update_texture` afterward. On Vulkan with CUDA, a create costs more than an update, so a release and re-create at every refresh is a frame-thread cost.
- Run expensive probes only while the window that shows them is open.
