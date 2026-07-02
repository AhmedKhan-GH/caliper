# Debug an applet

An applet is a shared library the host loads at runtime, so you don't debug it as
its own program — you attach a debugger to the running **`caliper`** host. The
dylib carries your own symbols (built `Debug`, they're right there), so once
attached you get full source-level stepping, breakpoints, and variable
inspection inside `on_init` / `on_frame` / `on_cleanup`.

## Attach a debugger to the running host

Build both the host and your applet with debug info (the default `Debug`
configuration), then launch and attach:

**LLDB (command line)**

```bash
cmake --build build --target caliper hello_applet
./build/caliper &                      # note the PID it prints, or use pgrep
lldb -p "$(pgrep -f 'build/caliper')"
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
loaded lazily when you open the applet card, a breakpoint in `on_frame` won't
bind until the applet is opened in the app — CLion resolves it the moment the
dylib is mapped in, which is expected. Breakpoints in `on_init` bind at open
time; to catch very early setup, break in `on_init` and step from there.

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

Loader diagnostics from the host itself are prefixed `[applet]` — that is where
you'll see `Loaded: …` for applets the current v1 loader accepts, and
`missing applet_info()` for epoch-2 dylibs the v1 loader skips (expected until
loader v2). Grep the host's stdout for `[applet]` when an applet fails to appear.

## Where per-applet data lives

The host hands you a writable directory via `host.data_dir()`, and that is the
**only** place an applet should read or write files — never build your own paths
or drop dotfiles in `$HOME`. On the current host this resolves under the app-data
root:

- **macOS:** `~/Library/Application Support/Caliper/`
- **Linux:** `$XDG_DATA_HOME/Caliper/` (or `~/.local/share/Caliper/`)
- **Windows:** `%APPDATA%/Caliper/`

Under loader v2 the host namespaces this directory per applet id, so two applets
never collide; either way, `host.data_dir()` is the accessor and your code stays
the same. The `signal_scope` exemplar shows the pattern — it builds
`data_dir() + "/settings.txt"` and persists a single value across relaunches.

When a stored file looks wrong, inspect it directly on disk under that root
(it's plain files, not a database), delete it to reset to defaults, and relaunch.
