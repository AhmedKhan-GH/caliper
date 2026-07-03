# SignalScope — the exemplar Caliper applet

A live multi-lead signal viewer whose only real purpose is to be copied: it
demonstrates every idiom of writing an applet against the Caliper SDK
(ABI epoch 2), in ~200 lines.

> **Status:** compiles once Phase 1 of `PLATFORM.md` lands (the SDK headers
> and sugar layer it uses are built by
> `docs/superpowers/plans/2026-07-01-platform-phase0-phase1.md`, which wires
> this example into the build at Task 11). The code is complete now so it can
> be read as the reference while the platform work proceeds.

## Anatomy

| File | Role |
|---|---|
| `signal_scope.cpp` | The applet. One class + one macro. The numbered `EXEMPLAR` comments are the curriculum. |
| `signal_scope.caliper.toml` | The manifest — identity, ABI epoch, required/optional services. Checked by the host **before** your code loads; must agree with the macro's fields (the loader verifies). |
| `CMakeLists.txt` | The entire build. In-tree it links `caliper::sdk` + `caliper::ui_stack`; an independent repo swaps those for a `find_package`/CPM fetch of a tagged SDK release — nothing else changes. |

## The idioms it teaches (EXEMPLAR 1–8 in the source)

1. Include one SDK header, then program **raw ImGui/ImPlot** — no wrapper layer.
2. `on_init` receives your `Host&` for life; log through the host, not stdout.
3. **Probe optional services, assume required ones** — required is enforced by
   the manifest gate before you load; optional returns `nullptr` and you
   degrade gracefully (shown with `caliper.metrics.v1`, which arrives in Phase 2).
4. Persist only under `host.data_dir()` — your per-applet sandbox.
5. Animate from `frame.delta_sec`, size UI in logical units, treat
   `fb_width/fb_height` as physical pixels (`dpi_scale` converts).
6. **Never block the frame thread** — the Anti-patterns section deliberately
   violates this so you can watch the host's watchdog flag it (long work
   belongs in `caliper.jobs.v1` once Phase 2 ships it).
7. `on_cleanup` is symmetric with `on_init`: persist, release, log.
8. `CALIPER_APPLET(...)` is the entire ABI surface of the dylib — descriptor,
   exception-safe C bridges, and UI-context/allocator connection, generated.

## Run it

After Phase 1: build the repo, launch `caliper`, click the SignalScope card.
Pause/speed/leads exercise the controls; the watchdog demo is under
"Anti-patterns"; settings persist across relaunches via the data dir.
