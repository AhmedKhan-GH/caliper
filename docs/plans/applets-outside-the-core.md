# Applets outside the core

The host should ship with no applets and name none. Applets are built, stored
and installed outside the core repository, and adding one never means editing
the host. This page records what already works, what is still hard-coded, and
the changes that remove it. The decisions are
[D19](../decisions/d019-host-names-no-applet.md),
[D20](../decisions/d020-applet-search-order.md),
[D21](../decisions/d021-shipped-applets-in-the-app-bundle.md) and
[D22](../decisions/d022-third-party-applet-signing.md). The registry is a separate
plan, [Applet registry](registry.md).

Two parts of this plan are undecided: the `caliper` command-line tool (`caliper dev`,
`caliper new`, `caliper install`) and the `.caliperapp` bundle format. The
[roadmap](roadmap.md) lists them under "Undecided"; the sections below describe
them as proposed.

## What already works

- The loader finds applets by scanning folders, not by name. It looks in
  `<exe>/applets/` and in the user data folder
  (`~/Library/Application Support/Caliper/applets/` on macOS).
- Each applet's `caliper.toml` is checked before the `dlopen`: the ABI epoch,
  the minimum host version and the services it needs
  (see [Trust model](../explanations/trust-model.md)).
- Each applet gets its own data folder, named by its manifest id.
- `sdk/` holds the headers, the C service tables, `caliper-sdk-config.cmake` and
  a testing fixture.

## What is still hard-coded

| Where | What | Effect |
|---|---|---|
| `src/main.cpp`, the default dock layout (about lines 298 to 316) | Lists window titles of specific applets ("EmbedScope: Cloud", "GPTScope: Logit Lens", "FlowScope: Field" and others) and places each one in the centre, the side column or the bottom | A new applet gets no layout unless the host is edited |
| Root `CMakeLists.txt`, the Applets section | Builds every folder under `applets/*` and makes the `caliper` target depend on each; adds `hello`, `sine_scope` and `opengllama` by name | The core build compiles applets |
| Root `CMakeLists.txt`, the stale-library cleanup | Deletes any applet library or manifest in the build's `applets/` folder that is not on its own list | An applet built elsewhere and copied in is deleted on the next configure |

## The changes

### Layout by role

Applets declare a role for each of their windows, and the host lays out roles,
never names. A proposed `caliper.ui.v1` addition, called next to the window's
`Begin()`:

```c
ui->dock_hint("EmbedScope: Cloud", CALIPER_DOCK_CENTRAL);
ui->dock_hint("EmbedScope: Training", CALIPER_DOCK_SIDE_TOP);
ui->dock_hint("EmbedScope: Tensors", CALIPER_DOCK_SIDE_BOTTOM);
```

The hint lives in code beside the title, so the two cannot drift apart the way
a list in the manifest could. Windows with no hint dock in the centre. A layout
the user has arranged still wins over every hint.

### The SDK as an installed package

`cmake --install` produces `caliper-sdk` with:

- `caliper::sdk`: the headers and the C service tables;
- `caliper::ui_stack`: the pinned ImGui, ImPlot and ImPlot3D;
- `caliper_add_applet()`: builds the module, validates the manifest, makes the
  bundle and registers the conformance test.

The SDK ships as a release asset beside the app. A separate SDK repository can
wait until outside contributors exist.

The host hands its ImGui context to every applet, so an applet must be built
against exactly the host's ImGui version and `imconfig`. The SDK ships that
pinned set, and the conformance test checks its hash. Without the check, a
mismatch crashes in ways that are hard to trace.

### An empty core build

- The Applets loop and the stale-library cleanup leave the root `CMakeLists.txt`.
- `CALIPER_BUILD_APPLETS` goes away.
- First-party applets move to one `caliper-applets` repository with their
  history ([D8](../decisions/d008-applets-keep-their-history.md)), rather than one repository each.
- `hello` stays in this repository as the loader-test fixture under `tests/`.
- `examples/hello` and `examples/sine_scope` stay, because the tutorials embed
  them. CI builds them against the installed SDK, as an outside developer
  would; they are no longer dependencies of the `caliper` target.

### Loading during development

- `--applets <dir>` and `CALIPER_APPLET_PATH` add folders to the search, so a
  dev build loads applets from their own build folders without copying.
- `caliper dev <folder>` ([D9](../decisions/d009-cli-as-host-subcommands.md)) loads one applet from its build folder and reloads
  it when its library is rebuilt.
- `caliper new <name>` creates an applet from a template: manifest, CMake and CI.

### Installing

- Copying a `.caliperapp` folder into the user applets folder keeps working.
- `caliper install <path|url>` checks the sha256, places the bundle and handles
  the quarantine flag under the signing policy (D22).
- A Browse tab reading the registry comes later ([Applet registry](registry.md)).

## Where applets are stored

| What | Where |
|---|---|
| Source | The applet's own repository. First-party applets in `caliper-applets`; private applets in private repositories |
| Releases | The applet repository's releases, as `<id>-<version>-<platform>.caliperapp.zip`; the registry points to them |
| Shipped with the app | `Caliper.app/Contents/PlugIns/`, signed with the app (D21) |
| Installed by the user | macOS `~/Library/Application Support/Caliper/applets/<id>.caliperapp/`; Windows `%APPDATA%\Caliper\applets\`; Linux `~/.local/share/caliper/applets/` |
| Applet data | `~/Library/Application Support/Caliper/data/<id>/` |
| Rebuildable caches | `~/Library/Caches/Caliper/<id>/` |
| Shared runtimes | `~/Library/Application Support/Caliper/runtimes/` ([D5](../decisions/d005-runtime-packs.md)) |
| During development | The applet's build folder, found through `--applets`, `CALIPER_APPLET_PATH` or `caliper dev` |

The host scans `<exe>/applets/` today, which inside a macOS app is
`Contents/MacOS/applets/`. Apple expects loadable code in `Contents/PlugIns/`,
and anything written into the app bundle breaks its signature. Shipped applets
therefore move to `Contents/PlugIns/`, and the app bundle is never written after
it ships.

### Search order

1. Development paths (`--applets`, `CALIPER_APPLET_PATH`, `caliper dev`)
2. The user applets folder
3. The applets shipped with the app

When two applets share an id, the higher version wins. A developer can override
a shipped applet, and a user-installed update replaces the shipped copy without
touching the app.

## Signing on macOS

A notarized host with the hardened runtime refuses to load libraries signed by a
different team. Two options:

- The host carries the `com.apple.security.cs.disable-library-validation`
  entitlement and accepts third-party applets, relying on the manifest checks,
  the sha256 and the trust model.
- Only applets signed by the same team load; others must be re-signed locally.

Downloaded bundles also carry Gatekeeper's quarantine flag, which
`caliper install` must handle. The policy is D22 and must be decided before the
first third-party applet ships.

## Checks that keep it this way

- **No applets present.** CI builds the host, starts it headless with an empty
  applets folder, and checks that the launcher shows its empty state.
- **The host names no applet.** CI collects the ids and window titles from every
  known `caliper.toml` and applet source, searches `src/` and `include/` for
  them, and fails on a match.

## Order of work

1. Install the SDK and build one applet outside the repository against it.
2. Remove the applets from the core build, and replace the dock list with role hints.
3. Add the development paths, `caliper dev` and `caliper install`.
4. Move the first-party applets to `caliper-applets`.
5. Add the registry and the Browse tab.
