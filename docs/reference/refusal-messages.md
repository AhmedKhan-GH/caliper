# Refusal Messages

Before the host loads any applet code, it runs a pure **negotiation** check over
the applet's manifest and the host's own capabilities (see
[Manifest](../applets/manifest.md) and [ABI (epoch 2)](../applets/applet-contract.md)). If the applet is
incompatible, negotiation refuses it and the landing screen shows one of the
messages below as the applet's status text — no code from the applet is
executed.

The checks run in a fixed order (see [Compatibility](../explanations/compatibility.md)): **ABI epoch → minimum host
version → UI-stack hash → libtorch version → required services → data schema**.
The first failing check wins, so only one message is ever shown. These strings
are contractual: the loader and the landing screen reuse them verbatim.

## Negotiation refusals

| Message template | When it fires | What to do |
|------------------|---------------|------------|
| `Built for ABI epoch {applet_epoch}; this host speaks {host_epoch} — check for an applet update.` | The applet's `abi_epoch` does not equal the host's ABI epoch. | Install a build of the applet that targets this host's epoch, or update the host to one that speaks the applet's epoch. |
| `Requires host {min_host} or newer; this host is {host_version}.` | The manifest sets a `min_host` floor and the host's version is older than it (compared numerically, not lexically). | Update the host to at least `{min_host}`. Only fires when `min_host` is set; an empty `min_host` imposes no floor. |
| `Built against UI stack {applet_ui_stack}; this host uses {host_ui_stack} — rebuild the applet against this host's SDK.` | The manifest's `[build].ui_stack` differs from the host's UI-stack hash. Fires only when both are present; a host built without a hash checks nothing. | Rebuild the applet against the SDK installed with this host. |
| `Built against libtorch {applet_torch}; this host provides {host_torch}.` | The manifest's `[compat].torch` differs from the libtorch version the host provides. | Install a build of the applet made against `{host_torch}`, or run it on a host that provides `{applet_torch}`. |
| `Built against libtorch {applet_torch}; this host provides no libtorch.` | The manifest sets `[compat].torch` and the host was built without libtorch. | Run the applet on a host built with libtorch `{applet_torch}`. |
| `Requires a capability this host doesn't have: {service_id}.` | A service listed in the manifest's `required` array is not vended by this host. `{service_id}` is the **first** missing service in declaration order. | Run the applet on a host that provides `{service_id}`, or use an applet build that doesn't require it. Optional services never cause a refusal. |
| `Stored data is schema {stored}; this applet reads schema {applet}.` | The manifest's `[data].schema` is lower than the schema recorded in the applet's data folder, which a newer version of the applet has opened. An applet without the key is never checked. | Install a version of the applet whose schema is at least `{stored}`. |

Placeholders in `{braces}` are filled at runtime: `{applet_epoch}` / `{host_epoch}`
are integers, `{min_host}` / `{host_version}` are semver strings like `0.6.0`,
`{service_id}` is a service id like `caliper.jobs.v1`, `{applet_torch}` /
`{host_torch}` are libtorch version strings like `2.5.1`, `{applet_ui_stack}` /
`{host_ui_stack}` are the first 12 characters of each UI-stack hash, and
`{stored}` / `{applet}` are data-schema integers.

The host records the highest data schema that has opened `data/<id>/` in the
file `.caliper-data.toml` inside that folder. `launch()` checks the stamp again
before it opens the library, so a stamp raised since the scan still refuses
the applet with the same message and the same `Refused` status. It raises the
stamp before `create()` and `initialize()` run.

The required-services check runs a second time inside `launch()`. After the
`dlopen` and the descriptor checks, the loader walks the binary's own
`required_services` list (set by the `CALIPER_APPLET` macro's `.services`)
against the host's services. A service missing there gives the same
`Requires a capability this host doesn't have: {service_id}.` message and the
same `Refused` status, before `create()` runs; the binary stays mapped. This
catches a manifest whose `required` list leaves out a service the binary needs.

## Loader statuses

Negotiation is only the first gate. Once an applet passes it, the **loader**
is what actually finds the binary, `dlopen`s it, checks the
descriptor against the manifest, and runs the lifecycle under the crash guard.
Each of those steps has its own failure, and the loader sorts them into two
statuses:

- **Failed** — the applet is broken but the host is fine: a bad manifest, a
  missing or unloadable binary, a descriptor that contradicts its manifest, or a
  lifecycle hook that returned an honest "no". Re-launchable once the cause is
  fixed.
- **Quarantined** — the applet *faulted* (a signal: SIGSEGV/SIGBUS/SIGFPE/SIGILL,
  including a stack overflow, or a C++ exception escaping an entry point) while
  the host held it, or a job it submitted faulted. The loader never calls the
  faulted instance again — no retry — and the dylib stays mapped. What happens
  next depends on the kind of fault:
    - **After a C++ exception** the stack unwound normally and the process is
      unharmed. The loader keeps the instance, uncalled, and the Inspector
      offers `⌘R Restart`: the loader runs the old instance's `cleanup()` (if
      its `initialize()` had succeeded) and `destroy()`, both guarded, then
      launches a fresh one in the same process. Restart reuses the library
      already loaded, so a rebuilt dylib needs a Caliper restart.
    - **After a signal** memory may be corrupt. The instance is abandoned with
      no `cleanup`/`destroy` (running destructors inside a corrupted image is
      worse than the leak), and the applet stays quarantined until Caliper
      itself restarts; the Inspector says `restart Caliper to reload`. On
      Windows every SEH exception counts as a signal, including a C++ throw
      MSVC could not unwind through the C ABI.

Ordering matters: `scan()` is manifest-first and never `dlopen`s, so parse,
missing-binary, and negotiation results are all decided before any applet code
runs. Everything from `load failed` downward happens only inside `launch()`.

A copy can also be **Shadowed**. Of the copies of one applet id that pass
negotiation, the loader keeps one `Ready`: the highest version, and at equal
versions the copy found first in the [search order](cli.md#the-search-order).
Every other passing copy is `Shadowed`, with the text
`Version {version} in {folder} is used instead.`, where `{version}` and
`{folder}` name the chosen copy. A shadowed copy is never launched, and the
status bar's count of instruments leaves it out. A refused or failed copy keeps
its own reason, so the landing screen shows why a newer copy was passed over.

### Failed messages

| Message template | When it fires | What to do |
|------------------|---------------|------------|
| _(manifest parser text, passed through verbatim)_ | `scan()` could not parse `caliper.toml` — malformed TOML or a missing/invalid required field. The error is whatever the [manifest](../applets/manifest.md) parser reported. | Fix the manifest as the message describes; see [Manifest](../applets/manifest.md) for the required fields and formats. |
| `applet.id is not a valid applet id (lowercase reverse-DNS, [a-z0-9.-]): {id}` | The parser's own message for an `id` that breaks the [id rules](../applets/manifest.md#applet-ids): a character outside `[a-z0-9.-]`, an empty segment (`..`, a leading or trailing dot), or more than 128 characters. | Rename the applet to a lowercase reverse-DNS id, and use the same id in the descriptor. |
| `applet id is not path-safe: {id}` | The loader's own check of the id, made before it builds the applet's data folder path. The parser refuses the same ids first, so this appears only for a manifest that reached the loader another way. | As above. |
| `applet binary not found next to {manifest_filename}` | `scan()` found the manifest but no sibling shared library. The loader looks for `<stem>` and `lib<stem>` with the platform extension (`.dylib`/`.so`/`.dll`) beside `<stem>.caliper.toml`. | Ship the compiled applet library next to its manifest, named to match the manifest's stem. |
| `load failed: {dlerror}` | `dlopen`/`LoadLibrary` failed at `launch()` — a corrupt image, a wrong architecture, or an unresolved dependency. `{dlerror}` is the OS loader's own text. | Rebuild the applet for this platform/arch and make sure its dependencies resolve. |
| `missing export caliper_applet_descriptor` | The library loaded but does not export the one required symbol. Usually a non-applet library, or one built without `CALIPER_APPLET` / the `CALIPER_EXPORT` visibility attribute. | Build with the SDK's `CALIPER_APPLET` macro (or export the descriptor by hand) so the symbol is visible. |
| `descriptor missing or truncated` | The descriptor getter returned NULL, or a `struct_size` smaller than this host's `CaliperAppletDescriptor` — an ABI too old to trust. | Rebuild against the current SDK headers for this epoch. |
| `descriptor ABI epoch disagrees with manifest` | The compiled-in `abi_epoch` does not match the epoch the host negotiated from the manifest. The binary and its manifest were built against different epochs. | Rebuild the binary and manifest together against one SDK epoch. |
| `descriptor id disagrees with manifest` | The descriptor's `id` is NULL or differs from the manifest's `id`. The manifest describes a different applet than the binary. | Make the manifest `id` and the descriptor `id` identical (rebuild if the descriptor is stale). |
| `descriptor version disagrees with manifest` | The descriptor's `version` is NULL or differs from the manifest's `version`. | Bump manifest and descriptor to the same version and rebuild. |
| `descriptor function table incomplete` | One of the required `api` entry points (`create`, `destroy`, `initialize`, `frame`, `cleanup`) is NULL. | Provide all five lifecycle functions; the `CALIPER_APPLET` macro wires them for you. |
| `could not record the data schema in {stamp_path}` | The manifest declares `[data].schema`, and the host could not write the data-schema stamp in the applet's data folder. The library is open, but none of the applet's entry points has run. | Make the applet's data folder writable. |
| `create() returned null` | `create()` ran cleanly but returned no instance — the applet declined to construct itself (e.g. an internal precondition failed). | Investigate the applet's `create`; return a valid instance or surface the real error from `initialize` instead. |
| `initialize() returned false` | `initialize()` ran cleanly but reported failure. The loader then calls `destroy()` on the instance and marks the applet Failed. | Check the applet's own logs for why init refused (a required service or resource it couldn't obtain). |

### Quarantined messages

| Message template | When it fires | What to do |
|------------------|---------------|------------|
| `crashed in create(): {fault}` | The applet faulted inside `create()`. The instance never existed; nothing to clean up. | Fix the fault in the applet's constructor path, then restart it (see below). |
| `crashed in initialize(): {fault}` | The applet faulted inside `initialize()`. After a signal the instance is abandoned — `destroy()` is **not** called on it; after an exception it is kept for Restart, which calls `destroy()` (not `cleanup()`, since `initialize()` never succeeded). | Fix the fault in the applet's init path, then restart it. |
| `crashed in frame(): {fault}` | The applet faulted during a `frame()` call. The instance's `frame()` is never called again; after a signal the instance is abandoned, after an exception it is kept for Restart. | Fix the fault in the applet's per-frame path, then restart it. |
| `crashed during teardown: {fault}` | The applet faulted in `cleanup()` or `destroy()` while shutting down. Teardown stops; the dylib is left mapped. After an exception in `cleanup()`, the `destroy()` still owed is made by Restart. | Fix the fault in the applet's teardown path, then restart it. |
| `crashed in background job "{label}": {fault}` | A job the applet submitted through `caliper.jobs.v1` — from one of its entry points, or from another of its jobs — faulted or threw (see [jobs.v1](../applets/services/caliper-jobs-v1.md)). The loader quarantines the applet before its next `frame()` or at teardown, whichever comes first; a `frame()` already running finishes. A job that threw leaves the applet restartable; a job that took a signal does not. | Fix the fault in the job function, then restart the applet. |
| `crashed while restarting: {fault}` | Restart's guarded `cleanup()` or `destroy()` of the old instance faulted. The applet is quarantined again with this reason; after a throw from `cleanup()`, a later Restart tries the old instance's `destroy()` again. | Fix the fault in the applet's teardown path; after a signal, restart Caliper. |

`{fault}` is the crash guard's description of the fault: a signal such as
`SIGSEGV (invalid memory access)`, `C++ exception: <what()>`, or
`C++ exception (non-std type)`. `{label}` is the label the job was submitted
with, `{dlerror}` is the platform loader's message, `{manifest_filename}` is the
manifest file the applet was scanned from, `{id}` is the manifest's
`applet.id`, and `{stamp_path}` is the stamp file `.caliper-data.toml` in the
applet's data folder.

"Restart it" above means `⌘R Restart` in the Inspector when the fault was a
C++ exception, and a restart of Caliper when it was a signal (see
[Loader statuses](#loader-statuses)).

!!! note
    The landing screen shows these statuses. An applet that is not
    `Ready`/`Active` keeps its row in the Instruments pane, with `REFUSED`,
    `FAILED`, `QUARANTINED` or `SHADOWED` in the Status column. With the row selected, the
    Inspector's Status field shows the exact `status_text` from the tables above
    (the negotiation refusal, `Failed`, `Quarantined` or `Shadowed` message), and the
    `⏎ Launch` button is disabled. Beside it is the same text for a refused,
    failed or shadowed applet, `⌘R Restart` for one quarantined by a C++ exception, and
    `restart Caliper to reload` for one quarantined by a signal. A launch that
    fails is also logged in the Activity pane as `launch failed` with the
    reason, and a quarantine while the applet runs (in `frame()`, or from one
    of its jobs) as `quarantined`, preceded by a `job failed` row when a job
    caused it. Restarts are logged as `restart` and `restart failed`. The
    landing screen reads the statuses every frame, so a mid-session quarantine
    shows up as soon as the host returns to it.

## Spec applets

A [spec applet](../applets/manifest.md#spec-applets) has no library, so none of the steps that open or call one applies to it: there is no `dlopen`, no descriptor, no lifecycle hook, and it is never `QUARANTINED`. The loader lists it with the same statuses, `READY`, `REFUSED`, `FAILED` and `SHADOWED`, and chooses among copies of one id by the same rule, whatever their kind.

When the loader lists a spec applet it makes only the cheap checks, in this order: the manifest parses (its messages are the parser's, as for a compiled applet, and include the keys a spec applet refuses), the id is path-safe, the spec file passes the file checks, its `caliper_spec` line is read, then `min_host`, the spec runtime and the spec format version. The first failure wins. The full validation of the spec runs when the applet is launched and in `caliper check`, through the builder's spec library, and its first error becomes the applet's reason.

### Spec applet refusals

| Message template | When it fires | What to do |
|------------------|---------------|------------|
| `Requires host {min_host} or newer; this host is {host_version}.` | The manifest sets `[compat].min_host` and the host is older, as for a compiled applet. | Update the host. |
| `Requires the spec runtime, which this build of Caliper does not have.` | The host was built without the builder (`CALIPER_WITH_BUILDER=OFF`, the default on Windows), so it cannot validate or open a spec. | Use a build of Caliper with the builder. |
| `Written in spec format {spec_version}; this host reads spec format {host_version} — update Caliper to open it.` | The spec's `caliper_spec` is a later version than this host reads. For an earlier version that the host does not read, the message ends after `{host_version}`. | Open the applet with a newer Caliper. |
| `A workspace holds spec applets only; a compiled applet is installed in the user applets folder.` | A compiled applet's manifest is in the open workspace's `applets/` folder or one folder below it. Its library is never opened. | Install the compiled applet with `caliper install`, and keep the workspace for spec applets. |

### Spec applet failures

| Message template | When it fires | What to do |
|------------------|---------------|------------|
| `spec file {spec} not found next to {manifest_filename}` | The file that `applet.spec` names is not in the manifest's folder. | Put the spec beside its manifest, or correct `applet.spec`. |
| `spec file {spec} is a symbolic link; a spec applet's spec is a regular file in its folder` | The spec is a symbolic link, wherever it points. The loader never follows one, so a spec cannot lead out of its folder. | Replace the link with the file. |
| `spec file {spec} is not a regular file` | The spec's name is a folder or another kind of file. | Put the spec file there. |
| `spec file {spec} is larger than 512 KiB` | The spec is longer than the spec format's limit. | Make the spec smaller. |
| `spec file {spec} has no line caliper_spec = <version> before its first table` | No top-level `caliper_spec` key comes before the first table header, so the version cannot be read without parsing the file. | Start the spec with `caliper_spec = 0`, as the builder writes it. |
| `spec file {spec} has a caliper_spec that is not a decimal integer` | The version line's value is not a decimal integer. | Write the version as a decimal integer. |
| `manifest is larger than 64 KiB` | A manifest one folder down, or in the workspace's folder, is longer than 64 KiB. | Shorten the manifest. |
| `manifest has a dotted key of more than 32 parts` | Such a manifest has a dotted key of more than 32 parts, which is refused before it is parsed. | Shorten the key. |
| `{spec}:{line}:{column}: {code}: {message}` | The spec fails the full validation at launch or in `caliper check`: the spec library's first error, with its place in the file, its stable code (such as `spec.unknown_source`) and its sentence. Without a known place it reads `{spec}: {code}: {message} (at {path})`. | Fix the spec as the message says; `caliper check <folder>` prints every error and warning. |

`{spec}` is the spec's file name, `{manifest_filename}` the manifest's, `{spec_version}` and `{host_version}` are spec format versions (the host reads version 0), and `{line}` and `{column}` count from 1.

A spec applet that passes every check opens on the app's spec page when it is launched ([The spec page](cli.md#the-spec-page)), and is `ACTIVE` while the page shows it. The loader itself only lists and validates it: a program that calls `AppletLoader::launch()` on one, rather than the app, gets `A spec applet opens as a page, which this version of Caliper does not draw yet.`, and the applet stays `READY`.
