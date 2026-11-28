# Refusal Messages

Before the host loads any instrument code, it runs a pure **negotiation** check on the instrument's manifest and the host's capabilities. See [Manifest](../instruments/manifest.md) and [Instrument Contract (ABI epoch 4)](../instruments/instrument-contract.md). If the instrument is incompatible, the host refuses it, runs no instrument code, and shows one message below as the status text on the landing screen.

The checks run in this fixed order: **ABI epoch → minimum host version → UI-stack hash → libtorch version → required services → data schema**. See [Compatibility and Epochs](../development/compatibility-and-epochs.md). The first failing check wins, so the host shows only one message. These strings are contractual, and the loader and the landing screen reuse them verbatim.

## Negotiation Refusals

| Message template | When it fires | What to do |
|------------------|---------------|------------|
| `Built for ABI epoch {instrument_epoch}; this host speaks {host_epoch} — check for an instrument update.` | The instrument's `abi_epoch` differs from the host's. | Install an instrument build for this host's epoch, or update the host. |
| `Requires host {min_host} or newer; this host is {host_version}.` | The manifest sets `min_host` and the host is older. The comparison is numeric. An empty `min_host` sets no floor. | Update the host to at least `{min_host}`. |
| `Built against UI stack {instrument_ui_stack}; this host uses {host_ui_stack} — rebuild the instrument against this host's SDK.` | The manifest's `[build].ui_stack` differs from the host's UI-stack hash. The check runs only when both exist. | Rebuild the instrument against this host's SDK. |
| `Built against libtorch {instrument_torch}; this host provides {host_torch}.` | The manifest's `[compat].torch` differs from the host's libtorch version. | Install an instrument build made against `{host_torch}`, or use a host that provides `{instrument_torch}`. |
| `Built against libtorch {instrument_torch}; this host provides no libtorch.` | The manifest sets `[compat].torch` and the host has no libtorch. | Use a host built with libtorch `{instrument_torch}`. |
| `Requires a capability this host doesn't have: {service_id}.` | A service in the manifest's `required` array is not vended by this host. `{service_id}` is the **first** missing service in declaration order. | Use a host that provides `{service_id}`, or an instrument build that does not require it. Optional services never cause a refusal. |
| `Stored data is schema {stored}; this instrument reads schema {instrument}.` | The manifest's `[data].schema` is lower than the schema that a newer instrument version stored in the data folder. An instrument without the key is never checked. | Install an instrument version with a schema of at least `{stored}`. |

Placeholders in `{braces}` are filled at runtime. `{min_host}` and `{host_version}` are semver strings such as `0.6.0`. `{service_id}` is a service id such as `caliper.jobs.v1`. `{instrument_torch}` and `{host_torch}` are libtorch versions such as `2.5.1`. `{instrument_ui_stack}` and `{host_ui_stack}` are the first 12 characters of each UI-stack hash. The epoch and schema placeholders are integers.

The host records the highest data schema that opened `data/<id>/` in the file `.caliper-data.toml` in that folder. `launch()` checks this stamp again before it opens the library. If the stamp rose after the scan, the instrument gets the same message and the same `Refused` status. The host raises the stamp before `create()` and `initialize()` run.

The required-services check also runs inside `launch()`. After `dlopen` and the descriptor checks, the loader compares the binary's own `required_services` list (set by `.services` in the `CALIPER_INSTRUMENT` macro) with the host's services. A missing service gives the same message and the same `Refused` status before `create()` runs, and the binary stays mapped. This catches a manifest whose `required` list omits a service that the binary needs.

## Loader Statuses

After an instrument passes negotiation, the **loader** finds the binary, calls `dlopen`, checks the descriptor against the manifest, and runs the lifecycle under the crash guard. It sorts failures into two statuses:

- **Failed**: The instrument is broken but the host is fine. Causes are a bad manifest, a missing or unloadable binary, a descriptor that contradicts its manifest, or a lifecycle hook that returned an honest "no". The instrument can launch again after the cause is fixed.
- **Quarantined**: The instrument faulted while the host held it, or a job it submitted faulted. A fault is a signal (SIGSEGV, SIGBUS, SIGFPE or SIGILL, including a stack overflow) or a C++ exception that escapes an entry point. The loader never calls the faulted instance again, and the dylib stays mapped.

What happens next depends on the kind of fault:

- **After a C++ exception**: The process is unharmed. The loader keeps the instance, uncalled, and the Inspector offers `⌘R Restart`. Restart runs the old instance's `cleanup()` (if its `initialize()` succeeded) and `destroy()`, both guarded. Then it launches a fresh instance in the same process. Restart reuses the loaded library, so a rebuilt dylib needs a Caliper restart.
- **After a signal**: Memory may be corrupt. The loader abandons the instance and calls neither `cleanup()` nor `destroy()`. The instrument stays quarantined until Caliper restarts, and the Inspector says `restart Caliper to reload`. On Windows, every SEH exception counts as a signal, including a C++ throw that MSVC cannot unwind through the C ABI.

`scan()` is manifest-first and never calls `dlopen`. Thus the host decides the parse, missing-binary and negotiation results before any instrument code runs. Everything from `load failed` downward happens only inside `launch()`.

A copy can also be **Shadowed**. Of the copies of one instrument id that pass negotiation, the loader keeps one `Ready`. It picks the highest version, and at equal versions the copy found first in the [search order](cli.md#search-order). Every other passing copy is `Shadowed`, with the text `Version {version} in {folder} is used instead.` The host never launches a shadowed copy, and the status bar's instrument count leaves it out. A refused or failed copy keeps its own reason.

### Failed Messages

| Message template | When it fires | What to do |
|------------------|---------------|------------|
| _(manifest parser text, passed through verbatim)_ | `scan()` could not parse `caliper.toml`. The TOML is malformed, or a required field is missing or invalid. | Fix the manifest as the message says. See [Manifest](../instruments/manifest.md). |
| `manifest missing [instrument] table` | The manifest has no `[instrument]` table. | Add the `[instrument]` table. |
| `instrument.kind is not a string` | `instrument.kind` is present and not a string. | Set `instrument.kind` to `"library"` or `"spec"`. |
| `instrument.kind is not "library" or "spec": {word}` | `instrument.kind` names an unknown kind. | Use `"library"` or `"spec"`. |
| `manifest missing instrument.id` | `instrument.id` is absent or empty. | Set `instrument.id`. |
| `instrument.id is not a valid instrument id (lowercase reverse-DNS, [a-z0-9.-]): {id}` | The `id` breaks the [id rules](../instruments/manifest.md#instrument-ids): a character outside `[a-z0-9.-]`, an empty segment, or more than 128 characters. | Rename the instrument to a lowercase reverse-DNS id, and use the same id in the descriptor. |
| `manifest missing instrument.name` | `instrument.name` is absent or empty. | Set `instrument.name`. |
| `manifest missing instrument.version` | `instrument.version` is absent or empty. | Set `instrument.version`. |
| `instrument.version is not strict semver x.y.z: {version}` | `instrument.version` is not three numeric parts. | Use strict semver `x.y.z`. |
| `manifest missing instrument.spec, the spec file of a kind = "spec" instrument` | `instrument.spec` is absent for `kind = "spec"`. | Set `instrument.spec` to the spec file name. |
| `instrument.spec is not a plain file name in the instrument's folder: {file}` | `instrument.spec` contains a path separator or invalid characters. | Use a plain file name in the instrument's folder. |
| `compat.abi_epoch does not apply to a kind = "spec" instrument, which has no library` | `compat.abi_epoch` is present for `kind = "spec"`. | Remove `compat.abi_epoch`. |
| `compat.torch does not apply to a kind = "spec" instrument, which has no library` | `compat.torch` is present for `kind = "spec"`. | Remove `compat.torch`. |
| `[build] does not apply to a kind = "spec" instrument, which has no library` | `[build]` is present for `kind = "spec"`. | Remove `[build]`. |
| `[services] does not apply to a kind = "spec" instrument, which calls no service` | `[services]` is present for `kind = "spec"`. | Remove `[services]`. |
| `[data] does not apply to a kind = "spec" instrument, which keeps no data of its own` | `[data]` is present for `kind = "spec"`. | Remove `[data]`. |
| `instrument id is not path-safe: {id}` | The loader's own id check, made before it builds the data folder path. The parser refuses the same ids first. | Rename to a valid lowercase reverse-DNS id. |
| `instrument binary not found next to {manifest_filename}` | `scan()` found the manifest but no sibling library. The loader looks for `<stem>` and `lib<stem>` with the platform extension (`.dylib`, `.so` or `.dll`) beside `<stem>.caliper.toml`. | Put the compiled library next to its manifest, named after the manifest's stem. |
| `load failed: {dlerror}` | `dlopen` or `LoadLibrary` failed at `launch()`: a corrupt image, a wrong architecture, or an unresolved dependency. `{dlerror}` is the OS loader's text. | Rebuild the instrument for this platform and architecture, and resolve its dependencies. |
| `missing export caliper_instrument_descriptor` | The library does not export the required symbol. It is usually not an instrument library, or it was built without `CALIPER_INSTRUMENT` or `CALIPER_EXPORT`. | Build with the SDK's `CALIPER_INSTRUMENT` macro, or export the descriptor by hand. |
| `descriptor missing or truncated` | The descriptor getter returned NULL, or `struct_size` is smaller than this host's `CaliperInstrumentDescriptor`. | Rebuild against the current SDK headers. |
| `descriptor ABI epoch disagrees with manifest` | The compiled-in `abi_epoch` differs from the manifest's epoch. | Rebuild the binary and manifest together against one SDK epoch. |
| `descriptor id disagrees with manifest` | The descriptor's `id` is NULL or differs from the manifest's `id`. | Make both ids identical. |
| `descriptor version disagrees with manifest` | The descriptor's `version` is NULL or differs from the manifest's `version`. | Set the same version in both and rebuild. |
| `descriptor function table incomplete` | One of the required `api` entry points (`create`, `destroy`, `initialize`, `frame`, `cleanup`) is NULL. | Provide all five functions. The `CALIPER_INSTRUMENT` macro wires them. |
| `could not record the data schema in {stamp_path}` | The manifest declares `[data].schema`, and the host could not write the stamp in the data folder. No entry point has run. | Make the data folder writable. |
| `create() returned null` | `create()` ran cleanly but returned no instance. | Return a valid instance, or report the real error from `initialize()`. |
| `initialize() returned false` | `initialize()` ran cleanly but reported failure. The loader calls `destroy()` on the instance. | Check the instrument's logs for the reason. |

### Quarantined Messages

| Message template | When it fires | What to do |
|------------------|---------------|------------|
| `crashed in create(): {fault}` | The instrument faulted inside `create()`. The instance never existed, so nothing needs cleanup. | Fix the constructor path, then restart. |
| `crashed in initialize(): {fault}` | The instrument faulted inside `initialize()`. After a signal, `destroy()` is not called. After an exception, Restart calls `destroy()` but not `cleanup()`. | Fix the init path, then restart. |
| `crashed in frame(): {fault}` | The instrument faulted during `frame()`. The loader never calls `frame()` again. | Fix the per-frame path, then restart. |
| `crashed during teardown: {fault}` | The instrument faulted in `cleanup()` or `destroy()`. Teardown stops. After an exception in `cleanup()`, Restart makes the `destroy()` call that is still owed. | Fix the teardown path, then restart. |
| `crashed in background job "{label}": {fault}` | A job that the instrument submitted through `caliper.jobs.v1` faulted or threw. See [`caliper.jobs.v1`](../instruments/services/caliper-jobs-v1.md). The loader quarantines the instrument before the next `frame()` or at teardown. A `frame()` that already runs finishes. After a throw the instrument can restart, but not after a signal. | Fix the job function, then restart. |
| `crashed while restarting: {fault}` | The guarded `cleanup()` or `destroy()` of the old instance faulted during Restart. After a throw from `cleanup()`, a later Restart tries `destroy()` again. | Fix the teardown path. After a signal, restart Caliper. |

`{fault}` is the crash guard's description: a signal such as `SIGSEGV (invalid memory access)`, `C++ exception: <what()>`, or `C++ exception (non-std type)`. `{label}` is the label of the job. `{dlerror}`, `{manifest_filename}`, `{id}` and `{stamp_path}` name the OS loader message, the scanned manifest file, the manifest's `instrument.id` and the `.caliper-data.toml` file.

"Restart" in the table means `⌘R Restart` in the Inspector after a C++ exception, and a restart of Caliper after a signal. See [Loader Statuses](#loader-statuses).

!!! note
    An instrument that is not `Ready` or `Active` keeps its row in the Instruments pane, with `REFUSED`, `FAILED`, `QUARANTINED` or `SHADOWED` in the Status column. When you select the row, the Inspector's Status field shows the exact `status_text` from the tables above, and `⏎ Launch` is disabled. A quarantined instrument also shows `⌘R Restart` or `restart Caliper to reload`. The Activity pane logs `launch failed` with the reason, `quarantined` (after a `job failed` row if a job caused it), `restart` and `restart failed`. The landing screen reads the statuses every frame, so a quarantine during a session shows at once.

## Spec-Based Instruments

An [instrument written as a spec](../instruments/manifest.md#spec-based-instruments) has no library. Thus it has no `dlopen`, no descriptor and no lifecycle hook, and it is never `QUARANTINED`. The loader lists it as `READY`, `REFUSED`, `FAILED` or `SHADOWED`, and it picks among copies of one id by the same rule as for library instruments.

The loader makes only cheap checks when it lists a spec-based instrument, in this order: the manifest parses, the id is path-safe, the spec file passes the file checks, the `caliper_spec` line is read, then `min_host`, the spec runtime and the spec format version pass. The first failure wins. The manifest messages are the parser's, and they include the keys a `kind = "spec"` manifest refuses. The full validation runs at launch and in `caliper check`, and its first error becomes the instrument's reason.

### Spec Refusals

| Message template | When it fires | What to do |
|------------------|---------------|------------|
| `Requires host {min_host} or newer; this host is {host_version}.` | The manifest sets `[compat].min_host` and the host is older. | Update the host. |
| `Requires the spec runtime, which this build of Caliper does not have.` | The host was built without the builder (`CALIPER_WITH_BUILDER=OFF`, the default on Windows). | Use a build of Caliper with the builder. |
| `Written in spec format {spec_version}; this host reads spec format {host_version} — update Caliper to open it.` | The spec's `caliper_spec` is later than the version the host reads (`2`). For an earlier unread version (`{spec_version} < {host_version}`), the message ends after `{host_version}.` | Open the instrument with a matching Caliper or update the spec to `caliper_spec = 2`. |
| `A workspace holds spec-based instruments only; a library instrument is installed in the user instruments folder.` | A library instrument's manifest is in the open workspace's `instruments/` folder or one folder below it. The loader never opens its library. | Install it with `caliper install`, and keep the workspace for spec-based instruments. |

### Spec Failures

| Message template | When it fires | What to do |
|------------------|---------------|------------|
| `spec file {spec} not found next to {manifest_filename}` | The file that `instrument.spec` names is not in the manifest's folder. | Put the spec beside its manifest, or correct `instrument.spec`. |
| `spec file {spec} is a symbolic link; an instrument's spec is a regular file in its folder` | The spec is a symbolic link. The loader never follows a link, so a spec cannot lead out of its folder. | Replace the link with the file. |
| `spec file {spec} is not a regular file` | The spec's name is a folder or another kind of file. | Put the spec file there. |
| `spec file {spec} is larger than 512 KiB` | The spec exceeds the spec format's limit. | Make the spec smaller. |
| `spec file {spec} has no line caliper_spec = <version> before its first table` | No top-level `caliper_spec` key comes before the first table header. | Start the spec with `caliper_spec = 2`, as the builder writes it. |
| `spec file {spec} has a caliper_spec that is not a decimal integer` | The value of the version line is not a decimal integer. | Write a decimal integer. |
| `manifest is larger than 64 KiB` | A manifest one folder down, or in the workspace's folder, exceeds 64 KiB. | Shorten the manifest. |
| `manifest has a dotted key of more than 32 parts` | Such a manifest has a dotted key of more than 32 parts. The loader refuses it before it parses the file. | Shorten the key. |
| `{spec}:{line}:{column}: {code}: {message}` | The spec fails the full validation. The text is the spec library's first error, with its place, its stable code (such as `spec.unknown_source`) and its sentence. Without a known place it reads `{spec}: {code}: {message} (at {path})`. | Fix the spec as the message says. `caliper check <folder>` prints every error and warning. |

`{spec}` is the spec's file name. `{spec_version}` and `{host_version}` are spec format versions, and the host reads version `2`. `{line}` and `{column}` count from 1.

An instrument written as a spec that passes every check opens on the app's spec page at launch ([Spec Page](../builder/workspace.md#spec-page)), and it is `ACTIVE` while the page shows it. If a program calls `InstrumentLoader::launch()` on a spec-based instrument instead of the app, it gets `An instrument written as a spec opens as a page, which this version of Caliper does not draw yet.` and the instrument stays `READY`.

## Renderer Refusals

Caliper draws with Metal on macOS and Vulkan on Windows and has no fallback renderer until the WebGPU fallback through Dawn lands ([No Fallback Renderer](../instruments/rendering.md#no-fallback-renderer)). These refusals come before any instrument loads.

| Message template | Who prints it | When it fires |
|------------------|---------------|---------------|
| `caliper: the {renderer} renderer could not start ({reason}), and there is no fallback renderer until the WebGPU fallback through Dawn lands` | The app (`caliper`, `caliper dev`) on standard error, then exit code 1 | The platform's renderer failed to start on its window. |
| `caliper: shot cannot draw offscreen: the {renderer} renderer could not start ({reason}), and there is no fallback renderer until the WebGPU fallback through Dawn lands` | `caliper shot`, then exit code 1 | The renderer failed to make its offscreen canvas. |
| `attach_canvas: the {renderer} renderer could not start ({reason}), and there is no fallback renderer until the WebGPU fallback through Dawn lands` | Caliper Embed, as `caliper_core_last_error` after `caliper_core_attach_canvas` returns 0 | The renderer failed to make the canvas. The core stays usable. |
| `CALIPER_RENDERER=gl asks for the OpenGL renderer, which was removed; unset it, and Caliper draws with Metal on macOS and Vulkan on Windows` | The app and `shot` after their prefixes (`caliper: `, `caliper: shot cannot draw offscreen: `), exit code 1; Caliper Embed after `caliper_core_create: `, in the embedder's log, and `caliper_core_create` returns `NULL` | `CALIPER_RENDERER=gl` is set. |
| `this platform has no renderer: Caliper draws with Metal on macOS and Vulkan on Windows` | The same three, with the same prefixes | Neither Metal nor Vulkan is built, as on Linux. |

`{renderer}` is `Metal` or `Vulkan`. `{reason}` is the backend's own, such as `no Metal device`, `no Vulkan loader`, `no Vulkan device with a graphics queue` or `ImGui's Metal backend did not start`, and is `CALIPER_TEST_RENDERER_FAIL=1 made its start fail` when that test variable forces the failure. When the app cannot create its window, the reason is `GLFW could not create a window`.
