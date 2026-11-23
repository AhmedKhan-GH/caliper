# Refusal Messages

Before the host loads any applet code, it runs a pure **negotiation** check on the applet's manifest and the host's capabilities. See [Manifest](../applets/manifest.md) and [ABI (epoch 3)](../applets/applet-contract.md). If the applet is incompatible, the host refuses it, runs no applet code, and shows one message below as the status text on the landing screen.

The checks run in this fixed order: **ABI epoch → minimum host version → UI-stack hash → libtorch version → required services → data schema**. See [Compatibility](../development/compatibility-and-epochs.md). The first failing check wins, so the host shows only one message. These strings are contractual, and the loader and the landing screen reuse them verbatim.

## Negotiation Refusals

| Message template | When it fires | What to do |
|------------------|---------------|------------|
| `Built for ABI epoch {applet_epoch}; this host speaks {host_epoch} — check for an applet update.` | The applet's `abi_epoch` differs from the host's. | Install an applet build for this host's epoch, or update the host. |
| `Requires host {min_host} or newer; this host is {host_version}.` | The manifest sets `min_host` and the host is older. The comparison is numeric. An empty `min_host` sets no floor. | Update the host to at least `{min_host}`. |
| `Built against UI stack {applet_ui_stack}; this host uses {host_ui_stack} — rebuild the applet against this host's SDK.` | The manifest's `[build].ui_stack` differs from the host's UI-stack hash. The check runs only when both exist. | Rebuild the applet against this host's SDK. |
| `Built against libtorch {applet_torch}; this host provides {host_torch}.` | The manifest's `[compat].torch` differs from the host's libtorch version. | Install an applet build made against `{host_torch}`, or use a host that provides `{applet_torch}`. |
| `Built against libtorch {applet_torch}; this host provides no libtorch.` | The manifest sets `[compat].torch` and the host has no libtorch. | Use a host built with libtorch `{applet_torch}`. |
| `Requires a capability this host doesn't have: {service_id}.` | A service in the manifest's `required` array is not vended by this host. `{service_id}` is the **first** missing service in declaration order. | Use a host that provides `{service_id}`, or an applet build that does not require it. Optional services never cause a refusal. |
| `Stored data is schema {stored}; this applet reads schema {applet}.` | The manifest's `[data].schema` is lower than the schema that a newer applet version stored in the data folder. An applet without the key is never checked. | Install an applet version with a schema of at least `{stored}`. |

Placeholders in `{braces}` are filled at runtime. `{min_host}` and `{host_version}` are semver strings such as `0.6.0`. `{service_id}` is a service id such as `caliper.jobs.v1`. `{applet_torch}` and `{host_torch}` are libtorch versions such as `2.5.1`. `{applet_ui_stack}` and `{host_ui_stack}` are the first 12 characters of each UI-stack hash. The epoch and schema placeholders are integers.

The host records the highest data schema that opened `data/<id>/` in the file `.caliper-data.toml` in that folder. `launch()` checks this stamp again before it opens the library. If the stamp rose after the scan, the applet gets the same message and the same `Refused` status. The host raises the stamp before `create()` and `initialize()` run.

The required-services check also runs inside `launch()`. After `dlopen` and the descriptor checks, the loader compares the binary's own `required_services` list (set by `.services` in the `CALIPER_APPLET` macro) with the host's services. A missing service gives the same message and the same `Refused` status before `create()` runs, and the binary stays mapped. This catches a manifest whose `required` list omits a service that the binary needs.

## Loader Statuses

After an applet passes negotiation, the **loader** finds the binary, calls `dlopen`, checks the descriptor against the manifest, and runs the lifecycle under the crash guard. It sorts failures into two statuses:

- **Failed**: The applet is broken but the host is fine. Causes are a bad manifest, a missing or unloadable binary, a descriptor that contradicts its manifest, or a lifecycle hook that returned an honest "no". The applet can launch again after the cause is fixed.
- **Quarantined**: The applet faulted while the host held it, or a job it submitted faulted. A fault is a signal (SIGSEGV, SIGBUS, SIGFPE or SIGILL, including a stack overflow) or a C++ exception that escapes an entry point. The loader never calls the faulted instance again, and the dylib stays mapped.

What happens next depends on the kind of fault:

- **After a C++ exception**: The process is unharmed. The loader keeps the instance, uncalled, and the Inspector offers `⌘R Restart`. Restart runs the old instance's `cleanup()` (if its `initialize()` succeeded) and `destroy()`, both guarded. Then it launches a fresh instance in the same process. Restart reuses the loaded library, so a rebuilt dylib needs a Caliper restart.
- **After a signal**: Memory may be corrupt. The loader abandons the instance and calls neither `cleanup()` nor `destroy()`. The applet stays quarantined until Caliper restarts, and the Inspector says `restart Caliper to reload`. On Windows, every SEH exception counts as a signal, including a C++ throw that MSVC cannot unwind through the C ABI.

`scan()` is manifest-first and never calls `dlopen`. Thus the host decides the parse, missing-binary and negotiation results before any applet code runs. Everything from `load failed` downward happens only inside `launch()`.

A copy can also be **Shadowed**. Of the copies of one applet id that pass negotiation, the loader keeps one `Ready`. It picks the highest version, and at equal versions the copy found first in the [search order](cli.md#search-order). Every other passing copy is `Shadowed`, with the text `Version {version} in {folder} is used instead.` The host never launches a shadowed copy, and the status bar's instrument count leaves it out. A refused or failed copy keeps its own reason.

### Failed Messages

| Message template | When it fires | What to do |
|------------------|---------------|------------|
| _(manifest parser text, passed through verbatim)_ | `scan()` could not parse `caliper.toml`. The TOML is malformed, or a required field is missing or invalid. | Fix the manifest as the message says. See [Manifest](../applets/manifest.md). |
| `applet.id is not a valid applet id (lowercase reverse-DNS, [a-z0-9.-]): {id}` | The `id` breaks the [id rules](../applets/manifest.md#applet-ids): a character outside `[a-z0-9.-]`, an empty segment, or more than 128 characters. | Rename the applet to a lowercase reverse-DNS id, and use the same id in the descriptor. |
| `applet id is not path-safe: {id}` | The loader's own id check, made before it builds the data folder path. The parser refuses the same ids first. | As above. |
| `applet binary not found next to {manifest_filename}` | `scan()` found the manifest but no sibling library. The loader looks for `<stem>` and `lib<stem>` with the platform extension (`.dylib`, `.so` or `.dll`) beside `<stem>.caliper.toml`. | Put the compiled library next to its manifest, named after the manifest's stem. |
| `load failed: {dlerror}` | `dlopen` or `LoadLibrary` failed at `launch()`: a corrupt image, a wrong architecture, or an unresolved dependency. `{dlerror}` is the OS loader's text. | Rebuild the applet for this platform and architecture, and resolve its dependencies. |
| `missing export caliper_applet_descriptor` | The library does not export the required symbol. It is usually not an applet library, or it was built without `CALIPER_APPLET` or `CALIPER_EXPORT`. | Build with the SDK's `CALIPER_APPLET` macro, or export the descriptor by hand. |
| `descriptor missing or truncated` | The descriptor getter returned NULL, or `struct_size` is smaller than this host's `CaliperAppletDescriptor`. | Rebuild against the current SDK headers. |
| `descriptor ABI epoch disagrees with manifest` | The compiled-in `abi_epoch` differs from the manifest's epoch. | Rebuild the binary and manifest together against one SDK epoch. |
| `descriptor id disagrees with manifest` | The descriptor's `id` is NULL or differs from the manifest's `id`. | Make both ids identical. |
| `descriptor version disagrees with manifest` | The descriptor's `version` is NULL or differs from the manifest's `version`. | Set the same version in both and rebuild. |
| `descriptor function table incomplete` | One of the required `api` entry points (`create`, `destroy`, `initialize`, `frame`, `cleanup`) is NULL. | Provide all five functions. The `CALIPER_APPLET` macro wires them. |
| `could not record the data schema in {stamp_path}` | The manifest declares `[data].schema`, and the host could not write the stamp in the data folder. No entry point has run. | Make the data folder writable. |
| `create() returned null` | `create()` ran cleanly but returned no instance. | Return a valid instance, or report the real error from `initialize()`. |
| `initialize() returned false` | `initialize()` ran cleanly but reported failure. The loader calls `destroy()` on the instance. | Check the applet's logs for the reason. |

### Quarantined Messages

| Message template | When it fires | What to do |
|------------------|---------------|------------|
| `crashed in create(): {fault}` | The applet faulted inside `create()`. The instance never existed, so nothing needs cleanup. | Fix the constructor path, then restart. |
| `crashed in initialize(): {fault}` | The applet faulted inside `initialize()`. After a signal, `destroy()` is not called. After an exception, Restart calls `destroy()` but not `cleanup()`. | Fix the init path, then restart. |
| `crashed in frame(): {fault}` | The applet faulted during `frame()`. The loader never calls `frame()` again. | Fix the per-frame path, then restart. |
| `crashed during teardown: {fault}` | The applet faulted in `cleanup()` or `destroy()`. Teardown stops. After an exception in `cleanup()`, Restart makes the `destroy()` call that is still owed. | Fix the teardown path, then restart. |
| `crashed in background job "{label}": {fault}` | A job that the applet submitted through `caliper.jobs.v1` faulted or threw. See [jobs.v1](../applets/services/caliper-jobs-v1.md). The loader quarantines the applet before the next `frame()` or at teardown. A `frame()` that already runs finishes. After a throw the applet can restart, but not after a signal. | Fix the job function, then restart. |
| `crashed while restarting: {fault}` | The guarded `cleanup()` or `destroy()` of the old instance faulted during Restart. After a throw from `cleanup()`, a later Restart tries `destroy()` again. | Fix the teardown path. After a signal, restart Caliper. |

`{fault}` is the crash guard's description: a signal such as `SIGSEGV (invalid memory access)`, `C++ exception: <what()>`, or `C++ exception (non-std type)`. `{label}` is the label of the job. `{dlerror}`, `{manifest_filename}`, `{id}` and `{stamp_path}` name the OS loader message, the scanned manifest file, the manifest's `applet.id` and the `.caliper-data.toml` file.

"Restart" in the table means `⌘R Restart` in the Inspector after a C++ exception, and a restart of Caliper after a signal. See [Loader statuses](#loader-statuses).

!!! note
    An applet that is not `Ready` or `Active` keeps its row in the Instruments pane, with `REFUSED`, `FAILED`, `QUARANTINED` or `SHADOWED` in the Status column. When you select the row, the Inspector's Status field shows the exact `status_text` from the tables above, and `⏎ Launch` is disabled. A quarantined applet also shows `⌘R Restart` or `restart Caliper to reload`. The Activity pane logs `launch failed` with the reason, `quarantined` (after a `job failed` row if a job caused it), `restart` and `restart failed`. The landing screen reads the statuses every frame, so a quarantine during a session shows at once.

## Spec Applets

A [spec applet](../applets/manifest.md#spec-applets) has no library. Thus it has no `dlopen`, no descriptor and no lifecycle hook, and it is never `QUARANTINED`. The loader lists it as `READY`, `REFUSED`, `FAILED` or `SHADOWED`, and it picks among copies of one id by the same rule as for compiled applets.

The loader makes only cheap checks when it lists a spec applet, in this order: the manifest parses, the id is path-safe, the spec file passes the file checks, the `caliper_spec` line is read, then `min_host`, the spec runtime and the spec format version pass. The first failure wins. The manifest messages are the parser's, and they include the keys a spec applet refuses. The full validation runs at launch and in `caliper check`, and its first error becomes the applet's reason.

### Spec Applet Refusals

| Message template | When it fires | What to do |
|------------------|---------------|------------|
| `Requires host {min_host} or newer; this host is {host_version}.` | The manifest sets `[compat].min_host` and the host is older. | Update the host. |
| `Requires the spec runtime, which this build of Caliper does not have.` | The host was built without the builder (`CALIPER_WITH_BUILDER=OFF`, the default on Windows). | Use a build of Caliper with the builder. |
| `Written in spec format {spec_version}; this host reads spec format {host_version} — update Caliper to open it.` | The spec's `caliper_spec` is later than the version the host reads. For an earlier unread version, the message ends after `{host_version}`. | Open the applet with a newer Caliper. |
| `A workspace holds spec applets only; a compiled applet is installed in the user applets folder.` | A compiled applet's manifest is in the open workspace's `applets/` folder or one folder below it. The loader never opens its library. | Install it with `caliper install`, and keep the workspace for spec applets. |

### Spec Applet Failures

| Message template | When it fires | What to do |
|------------------|---------------|------------|
| `spec file {spec} not found next to {manifest_filename}` | The file that `applet.spec` names is not in the manifest's folder. | Put the spec beside its manifest, or correct `applet.spec`. |
| `spec file {spec} is a symbolic link; a spec applet's spec is a regular file in its folder` | The spec is a symbolic link. The loader never follows a link, so a spec cannot lead out of its folder. | Replace the link with the file. |
| `spec file {spec} is not a regular file` | The spec's name is a folder or another kind of file. | Put the spec file there. |
| `spec file {spec} is larger than 512 KiB` | The spec exceeds the spec format's limit. | Make the spec smaller. |
| `spec file {spec} has no line caliper_spec = <version> before its first table` | No top-level `caliper_spec` key comes before the first table header. | Start the spec with `caliper_spec = 0`, as the builder writes it. |
| `spec file {spec} has a caliper_spec that is not a decimal integer` | The value of the version line is not a decimal integer. | Write a decimal integer. |
| `manifest is larger than 64 KiB` | A manifest one folder down, or in the workspace's folder, exceeds 64 KiB. | Shorten the manifest. |
| `manifest has a dotted key of more than 32 parts` | Such a manifest has a dotted key of more than 32 parts. The loader refuses it before it parses the file. | Shorten the key. |
| `{spec}:{line}:{column}: {code}: {message}` | The spec fails the full validation. The text is the spec library's first error, with its place, its stable code (such as `spec.unknown_source`) and its sentence. Without a known place it reads `{spec}: {code}: {message} (at {path})`. | Fix the spec as the message says. `caliper check <folder>` prints every error and warning. |

`{spec}` is the spec's file name. `{spec_version}` and `{host_version}` are spec format versions, and the host reads version 0. `{line}` and `{column}` count from 1.

A spec applet that passes every check opens on the app's spec page at launch ([Spec Page](../builder/workspace.md#spec-page)), and it is `ACTIVE` while the page shows it. If a program calls `AppletLoader::launch()` on a spec applet instead of the app, it gets `A spec applet opens as a page, which this version of Caliper does not draw yet.` and the applet stays `READY`.

## Renderer Refusals

Caliper draws with Metal on macOS and Vulkan on Windows and has no fallback renderer until the WebGPU fallback through Dawn lands ([No Fallback Renderer](../applets/rendering.md#no-fallback-renderer)). These refusals come before any applet loads.

| Message template | Who prints it | When it fires |
|------------------|---------------|---------------|
| `caliper: the {renderer} renderer could not start ({reason}), and there is no fallback renderer until the WebGPU fallback through Dawn lands` | The app (`caliper`, `caliper dev`) on standard error, then exit code 1 | The platform's renderer failed to start on its window. |
| `caliper: shot cannot draw offscreen: the {renderer} renderer could not start ({reason}), and there is no fallback renderer until the WebGPU fallback through Dawn lands` | `caliper shot`, then exit code 1 | The renderer failed to make its offscreen canvas. |
| `attach_canvas: the {renderer} renderer could not start ({reason}), and there is no fallback renderer until the WebGPU fallback through Dawn lands` | Caliper Embed, as `caliper_core_last_error` after `caliper_core_attach_canvas` returns 0 | The renderer failed to make the canvas. The core stays usable. |
| `CALIPER_RENDERER=gl asks for the OpenGL renderer, which was removed; unset it, and Caliper draws with Metal on macOS and Vulkan on Windows` | The app and `shot` after their prefixes (`caliper: `, `caliper: shot cannot draw offscreen: `), exit code 1; Caliper Embed after `caliper_core_create: `, in the embedder's log, and `caliper_core_create` returns `NULL` | `CALIPER_RENDERER=gl` is set. |
| `this platform has no renderer: Caliper draws with Metal on macOS and Vulkan on Windows` | The same three, with the same prefixes | Neither Metal nor Vulkan is built, as on Linux. |

`{renderer}` is `Metal` or `Vulkan`. `{reason}` is the backend's own, such as `no Metal device`, `no Vulkan loader`, `no Vulkan device with a graphics queue` or `ImGui's Metal backend did not start`, and is `CALIPER_TEST_RENDERER_FAIL=1 made its start fail` when that test variable forces the failure. When the app cannot create its window, the reason is `GLFW could not create a window`.
