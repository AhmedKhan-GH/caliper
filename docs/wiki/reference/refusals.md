# Refusal messages

Before the host loads any applet code, it runs a pure **negotiation** check over
the applet's manifest and the host's own capabilities (see
[Manifest](manifest.md) and [ABI (epoch 2)](abi.md)). If the applet is
incompatible, negotiation refuses it and the host shows one of the messages
below on the applet's failure card — no code from the applet is executed.

The checks run in a fixed order (PLATFORM.md §14): **ABI epoch → minimum host
version → required services**. The first failing check wins, so only one message
is ever shown. These strings are contractual: the loader and its cards reuse
them verbatim.

## Negotiation refusals

| Message template | When it fires | What to do |
|------------------|---------------|------------|
| `Built for ABI epoch {applet_epoch}; this host speaks {host_epoch} — check for an applet update.` | The applet's `abi_epoch` does not equal the host's ABI epoch. | Install a build of the applet that targets this host's epoch, or update the host to one that speaks the applet's epoch. |
| `Requires host {min_host} or newer; this host is {host_version}.` | The manifest sets a `min_host` floor and the host's version is older than it (compared numerically, not lexically). | Update the host to at least `{min_host}`. Only fires when `min_host` is set; an empty `min_host` imposes no floor. |
| `Requires a capability this host doesn't have: {service_id}.` | A service listed in the manifest's `required` array is not vended by this host. `{service_id}` is the **first** missing service in declaration order. | Run the applet on a host that provides `{service_id}`, or use an applet build that doesn't require it. Optional services never cause a refusal. |

Placeholders in `{braces}` are filled at runtime: `{applet_epoch}` / `{host_epoch}`
are integers, `{min_host}` / `{host_version}` are semver strings like `0.6.0`, and
`{service_id}` is a service id like `caliper.jobs.v1`.

## Loader statuses

Negotiation is only the first gate. Once an applet passes it, the **loader**
(PLATFORM.md §14/§15) is what actually finds the binary, `dlopen`s it, checks the
descriptor against the manifest, and runs the lifecycle under the crash guard.
Each of those steps has its own failure, and the loader sorts them into two
statuses:

- **Failed** — the applet is broken but the host is fine: a bad manifest, a
  missing or unloadable binary, a descriptor that contradicts its manifest, or a
  lifecycle hook that returned an honest "no". Re-launchable once the cause is
  fixed.
- **Quarantined** — the applet *faulted* (a signal: SIGSEGV/SIGBUS/SIGFPE/SIGILL)
  while the host held it. Process memory is now suspect, so the loader abandons
  the instance and **never calls it again this session** — no retry, no
  `cleanup`/`destroy` on the faulted instance, and the dylib is left mapped
  (running static destructors inside a corrupted image is worse than the leak).
  Recovery is a host restart.

Ordering matters: `scan()` is manifest-first and never `dlopen`s, so parse,
missing-binary, and negotiation results are all decided before any applet code
runs. Everything from `load failed` downward happens only inside `launch()`.

### Failed messages

| Message template | When it fires | What to do |
|------------------|---------------|------------|
| _(manifest parser text, passed through verbatim)_ | `scan()` could not parse `caliper.toml` — malformed TOML or a missing/invalid required field. The error is whatever the [manifest](manifest.md) parser reported. | Fix the manifest as the message describes; see [Manifest](manifest.md) for the required fields and formats. |
| `applet binary not found next to {manifest_filename}` | `scan()` found the manifest but no sibling shared library. The loader looks for `<stem>` and `lib<stem>` with the platform extension (`.dylib`/`.so`/`.dll`) beside `<stem>.caliper.toml`. | Ship the compiled applet library next to its manifest, named to match the manifest's stem. |
| `load failed: {dlerror}` | `dlopen`/`LoadLibrary` failed at `launch()` — a corrupt image, a wrong architecture, or an unresolved dependency. `{dlerror}` is the OS loader's own text. | Rebuild the applet for this platform/arch and make sure its dependencies resolve. |
| `missing export caliper_applet_descriptor` | The library loaded but does not export the one required symbol. Usually a non-applet library, or one built without `CALIPER_APPLET` / the `CALIPER_EXPORT` visibility attribute. | Build with the SDK's `CALIPER_APPLET` macro (or export the descriptor by hand) so the symbol is visible. |
| `descriptor missing or truncated` | The descriptor getter returned NULL, or a `struct_size` smaller than this host's `CaliperAppletDescriptor` — an ABI too old to trust. | Rebuild against the current SDK headers for this epoch. |
| `descriptor ABI epoch disagrees with manifest` | The compiled-in `abi_epoch` does not match the epoch the host negotiated from the manifest. The binary and its manifest were built against different epochs. | Rebuild the binary and manifest together against one SDK epoch. |
| `descriptor id disagrees with manifest` | The descriptor's `id` is NULL or differs from the manifest's `id`. The manifest describes a different applet than the binary. | Make the manifest `id` and the descriptor `id` identical (rebuild if the descriptor is stale). |
| `descriptor version disagrees with manifest` | The descriptor's `version` is NULL or differs from the manifest's `version`. | Bump manifest and descriptor to the same version and rebuild. |
| `descriptor function table incomplete` | One of the required `api` entry points (`create`, `destroy`, `initialize`, `frame`, `cleanup`) is NULL. | Provide all five lifecycle functions; the `CALIPER_APPLET` macro wires them for you. |
| `create() returned null` | `create()` ran cleanly but returned no instance — the applet declined to construct itself (e.g. an internal precondition failed). | Investigate the applet's `create`; return a valid instance or surface the real error from `initialize` instead. |
| `initialize() returned false` | `initialize()` ran cleanly but reported failure. The loader then calls `destroy()` on the instance and marks the card Failed. | Check the applet's own logs for why init refused (a required service or resource it couldn't obtain). |

### Quarantined messages

| Message template | When it fires | What to do |
|------------------|---------------|------------|
| `crashed in create(): {signal}` | The applet faulted inside `create()`. The instance never existed; nothing to clean up. | Fix the fault in the applet's constructor path; restart the host to try again. |
| `crashed in initialize(): {signal}` | The applet faulted inside `initialize()`. The instance is abandoned — `destroy()` is **not** called on a faulted instance. | Fix the fault in the applet's init path; restart the host. |
| `crashed in frame(): {signal}` | The applet faulted during a `frame()` call. The instance is abandoned and `frame()` is never called again this session. | Fix the fault in the applet's per-frame path; restart the host. |
| `crashed during teardown: {signal}` | The applet faulted in `cleanup()` or `destroy()` while shutting down. Teardown stops; the dylib is left mapped. | Fix the fault in the applet's teardown path; restart the host. |

`{signal}` is the crash guard's description of the fault, e.g.
`SIGSEGV (invalid memory access)`. `{dlerror}` is the platform loader's message,
and `{manifest_filename}` is the manifest file the applet was scanned from.

!!! note
    The host now renders these statuses on the landing page. Any applet that is
    not `Ready`/`Active` keeps its card, but the card body is prefixed with
    `[unavailable] <reason>` — where `<reason>` is the exact `status_text` from
    the tables above (the negotiation refusal, `Failed`, or `Quarantined`
    message) — followed by the applet's normal summary. The card stays visible
    but will not launch until the cause is fixed, and the text is refreshed each
    time you return to the landing page so a mid-session quarantine shows up
    immediately.
