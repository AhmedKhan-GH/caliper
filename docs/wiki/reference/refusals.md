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

!!! note
    Loader-produced failure messages (missing binary, descriptor mismatch, crash
    quarantine) join this page at Tasks 12–13.
