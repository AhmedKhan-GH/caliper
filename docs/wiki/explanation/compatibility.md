# Compatibility & epochs

*Adapted from `PLATFORM.md` §6b (growth rules) and §14 (versioning policy). The spec is the source of truth.*

The host is a **service registry, not a struct of fields**. Applets obtain capabilities by name through a single extension point:

```c
struct CaliperHost {
    uint32_t    struct_size;
    uint32_t    abi_epoch;          // epoch this host is speaking
    uint32_t    host_version;       // (major<<16)|(minor<<8)|patch, informational
    const char* applet_data_dir;    // per-applet sandboxed storage, UTF-8
    /* THE extension point. Returns a service table or NULL. The pointer is
       valid for the applet's lifetime. Unknown ids return NULL — never UB. */
    const void* (*get_service)(const CaliperHost* host, const char* service_id);
};
```

## Growth rules (the platform's constitution)

These four rules are what let the contract grow for years without breaking a single shipped applet:

1. **A published service struct is immutable.** Capability additions ship as a *new id* (`caliper.metrics.v2`) alongside the old one, which keeps working.
2. **Hosts may provide any set of services.** Applets declare `required_services` (refusal happens at the manifest check, with a friendly card) and probe optional ones at runtime.
3. **`struct_size` is always the first field.** A reader never touches bytes beyond the writer's declared size — this is how new fields get appended without an epoch bump.
4. **The ABI epoch bumps only for** entry-point changes, `CaliperHost` layout changes, or UI-stack pin changes. Target cadence: at most one per year after stabilization.

## Versioning policy

Every moving part has its own scheme, so a change in one rarely forces a change in another:

| Thing | Scheme | Breaks when | Cadence target |
|---|---|---|---|
| **ABI epoch** | integer | entry / `CaliperHost` / UI-pin changes | ≤ 1/year post-stabilization; host supports N, and N−1 where feasible |
| **SDK** | semver `0.x` → `1.x` | minor = additive (new services, sugar, viz) within an epoch | monthly-ish while building |
| **Services** | id suffix `.v1`, `.v2` | never — old ids keep working alongside new | as needed |
| **Host app** | semver | UI/features; never silently drops epochs/services (deprecation window ≥ 2 releases) | monthly-ish |
| **Applets** | semver, theirs | their business entirely | theirs |
| **Runtime packs** | upstream version + platform | n/a (side-by-side installs; one per process per session) | tracks upstream |

## Negotiation at load

Every check happens **before `dlopen`**, in order. The first failure renders a reasoned card (see [Refusal messages](../reference/refusals.md)) instead of a loader crash:

1. platform binary present
2. epoch supported
3. `min_host` satisfied
4. required services available
5. runtime packs resolvable (download prompt)
6. *then* `dlopen`
7. descriptor sanity — id / version / epoch agree with the manifest
8. `create` / `initialize`

!!! note "Why epochs, not a rolling ABI"
    The UI-stack pin (the exact imgui/implot commits) *is* part of the epoch, because applets write raw ImGui against those headers. Bumping the pin is therefore an epoch bump — rare, CI-flagged, and the cost is an applet rebuild, not a silent break. See the [Architecture](architecture.md) overview and `PLATFORM.md` §9.
