# caliper.device.v1

Service id `caliper.device.v1` — the host's negotiated compute device (PLATFORM.md §7.3). This page embeds the header verbatim; the docs build fails if the file moves.

```c
--8<-- "sdk/include/caliper/services/device_v1.h"
```

## Semantics

**The host negotiates; the applet maps.** The host detects one compute device
and vends it through this service. The applet reads the `kind` and maps it to
*its own framework's* device — the host stays framework-agnostic. This is a hard
platform rule (PLATFORM.md D11): **the host never links torch, DuckDB-for-ML, or
any ML framework.** On Apple the detection is Metal-API-only; CUDA detection
lands with real hardware in Phase 4.

**Kinds name the memory/API domain, not a backend.** The enum is
`CALIPER_DEV_CPU = 0`, `CALIPER_DEV_CUDA = 1`, `CALIPER_DEV_METAL = 2`.
`METAL` deliberately does **not** say "MPS": it names the Metal memory/API
domain, which covers torch-MPS, MLX, and ggml-Metal alike (spec §7.2). "MPS" is
a torch-specific backend name and must never appear in the ABI. An applet that
uses libtorch maps `CALIPER_DEV_METAL → torch::kMPS` itself:

```cpp
torch::Device dev =
    device.kind == CALIPER_DEV_METAL && torch::hasMPS()
        ? torch::Device(torch::kMPS)
        : torch::Device(torch::kCPU);
```

Note the `torch::hasMPS()` check: the host reports the *memory domain it
detected*, but the applet still confirms its own framework can use it, and falls
back to CPU otherwise. The kind is a negotiation input, not a promise that every
framework is built for it.

**Fields.** `index` is `0` for CPU and METAL (it carries a CUDA device ordinal
when that lands). `name` is a **host-owned** string valid for the process
lifetime — do not free it, and do not assume it outlives the host. `free_memory_hint`
is a best-effort byte count; `0` means "unknown", not "no memory". Any function
pointer may be null on an older host; the [`caliper::Device`](../../tutorials/first-applet.md)
sugar null-guards each one and defaults the whole snapshot to CPU when the host
does not vend the service.
