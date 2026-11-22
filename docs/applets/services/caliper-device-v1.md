# caliper.device.v1

Service id `caliper.device.v1` gives the compute device that the host negotiates.

```c
--8<-- "sdk/include/caliper/services/device_v1.h"
```

## Semantics

The host negotiates the device and the applet maps it. The host detects one compute device and vends it through this service. The applet reads the `kind` and maps it to the device of its own framework. This follows an [architectural rule](../../introduction/architecture.md#design-decisions): the host never links torch or any other ML framework.

On Apple, the host uses the Metal API only to detect the device. On other systems, the host detects CUDA devices through the CUDA driver API, which it loads at run time. Thus it links no CUDA toolkit. If there is no NVIDIA driver or no device, the service reports the CPU.

A kind names a memory and API domain, not a backend. The enum is `CALIPER_DEV_CPU = 0`, `CALIPER_DEV_CUDA = 1` and `CALIPER_DEV_METAL = 2`. `METAL` does not say "MPS" because it names the Metal memory and API domain. This domain covers torch-MPS, MLX and ggml-Metal (see [Types that never cross the boundary](../../development/compatibility-and-epochs.md#excluded-boundary-types)). "MPS" is a torch-specific backend name and must not appear in the ABI. An applet that uses libtorch maps `CALIPER_DEV_METAL` to `torch::kMPS` itself:

```cpp
torch::Device dev =
    device.kind == CALIPER_DEV_METAL && torch::hasMPS()
        ? torch::Device(torch::kMPS)
        : torch::Device(torch::kCPU);
```

The `torch::hasMPS()` check is necessary. The host reports the memory domain that it detected, but the applet must confirm that its own framework can use it. If the framework cannot, the applet falls back to the CPU. The kind is an input to negotiation. It does not promise that every framework supports it.

### Fields

`index` is `0` for CPU and METAL. For CUDA, it is the device ordinal, and the host reports device `0`. `name` is a host-owned string that stays valid for the process lifetime. Do not free it, and do not keep it after the host exits. `free_memory_hint` is a best-effort byte count, and `0` means "unknown", not "no memory".

Any function pointer can be null on an older host. The [`caliper::Device`](../first-applet.md) sugar checks each pointer for null. If the host does not vend the service, it sets the whole snapshot to the CPU.
