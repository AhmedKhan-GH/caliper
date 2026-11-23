# Applet Contract

The applet contract is frozen, and `sdk/include/caliper/abi.h` is its single source. Any change to it is an epoch bump. [Compatibility and epochs](../development/compatibility-and-epochs.md) explains how the contract grows.

The contract uses C types only. It has no STL, exceptions, or third-party types. Each struct starts with `struct_size`. The `CALIPER_EXPORT` macro marks a function for export: `__declspec(dllexport)` on Windows and default visibility on the other platforms.

## Types

<!-- c-api types sdk/include/caliper/abi.h CaliperAppletDescriptor -->

<!-- c-api types sdk/include/caliper/abi.h CaliperAppletAPI -->

<!-- c-api types sdk/include/caliper/abi.h CaliperHost -->

<!-- c-api types sdk/include/caliper/abi.h CaliperFrameInfo -->

## Calls

Every epoch-3 applet exports exactly one symbol. The `CALIPER_APPLET` macro in `caliper.hpp` generates it, or you can write it by hand in C. The `CALIPER_DESCRIPTOR_SYMBOL` macro holds the name of the symbol. The header declares the symbol only in a comment, thus the generated tables do not list it.

| Call | Signature | Description |
|---|---|---|
| `caliper_applet_descriptor` | `const CaliperAppletDescriptor* (void)` | Return the descriptor of the applet. |

The other calls are the entry points of the applet and the `get_service` call of the host.

<!-- c-api calls sdk/include/caliper/abi.h -->

The pointer from `get_service` stays valid for the lifetime of the applet. Future entry points are appended to `CaliperAppletAPI` and guarded by `struct_size`.

## Ids

<!-- c-api ids sdk/include/caliper/abi.h -->
