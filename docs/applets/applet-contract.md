# Applet Contract

The header below is the frozen applet contract, and it is the single source for the contract. Any change to it is an epoch bump. [Compatibility and epochs](../development/compatibility-and-epochs.md) explains how the contract grows.

```c
--8<-- "sdk/include/caliper/abi.h"
```

## Types

The contract uses C types only. It has no STL, exceptions, or third-party types. Each struct starts with `struct_size`.

| Type | Role | Fields |
|---|---|---|
| `CaliperAppletDescriptor` | Describes the applet to the host. | `abi_epoch`, `id`, `version`, `name`, `summary`, `tag`, `required_services`, `api` |
| `CaliperAppletAPI` | Holds the entry points of the applet. | `create`, `destroy`, `initialize`, `frame`, `cleanup` |
| `CaliperHost` | Gives the applet access to the host. | `abi_epoch`, `host_version`, `applet_data_dir`, `get_service` |
| `CaliperFrameInfo` | Describes one frame. | `fb_width`, `fb_height`, `dpi_scale`, `time_sec`, `delta_sec` |

The descriptor `abi_epoch` must equal an epoch that the host supports. The values `id` and `version` match the manifest. The `id` uses reverse-DNS form. The `version` is the applet semver. The `required_services` list ends with `NULL`, and the pointer itself may be `NULL`.

The host is a service registry and not a struct of fields. The `host_version` value is `(major<<16)|(minor<<8)|patch`. The `applet_data_dir` value is the per-applet sandbox directory. It is UTF-8 and the host owns it. The `struct_size` field of `CaliperHost` is `sizeof(CaliperHost)` as the host built it.

The `fb_width` and `fb_height` values are physical framebuffer pixels. ImGui coordinates are logical units, and physical equals logical times `dpi_scale`.

## Calls

Every epoch-2 applet exports exactly one symbol, `caliper_applet_descriptor`. The `CALIPER_APPLET` macro in `caliper.hpp` generates it, or you can write it by hand in C. The `CALIPER_DESCRIPTOR_SYMBOL` macro holds the name of the symbol.

| Call | Signature | Role |
|---|---|---|
| `caliper_applet_descriptor` | `const CaliperAppletDescriptor* (void)` | Returns the descriptor of the applet. |
| `create` | `void* (void)` | Returns a new applet instance. |
| `destroy` | `void (void* self)` | Destroys the instance. |
| `initialize` | `bool (void* self, const CaliperHost* host)` | Starts the instance with the host. |
| `frame` | `void (void* self, const CaliperFrameInfo* info)` | Runs once for each frame. |
| `cleanup` | `void (void* self)` | Cleans up the instance. |
| `get_service` | `const void* (const CaliperHost* host, const char* service_id)` | Returns a service table or `NULL`. |

The pointer from `get_service` stays valid for the lifetime of the applet. An unknown service id returns `NULL` and never causes undefined behavior. Future entry points are appended to `CaliperAppletAPI` and guarded by `struct_size`.
