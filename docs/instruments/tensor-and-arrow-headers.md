# Tensor and Arrow Headers

The SDK has two interchange headers: `caliper/tensor.h` for tensors and `caliper/arrow_c.h` for tables.

## Tensor Header

`CaliperTensor` is a type, not a service. It is a plain struct that describes a block of memory with a dtype, a shape, strides in elements, a device, and an optional stream. The layout follows DLPack, so a torch, numpy, or mlx tensor is a cast away and not a copy. The layout is frozen, and thus GPU-resident tensors cross the ABI with no CPU staging.

An instrument passes a `CaliperTensor` into a service. No service returns one. These services take it:

- [`caliper.metrics.v1`](services/caliper-metrics-v1.md): `image()` accepts a CPU-resident, contiguous, HWC `u8` tensor.
- [`caliper.tensor_bridge.v1`](services/caliper-tensor-bridge-v1.md) and its revisions `v1_1` and `v1_2`: they turn a tensor on any supported device into a texture.

### Types

<!-- c-api types sdk/include/caliper/tensor.h level=4 -->

## Arrow Header

[`caliper.data.v1`](services/caliper-data-v1.md) returns tabular results as structures of the [Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html). The interface is a specification and not a library, so no Arrow library type crosses the ABI.

### Overview

The Arrow C Data Interface is a small, stable ABI that passes columnar data between components with no shared library. It has three plain C structs:

- `ArrowSchema`: the column types.
- `ArrowArray`: one batch of column buffers.
- `ArrowArrayStream`: an iterator that yields batches.

Each struct carries a `release` callback. The producer fills a struct and sets `release`. The consumer drains the struct and must call `release(self)` exactly once, which sets the callback to NULL. A struct whose `release` is NULL is empty. A stream signals its end the same way.

### Tabular Boundary

The Caliper ABI is a frozen C ABI. No third-party type can appear in a service header, because the instrument and the host would then need to agree on the version and layout of that library. Thus no DuckDB or Arrow C++ type crosses the boundary. Both sides compile their own copy of the same frozen structs and interoperate by layout alone. DuckDB exports its results as these structs, so a SQL result reaches an instrument through `data.v1` with no conversion.

### Vendored Header

The definitions in `caliper/arrow_c.h` are a verbatim copy of the specification. The doc comments are Caliper's. The header is the only header that [`data_v1.h`](services/caliper-data-v1.md) can include, because the frozen `data.v1` struct references `ArrowArrayStream*`.

The include guards are the `ARROW_C_DATA_INTERFACE` and `ARROW_C_STREAM_INTERFACE` guards of the specification, and not `#pragma once`. Thus another vendored copy of these definitions, for example inside DuckDB or torch, can share a translation unit with this header. The first copy that the compiler includes wins, and the other copies compile to nothing.

### Types

Most instruments never use these structs directly. The `drain_numeric` helper of the [`caliper::Data`](cpp-sugar.md) sugar walks the stream and releases it. Use the raw interface only for non-numeric columns or for control of the stream.

<!-- c-api types sdk/include/caliper/arrow_c.h level=4 -->

### Flags

<!-- c-api ids sdk/include/caliper/arrow_c.h ARROW_FLAG_* -->
