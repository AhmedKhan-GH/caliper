# Arrow C Data Interface (`caliper/arrow_c.h`)

Tabular results from [`caliper.data.v1`](services/data-v1.md) cross the frozen ABI
as **Apache Arrow C Data Interface** structures. This page explains why, and what
the vendored header is.

## What it is

The [Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html)
is a tiny, stable, language-agnostic ABI for passing columnar data between
components **without a shared library dependency**. It is three plain C structs —
`ArrowSchema` (the column types), `ArrowArray` (one batch of column buffers), and
`ArrowArrayStream` (an iterator yielding batches) — each carrying a `release`
callback. There are no Arrow library types involved: it is `struct`s and function
pointers, designed by the Arrow project to be **copied verbatim** into downstream
projects.

**Ownership rule (from the spec):** the producer fills a struct and sets
`release`; the consumer drains it and **must call `release(self)` exactly once**
(which sets the callback to NULL). A struct whose `release` is NULL is empty —
that is also how end-of-stream is signalled.

## Why it is the ABI's tabular boundary

Caliper's contract is a **frozen C ABI** (PLATFORM.md §3): no third-party type may
appear in a service header, or the applet and host would have to agree on that
library's version and layout. DuckDB is the host's query engine and libtorch is a
common applet dependency, but **no DuckDB or Arrow C++ type crosses the
boundary**. The Arrow C Data Interface is the ratified escape hatch: it is a
*specification*, not a library, so both sides compile their own copy of the same
frozen structs and interoperate by layout alone. DuckDB exports its results as
these structs natively, so `data.v1` gets a zero-invention path from SQL result to
applet.

## The vendored header

`sdk/include/caliper/arrow_c.h` is the spec's definitions copied verbatim. Two
details matter:

- It is the **one** header [`data_v1.h`](services/data-v1.md) is allowed to
  include — the frozen `data.v1` struct references `ArrowArrayStream*` from here.
- Its include guards are the spec's own **`ARROW_C_DATA_INTERFACE` /
  `ARROW_C_STREAM_INTERFACE`** (not `#pragma once`). That is deliberate: another
  vendored copy of these definitions (inside DuckDB, torch, or another dependency)
  in the same translation unit **coexists** without redefinition, because whichever
  copy is included first wins and the rest compile to nothing.

Most applets never touch these structs directly — the [`caliper::Data`](sugar.md)
sugar's `drain_numeric` helper walks the stream and releases it for you. Reach for
the raw interface only when you need non-numeric columns or streaming control.

Upstream spec: <https://arrow.apache.org/docs/format/CDataInterface.html>.
