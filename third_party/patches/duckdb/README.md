# Local DuckDB patches

`cmake/Dependencies.cmake` applies these to the pinned DuckDB submodule at
configure time, the same way as the llama.cpp patches: skipped when already
applied, a configure error when they neither apply nor are applied.

Base commit: **`9e248008`** (the submodule pin).

- `0001-fmt-drop-stdext-checked-array-iterator.patch`: DuckDB's bundled fmt
  uses `stdext::checked_array_iterator` whenever `_SECURE_SCL` is defined, which
  every MSVC defines. MSVC 14.51 (Visual Studio 2026) no longer has that type,
  so `fmt/format.h` fails to compile on Windows. The patch keeps fmt's
  plain-pointer branch on every compiler. Drop it once the pin moves to a DuckDB
  whose fmt no longer has the checked branch.
