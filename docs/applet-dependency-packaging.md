# Applet Dependency Packaging

## Current State

All dependencies live in the top-level CMake build. Every applet links against everything Caliper provides, even if only one applet uses a given library.

## Proposed Split

### Caliper-level dependencies (shared by any applet)

- imgui
- implot / implot3d
- imgui-node-editor
- ImGuiFileDialog
- GLFW / OpenGL
- stb_image
- nlohmann_json

### Applet-owned dependencies (fetched per applet)

| Applet | Dependency | Reason |
|--------|-----------|--------|
| circuitnet | duckdb, parquet extension, core_functions extension | Only circuitnet uses SQL/analytics |
| opengllama | llama.cpp | Only opengllama uses LLM inference |

### How it would work

Each applet's `CMakeLists.txt` uses `FetchContent` to pull its own heavy dependencies. The applet builds as a shared library with those deps statically linked in. Caliper's top-level build only provides the SDK (imgui, windowing, plugin interface).

```
applets/
  circuitnet/
    CMakeLists.txt      # FetchContent(duckdb ...)
    ...
  opengllama/
    CMakeLists.txt      # FetchContent(llama.cpp ...)
    ...
```

### Benefits

- New applets don't bloat the core build
- Applets can pin their own dependency versions independently
- Possible to distribute applets as standalone .dylib files

### Tradeoffs

- Two applets using the same dep get duplicate copies
- Slightly more complex per-applet CMake
- Top-level build no longer builds everything in one shot (or it does via `add_subdirectory` but each applet manages its own fetches)
