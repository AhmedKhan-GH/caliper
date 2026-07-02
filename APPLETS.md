> **Superseded (Phase 1, PLATFORM.md §17):** applets now use ABI epoch 2 —
> `caliper_applet_descriptor()` + `CALIPER_APPLET` macro + `<name>.caliper.toml`
> manifest. See `examples/hello/` for the canonical minimal applet. The v1
> `applet_info`/six-function ABI described below no longer exists.

# Creating Applets for Caliper

Caliper applets are shared libraries (`.dylib` on macOS, `.so` on Linux, `.dll` on Windows) that the host discovers and loads at runtime via `dlopen`. Each applet exports a fixed set of C functions defined by the applet ABI, keeping the boundary simple and stable.

---

## Directory layout

```
applets/my_applet/
├── CMakeLists.txt       # Build configuration
├── plugin.cpp           # C ABI exports (boilerplate)
├── my_applet.h          # Your applet class
├── my_applet.cpp        # Your applet logic
└── ...                  # Any other source files you need
```

---

## Step 1: Write your applet class

Your class needs three methods:

```cpp
// my_applet.h
#pragma once
#include <memory>

class MyApplet {
public:
    MyApplet();
    ~MyApplet();

    bool initialize();              // Called once after context setup
    void draw_ui(int win_w, int win_h);  // Called every frame
    void cleanup();                 // Called before destruction

private:
    struct State;
    std::unique_ptr<State> s_;
};
```

The pimpl pattern (`struct State`) is recommended to keep all internal state out of the header, but not required. The host never sees your headers.

Implement your UI with standard ImGui/ImPlot/ImPlot3D calls in `draw_ui()`. The host has already set up the frame for you.

---

## Step 2: Write `plugin.cpp`

This file is boilerplate. Copy it, change the class name and the return value of `applet_info()`:

```cpp
#include "applet_api.h"
#include "my_applet.h"

#include <imgui.h>
#include <implot.h>
#include <implot3d.h>

extern "C" {

APPLET_API CaliperAppletInfo applet_info() {
    return {
        "My Applet",                    // name (shown on landing page)
        "1.0",                          // version
        "Short description of what "
        "this applet does.",            // description
        "Category",                     // tag (e.g. "ECG", "Audio", "Vision")
        CALIPER_APPLET_ABI
    };
}

APPLET_API void* applet_create() {
    return new MyApplet();
}

APPLET_API void applet_destroy(void* ctx) {
    delete static_cast<MyApplet*>(ctx);
}

APPLET_API bool applet_initialize(void* ctx, const CaliperHostContext* host) {
    ImGui::SetCurrentContext(host->imgui);
    ImPlot::SetCurrentContext(host->implot);
    ImPlot3D::SetCurrentContext(host->implot3d);
    return static_cast<MyApplet*>(ctx)->initialize();
}

APPLET_API void applet_draw_ui(void* ctx, int w, int h) {
    static_cast<MyApplet*>(ctx)->draw_ui(w, h);
}

APPLET_API void applet_cleanup(void* ctx) {
    static_cast<MyApplet*>(ctx)->cleanup();
}

} // extern "C"
```

The `applet_initialize` function is critical: it sets ImGui/ImPlot/ImPlot3D contexts so your applet renders into the host's window rather than creating its own.

---

## Step 3: Write `CMakeLists.txt`

```cmake
add_library(my_applet SHARED
    plugin.cpp
    my_applet.cpp
)

target_include_directories(my_applet PRIVATE
    ${CMAKE_SOURCE_DIR}/src          # For applet_api.h
    ${CMAKE_CURRENT_SOURCE_DIR}      # For your own headers
)

target_link_libraries(my_applet PRIVATE
    caliper_applet_sdk               # ImGui, ImPlot, ImPlot3D, ImGuiFileDialog
    # Add any extra dependencies below:
    # duckdb_static
    # "${TORCH_LIBRARIES}"
)

target_compile_definitions(my_applet PRIVATE CALIPER_APPLET_EXPORT)

set_target_properties(my_applet PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# macOS: set rpath if linking LibTorch
if(APPLE)
    set_target_properties(my_applet PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
endif()
```

The `caliper_applet_sdk` interface target provides ImGui, ImPlot, ImPlot3D, and ImGuiFileDialog headers and libraries. Add anything else your applet needs.

---

## Step 4: Register the applet

Add one line to the root `CMakeLists.txt` under the Applets section:

```cmake
add_subdirectory(applets/my_applet)
```

Build. Your shared library appears in `cmake-build-debug/applets/` and the host picks it up automatically.

---

## How the host discovers applets

On startup, the host scans two directories for shared libraries:

1. **Next to the executable**: `<exe_dir>/applets/` (dev builds and bundled distribution)
2. **User data directory**: `~/Library/Application Support/Caliper/applets/` (macOS) or equivalent (user-installed applets)

For each `.dylib`/`.so`/`.dll` found, the host:

1. Calls `dlopen` to load the library
2. Resolves `applet_info` via `dlsym`
3. Checks that `info.abi == CALIPER_APPLET_ABI`
4. Resolves the remaining 5 function pointers
5. Adds a card to the landing page using `info.name`, `info.description`, and `info.tag`

When a user clicks the card, the host calls `applet_create()` then `applet_initialize()`, passing its UI contexts. Each frame calls `applet_draw_ui()`. Pressing Escape tears down the applet and returns to the landing page.

---

## ABI contract

The applet ABI is defined in `src/applet_api.h`. Every applet must export these six C functions:

| Function | Signature | Purpose |
|----------|-----------|---------|
| `applet_info` | `CaliperAppletInfo (void)` | Return metadata (name, version, description, tag, ABI version) |
| `applet_create` | `void* (void)` | Allocate and return your applet instance |
| `applet_destroy` | `void (void* ctx)` | Free the instance |
| `applet_initialize` | `bool (void* ctx, const CaliperHostContext* host)` | Set UI contexts and initialize |
| `applet_draw_ui` | `void (void* ctx, int w, int h)` | Render one frame |
| `applet_cleanup` | `void (void* ctx)` | Release resources before destroy |

The `void* ctx` is your applet object, opaque to the host. All C++ complexity stays inside the shared library.

---

## What `caliper_applet_sdk` provides

The SDK interface target gives your applet access to:

- **ImGui** (immediate-mode UI)
- **ImPlot** (2D plotting)
- **ImPlot3D** (3D plotting)
- **ImGuiFileDialog** (native file/directory picker)
- **DuckDB headers** (link `duckdb_static` separately if you use it)

---

## Distribution

To ship specific applets, copy their `.dylib`/`.so`/`.dll` files into the `applets/` directory alongside the host binary. The host loads whatever it finds — no configuration file needed. Users can add or remove applets by adding or removing shared library files.
