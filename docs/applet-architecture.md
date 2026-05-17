# Caliper Applet Architecture

This document describes how to convert Caliper from a monolithic binary into a framework that loads applets as shared libraries (`.dylib` / `.so` / `.dll`) at runtime.

---

## Overview

Today every applet (ECG Explorer, Node Editor, UCDH PreE) is compiled directly into the `caliper` executable. The goal is to split the project into two pieces:

1. **Caliper Framework** — the executable. Owns the window, OpenGL context, ImGui state, libtorch device, and the landing screen. Discovers and loads applet shared libraries from a `modules/` directory.
2. **Applets** — shared libraries. Each implements a standard C interface and is built as its own CMake target. Can be developed, compiled, and distributed independently of the framework.

```
caliper                    # framework executable
modules/
  ecg_explorer.dylib       # applet
  node_editor.dylib         # applet
  data_explorer.dylib       # applet
  my_custom_applet.dylib    # user-built applet
```

---

## Step 1: Define the Applet Interface

Create `include/caliper/applet.h` as the public API that all applets implement. This header ships with the framework SDK.

```cpp
// include/caliper/applet.h
#pragma once
#include <cstdint>

struct CaliperContext {
    int framebuffer_width;
    int framebuffer_height;
    // Extend later: torch device handle, shared texture IDs, etc.
};

class CaliperApplet {
public:
    virtual ~CaliperApplet() = default;

    virtual const char* name() const = 0;
    virtual const char* description() const = 0;

    virtual bool initialize(const CaliperContext& ctx) = 0;
    virtual void draw_ui(const CaliperContext& ctx) = 0;
    virtual void cleanup() = 0;

    virtual bool should_exit() const = 0;
    virtual void reset_exit_flag() = 0;
};

// Every applet shared library must export these two C functions.
// The extern "C" linkage prevents name mangling so dlsym/GetProcAddress can find them.
extern "C" {
    CaliperApplet* caliper_create_applet();
    void caliper_destroy_applet(CaliperApplet* applet);
}
```

**Why a C boundary?** C++ vtable layouts vary between compilers and even compiler versions. The `extern "C"` factory functions are the only symbols looked up by name — once you have the `CaliperApplet*`, virtual dispatch works normally because the applet and framework share the same abstract base class header. As long as both sides are built with the same compiler and C++ standard, the vtable is binary-compatible.

**ABI rule:** `CaliperApplet` must remain a pure interface — no data members, no non-virtual methods that access state. Adding a new virtual method to the end is safe (existing applets simply don't override it). Removing or reordering virtuals breaks the vtable and is a breaking change.

---

## Step 2: Build the Platform Loader

Create `src/applet_loader.h` and `src/applet_loader.cpp`. This is framework-internal code, not part of the public SDK.

```cpp
// src/applet_loader.h
#pragma once
#include "caliper/applet.h"
#include <string>
#include <vector>
#include <memory>

struct LoadedApplet {
    void* handle;                              // dlopen / HMODULE handle
    CaliperApplet* applet;                     // instance returned by factory
    void (*destroy_fn)(CaliperApplet*);        // pointer to caliper_destroy_applet
    std::string path;                          // filesystem path for diagnostics
};

class AppletLoader {
public:
    // Scan a directory for shared libraries and load each one.
    // Returns the number of applets successfully loaded.
    int scan_and_load(const std::string& modules_dir);

    // Access loaded applets.
    const std::vector<LoadedApplet>& applets() const;

    // Unload all applets (calls destroy, then dlclose).
    void unload_all();

    ~AppletLoader();

private:
    bool load_one(const std::string& path);
    std::vector<LoadedApplet> applets_;
};
```

### Platform abstraction (inside `applet_loader.cpp`)

```cpp
#ifdef _WIN32
  #include <windows.h>
  #define LIB_OPEN(path)       LoadLibraryA(path)
  #define LIB_SYM(handle, sym) GetProcAddress((HMODULE)(handle), sym)
  #define LIB_CLOSE(handle)    FreeLibrary((HMODULE)(handle))
  #define LIB_EXT              ".dll"
#else
  #include <dlfcn.h>
  #define LIB_OPEN(path)       dlopen(path, RTLD_LAZY)
  #define LIB_SYM(handle, sym) dlsym(handle, sym)
  #define LIB_CLOSE(handle)    dlclose(handle)
  #ifdef __APPLE__
    #define LIB_EXT            ".dylib"
  #else
    #define LIB_EXT            ".so"
  #endif
#endif
```

### Loading a single applet

```cpp
bool AppletLoader::load_one(const std::string& path) {
    void* handle = LIB_OPEN(path.c_str());
    if (!handle) {
        std::cerr << "Failed to load " << path << std::endl;
        return false;
    }

    auto create_fn  = (CaliperApplet*(*)()) LIB_SYM(handle, "caliper_create_applet");
    auto destroy_fn = (void(*)(CaliperApplet*)) LIB_SYM(handle, "caliper_destroy_applet");

    if (!create_fn || !destroy_fn) {
        std::cerr << "Missing entry points in " << path << std::endl;
        LIB_CLOSE(handle);
        return false;
    }

    CaliperApplet* applet = create_fn();
    if (!applet) {
        LIB_CLOSE(handle);
        return false;
    }

    applets_.push_back({handle, applet, destroy_fn, path});
    return true;
}
```

### Scanning a directory

```cpp
int AppletLoader::scan_and_load(const std::string& modules_dir) {
    int count = 0;
    for (auto& entry : std::filesystem::directory_iterator(modules_dir)) {
        if (entry.path().extension() == LIB_EXT) {
            if (load_one(entry.path().string()))
                count++;
        }
    }
    return count;
}
```

### Unloading

```cpp
void AppletLoader::unload_all() {
    for (auto& la : applets_) {
        la.applet->cleanup();
        la.destroy_fn(la.applet);
        LIB_CLOSE(la.handle);
    }
    applets_.clear();
}

AppletLoader::~AppletLoader() {
    unload_all();
}
```

---

## Step 3: Convert an Existing Applet

Take `NodeEditorApplet` as the example. Today it looks like:

```
src/node_editor_applet.h      class declaration
src/node_editor_applet.cpp    implementation
```

Both files are compiled into the `caliper` executable.

### 3a. Make it implement `CaliperApplet`

```cpp
// modules/node_editor/node_editor_applet.h
#pragma once
#include "caliper/applet.h"

class NodeEditorApplet : public CaliperApplet {
public:
    const char* name() const override { return "Node Sandbox"; }
    const char* description() const override {
        return "Signal-processing node graph editor";
    }

    bool initialize(const CaliperContext& ctx) override;
    void draw_ui(const CaliperContext& ctx) override;
    void cleanup() override;

    bool should_exit() const override { return exit_requested_; }
    void reset_exit_flag() override { exit_requested_ = false; }

private:
    struct State;
    State* s_ = nullptr;
    bool exit_requested_ = false;
};
```

### 3b. Export the C factory functions

```cpp
// modules/node_editor/node_editor_applet.cpp (at the bottom)

extern "C" CaliperApplet* caliper_create_applet() {
    return new NodeEditorApplet();
}

extern "C" void caliper_destroy_applet(CaliperApplet* a) {
    delete a;
}
```

### 3c. Give it its own CMake target

```cmake
# modules/node_editor/CMakeLists.txt
add_library(node_editor MODULE
    node_editor_applet.cpp
)

target_include_directories(node_editor PRIVATE
    ${CMAKE_SOURCE_DIR}/include          # for caliper/applet.h
)

target_link_libraries(node_editor PRIVATE
    imgui_node_editor                     # its specific dependency
    imgui
)

# Place the output alongside the caliper binary
set_target_properties(node_editor PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/modules"
    PREFIX ""                             # produce "node_editor.dylib", not "libnode_editor.dylib"
)
```

> **`MODULE` vs `SHARED`:** CMake's `MODULE` library type produces a shared library that is loaded at runtime via `dlopen` and is never linked against at compile time. This is exactly what we want. `SHARED` is for libraries linked at build time.

---

## Step 4: Update the Framework CMake

### 4a. Remove applet sources from the executable

```cmake
# CMakeLists.txt (updated)
add_executable(caliper
    src/main.cpp
    src/intro_screen.cpp
    src/dataset.cpp
    src/app_paths.cpp
    src/applet_loader.cpp            # new
)
```

`node_editor_applet.cpp`, `ucdh_pree_applet.cpp`, and eventually the ECG code are no longer listed here.

### 4b. Add the modules subdirectory

```cmake
# At the bottom of the root CMakeLists.txt
add_subdirectory(modules/node_editor)
add_subdirectory(modules/data_explorer)
add_subdirectory(modules/ecg_explorer)
```

### 4c. Create the public include directory

```cmake
target_include_directories(caliper PUBLIC
    ${CMAKE_SOURCE_DIR}/include      # ships caliper/applet.h
)
```

### 4d. Link dl on POSIX

```cmake
if(NOT WIN32)
    target_link_libraries(caliper PRIVATE dl)
endif()
```

---

## Step 5: Update the Main Loop

Replace the hardcoded `AppPage` enum and `if/else` chain with dynamic dispatch.

### Before (current)

```cpp
enum class AppPage { Landing, ECGApp, NodeEditor, UCDHPreE };

// In the render loop:
if (page_ == AppPage::NodeEditor) {
    node_editor_.draw_ui(dw, dh);
    if (node_editor_.should_exit()) {
        node_editor_.reset_exit_flag();
        page_ = AppPage::Landing;
    }
}
```

### After

```cpp
// In CaliperApp:
AppletLoader loader_;
CaliperApplet* active_applet_ = nullptr;

// In initialize():
loader_.scan_and_load("modules/");

// The landing screen reads loader_.applets() to build its card grid
// dynamically instead of using a hardcoded AppletKind enum.

// In the render loop:
if (active_applet_) {
    CaliperContext ctx{dw, dh};
    active_applet_->draw_ui(ctx);
    if (active_applet_->should_exit()) {
        active_applet_->reset_exit_flag();
        active_applet_ = nullptr;
        glfwSetWindowTitle(window_, "Caliper");
    }
} else {
    // Landing screen
    intro_.draw_ui(dw, dh);
    if (intro_.should_launch()) {
        intro_.reset_launch_flag();
        int sel = intro_.selected_applet_index();
        auto& loaded = loader_.applets();
        if (sel >= 0 && sel < (int)loaded.size()) {
            active_applet_ = loaded[sel].applet;
            CaliperContext ctx{dw, dh};
            active_applet_->initialize(ctx);
            glfwSetWindowTitle(window_,
                (std::string("Caliper - ") + active_applet_->name()).c_str());
        }
    }
}
```

---

## Step 6: Update the Landing Screen

The `IntroScreen` currently uses a hardcoded `AppletKind` enum. Change it to accept a list of applet names dynamically:

```cpp
// In IntroScreen:
void set_available_applets(const std::vector<std::string>& names);
int selected_applet_index() const;  // replaces selected_applet() -> AppletKind
```

The landing screen renders one card per entry. The framework passes in names from `loader_.applets()` after scanning the modules directory.

---

## File Layout After Migration

```
caliper/
├── CMakeLists.txt
├── include/
│   └── caliper/
│       └── applet.h                  # public SDK header
├── src/
│   ├── main.cpp                      # framework core
│   ├── intro_screen.h/cpp            # landing page (stays in framework)
│   ├── applet_loader.h/cpp           # dlopen machinery
│   ├── dataset.h/cpp                 # shared data types (consider moving to SDK)
│   └── app_paths.h/cpp               # stays in framework
├── modules/
│   ├── ecg_explorer/
│   │   ├── CMakeLists.txt
│   │   ├── ecg_explorer.cpp
│   │   └── dsp.h                     # signal processing, moved from main.cpp
│   ├── node_editor/
│   │   ├── CMakeLists.txt
│   │   └── node_editor_applet.cpp
│   └── data_explorer/
│       ├── CMakeLists.txt
│       └── ucdh_pree_applet.cpp
├── cmake/
│   ├── Dependencies.cmake
│   └── wrappers/
├── third_party/
└── docs/
    └── applet-architecture.md        # this document
```

---

## Writing a New Applet from Scratch

A minimal applet in a single file:

```cpp
// modules/hello/hello_applet.cpp
#include "caliper/applet.h"
#include <imgui.h>

class HelloApplet : public CaliperApplet {
    bool exit_ = false;
public:
    const char* name() const override { return "Hello World"; }
    const char* description() const override { return "Minimal example applet"; }

    bool initialize(const CaliperContext&) override { return true; }

    void draw_ui(const CaliperContext& ctx) override {
        ImGui::SetNextWindowSize({(float)ctx.framebuffer_width,
                                  (float)ctx.framebuffer_height});
        ImGui::SetNextWindowPos({0, 0});
        ImGui::Begin("Hello", nullptr,
                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
        ImGui::Text("Hello from a Caliper applet!");
        if (ImGui::Button("Back to Menu")) exit_ = true;
        ImGui::End();
    }

    void cleanup() override {}
    bool should_exit() const override { return exit_; }
    void reset_exit_flag() override { exit_ = false; }
};

extern "C" CaliperApplet* caliper_create_applet() {
    return new HelloApplet();
}

extern "C" void caliper_destroy_applet(CaliperApplet* a) {
    delete a;
}
```

```cmake
# modules/hello/CMakeLists.txt
add_library(hello MODULE hello_applet.cpp)
target_include_directories(hello PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(hello PRIVATE imgui)
set_target_properties(hello PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/modules"
    PREFIX ""
)
```

Add one line to the root `CMakeLists.txt`:

```cmake
add_subdirectory(modules/hello)
```

Build, and `hello.dylib` appears in `build/modules/`. Caliper picks it up on next launch.

---

## Migration Order

Recommended order to extract applets, from easiest to hardest:

1. **Node Editor** — already self-contained behind a clean `initialize/draw_ui/cleanup` interface. Zero shared state with the framework beyond ImGui. Extractable as-is.

2. **UCDH PreE** — same clean interface. Only dependency is DuckDB + ImGui. Straightforward extraction.

3. **ECG Explorer** — the most involved. Currently lives inline in `main.cpp` as ~800 lines spanning `draw_ecg_ui()`, `BackgroundProcessor`, DSP functions, dataset state, and processing params. Extraction requires:
   - Moving the `dsp` namespace, `BackgroundProcessor`, `ProcessingParams`, and related state into the applet's own files.
   - Deciding whether `dataset.h/cpp` stays in the framework SDK (shared across applets that deal with ECG data) or moves into this applet exclusively.

---

## Future Extensions

Things you can add to `CaliperContext` and `CaliperApplet` later without breaking existing applets:

- **Torch device**: Pass the active `torch::Device` through `CaliperContext` so applets can run inference on the user's GPU without device negotiation.
- **Shared texture registry**: Let applets publish/consume named OpenGL textures for cross-applet visualization.
- **Applet metadata**: Add `virtual const char* version()`, `virtual const char* author()` for the landing screen.
- **Applet settings**: Add `virtual void draw_settings_ui()` for a per-applet settings panel.
- **Hot reload**: Watch the `modules/` directory with `inotify`/`FSEvents`/`ReadDirectoryChangesW`, unload the old `.dylib`, load the new one. Useful during development.
