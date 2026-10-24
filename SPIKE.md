# Caliper Python entry point: spike report

`pip install caliper`, then `caliper.show(tensor)` from a normal PyTorch script,
showing the tensor live in a Caliper window and updating each call, zero-copy for
Apple-GPU (MPS) tensors through DLPack.

Private copy of the public Caliper repo at commit 7eb015d, uncommitted. Machine:
Apple Silicon, macOS, 18 cores, 64 GB. Python 3.12 in `.venv` (uv), torch 2.14.1,
nanobind 3.1.0, scikit-build-core.

## 1. What works and what doesn't

Works, end to end, on this machine:

- `uv build --wheel` from the repo root produces
  `dist/caliper-0.1.0-cp312-cp312-macosx_14_0_arm64.whl` (11.9 MB; the module
  inside is 34.3 MB, DuckDB and ImGui statically linked). `otool -L` lists only
  system frameworks, libc++, libSystem and libobjc; `nm -gU` exports exactly
  `_PyInit__native`; zero `at::`/`c10::`/`torch::` symbols; minos 14.0.
- `import torch, caliper` in one process with torch 2.14.1: no clash, because
  libcaliper carries no libtorch (section 2).
- `caliper.show(t, name=...)` opens a 1280x800 GLFW/AppKit window on first use and
  updates the named view in place; non-blocking; `caliper.spin()` keeps the window
  alive after the script. 2-D float tensors show as a colormapped heatmap (viridis,
  magma, rdbu; autoscaled or fixed vmin/vmax); `(H,W,3|4)` and `(C,H,W)` images as
  RGBA; `(C,H,W)` activations with other C as a channel mosaic; `(N,C,H,W)` shows
  item 0. Conversions run on the tensor's device, never `.cpu()`.
- MPS tensors go through `t.__dlpack__()` zero-copy: the `kDLMetal` `data` pointer
  is the storage `MTLBuffer`, `byte_offset` is honoured, and the renderer's
  colormap compute kernel (or blit for RGBA) reads that buffer in place. Evidence in
  section 4. CPU tensors are staged (one copy into the texture).
- `examples/python/train_show.py`: a CNN trained on MPS for 30 s with
  `caliper.show` on `fc1.weight` (784x64) and a conv activation mosaic (84x84)
  every step: 4775 updates per view, both `compute-imported`.
- Tests: `python/tests` 26 passed (import; CPU show; MPS show incl. a row slice
  with nonzero `byte_offset`; ramp vs CPU reference byte-exact through both the
  texture readback and the composited canvas; RGBA images; `creates == 1` across
  50 autoscaled shows; non-finite vmin/vmax and `##` names rejected; non-main
  thread rejected). C++: `caliper_tests` 163/163, `caliper_embed_tests` 21/21 (7
  new view cases), `caliper_gfx_tests` 49/49, all torch-free.

Does not work, or works with a caveat:

- **The window is only live while the script calls `show()` or `spin()`.** AppKit
  needs the main thread, which is the training thread, so events are polled and a
  frame is rendered (at most 60 Hz) inside `show`; during a long step or a `sleep`
  the window does not repaint or respond and macOS shows the beachball after a few
  seconds. This is the matplotlib interactive-mode model, not a background viewer.
  Section 7 lists the fixes.
- **Every `show` on MPS drains torch's queue.** Either through
  `torch.mps.synchronize()` (fixed vmin/vmax) or through the autoscale
  `aminmax().item()` (same drain, different call). In the example this is 3.6 of
  the 4.1 ms a show-step costs and takes training from 331 to 159 steps/s with two
  shows per step. `--every 10` recovers most of it. The native zero-copy part is
  0.2 ms per view.
- `wait_gpu=False` (0.01 ms instead of 0.17 to 0.27 ms) keeps the tensor alive
  until the next update but cannot stop an in-place rewrite (`opt.step()`) racing
  the renderer's kernel; the result is a torn frame, not a crash.
- Only 2-D f32 and `(H,W,4)` u8 are zero-copy; f16/bf16 heatmaps and 3-channel
  images are converted on the GPU first (a device copy, no host copy).
- One core per process (embed v0): one window with N views. Window-mode calls must
  come from the main thread. Resize/scale/focus/mouse/scroll/text events are wired
  and the window was checked by eye; keys are not mapped. A `[feed] macOS provider`
  line goes to stderr from the core's feed provider, outside the log hook.
- The wheel is cp312 arm64 only, and the sdist is not viable (the C++ sources live
  outside `python/`); the package has to be built from this tree. Two FetchContent
  clones (tomlplusplus, doctest) still happen at configure time.
- The repo-root `imgui.ini` (present in the copy before the spike) was appended to
  by an early Python run from the repo root; the core now writes layout only under
  `data_dir` (the Python layer uses a per-process temp dir), so this no longer
  happens.

### Files

- Build: `CMakeLists.txt`, `cmake/Dependencies.cmake`, `tests/CMakeLists.txt`,
  `examples/embed_host/CMakeLists.txt`, `CMakePresets.json` (preset
  `release-notorch`).
- Core: `include/caliper/embed.h` (v1.2 view section), `src/host/embed/embed_core.cpp`
  (view registry, draw, ini path), `src/host/tensor_bridge.{h,cpp}` (`set_mapping`,
  `bridge_last_reject`), `src/host/host_services.{h,cpp}` (`services_bridge`),
  `src/host/renderer/host_renderer.h` + `metal_renderer.mm` (`device_update_wait`),
  `tests/test_embed.cpp` (7 view cases, sine-scope guard).
- Python: `pyproject.toml`, `python/CMakeLists.txt`, `python/src/_native.mm`,
  `python/src/dlpack.h` (vendored), `python/caliper/{__init__,_convert}.py`,
  `python/tests/*.py`, `python/bench/bench_show.py`, `examples/python/train_show.py`.
- Output: `dist/caliper-0.1.0-cp312-cp312-macosx_14_0_arm64.whl`.

### How to run

```sh
export PATH="$HOME/Applications/CLion.app/Contents/bin/cmake/mac/aarch64/bin:$HOME/Applications/CLion.app/Contents/bin/ninja/mac/aarch64:$PATH"
cmake --preset release-notorch && cmake --build --preset release-notorch   # C++ side, torch-free
./cmake-build-notorch/tests/caliper_embed_tests
MACOSX_DEPLOYMENT_TARGET=14.0 uv build --wheel --no-build-isolation --python .venv/bin/python --out-dir dist
uv pip install --python .venv/bin/python --reinstall --no-deps dist/caliper-*.whl
.venv/bin/python -m pytest python/tests -q
.venv/bin/python python/bench/bench_show.py
.venv/bin/python examples/python/train_show.py --seconds 30
```

## 2. The torch-free build

### Finding

libcaliper (core + Metal renderer + embed C ABI) never included a torch header.
The grep over `src/host/**`, `src/host/renderer/**`, `src/host/embed/**` finds the
word "torch" only in comments. The renderers consume `CaliperTensor.data` as a raw
`id<MTLBuffer>` (Metal) or a CUDA device pointer (Vulkan), and
`sdk/include/caliper/adapters/torch.hpp` is applet-side only. libtorch 2.5.1 reached
libcaliper, the `caliper` exe and `embed_host` purely through
`${CALIPER_DEPENDENCY_LIBS}`, to which `cmake/Dependencies.cmake` appended
`${TORCH_LIBRARIES}`.

### Changes

- `CMakeLists.txt`: options `CALIPER_WITH_TORCH` (ON), `CALIPER_BUILD_APPLETS`
  (ON; forced OFF when torch is OFF because every applet links libtorch),
  `CALIPER_BUILD_PYTHON` (OFF; `add_subdirectory(python)`). The applet glob,
  `examples/sine_scope` (links torch) and the MSVC torch-DLL copy are gated.
- `cmake/Dependencies.cmake`: the whole "Category 4: PyTorch" block is inside
  `if(CALIPER_WITH_TORCH)`; `TORCH_LIBRARIES` is no longer appended to
  `CALIPER_DEPENDENCY_LIBS` in either branch (applets and torch tests already link
  `"${TORCH_LIBRARIES}"` themselves).
- `tests/CMakeLists.txt`: `caliper_torch_tests` and the applet learning-core tests
  are inside `if(CALIPER_WITH_TORCH)`; `caliper_embed_tests` depends on
  `sine_scope` only with torch, otherwise it is compiled with
  `CALIPER_TESTS_NO_TORCH_APPLETS=1` and the sine-scope swap cases compile out.
- Link trim: libtorch was never the only unwanted passenger. libcaliper
  PRIVATE-linked the whole `${CALIPER_DEPENDENCY_LIBS}` list, and because it is a
  static library that list propagated to every embedder: imgui-node-editor,
  ImGuiColorTextEdit, ImTerm (chrome widgets only the `caliper` exe uses).
  `cmake/Dependencies.cmake` now also builds `CALIPER_CORE_LIBS` = `OpenGL::GL`,
  `-framework Cocoa/IOKit/CoreVideo`, `libglew_static`, `glfw`; libcaliper links
  that, the exe keeps the full list. `caliper_embed_tests` and `embed_host` link
  only `libcaliper caliper::sdk`, exactly what the Python extension links.
- `CMakePresets.json`: `release-notorch` configure/build/test presets
  (`cmake-build-notorch`, Release, tests on, torch off).
- slang and llama.cpp are configured unconditionally by `Dependencies.cmake` even
  though no in-tree target that libcaliper needs uses them (llama.cpp only backs the
  opt-in legacy opengllama applet). They cost about two thirds of a 12.7 s
  configure, and slang clones boost_regex and mimalloc over the network at
  configure time, which a pip build should not do. Gated behind
  `CALIPER_WITH_SLANG` / `CALIPER_WITH_LLAMA` (default ON; the Python build and
  the `release-notorch` preset set both OFF; `CALIPER_BUILD_OPENGLLAMA` refuses
  when llama is OFF). Fresh torch-free configure 6.5 s (was 12.7 s), no
  boost_regex/mimalloc clone. Two FetchContent clones remain: tomlplusplus
  (`caliper_host_lib`) and doctest (tests only); vendor tomlplusplus for offline
  pip installs. Cold torch-free build of everything (libcaliper, exe, embed_host,
  all test binaries): 462 steps, 120 s at `-j 8`.

### Verification (torch-free tree `cmake-build-notorch`)

```
$ nm -gU cmake-build-notorch/libcaliper.a | grep -ci "torch\|c10::\|_ZN2at"      -> 0
$ nm -u  cmake-build-notorch/libcaliper.a | grep -ci "torch\|c10\|_ZN2at"        -> 0
  (same 0 for libcaliper_host_lib.a, libcaliper_metal_backend.a, libcaliper_app_paths.a)
$ otool -L <bin> | grep -ciE 'libtorch|libc10|libomp'
  tests/caliper_embed_tests 0 | examples/embed_host/embed_host 0 | caliper 0 | tests/caliper_tests 0
$ nm <bin> | c++filt | grep -cE '[ (<,](at|c10|torch)::|_ZN3c10|_ZN2at|_ZN5torch'  -> 0 for all four
$ grep -rnE '#\s*include\s*[<"](torch|ATen|c10)' src include                      -> 0
```

A case-insensitive `torch` grep over the binaries gives false positives from
DuckDB symbol names (`OperatorChildren`, `VectorChildIndex`), hence the
namespace-anchored patterns. `otool -L` lists only system frameworks, libc++ and
libobjc; the torch-free embed binaries carry no `LC_RPATH` at all.

Build times: cold torch-free build of libcaliper + both test binaries + hello
applet 1 min 30 s (451 ninja steps, `-j 8`); incremental 1 to 5 s. `libcaliper.a`
is 276 KB (its own six objects; dependencies are separate archives). The default
`cmake --preset release` still configures with torch ON (vendored libtorch 2.5.1
found, nine applets discovered) and a cold `libcaliper` there builds in 123 s with
zero torch symbols; `caliper_embed_tests` passes 15/15 with torch ON.

Tests torch-free: `caliper_tests` 163/163. `caliper_embed_tests` 13/13 with the two
sine-scope cases compiled out (they load `dev.example.sine-scope`, a torch applet;
without the guard the first `REQUIRE` leaves the core live and the one-core-per-
process gate fails every later case).

## 3. The tensor view: host-side, inside libcaliper, behind the embed ABI

Choice: a built-in **host-side view** implemented in `src/host/embed/embed_core.cpp`
and exposed as embed ABI v1.2 functions (`caliper_core_view_tensor` and friends in
`include/caliper/embed.h`), not an applet.

Why not an applet: an applet is a dylib found by manifest scan and dlopen'd, which
means shipping a dylib plus a `.caliper.toml` in the wheel, pointing `applets_dir`
at it, and inventing a channel from Python into the applet instance for the tensor
handoff. The core already owns the frame (`caliper_core_frame`), the bridge
(`caliper.tensor_bridge.v1_2`, which has the MTLBuffer import and offset update
the zero-copy path needs) and the ImGui context. Drawing a few `ImGui::Image`
windows there is ~300 lines, needs no new file in the wheel, and keeps the Python
extension a thin wrapper over C functions: no ImGui, renderer or torch type crosses
into nanobind.

### The ABI as landed (`include/caliper/embed.h`, v1.2 section; `CALIPER_EMBED_API_VERSION` stays 1, no existing struct changed)

```c
#define CALIPER_EMBED_HAVE_VIEWS 1
enum { CALIPER_VIEW_FLAG_RAW = 1 << 0,        /* draw 1:1 at (0,0), no chrome (tests) */
       CALIPER_VIEW_FLAG_WAIT_GPU = 1 << 1 };  /* block until the device update completed */
typedef struct CaliperViewDesc {
    size_t struct_size; const char* name; const CaliperTensor* tensor;
    uint64_t byte_offset; int32_t colormap; float vmin, vmax; uint32_t flags;
} CaliperViewDesc;
typedef struct CaliperViewInfo {
    size_t struct_size; const char* path; int32_t width, height; int32_t device;
    uint64_t updates; double last_update_ms; int32_t imported; uint64_t creates;
} CaliperViewInfo;
int caliper_core_view_tensor(CaliperCore*, const CaliperViewDesc*);
int caliper_core_view_remove(CaliperCore*, const char* name);
int caliper_core_view_count(CaliperCore*);
int caliper_core_view_info(CaliperCore*, const char* name, CaliperViewInfo* out);
int caliper_core_view_readback(CaliperCore*, const char* name, void* buf, size_t buf_bytes);
int caliper_core_colormap_lut(int32_t colormap, uint32_t out256[256]);
```

### Behaviour

- Accepted inputs: 2-D f32 `(H,W)` colormapped (viridis, magma, rdbu; vmin/vmax),
  3-D u8 `(H,W,4)` RGBA. Everything else is refused with a `view_tensor('<name>'):
  ...` reason in `caliper_core_last_error`; the Python layer
  (`python/caliper/_convert.py`) does the conversions on the tensor's device.
- CPU tensors: `update_texture` (CPU-staged copy, path `cpu-staged`).
- METAL tensors: `import_allocation(MTLBuffer, byte_offset + extent,
  CALIPER_ALLOC_HANDLE_MTLBUFFER)` cached per view by buffer pointer, then
  `update_texture_from_alloc(tex, alloc, byte_offset, desc)`. If the device path
  is refused the call fails; a METAL tensor never falls back to a host copy
  (zero-copy honesty). Path `compute-imported` / `blit-imported`, `imported = 1`.
- A vmin/vmax/colormap change updates the bridge entry's pinned mapping
  (`TensorBridge::set_mapping`, host-internal) without recreating the texture
  (`creates` stays 1); a shape or dtype change recreates it.
- `WAIT_GPU`: the Metal renderer remembers the last device-update command buffer
  (`last_update_cb_`) and `device_update_wait()` blocks on it.
- Drawing: after the applet frame (or alone), one ImGui window per view, tiled,
  small tensors magnified by a whole-number factor with nearest sampling, an info
  line (`WxH dtype device path updates=N last=ms`). `RAW` draws the texture 1:1
  at (0,0) with no chrome for byte-exact `read_pixels` tests.
- Views are destroyed in shutdown before the renderer is detached from the bridge.

Other host-internal additions: `services_bridge()` (`host_services.h`),
`bridge_last_reject()` so the bridge's refusal text reaches `last_error`,
`HostRenderer::device_update_wait()` (default no-op).

### C++ tests (`tests/test_embed.cpp`, torch-free `caliper_embed_tests`): 21/21 cases, 246 assertions

- `colormap_lut` equals the bridge's table.
- RAW f32 ramp (CPU): texture readback and canvas `read_pixels` byte-exact
  (asserted `== 0` max diff, not `<= 1`).
- METAL f32 at `byte_offset` 64 (an `MTLBuffer` made through the ObjC runtime
  from a C++ TU): path `compute-imported`, `imported == 1`, readback and canvas
  byte-exact; re-import from a second buffer at offset 128; vmin/vmax change
  (`creates` stays 1, `updates` increments); colormap change; in-place rewrite of
  the same buffer; shape change (`creates` becomes 2).
- METAL `(H,W,4)` u8 at offset 256: `blit-imported`, readback equals the bytes.
- Cost at 1024x1024 f32, mean of 20 updates: 0.97 ms CPU-staged vs 0.34 ms
  zero-copy including the GPU wait; 64x48 update 0.129 ms with `WAIT_GPU`,
  0.009 ms without; create + first wait 4.6 ms (shader warm; the first METAL f32
  show in a process compiles the colormap shader: tens of milliseconds with a
  cold OS shader cache, 1.6 ms warm, see section 6).
- Windowed view draws; removing it restores the plain background.
- Gates: show before a canvas, unknown name, C=3 on METAL (error names RGBA),
  misaligned offset, short struct, empty name, non-contiguous tensor, bad
  colormap all return 0; shutdown with live views is clean and a fresh core works.
- `caliper_tests` 163/163 and `caliper_gfx_tests` 49/49 still pass with the
  renderer/bridge additions.

## 4. Zero-copy evidence

### DLPack from torch 2.14.1 on MPS (ctypes, this machine)

```
t = torch.arange(12, dtype=torch.float32, device="mps").reshape(3, 4)
capsule name            b'dltensor'            (legacy DLManagedTensor)
device_type             8                      (kDLMetal)
data                    0x7992560e00           == t.untyped_storage().data_ptr()
byte_offset             0
s = t[1:, 1:3]
data                    0x7992560e00           (same MTLBuffer)
byte_offset             20                     (storage_offset 5 * 4 bytes)
__dlpack__(max_version=(1,0)) -> capsule b'dltensor_versioned'
```

`data` is the storage's `id<MTLBuffer>` object pointer (what
`adapters/torch.hpp` obtains as `storage().mutable_data()`), and views carry the
offset separately. The bridge's v1.2 import path
(`import_allocation(MTLBuffer, size, CALIPER_ALLOC_HANDLE_MTLBUFFER)` +
`update_texture_from_alloc(tex, alloc, offset_bytes, desc)`) takes exactly this
pair, so slices work without the `storage_offset() == 0` rejection the C++ adapter
has.

### What the renderer does with it

`metal_renderer.mm`: `import_external_allocation` is an ARC strong retain of the
MTLBuffer (no copy); `tex_update_from_imported` dispatches the colormap compute
kernel with `[enc setBuffer:src offset:byte_offset]` (f32) or a blit
`copyFromBuffer:sourceOffset:` (u8 RGBA) straight into the texture. No
`contents()` read, no host staging buffer. The path string the renderer records
(`compute-imported` / `blit-imported`) is surfaced through `CaliperViewInfo.path`
and `imported == 1`, and the Python tests assert it.

### Measurements that corroborate it (section 6 has the full table)

- `CaliperViewInfo.path == "compute-imported"` and `imported == 1` for every MPS
  show in the tests, the bench and the 30 s example (4775 updates per view).
- Per-call cost of the device path is nearly flat across a 64x range of sizes
  (0.17 ms at 256x256, 0.17 ms at 1024x1024, 0.27 ms at 2048x2048, GPU wait
  included; 0.01 ms without the wait), while the CPU-staged path scales with the
  bytes moved (0.05, 0.68, 2.99 ms). A host copy of 16.8 MB would show up as
  milliseconds.
- `torch.mps.current_allocated_memory()` and RSS do not grow over 50 shows at any
  size (deltas 0.0 MB; the 256x256 row's +0.5 MB RSS is first-use allocation of
  the renderer's pipeline, not per-call).
- The C++ test imports an `MTLBuffer` at `byte_offset` 64, 128 and 256 and reads
  back byte-exact pixels, so the offset really is applied inside the kernel and not
  via a repacked copy.
- Design-level: `metal_renderer.mm` has no `memcpy`/`contents` on this path, and
  `embed_core.cpp` refuses to fall back to a host copy for a METAL tensor; a
  refusal is an error, never a silent copy.

## 5. Sync approach

DLPack defines no Metal stream semantics (`kDLMetal` carries a device and a
buffer, nothing about which `MTLCommandQueue` wrote it), and `CaliperTensor.stream`
would need torch's MPS command queue, which torch 2.14.1 does not expose from
Python (`torch.mps` has `synchronize`, `Event` with `record/query/wait/synchronize`,
`compile_shader`, memory counters; `torch._C` has `_mps_deviceSynchronize`,
`_mps_recordEvent`, `_mps_waitForEvent`, `_mps_setStream`, no queue handle). The
C++ adapter (`sdk/include/caliper/adapters/torch.hpp`, lines 83 to 111) gets the
queue through `torch::mps::get_command_buffer()` under `dispatch_sync_f` on torch's
stream dispatch queue; that is a libtorch C++ call and this build links no torch.

Two hazards, two fences:

1. Producer-side: torch kernels that write the tensor may still be queued when
   Caliper's compute kernel reads the buffer. Fence: `torch.mps.synchronize()` in
   `caliper.show` before the handoff (a full device drain, the same v1 story as
   `synced_to_tensor`). Its cost is measured in section 6 and is the dominant cost
   of `show` on MPS.
2. Consumer-side: Caliper's colormap/blit is committed on the renderer's own
   `MTLCommandQueue` and the call returns before it runs. The next training step
   can rewrite (or the allocator can reuse) the buffer before the GPU reads it; the
   result is a torn or stale frame, never a crash (the import retains the
   `MTLBuffer`, and the command buffer retains every resource it references until
   it retires). Fence: `CALIPER_VIEW_FLAG_WAIT_GPU` (the Python default
   `wait_gpu=True`) makes the renderer wait for that command buffer to complete
   before `view_tensor` returns. Measured separately in section 6; `wait_gpu=False`
   is the fast-and-tearable option.

No `MTLSharedEvent` ordering is possible from Python today. The renderer already
supports it (`order_after_producer` honours a non-NULL `CaliperTensor.stream`), so
the production route (section 7) is to obtain torch's queue and skip both fences.

## 6. Measurements

All on this machine with the installed wheel, machine otherwise idle, torch 2.14.1.
Milliseconds, median of 50 calls, offscreen 1280x800 canvas
(`python/bench/bench_show.py`).

### `caliper.show` per call, f32 heatmap, fixed vmin/vmax

| size | device | MB | show total | convert | mps sync | native (GPU wait) | poll+frame | native (no wait) | total with autoscale | path | imported | Δ device mem | Δ RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256x256 | mps | 0.3 | 0.390 | 0.003 | 0.001 | 0.169 | 0.213 | 0.009 | 0.558 | compute-imported | yes | 0.0 MB | +0.5 MB |
| 256x256 | cpu | 0.3 | 0.261 | 0.002 | 0 | 0.047 | 0.210 | 0.039 | 0.080 | cpu-staged | no | 0.0 | +0.1 |
| 1024x1024 | mps | 4.2 | 0.387 | 0.003 | 0.001 | 0.174 | 0.207 | 0.013 | 0.603 | compute-imported | yes | 0.0 | 0.0 |
| 1024x1024 | cpu | 4.2 | 0.929 | 0.004 | 0 | 0.680 | 0.230 | 0.668 | 0.726 | cpu-staged | no | 0.0 | 0.0 |
| 2048x2048 | mps | 16.8 | 0.535 | 0.003 | 0.001 | 0.270 | 0.255 | 0.014 | 0.592 | compute-imported | yes | 0.0 | 0.0 |
| 2048x2048 | cpu | 16.8 | 3.272 | 0.010 | 0 | 2.991 | 0.279 | 3.126 | 3.255 | cpu-staged | no | 0.0 | 0.0 |

Reading the table: "mps sync" is 0.001 ms here because the bench tensor is idle
(nothing queued on torch's stream); in a training loop the same call is the
pipeline drain, see the example below. "total with autoscale" adds
`torch.aminmax(...).item()`, which on MPS is itself a drain plus a tiny kernel
(about 0.2 ms idle). The MPS path with the GPU wait is bounded by the colormap
kernel's round trip and barely moves with size; the CPU path is a memcpy plus a
CPU LUT pass and scales linearly.

### Frame time

- `frame()` alone, 2 views (1024x1024 f32 + 512x512 RGBA), 200 frames: median
  0.19 ms, p95 0.31 ms, max 0.99 ms (offscreen composite; windowed adds the
  CAMetalLayer present).
- Cold start: core create + attach 31 ms; first frame 5.6 ms; first f32
  `view_tensor` in a process compiles the colormap shader (1.6 ms with a warm OS
  shader cache, about 40 ms cold, then 0.18 ms); first u8 view 0.3 ms.
- C++ (`caliper_embed_tests`): 1024x1024 f32 update, mean of 20: 0.97 ms
  CPU-staged vs 0.34 ms zero-copy with the GPU wait; 64x48: 0.129 ms with the
  wait, 0.009 ms without.

### The training example (`examples/python/train_show.py --seconds 30`, MPS)

Tiny CNN, Adam, synthetic data; `show(fc1.weight)` (784x64 heatmap, autoscaled)
and `show(activation)` (conv1 output as an 84x84 channel mosaic) every step.

| | steps/s |
|---|---|
| no show (2 s baseline) | 331 |
| two shows per step | 159 (48 %) |

Per show-step, median: total 4.08 ms = convert + autoscale 3.56 + mps sync 0.01 +
native 0.40 (two views, zero-copy, GPU wait) + poll/frame 0.06. The dominant term
is the autoscale `.item()`, which is the producer-side drain of torch's queued
work; with fixed vmin/vmax the same cost moves into `torch.mps.synchronize()`.
Agent C's run of the same script measured 690 to 301 steps/s (44 %) and 92 % with
`--every 10`; absolute rates vary with machine load, the ratio does not. On CPU
(`--cpu`) the ratio was similar (150 to 96 steps/s) but noisy under concurrent
builds.

Both views report `compute-imported`, `imported`, 4775 updates each, `creates` 1.

## 7. What a production version needs

- **An event loop that does not depend on the training loop.** The spike pumps
  GLFW events and renders inside `caliper.show` (rate-limited) and in
  `caliper.spin`. AppKit requires the main thread, which is the user's training
  thread, so the window is only interactive while the script keeps calling `show`.
  Options: (a) a render thread that owns the ImGui context and the Metal queue,
  with `show` queueing a handoff (Metal encoding from a second thread is fine;
  input would still need the main-thread run loop, e.g. a `CFRunLoop` source or a
  Python `threading` pump the user installs); (b) a separate viewer process with
  the tensor passed as an `MTLSharedEvent`/IOSurface-backed buffer, which also
  survives the training process exiting; (c) accept the matplotlib-style model
  (`plt.pause`) and document it. (b) is the Caliper-native answer (the host app
  already exists); (a) is the minimal in-process one.
- **Stream-ordered handoff instead of two full fences.** Build a tiny torch
  extension against the user's torch (`torch.utils.cpp_extension`, a dozen lines
  calling `torch::mps::get_dispatch_queue()`/`get_command_buffer()` as
  `adapters/torch.hpp` does) that returns the producer `MTLCommandQueue` pointer;
  pass it as `CaliperTensor.stream` and drop `torch.mps.synchronize()` and
  `WAIT_GPU`. The renderer's `order_after_producer` already implements the
  `MTLSharedEvent` side. Alternatively `torch.mps.Event.record()` plus `query()`
  polling makes the producer fence non-blocking without any extension.
- **Buffer lifetime and reuse.** torch's MPS caching allocator hands out
  sub-ranges of larger `MTLBuffer`s (`byte_offset` is normal, not a corner case)
  and reuses freed buffers immediately. The spike caches one import per view keyed
  by buffer pointer and re-imports on change; production should bound the import
  table, handle `release_allocation` on view removal, and consider `alloc_shared`
  (bridge-owned storage the training loop writes into via `torch.frombuffer`-style
  aliasing) for the weights-every-step case.
- **More layouts on the zero-copy path.** Today only 2-D f32 heatmaps and
  (H,W,4) u8 images are zero-copy; f16/bf16 and 3-channel images are converted on
  the tensor's device first (a GPU copy, no host copy). Adding f16/bf16 and RGB
  variants of the colormap/blit kernels removes those copies.
- **Packaging.** The wheel statically links DuckDB (metrics/artifact stores the
  Python entry point never uses) and GLEW/GL for the frozen GL fallback. A
  `CALIPER_WITH_STORES=OFF` style option and dropping GL for embed-only builds
  would cut the module from tens of MB to a few MB and the cold build from minutes
  to seconds. Wheels need `MACOSX_DEPLOYMENT_TARGET` pinned, a `cibuildwheel`
  job for arm64, and `delocate` is unnecessary (only system frameworks are linked).
- **Multiple cores / windows.** The embed core is one-per-process (process-global
  ImGui context and service registry). One window with N views is fine; two windows
  need the de-singletonisation the embed header already lists as not-yet.
- **Input.** Only mouse, scroll, resize, scale, focus and text reach ImGui; a full
  GLFW-key to ImGuiKey table is a copy from `imgui_impl_glfw`.

## 8. Live streaming

Second stage: `show()` no longer waits for anything, the viewer samples at its
own pace, and the window never freezes. Measured on the same machine (M5 Max,
18 cores, 64 GB, torch 2.14.1), machine otherwise idle unless a row says so.
Raw logs: `python/bench/results/` (`A-*` in-process, `B-*` separate process,
`P-*`/`S-*` final code).

### 8.1 The probe that decided the design

`python/bench/probe_foreign_mtlbuffer.py`: torch 2.14.1 accepts
`torch.from_dlpack` over a `kDLMetal` capsule whose `data` is an `MTLBuffer`
that torch did not allocate (here `newBufferWithBytesNoCopy` over a page-aligned
`MAP_SHARED` mmap, the shared-memory candidate). `snap.copy_(src)` and
`torch.aminmax(src, out=(lo, hi))` write into it on torch's own stream, verified
from the CPU side of the mapping. `torch.mps.Event.record()` commits torch's
pending command buffer, so `Event.query()` turns true with no `synchronize()`.
The whole snapshot path (copy, aminmax into a stats region, record, query) costs
0.057 ms CPU at 1024x1024, against the 3.6 ms drain of section 6. So a
training process can hand a GPU snapshot to another process through shared
memory with one GPU-to-GPU copy and no host copy, and it can signal completion
by polling.

### 8.2 Phase 1: two prototypes, measured

Both prototypes share the snapshot design: `show()` picks a free slot of a
3-slot ring, enqueues `copy_` (plus `aminmax(out=)` into the slot's stats) on
torch's stream, records an Event, and returns; completed snapshots are published
oldest-first by polling `Event.query()`; the consumer draws the newest published
one at up to 60 fps; when every free slot is in flight the new snapshot is
dropped (overwriting an in-flight slot was tried first and starved publishing:
20 published of about 1180 under a GPU backlog).

(A) In-process: slots are torch tensors (stats in a caliper-made shared
`MTLBuffer` the render thread reads through `contents`), a C++ render thread
inside `caliper._native` drains a publish queue and calls `view_tensor` +
`frame` at 60 Hz; GLFW events are queued from the main thread and applied on
the render thread. (B) Separate process: slots are POSIX shm segments
(`shm_open`, 3 slots + a GPU-written stats region each, a control segment with
the view table), wrapped by both processes with `newBufferWithBytesNoCopy`; the
viewer is `caliper-viewer`, an AppKit host over libcaliper that maps the
segments, creates its `MTLBuffer` on the renderer's device and feeds the
snapshot to the embed v1.2 view (`compute-imported`, in place).

Training loop for the table: the TinyCNN of `train_show.py` plus a synthetic
SxS f32 weight rewritten in place every step and shown every step; 10 s after a
1 s warm-up; two repeats.

A finding that changes how the table must be read: `torch.mps.Event.record()`
commits torch's command buffer, and on this small model that alone raises
throughput by 40 to 75 % (torch's adaptive commit holds many small kernels until
a sync; a periodic commit lets the GPU start early). A plain "no show" baseline
therefore flatters `show()`. Every row below is reported against both the plain
baseline and a commit-matched control (the same loop recording an Event at
60 Hz and taking no snapshot).

| MPS, steps/s | 256² | 1024² | 2048² |
|---|---|---|---|
| A plain baseline | 709 / 747 | 685 / 754 | 676 / 745 |
| A control (Event.record at 60 Hz) | 1190 / 1217 | 1188 / 1247 | 1242 / 1253 |
| A sampler 60 Hz, % baseline · % control | 162·96 / 161·99 | 168·97 / 162·98 | 175·95 / 163·97 |
| A sampler off, % baseline · % control | 154·92 / 161·99 | 158·91 / 158·96 | 153·83 / 160·95 |
| B plain baseline | 762 | 764 / 752 | 736 / 740 |
| B sampler 60 Hz, % baseline · % control | 136·95.7 | 139·97.2 / 137·95.5 | 142·96.9 / 137·94.0 |
| B sampler off, % baseline · % control | 148·96.5 | 149·97.1 / 141·97.1 | 153·94.9 / 148·94.1 |

CPU tensors at 256²: A sampler 98 / 97 % of baseline, B 98.1 %. The spike's
in-process `show()` ran `train_show.py`'s two views at 48 % of baseline; B runs
the same two views at 137 % of baseline (96.8 % of the control).

| per call, median | A | B |
|---|---|---|
| `show()` taking a snapshot, MPS | 35 to 54 µs, flat 256² to 2048² | 50 to 57 µs, flat |
| `show()` skipped by the sampler | 1.7 to 2.2 µs | 2.0 µs |
| `show()` taking a snapshot, CPU 256² | 65 to 86 µs | 72 µs |
| step to screen, sampler on (median / p95) | 28 to 40 / 31 to 47 ms | 19 / 29 ms |
| step to screen, sampler off | 2 to 22 ms | 4.3 to 5.7 / 7 to 8 ms |
| viewer frame rate during training | 60 Hz (render thread) | 59.9 to 60 fps |

Latency is dominated by the snapshot waiting behind torch's queued work (B
measured 13 of its 19 ms there; with a sync every step it drops to 9.9 ms), then
by the wait for the next 60 Hz tick. `show()` has no back-pressure by design: in
a GPU loop with no sync point at all, latency grows with torch's run-ahead until
the loop hits one (`loss.item()`, logging); a benchmark loop without any sync
queued 228k 16 MB kernels per second and torch's own encoded-command queue grew
RSS at 85 MB/s with caliper and 250 MB/s without it (`B-backlog.txt`).

No host copy, both prototypes: snapshot cost flat with size (a 16.8 MB host
copy would be milliseconds); over 200 snapshots at every size
`torch.mps.current_allocated_memory()` changed by 0.000 MB and trainer RSS by at
most +0.45 MB (A) / +0.12 MB (B), viewer RSS by at most +0.66 MB; every MPS view
reports `compute-imported` with `imported = 1` (`blit-imported` for RGBA); the
code-path grep finds `.cpu()` only for devices other than CPU/MPS,
`synchronize()` only in `flush()`/`close()`, and one 13 KB `memcpy` of the
control table. A's tearing detector (decode each read-back frame, check it
shows a value that was published) saw 0 bad frames in about 450 with the real
ring and 18 in 441 with a deliberately unsafe control. The two clocks the
latency stamp relies on (`time.monotonic_ns()` and `CLOCK_UPTIME_RAW`) agree to
within 1 µs.

Liveness, a 5 s `time.sleep` in the training loop with live content:

| | spike | A, render thread | A + SIGALRM pump | A + `run(train)` | B |
|---|---|---|---|---|---|
| canvas repaints | no (identical screenshots) | yes, 300 frames | yes | yes | yes, 59.5 to 60 fps |
| input handled | after the sleep (4.5 s late) | after the sleep | within 16 ms, but 5 s late during a long C call | 0 to 0.3 ms | at once (own process) |
| close | after the sleep | after the sleep | 118 ms | 103 ms | at once |
| verdict | | repaints only | `DataLoader(num_workers=2)` hangs with any 60 Hz `setitimer`, reproduced without caliper; cannot ship | needs training off the main thread | |

Screenshots: `A-liveness-{spike,none,timer,timer-ccall,run}-t{1,3}s.png`,
`B-liveness-t{1,3,4.5}.png`. Neither prototype drove the real mouse (no
Accessibility permission); A used in-process event probes, B had the viewer
post a move and click through its own AppKit queue (`focused: yes`, `hovered:
yes` during the sleep). Window dragging was not exercised. B's viewer keeps
drawing its last frames at 60 fps after the trainer is SIGKILLed and reports
the client as exited.

### 8.3 Decision

Against the commit-matched control the sampler rows read 94 to 99 % (A 95 to
99 %, B 94.0 to 97.2 %; the one B repeat below 95 % is 2048² at 94.0 %, within
the run-to-run spread of its 96.9 % twin), and 136 to 175 % of the plain
baseline. Both transports are zero-copy. Only the separate process satisfies "the window and
interaction never freeze whatever the training loop does": in-process, AppKit
input needs the main thread, which is the training thread, and the only
mitigation that works (`run(train)`, training on a worker thread) changes how
the script is run; the signal-timer pump is disqualified by the DataLoader
hang. The founder's decision: build both modes behind one view layer in
libcaliper, the separate-process "attach to app" mode first as the primary one
(it also survives the script and is what the Caliper app's landing will attach
to), the in-process `caliper.window()` second with its limits documented.

### 8.4 Architecture

One view layer in libcaliper, two transports, routing by context.

```
training script                         viewer (caliper-viewer, or the Caliper app later)
  caliper.show/points/series/histogram     per frame (60 Hz NSTimer):
    -> _ring.SnapshotRing                    caliper_core_live_poll(core)
       prepare on device (no .item())          Source::sync/take  (live_session.mm: shm)
       slot.copy_(x), aminmax(out=stats)       -> caliper_core_live_apply(core, &snap)
       torch.mps.Event.record()                   heatmap/image -> v1.2 TensorView
       poll Event.query(), publish              points -> geometry pipeline, in place
    -> Transport.publish                        series/histogram -> CPU-read, ImPlot
       SessionTransport (live.py): shm seq    caliper_core_frame(core)
       WindowTransport (_inproc.py): queue  in-process: the render thread does the same
```

- `include/caliper/live_views.h` (embed ABI v1.3, additive): `CaliperLiveSnapshot`
  (name, kind, payload tensor + byte offset, optional aux tensor, CPU-readable
  stats, colormap/vmin/vmax/flags, series_total, points_dims, label_count,
  seq, stamp) and `caliper_core_live_apply`, `_view_info`, `_view_readback`,
  `_view_values`, `_view_read_done`, `_set_paused`, `_view_remove`, `_view_count`.
  Implemented in `src/host/live/live_views*.{cpp,mm}`: heatmap and image
  delegate to the v1.2 `TensorView` of the same name (so `view_info`/
  `view_readback` keep working and the path strings stay `compute-imported`/
  `blit-imported`); points draw every change straight from the imported
  payload buffer through the renderer's geometry pipeline (`geom_create_view_ex`
  with depth, `geom_draw_primitives` with point topology, a 16-colour
  categorical LUT for labels, orbit/pan/zoom, Fit, hover pick); series and
  histogram copy the values at apply from the CPU-visible side of the payload
  and draw with ImPlot. One pause flag per core freezes the display only.
- `include/caliper/live.h` + `src/host/live/live_shm.h`, `live_client.mm`,
  `live_source.h`, `live_session.mm`, `live_core.cpp`: the session protocol
  (v2) and both ends of it. A session is a control segment (`/clp-<pid>-ctl`:
  pid, script, start time, a seqlocked table of views, the viewer heartbeat), one
  segment per view (`/clp-<pid>-<n>`: header with the published seq/slot and
  the viewer's two claims, 3 slots of payload | aux | stats, page aligned), and
  a session file `<app-data>/live/<pid>.json` (pid, script, start time, token,
  view names, segment names). `caliper_live_list_sessions` lists and reclaims
  stale ones (dead pid: file removed, names unlinked); `caliper_core_live_attach`
  / `_attach_pid` / `_detach` / `_poll` / `_picker` are what the Caliper app's
  landing will call. `live_core.cpp` holds the viewer side behind an abstract
  `Source` (sync / take / drew / heartbeat / producer_alive), so an in-process
  Source could reuse it.
- `src/host/live/viewer_main.mm` → `caliper-viewer`: an AppKit shell like
  `examples/embed_host`; `--session <token>` attaches at once, no flag shows the
  session picker; Detach, Pause, a "SESSION ENDED" banner when the client is
  gone (frames kept); `--stats`, `--exit-after`, `--latency-log`, `--headless
  --dump-dir` (offscreen, per-view dumps for tests). Ships in the wheel at
  `caliper/bin/caliper-viewer` (41 MB: libcaliper with DuckDB statically linked).
- Python (`python/caliper`): `_ring.py` (the one `SnapshotRing` and the
  `Transport` protocol: `attached`, `alloc`, `reading_slot`/`claimed_slots`,
  `publish`, `remove`), `_convert.py` (device-side preparation per kind),
  `live.py` (`SessionTransport`: segments, session file, `attached()` as the
  bound C method, viewer launch), `_inproc.py` + `python/src/_native.mm`
  (`WindowTransport` + `Window`: GLFW window, render thread, `run(fn)`,
  offscreen `render()`), `_watch.py` (module hooks), `__init__.py` (routing).
  Every slot in both transports is CPU-readable shared memory (shm pages, or
  caliper-made shared `MTLBuffer`s) handed to torch through DLPack, so the
  view layer reads stats, series and hover values without any host copy.

### 8.5 The API

| call | what it does | returns |
|---|---|---|
| `show(t, name="tensor", cmap="viridis", vmin=None, vmax=None, rate=None)` | 2-D → heatmap; (C,H,W) → channel mosaic; (H,W,3/4), (3/4,H,W) → RGBA; (N,…) → item 0; autoscale on the GPU unless vmin/vmax | True if a snapshot was taken (a `ShowStats` in offscreen mode) |
| `points(z, labels=None, name="points", classes=None, cmap=, vmin=, vmax=, rate=)` | (N,2) or (N,3) → 2-D pan/zoom or 3-D orbit view; `labels` + `classes` → categorical colours, else continuous colormap | True/False |
| `series(value, name="series", rate=None)` | appends one scalar per call (a 0-d GPU tensor stays on the GPU: a 4-byte copy into the ring, no `.item()`); the view updates at the rate | True/False |
| `histogram(t, name="histogram", bins=64, rate=None)` | bins on the device (aminmax + floor + index_add_; `torch.histc` blocks on MPS, so it is not used there); counts match `histc` | True/False |
| `watch(model, activations=True, gradients=False, every=1, include=None, rate=None)` | forward hooks → `<module>.act` mosaics, gradient hooks → `<module>.<param>.grad` heatmaps | handle with `.remove()` |
| `session(auto_launch=False, viewer_path=None, script_name=None)` | open the live session now (the first call opens it anyway); `auto_launch` spawns `caliper-viewer` | `SessionTransport` |
| `window(width=1280, height=800, title="caliper", *, offscreen=False, fps=60, rate=None, activate=True)` | the in-process window (main thread only); `offscreen=True` for `render()` | `Window` with `.run`, `.stats`, `.render`, `.close` |
| `run(fn, *args)` | run `fn` on a worker thread while the main thread pumps AppKit for the window | `fn`'s result |
| `offscreen(w, h)` | the synchronous test mode: each call snapshots, publishes, applies and frames before returning | |
| `spin(seconds=None)`, `flush()`, `close()`, `remove(name)`, `stats()` | wait for the window or the viewer; publish pending snapshots; shut down; drop a view; counters | |
| `view_info`, `view_readback`, `read_pixels`, `render`, `live_view_info`, `live_view_values`, `live_view_readback`, `live_pause`, `colormap_lut` | test and measurement surface | |

Routing: every call offers the snapshot to each active transport (the session,
opened by the first call; the window, if `window()` was called). A transport
whose `attached()` is false is skipped before any torch work. `rate` is per
view (default 60 Hz); `series` appends every call and publishes at the rate.
Environment: `CALIPER_DATA_DIR` (session-file root and the window's data dir),
`CALIPER_VIEWER` (viewer binary), `CALIPER_PY_LOG`, `CALIPER_PY_LIVE_APPLY=0`
(route heatmaps through the v1.2 call instead of `live_apply`, a debugging
switch).

### 8.6 Zero-copy and no-stall evidence (final code)

- No synchronize, no `.item()`, no `.cpu()` on any live path: the tests patch
  `torch.mps.synchronize`, `.item`, `.cpu` and `.numpy` to raise during
  `show`/`points`/`series`/`histogram` with queued GPU work; the only `.cpu()`
  in the package is for devices other than CPU/MPS, the only `synchronize()` is
  in `flush()`/`close()`.
- Calls return with the snapshot still pending: 28 to 129 µs per kind with the
  GPU busy (WP), 32 to 52 µs median in the example (`final-example-*.txt`);
  skipped calls 0.3 to 7 µs; with nothing attached 0.13 to 0.21 µs (the bound C
  `attached()` is 0.04 µs).
- Memory over 200 snapshots at every size and in every mode: device memory
  +0.0 MB, RSS under 0.5 MB; every MPS heatmap `compute-imported`, image
  `blit-imported`, points `points-imported`; series/histogram `cpu-read` (the
  viewer copies 4 KB per apply from shared memory).
- Tearing: 0 torn and 0 stale frames in 471 read-backs on the final ring; the
  unsafe control (no claims) shows 16.
- Liveness: the viewer runs at 59.5 to 60 fps through a 5 s stall of the
  trainer and after the trainer is SIGKILLed ("SESSION ENDED", frames kept);
  with the viewer SIGKILLed, `attached()` turns false in 0.98 to 1.00 s and
  training continues; the in-process window repaints at 60 fps through a stall
  and handles input in 0.1 ms under `run()` (4.5 s late without it).
- App Nap: an occluded in-process window throttled training to 56 to 60 % of
  the control and the render thread to 9 fps (215 ms gaps); an occluded viewer
  dropped to 5 to 11 fps. Both processes now hold an `NSProcessInfo` activity
  (`UserInitiatedAllowingIdleSystemSleep | LatencyCritical`) while a window is
  open; the same row then runs at 100.5 % of the control at 61 fps.

### 8.7 Sync design

- Producer side: a snapshot is `copy_` (plus `aminmax(out=)` into the slot's
  stats, or the histogram binning) enqueued on torch's stream, then
  `torch.mps.Event.record()`, which commits torch's command buffer. The ring
  polls `Event.query()` oldest-first on every later call (and from the
  transport's own thread in the session mode) and publishes the newest
  completed snapshot; nothing ever waits.
- Slots: 3 per view (1 for series). A write goes to a slot that is neither the
  published one, nor claimed by the consumer, nor pending; when none is free
  the new snapshot is dropped (counted). The consumer claims a slot with a
  seq_cst store-then-recheck on the segment header, moves a claim only to a
  newer published slot of the same view, and only after the GPU read of the
  previous snapshot retired (`caliper_core_live_view_read_done`, which polls
  the renderer's retained command buffer and never waits). Points hold two
  claims (the current payload is drawn on every change; the superseded one
  until its last draw retired); heatmap/image/histogram one; series none.
- The commit effect: `Event.record()` commits torch's command buffer at up to
  60 Hz per view, and on this small model that alone raises throughput by 40
  to 75 %. Benchmarks therefore report both the plain baseline and a
  commit-matched control; the overhead of the snapshots is the gap to the
  control.
- No back-pressure by design: in a GPU loop without any sync point,
  step-to-screen latency grows with torch's run-ahead until the loop hits one.

### 8.8 Measured overhead (final code, idle machine, `final-example-*.txt`)

`examples/python/train_show.py`, TinyCNN on MPS, four calls per step
(`fc1.weight` 784x64 heatmap, conv1 activation mosaic 84x84, 64-point 3-D
embedding coloured by class, loss series), 15 s after a 4 s baseline and a 4 s
control:

| mode | steps/s | % of plain baseline | % of commit-matched control |
|---|---|---|---|
| none (session open, nobody attached; 4 calls/step at 0.3 to 0.9 µs) | 728.5 vs 743.2 | 98.0 | – |
| attach (caliper-viewer, 60 fps) | 1135.3 vs 735.2 / 1200.0 | 154.4 | 94.6 |
| window (`caliper.window()` + `run()`, 61 fps) | 1115.2 vs 726.5 / 1172.1 | 153.5 | 95.1 |

Per call in the example: snapshot 32 to 52 µs (heatmap 32, mosaic 46 to 52,
points 34, series 14 to 19), skipped 0.8 to 7 µs; about 875 snapshots per view
in 15 s (58 per second per view). Step to screen: 15.6 ms median in window mode
(render-thread stamp), 20.7 to 24 ms through the viewer. WP's and S's longer
tables (`P-*`, `S-*`): window mode 98.3 to 99.5 % of the control with the
sampler, 94.3 to 97.6 % without it; attach mode 98.1 to 99.4 %; CPU tensors at
256² 97 to 98 % of the plain baseline. The earlier prototypes' rows are in
section 8.2.

Tests at the end: `caliper_tests` 163/163, `caliper_embed_tests` 21/21
(22/22 with torch), `caliper_live_tests` 11/11, `caliper_live_views_tests`
13/13, `caliper_gfx_tests` 49/49, `caliper_export_tests` 12/12, both under the
`release-notorch` and the default `release` preset; pytest 86 passed, 1 skipped
(the session-guard test, which only runs without `caliper.live`), from the
staged tree and from the installed wheel. Four of the original 26 expectations
changed: CPU tensors report `compute-imported`/`blit-imported` (their slot is a
shared `MTLBuffer` the core reads in place), `show()` from a thread is allowed
(`run()` needs it; `window()`/`offscreen()` still require the main thread), and
`wait_gpu=` is gone (the producer never waits).

### 8.9 What's left

- One viewer per session (a single reader claim table); a second viewer needs
  per-reader claims.
- Shared-memory names leak if the trainer is SIGKILLed; the next listing
  reclaims them (dead pid), so in practice they last until a viewer lists.
- About 15 client-side drops per view while a viewer attaches (GPU busy while
  its resources are built); after a re-attach the first latency sample includes
  the old snapshot.
- Points views hit "ring full" more often than other kinds (the current slot
  stays claimed): 4 slots would help; the view still updates at about 50 Hz.
- `torch.from_dlpack` over a foreign `kDLMetal` buffer is undocumented torch
  behaviour (verified on 2.14.1); whether torch's command buffers retain the
  buffer is unknown, so `remove`/`close` wait for the view's events.
- The viewer runs a 60 Hz timer; a display link at the screen's refresh would
  cut pickup delay. Viewer smoothness while a model saturates the GPU is
  untested. Window dragging was not exercised programmatically.
- Shape and argument errors surface only when a snapshot is built, so a bad
  shape goes unnoticed until something attaches.
- In-process mode: `signal.signal()` fails in user code under `run()`, and
  Ctrl-C lands between bytecodes.
- Hover on a heatmap reads the slot after its claim ended (may show a newer
  value); the info overlay overlaps the Pause button in very small view windows.

### 8.10 Design note: instrument input slots

The existing applets (instrument scopes) draw from tensors their own workers
produce. To feed them from a live session, each applet would declare named
input slots in its manifest (`[inputs] name = "weights"`, kind, dtype, shape
pattern); the host binds a slot to a live view of a matching kind by name
(`caliper_core_live_bind(core, view, applet_slot)`), and the applet reads the
bound view's newest completed snapshot through a new `caliper.live_input.v1`
service (`acquire(slot) -> CaliperTensor + seq`, `release`), which is the same
`Source::take`/claim rule the viewer uses, so the slot's buffer is never
rewritten while the applet reads it. No applet change is needed for the view
kinds already drawn by the view layer; an applet that wants the raw tensor
(for example a scope that runs its own kernels over the weights) gets the
imported `MTLBuffer` and byte offset, zero-copy, and must call `release` before
the next frame. The binding UI belongs to the Caliper app's landing (the view
list per session next to the applet's slot list).
