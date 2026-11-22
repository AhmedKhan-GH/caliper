# Graphics Layer

The graphics layer is the part of the renderer core that every backend implements, and its programs share one integer colormap, the [colormap by keys](#colormap-by-keys). [Rendering](rendering.md) describes the backends that the layer sits behind.

The renderer core's graphics layer is in `src/host/gfx/`, in the namespace `caliper_gfx`. It is an internal C++17
layer shaped to WebGPU's model and default limits, and nothing of it crosses the ABI. The browser views draw through
it ([the WebGPU backend](rendering.md#webgpu-backend)). Natively the scene view and the points view draw through it, and the
renderers' other clients are to move onto the layer one at a time.

## Pieces

The layer offers these pieces:

- **Resources.** Buffers, textures, samplers, programs, render and compute pipelines and bind groups, named by typed
  64-bit handles. A released handle is refused rather than reaching another resource, and the object behind it is
  destroyed only after the last submit that used it has completed.
- **Draw lists.** A draw list holds the plain records of the draws for one target. Each record names a pipeline, up
  to four bind groups and four dynamic offsets, up to eight vertex buffers at any 4-byte-aligned offset (read per
  vertex, per instance or as a constant), an optional buffer of u32 indices, the counts and a pick id. A target has a
  colour attachment, an optional `RG32Uint` ID attachment and an optional `Depth32Float` depth attachment.
- **The uniform arena.** `DrawList::uniform` copies a parameter block into a 256-byte-aligned slice of the list's
  arena and returns its dynamic offset. The backend uploads each arena once per submit.
- **Frames.** A frame is an ordered list of producer waits, compute dispatches, buffer-to-texture copies and render
  passes. A submit returns a serial, and completion is polled. A blocking wait exists only where the backend allows
  one.
- **Validation.** Every descriptor and every frame is checked against the backend's limits before anything is
  encoded. One bad draw refuses the whole submit and leaves its targets untouched, and `last_error()` says why. A draw
  may not bind a texture that its own pass renders into.
- **Readback.** A texture region is read back asynchronously.
- **Import.** External memory becomes a layer buffer or texture, and the result reports the route: in place, copied
  on the GPU, copied from the host, or refused. A transient host import, whose memory is valid only until the call
  returns, is always copied. `mapped()` gives the CPU view of an import whose memory the CPU can read: an `MTLBuffer`
  in shared storage, or the backend's copy of host memory, where the backend maps buffers. `take_stats()` counts the
  imports, copies and bytes copied, so an info line can say what ran, and the GPU time of the submits completed since
  the last call with their number.

## Backends

Three backends exist. The Metal backend runs on the Metal renderer's own device and queue (`HostRenderer::native_device`
and `native_queue`), so the layer creates no second device, and its work is ordered with the renderer's by commit
order. It imports an `MTLBuffer` of that device, and page-aligned host memory that is not transient, in place, and it
copies other host memory. It runs the programs of the shader generator ([Shader authoring](shader-authoring.md)), and it binds the
buffer of storage sizes that their MSL reads. It keeps Metal's fast paths:

- bind groups are resolved once into Metal objects;
- dynamic offsets move with `setVertexBufferOffset`;
- state that repeats from one draw to the next is not set again;
- command buffers do not retain their resources, because the layer defers every release;
- a write goes straight into a buffer or a shared texture when no unfinished submit and no earlier queued write uses
  it, and is otherwise queued as a blit, so that writes always land in the order they were made.

The recording backend has no GPU. It records each frame and each resource operation, and it keeps buffer and texture
contents in host memory for headless tests. The WebGPU backend, `src/host/renderer/webgpu_backend.cpp`, runs in the
browser on the WebGPU renderer's device and queue, as [the WebGPU backend](rendering.md#webgpu-backend) describes; it copies
every import, because a browser cannot wrap host memory. A Vulkan backend is planned.

## Tests

The layer's tests run in three suites:

- `caliper_tests --test-case="gfx layer*"` runs the core on the recording backend. It covers the handles, the
  limits, the arena, deferred release, readback, the routes of import, and the refusal of bad frames.
- `caliper_gfx_tests --test-case="gfx/Metal layer*"` runs the layer's rows of the graphics suite on Metal. One row
  draws a list of quads with an indexed draw, depth, additive blending and the ID target. Another draws the
  generated points program from an imported `MTLBuffer`, and checks that the route is reported as in place, with
  zero bytes copied. Each target is read back through the layer and compared byte for byte with a CPU reference.
  Further rows run the generated index clamp as a dispatch and make every generated program's pipelines.
  `CALIPER_GFX_CAPTURE_DIR` writes PNG captures of the rows, and `CALIPER_GFX_LAYER_BENCH=1` runs a row that times
  the CPU encoding of 1,000 draws a frame on both backends.
- `web/test/wasm.spec.ts` runs the WebGPU backend in headless Chromium, on SwiftShader and on the machine's GPU, each
  with a default and a compatibility-mode device ([Testing](../development/testing.md)).

## Colormap by Keys

The renderer core decides a colormap's level with integer comparisons, so that no shader compiler's or adapter's
float arithmetic can move a value to another colour. WGSL allows a division 2.5 ULP from exact and lets an
implementation assume that no NaN or infinity occurs, so a shader that repeats the reference's float formula is exact
only where the compiler happens to keep the reference's operations. The keys make it exact by construction:

- **The key of a float.** A float's bits map to an unsigned key that orders floats numerically: `bits | 0x80000000`
  for a positive float and `~bits` for a negative one. The reference `map_f32_to_rgba8` is monotone in the value,
  because every IEEE operation in it is, so its level is monotone in the key.
- **The keys of a range.** For a range (vmin, vmax), the CPU finds by binary search over keys, for every level k from
  1 to 255, the first key whose reference level is at least k, evaluating `map_f32_to_rgba8` itself 5,900 to 7,900
  times for the ranges measured, in 65 to 83 microseconds on an M-series Max. A view keeps the blocks it uses, so a
  range is derived once; finding a block in the cache costs about 4 nanoseconds and copying it out about 46. A level
  that no float reaches gets the sentinel `0xffffffff`. A decreasing range flips every key, so that the level only
  grows with the key. A degenerate range, an infinite end, an infinite span or a NaN end
  maps every value to level 0, so all its keys are the sentinel.
- **The block.** The `Cmap` uniform block holds the range (for the guess), the flip, and one 16-byte entry per level:
  the level's first key, the next level's first key and the level's packed colour, 4,112 bytes in all, which the
  layer's uniform arena pads to 4,352. A level test and the level's colour are then one load of one entry. The entries
  are 16 bytes because WGSL gives every array in a uniform block a 16-byte stride, so flat arrays of 32-bit keys are
  not possible there, and indexing into a vector of four keys costs more.
- **The decision.** The shader tests NaN on the bits, guesses the level with the usual float formula, and accepts the
  guess only if its entry confirms it with two integer comparisons; it tries the two neighbouring levels next and a
  binary search of eight integer steps last. The guess only chooses which comparisons run. The colour is the entry's
  32-bit colour, converted to [0, 1] and back at the store, an error far below the half level that would change a
  byte.

The algorithm has two sources. The WGSL one is `cmap_color` in `src/host/gfx/shaders/common/colormap.wgsl`, which
every generated program that colours by value uses: the vertex stages of points, lines, boxes and meshes, the image
program, and `colormap_texture`, the compute form that writes an `rgba8unorm` texture from f32 values at an element
base and row pitch, which the texture path is to run. The C++ one is `src/host/gfx/colormap_keys.h`, header-only
C++17 with no GPU, whose decision functions are marked for a CUDA kernel to include. It holds the key of a float, the
derivation of a range's keys, the fill of the `Cmap` uniform block, the integer decision exactly as the shader takes it, the range of a mapping whose ends may come
from statistics, as the snapshot flags `AUTO_VMIN` and `AUTO_VMAX` take them, and `KeyBlockCache`, which keeps the
blocks of one owner by colormap and the exact bits of the range and drops the least recently used. A target that
derives keys links `src/host/tensor_bridge.cpp`, where the reference lives.

The keys follow whatever the reference computes. Its index for a NaN quotient, which an infinite end or span gives, is
defined as 0, and fusing its `t * 255 + 0.5` into one multiply-add, which the compiler does on arm64, changes no
level for any t in [0, 1], so the reference and its keys are the same on every platform.

Integer keys decide the colormap on every path, in the cheaper exact form: in the app on every native backend, in
the browser, on Dawn and in the CPU reference, which give bit-identical colours. Every generated program that colours
by value takes them, and the scene views fill their blocks from the shared cache, natively and in the browser. The
float formula is retired on the native path too, when the texture path moves onto the layer. The cheaper exact form
is the float guess compiled with fast math, one 16-byte entry per level in the 4,112-byte block, and a 16 by 8
workgroup for `colormap_texture`: 128 invocations, the most that a compatibility-mode WebGPU device allows in a
workgroup by default, so that one program serves every device. The decision rested on its cost on Metal, measured on
an M-series Max in quiet moments (`python/bench/results/step6-colormap-keys.txt`). As first built, with keys packed
four to a vector, the layer's safe math and an 8 x 8 workgroup, the compute form took 31.0 µs on a 1,024 x 1,024
heatmap and 79.2 µs on a 2,048 x 2,048 one, against 9.5 µs and 25.8 µs for the Metal renderer's float pass (load
3.49). Most of the difference was the precise division of the float guess, which the decision does not need, and the
rest the vector-packed reads. The cheaper exact form then measured 14.7 µs and 36.3 µs in the same window. As built
now, with one entry per level, fast math and a 16 by 8 workgroup, it takes, per pass over a heatmap, 9.5 µs against
7.7 µs for the float pass at 1,024 by 1,024 and 32.4 µs against 25.8 µs at 2,048 by 2,048, at a 1-minute load of
3.93: within the 20 µs that a step may add to an update's GPU time.

Fast math is a per-program option. The Metal backend compiles every program with fast math off, so that float
results follow IEEE rules as the CPU references do, and `colormap_texture` alone sets `fast_math` in the shader
manifest (`ProgramDesc::fast_math`): no rounding of its float guess can change a texel, because the guess only
chooses which integer comparisons run. The vertex-stage colormap of the scene programs keeps safe math, because
those programs place points and lines on pixel centres.

The renderers' own colormaps do not use the keys yet: the Metal renderer's compute pass `cmap_f32`, the Vulkan
renderer's `colormap.comp`, the CUDA kernel and the WebGPU renderer's heatmap pass with its float thresholds
(`src/host/renderer/webgpu_programs.h`, which draws through the layer's WebGPU backend) each repeat the float formula
or the thresholds, and each is replaced by `colormap_texture` when the texture path of its client moves onto the
layer. The browser's TypeScript views have no shader: their WebGPU renderers, with their WGSL and the thresholds they
searched for, were removed, and they draw only with Canvas 2D, which keeps a TypeScript port of the reference
(`web/src/render/colormap.ts`) that is compared with the C++.

What proves it:

- `caliper_tests --test-case="colormap keys*"` compares the C++ decision with `map_f32_to_rgba8` byte for byte on the
  boundary sets of `tests/shaders/colormap_cases.h`: fourteen ranges that increase, decrease, are degenerate, span one
  float step, have an infinite or NaN end, span subnormals, reach the largest floats or overflow, and for each range
  every key and its two neighbours, the range's ends and their neighbours, signed zeros, infinities, quiet,
  signalling and negative NaNs, subnormals, the largest floats and 65,536 random bit patterns. It also checks that a
  wrong guess never changes a level, the keys' order, the AUTO ends and the cache.
- `caliper_colormap_exhaustive` compares the C++ decision with the reference for all 2<sup>32</sup> inputs on each
  of those ranges, and `caliper_colormap_gpu_exhaustive` does the same for the generated `colormap_texture` on
  Metal, created through the layer with fast math as the layer compiles it. Both are ctest entries under the label
  `slow` that skip unless `CALIPER_SLOW_TESTS=1`, and they run in up to 6 threads.
- `caliper_gfx_tests --test-case="gfx/Metal colormap*"` runs the boundary sets through the layer's Metal backend with
  the generated programs: `colormap_texture` as a dispatch and the vertex stage of `scene_points` as one point a pixel
  wide on each pixel centre, each from a layer buffer and from an imported `MTLBuffer` that the layer reports as in
  place with zero bytes copied, compared byte for byte with the reference.
- `caliper_shader_tests` runs the colormap probe's MSL with fast math on and off, and `web/test/shaders.spec.ts` runs
  the probe, `colormap_texture` and the points' vertex stage in headless Chromium on default and compatibility-mode
  devices, against vectors that `caliper_colormap_cases` writes from the C++.
