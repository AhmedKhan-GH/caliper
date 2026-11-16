# Write a shader

The shaders of the graphics layer in `src/host/gfx/` are written once, in WGSL, and translated to MSL for Metal and to SPIR-V for Vulkan at developer time. The translated sources are committed, so building Caliper needs no Rust and no shader tool. Only a developer who changes a shader runs the translator, and a CI check makes sure that nobody forgets to. The renderers in `src/host/renderer/` still use their own hand-written shaders; so far only the graphics suite's layer and scene rows draw through these programs, on Metal.

## Where the sources are

| Path | Contents |
|---|---|
| `src/host/gfx/shaders/common/` | What the scene programs share: the declarations of the bind groups and varyings (`decls.wgsl`), the colormap decided by integer keys (`colormap.wgsl`), the helpers (`helpers.wgsl`) and the two shared fragment stages (`fragments.wgsl`) |
| `src/host/gfx/shaders/scene/` | The vertex and fragment stages of each kind of scene draw: points, lines, boxes, meshes and images |
| `src/host/gfx/shaders/compute/` | The index clamp, the edge gather and `colormap_texture`, the colormap's compute form for textures |
| `src/host/gfx/shaders/tests/` | Programs that only the tests use, such as the colormap probe |
| `src/host/gfx/shaders/manifest.json` | The programs: each one's sources in order, its outputs, the vertex and fragment pairs the checks make into pipelines, and `fast_math` for a program that Metal may compile with fast math |
| `src/host/gfx/shaders/generated/` | The committed outputs; never edit them by hand |

A program is its sources concatenated in the order the manifest gives. Programs are appended to the manifest, because a program's id is its position there.

The Metal backend compiles every program with fast math off, so that float results follow IEEE rules as the CPU references do: positions land on pixel centres, lighting stays within its two levels and NaN positions are hidden. A program whose output no float rounding can change may set `"fast_math": true` in the manifest, which the generator passes on as `ProgramDesc::fast_math`. Only `colormap_texture` sets it: its float guess only chooses which integer comparisons run, and an exhaustive test on Metal checks every input.

## What is generated

For each program, `tools/shaders/generate.py` writes three headers, each with a comment that names its sources and the Naga version and nothing that changes from one run to the next:

| File | Contents | Used by |
|---|---|---|
| `<program>_wgsl.h` | The assembled WGSL as a string | The browser and Dawn, which compile WGSL themselves |
| `<program>_msl.h` | The MSL as a string | The Metal backend, which compiles it at run time |
| `<program>_spv.h` | The SPIR-V as an array of 32-bit words | The Vulkan backend |

It also writes, for all programs together:

| File | Contents |
|---|---|
| `uniforms.h` | Every uniform block as a C++ struct, such as `Frame`, `PrimParams` and `Cmap`, with its size and every member's offset checked by `static_assert` |
| `programs.h` | `ProgramId`, and each program's entries and bindings as the layer's `EntryDesc` and `BindingDesc`, its vertex inputs, its fragment outputs and its checked pairs |
| `programs_metal.h`, `programs_vulkan.h`, `programs_webgpu.h` | The table of `ProgramDesc`s for one backend, indexed by `ProgramId`, which includes only that backend's sources |
| `programs.json` | The same description for tools and tests outside C++ |

The headers are C++17 and live in the namespace `caliper_gfx::shaders`. A target that includes them has `src/host` and `sdk/include` on its include path.

## Regenerate after a change

Edit the WGSL, then run the generator from the repository root:

```sh
python3 tools/shaders/generate.py
```

The first run builds the translator's driver with cargo, so a Rust toolchain is needed; the build goes to `tools/shaders/naga-driver/target/`, or to `CARGO_TARGET_DIR` when it is set. Commit the changed sources and the changed files under `generated/` together. Before pushing, check that they agree:

```sh
python3 tools/shaders/generate.py --check
```

The check regenerates every output in a temporary folder and fails when a file differs from the committed one, is missing or is no longer generated, or when a WGSL source does not compile. A compile error is reported with Naga's message at the line of the source file that holds it.

## The translator

The translator is Naga 30.0.2, pinned exactly in `tools/shaders/naga-driver/Cargo.toml` and with every other crate in its `Cargo.lock`. Naga is licensed under MIT or Apache-2.0. The driver is a small program on the `naga` crate rather than Naga's own command line, because the command line takes no Metal binding map and gives every resource the attribute `[[user(fake0)]]`, which Metal refuses. Changing the Naga version changes every output, so it is done in one commit with the regenerated files.

The driver translates with these rules, which the generated description records:

- **Metal indices.** Each binding that an entry point uses gets one Metal index, the same in every stage. Buffers, textures and samplers are numbered separately from 0, in (group, binding) order across the program. Buffer indices stay below 15. Vertex buffer slot i is Metal buffer index 16 + i, read through `[[stage_in]]` with `[[attribute(location)]]`.
- **The sizes buffer.** Naga's MSL gives every entry point that uses a runtime-sized storage array one more argument, a buffer of the arrays' byte sizes, at Metal buffer index 15 (`kMetalSizesBuffer`). It declares such an array as a one-element array and indexes past it. The programs read the sizes only through `arrayLength`, which none uses, but the Metal backend binds a buffer there.
- **SPIR-V.** Groups and bindings become descriptor sets and bindings unchanged. Clip space stays WebGPU's, with +y up and no flip in the shader; the Vulkan backend flips with a negative viewport height.
- **Versions.** MSL 2.4 and SPIR-V 1.3, which Vulkan 1.1 accepts.
- **Index checks.** Indices into fixed-size arrays, vectors and matrices are clamped to the last element; indices into storage buffers and texture loads are not checked, and the programs keep them in range.
- **Dynamic offsets.** Every uniform binding is dynamic, because the frame, draw and colormap blocks and the parameters of compute passes all live in the layer's uniform arena.
- **No override constants.** Naga's MSL writer cannot declare them as Metal function constants, so the driver refuses a program that has one.

The driver also compacts each program, so the MSL and SPIR-V hold only what the program's entry points reach.

## Write WGSL that every backend takes

WGSL already rules out what WebGPU lacks, such as point sizes and push constants. The programs also keep to WebGPU's compatibility mode, so that one program serves every device:

- No storage buffer is read in a vertex stage. Per-item data arrives in vertex buffers, at any offset and stride.
- A flat varying is declared `@interpolate(flat, either)`.
- A colour that comes from a value is decided by `cmap_color`, whose integer keys make it exact on every backend. The CPU fills the keys of its `Cmap` block with `src/host/gfx/colormap_keys.h` ([Rendering](../explanations/rendering.md#the-colormap-by-keys)).
- A uniform block keeps WGSL's layout. Its C++ struct is generated, so the CPU code that fills it never copies the layout by hand.

## A hand-written version on one platform

A program may have a hand-written native version on one platform, for example an MSL version that keeps a Metal-only fast path. In the manifest, drop that output from the program's `outputs` and name the header under `hand_written`, as a path below `src/host`:

```json
"outputs": ["wgsl", "spirv"],
"hand_written": {"msl": "gfx/metal/index_clamp_msl.h"}
```

The generator then leaves that file alone and the backend's table includes it in place of a generated header. The hand-written header defines the constant the generated one would, such as `kIndexClampMsl`, and takes the bindings and uniform structs the generator describes. The check covers only the generated outputs; the tests below cover the hand-written one as they cover the others.

## The checks

| Check | Where it runs | What it proves |
|---|---|---|
| `generate.py --check` | The Shaders workflow, with the driver built from the pinned crates and cached | The committed outputs are what the sources give, and every source compiles |
| `tools/shaders/test_generate.py` | The Shaders workflow | The check fails on a changed, missing or extra file and on a source that does not compile, and a hand-written output is left alone |
| `tools/shaders/validate_spirv.py` | The Shaders workflow, with SPIRV-Tools' `spirv-val` | Every SPIR-V program is valid for Vulkan 1.1 |
| `caliper_shader_tests` (`tests/shaders/`) | ctest on macOS, label `gfx` | Every MSL program compiles on the default Metal device, every compute entry and checked pair becomes a pipeline state, and the colormap matches `map_f32_to_rgba8` byte for byte on its boundary cases, with fast math on and off |
| `web/test/shaders.spec.ts` | The web package's Playwright tests | Every WGSL program compiles and every pipeline is created on a default and a compatibility-mode device, the colormap matches the C++ byte for byte on both through the probe, `colormap_texture` and the points' vertex stage, and the points, line and mesh programs draw a fixture |

The browser compares the colormap with vectors generated by the C++. Write them with `caliper_colormap_cases` and pass them on; without the file, the colormap rows and the fixture, whose keys come from it too, are skipped:

```sh
cmake-build-release/tests/caliper_colormap_cases /path/to/cases.bin
cd web && CALIPER_COLORMAP_CASES=/path/to/cases.bin npx playwright test test/shaders.spec.ts --project=webgpu
```
