# `caliper.geometry.v1_3`: instanced transforms (populations in one draw)

**Date:** 2026-07-10
**Status:** implementation spec — ready for execution (subagent-driven, Vulkan-first
run-proven on this Windows/NVIDIA box; Metal transcribed + reviewed, hardware pass
deferred to the mac session, per house protocol).
**Authority (requirement contract, upstream):**
`docs/superpowers/specs/2026-07-10-caliper-framework-remaining-work.md` §2 (the
protocol, the decided-in / rejected list, the §2.1 decision battery);
`docs/superpowers/specs/2026-07-10-twinscope-twin-exemplar-design.md` §4 (the fleet
requirement); `GEOMETRY.md` §1.1/§1.2 invariants + §11 row R3. **House register**
(structure, byte-exact row discipline, gate-battery style):
`docs/superpowers/specs/2026-07-10-geometry-v1_2-textured-mesh-design.md` and
`docs/superpowers/specs/2026-07-09-geometry-v1_1-execution-plan.md`.
**Checkbox discipline (inherited):** a box is checked only when the suite is green,
the path is run-proven by a logged artifact, and the fixing commit is named. A twin
divergence is STOP-and-diagnose, never a loosened comparison. Invariants at the
bottom never become checkboxes.

---

## 1. The rung in one paragraph

R3 is the last rung of the twin story: **populations of shapes.** One imported mesh ×
an imported `(N,16)` f32 column-major pose tensor draws a whole fleet in a single
call — `vkCmdDraw(consumed, N, 0, 0)` / `drawPrimitives:…instanceCount:N` — with the
per-instance model matrix pulled by instance index in the vertex shader, and an
optional imported `(N,)` f32 per-instance scalar tinting each unit through the draw's
existing colormap/`vmin`/`vmax` LUT. It is a **pure additive struct-growth revision**
in the exact shape v1_2 used: a new record `CaliperGeomDrawV1_3` = `V1_2` + an
appended instance tail carried by the existing `draw_primitives` + `draw_stride`
mechanism, one new service id, one new caps bit, `reserved0` still NULL, **no new
entry points**. Honest framing of what exists: both backends' draw calls already carry
the instance-count *slot* R3 turns on, but every shipped draw passes an instance count
of **1** (`vkCmdDraw(cb, count, 1, 0, 0)` at `vulkan_renderer.cpp:760` and `:1270`;
Metal's `drawPrimitives:vertexStart:vertexCount:` at `metal_renderer.mm:755`/`:973`
uses the non-instanced arity; the v1 "instanced points" naming at
`host_renderer.h:110-116` describes the vertex-pull architecture, not an N>1 path).
R3 is the framework's **first true N>1 instanced draw** — what it generalizes is the
existing draw-call slot and vertex-pull mechanism, not a shipped N>1 code path. With R2 (textures-on-meshes, shipped) and R3
merged, the roadmap's rule that **"R2+R3 together ARE the twin demo"** is satisfied:
TwinScope's hidden 50-variant batch becomes fifty housings on screen, each tinted by
its own live state.

**The three spec-pass decisions (resolved in this doc, not deferred):**

- **(a) Naming/layout —** `CaliperGeomDrawV1_3 = CaliperGeomDrawV1_2 + instance tail`,
  a new 256-byte record with a new minimum stride, prefix-identical to v1_2 (which is
  itself prefix-identical to the frozen 192-byte v1.1 `CaliperGeomDraw`). *Rationale:*
  this is byte-for-byte the mechanism v1_2 used to grow from v1_1
  (`geometry_v1_2.h:20-25`, a `base` member + appended fields, `static_assert` on size
  and tail offset) — a combined revision would gain nothing and break the one-rung-per-
  record cadence the ABI tests pin.
- **(c) LAMBERT normal matrix under instancing —** derive the normal from the
  **instance matrix's upper-3×3 composed with the existing per-draw double-precision
  normal matrix, then normalize**, and **restrict instanced LAMBERT draws to rigid +
  uniform-scale instance transforms** (rotation, uniform scale, translation — no shear,
  no non-uniform scale), **enforced by gate G14** (whole-frame refusal, tolerance
  `1e-4` — §5.1), never merely documented. *Rationale:* a full shader-side `inverse(transpose(mat3(…)))`
  of the combined transform is cross-backend non-deterministic (GLSL vs MSL differ at
  the ULP, which fails the byte bar); a second per-instance normal-matrix stream
  doubles the ABI and the host's N-matrix compute for no exemplar benefit; under the
  rigid+uniform-scale restriction the correct normal map is the instance rotation up to
  a scalar that `normalize` removes, so applying the raw instance upper-3×3 to the
  world normal before the per-draw normal matrix is exact-compose and stays a ±2-LSB
  tolerance row exactly as v1_2's Lambert row already is. Full CPU-reference recipe in
  §4.3.
- **(e) Branch —** `feat/geometry-v1_3`, a fresh branch. *Rationale:* v1_1 and v1_2
  each shipped on their own `feat/geometry-v1_N` branch; a combined name only earns its
  keep when a rung ships alongside another, which R3 does not.

**REJECTED (restated as rejected, not deferred): per-instance textures**
(texture-array / atlas ABI). The hero unit carries the draped field at texture
resolution (R2); the fleet carries scalar state through the LUT (R3). A future need for
per-instance textures must bring its own exemplar and its own rung — it is not on the
2-manifold ladder this revision extends.

---

## 2. ABI — the record decision (a)

### 2.1 The new record

The v1.1 `CaliperGeomDraw` prefix stays frozen at 192 bytes (`geometry_v1_1.h:115`);
the v1.2 record stays frozen at 216 (`geometry_v1_2.h:55-58`). v1.3 appends an
instance tail to the v1.2 record, in a new header `sdk/include/caliper/services/
geometry_v1_3.h` that includes `geometry_v1_2.h`:

```c
#define CALIPER_GEOMETRY_V1_3 "caliper.geometry.v1_3"

/* caps() bit 3: instanced draws are live. */
#define CALIPER_GEOM_CAP_INSTANCED (1u << 3)

typedef struct CaliperGeomDrawV1_3 {
    CaliperGeomDrawV1_2 base;          /* frozen 216-byte v1.2 record */
    CaliperAllocId instance_alloc;     /* (N,16) f32 column-major model matrices */
    uint64_t       instance_offset;    /* bytes, 4-byte aligned */
    uint64_t       instance_count;     /* N; 0 or instance_alloc==0 -> non-instanced */
    CaliperAllocId instance_attr_alloc;/* optional (N,) f32; 0 = no per-instance tint */
    uint64_t       instance_attr_offset;/* bytes, 4-byte aligned */
} CaliperGeomDrawV1_3;
```

No new color-mode constant is added: the per-instance tint reuses `COLORMAP` LUT
semantics (`colormap`/`vmin`/`vmax` already on the base record). No new topology,
shade, blend, or depth constants. `reserved0` remains NULL.

### 2.2 Exact layout (byte offsets, LP64 / MSVC x64 — all fields 8-byte)

`CaliperAllocId` and `CaliperTextureId` are 8-byte handles; the v1.2 record is 216
bytes with `uv_alloc@192`, `uv_offset@200`, `texture@208` (pinned in
`test_abi.cpp:194-198`). The instance tail therefore lands at:

| field | type | offset | size |
|---|---|---:|---:|
| `base` | `CaliperGeomDrawV1_2` | 0 | 216 |
| `instance_alloc` | `CaliperAllocId` (u64) | 216 | 8 |
| `instance_offset` | `uint64_t` | 224 | 8 |
| `instance_count` | `uint64_t` | 232 | 8 |
| `instance_attr_alloc` | `CaliperAllocId` (u64) | 240 | 8 |
| `instance_attr_offset` | `uint64_t` | 248 | 8 |
| **sizeof** | | | **256** |

`alignof(CaliperGeomDrawV1_3) == 8`. Header `static_assert`s (mirroring
`geometry_v1_2.h:55-58`): `sizeof == 256`; `offsetof(instance_alloc) == 216`;
`offsetof(base) == 0`; plus a re-assert that `sizeof(CaliperGeomDrawV1_2) == 216` so a
drift in the middle record is caught here too.

### 2.3 The service table

`CaliperGeometryV1_3` has the same slots as v1.2 (`geometry_v1_2.h:27-51`): identical
v1 prefix, `create_view_ex`, and a `draw_primitives` typed for the extended record
but with the same binary calling convention and still receiving `draw_stride`.
`reserved0` remains NULL. The host vends all four revisions (`kGeom1`, `kGeom11`,
`kGeom12`, and the new `kGeom13` — `host_services.cpp:333-340`). Minimum strides by
revision: v1.1 = 192, v1.2 = 216, v1.3 = 256. A v1.3 caller cannot expose an absent
tail (short-stride refusal), and a v1.1/v1.2 caller can never reach the instance tail
(their entry points route to the enum axis below with the lower ceiling).

### 2.4 ABI test pins to add (`test_abi.cpp`, extending 192-208)

- `static_assert(std::is_standard_layout_v<CaliperGeomDrawV1_3>)`.
- `sizeof(CaliperGeomDrawV1_3) == 256`; `sizeof(CaliperGeomDrawV1_2) == 216`;
  `sizeof(CaliperGeomDraw) == 192` (all three, so the chain is pinned in one place).
- `offsetof(CaliperGeomDrawV1_3, base) == 0`, `instance_alloc == 216`,
  `instance_offset == 224`, `instance_count == 232`, `instance_attr_alloc == 240`,
  `instance_attr_offset == 248`.
- Service-table parity: `offsetof(CaliperGeometryV1_3, draw_primitives) ==
  offsetof(CaliperGeometryV1_2, draw_primitives)` and same for `reserved0`,
  `struct_size`, the v1 prefix (mirror `test_abi.cpp:200-205`).
- `CALIPER_GEOMETRY_V1_3` string equals `"caliper.geometry.v1_3"`;
  `CALIPER_GEOM_CAP_INSTANCED == (1u<<3)`.
- **Widening/compat regression** — the true analog of the v1_2 precedent
  (`test_abi.cpp:214-274`, where `kStubGeom12` is the fixture's ONLY geometry table,
  `:258`): provide a stub `kStubGeom13` as the **only** geometry service (v1, v1_1,
  v1_2 all NULL — a v1.3-only host), then drive fully-poisoned frozen **192-byte
  v1.1-shaped** records and **216-byte v1.2-shaped** records through the SDK wrapper's
  respective overloads. Assert each call arrives widened: received stride ==
  `sizeof(CaliperGeomDrawV1_3)` (256), the 192/216-byte prefix intact byte-for-byte,
  and the appended tail all-zero. (The pixel-level half of the additive-default rule —
  a v1.3 record with `instance_alloc==0` drawing byte-identically to v1.2 — is §8
  row C.)

---

## 3. Host validator — the enum revision axis refactor (§2.1's "single-axis" fold-in)

The R2 hardening replaced a rejected two-bool encoding with a single `bool v12` axis in
`TensorBridge::geom_draw_primitives_impl` (`tensor_bridge.cpp:587-596`,
`tensor_bridge.h:151-159`). R3 **generalizes that one axis to an enum** — it does NOT
add a second bool (the two-bool encoding was the reviewed defect; do not reintroduce
it).

### 3.1 The refactor

Replace `bool v12` with a scoped enum declared next to the impl:

```cpp
enum class GeomRev : uint32_t { V1_1, V1_2, V1_3 };
```

`geom_draw_primitives_impl(..., GeomRev rev, ...)`. The three public entry points call
it with `GeomRev::V1_1` / `V1_2` / `V1_3` (replacing `/*v12=*/false` at
`tensor_bridge.cpp:576` and `/*v12=*/true` at `:584`, plus the new v1.3 wrapper).

### 3.2 What the enum derives (replacing lines 593-596)

- **`min_stride`** — `192 / 216 / 256` for `V1_1 / V1_2 / V1_3`
  (`sizeof(CaliperGeomDraw)` / `…V1_2` / `…V1_3`).
- **`max_color`** — `VERTEX_RGBA (2)` for `V1_1`, `COLOR_TEXTURE (3)` for `V1_2` **and**
  `V1_3` (v1.3 adds no color mode; the per-instance tint is not a color-mode value).
- **`instance_axis`** — `rev == GeomRev::V1_3`: only then is the instance tail read
  (`d13->instance_alloc` etc.); for `V1_1`/`V1_2` the tail bytes do not exist and are
  never dereferenced, exactly as the v1.2 `uv`/`texture` tail is only read when
  `rev >= V1_2` today (`tensor_bridge.cpp:751-763`).

The per-draw resolve loop (`tensor_bridge.cpp:642-791`) reinterprets the record as
`CaliperGeomDraw* d` / `CaliperGeomDrawV1_2* d12` / (new) `CaliperGeomDrawV1_3* d13`,
each at the same `record` base — a prefix cast, standard-layout-safe. The instance
gate block (§5) runs only under `GeomRev::V1_3`.

### 3.3 New host plumbing

- `tensor_bridge.h`: add `geom_draw_primitives_v1_3(...)`, update the
  `geom_draw_primitives_impl` signature to take `GeomRev`.
- `tensor_bridge.cpp`: `geom_caps()` (`:459-468`) gains
  `if (primitives && renderer_.supports_geometry_instanced()) c |=
  CALIPER_GEOM_CAP_INSTANCED;` (a new backend query mirroring
  `supports_geometry_textured()` at `host_renderer.h:130`).
- `host_services.cpp`: add `geo_draw_primitives_v13` (`:315-332` pattern), the
  `kGeom13` service (`:338-340` pattern), and register `CALIPER_GEOMETRY_V1_3` in
  `kIds` + the `service()` switch (`:427`).
- `HostGeomDraw` (`host_renderer.h:13-43`) grows resolved instance fields:
  `instance_alloc`/`instance_offset`/`instance_count`/`instance_attr_alloc`/
  `instance_attr_offset` (renderer ids + byte offsets, mirroring how `uv_alloc` holds a
  resolved renderer id — `tensor_bridge.cpp:776-777`). Defaults zero → non-instanced.
  **Tint-LUT resolution rule (binding contract):** `HostGeomDraw.lut256` is populated
  whenever the draw needs a LUT — base `color_mode == COLORMAP` (as today,
  `tensor_bridge.cpp:743-745`) **or** `instance_attr_alloc != 0` — resolved from the
  base record's `colormap` (gate G12 guarantees it resolves), **regardless of base
  color_mode**. A FLAT- or VERTEX_RGBA-based instanced-tint draw still carries a real
  LUT; today `lut256` stays null for those modes, which would leave the tint reading
  placeholder garbage (§6.2).
- SDK sugar `sdk/include/caliper/caliper.hpp` (`:433-437`): add
  `geom_draw_v1_3_defaults()` returning `{ .base = geom_draw_v1_2_defaults() }` with a
  zero instance tail.
- **SDK `caliper::Geometry` wrapper surface (pinned — TwinScope draws through it,
  `twin_scope.cpp:889/:912`).** The wrapper (`caliper.hpp:336-418`) grows exactly:
  - a `g13_` member resolved from `CALIPER_GEOMETRY_V1_3` in the constructor,
    following the `g12_` resolution pattern (`:344-345`);
  - `bool has_instanced()` returning `(caps() & CALIPER_GEOM_CAP_INSTANCED) != 0u`,
    mirroring `has_textured()` (`:364-366`);
  - a `draw_primitives(view, cam, const CaliperGeomDrawV1_3* draws, count, clear)`
    overload passing `sizeof(CaliperGeomDrawV1_3)` as stride, mirroring the v1.2
    overload (`:404-412`);
  - the **widening tier** `if (!g12_ && g13_) { g12_ =
    reinterpret_cast<const CaliperGeometryV1_2*>(g13_); widen_v12_draws_ = true; }`,
    placed BEFORE and chaining into the existing `!g11_ && g12_` tier (`:346-352`), so
    on a v1_3-only host both flags set. The overloads then widen to the **widest
    required record**: the v1.1 overload (`:391-403`) builds zero-tailed
    `CaliperGeomDrawV1_3` records (stride 256) when `widen_v12_draws_`, else zero-tailed
    `V1_2` records (stride 216) when only `widen_v11_draws_`; the v1.2 overload builds
    zero-tailed `V1_3` records (stride 256) when `widen_v12_draws_`. Prefix copied
    intact, tail zero — the v1_2 widening precedent (`:398-402`) extended one tier.

---

## 4. Semantics (b) + LAMBERT decision (c) + the CPU reference

### 4.1 Effective transform

Effective pipeline is `proj · view · draw_model · instance_matrix · vertex`. The host
premultiplies `mvp_draw = proj · view · draw_model` per draw exactly as today
(`vulkan_renderer.cpp:1092-1095`, `metal_renderer.mm:916-919`). The **instance matrix
cannot be premultiplied host-side** (it varies per instance), so it is pulled in the
vertex shader by instance index and applied to the world position **first**:

```
world'   = instance_matrix * vec4(world_pos, 1.0)   // instance in [0,1] NDC-agnostic model space
gl_Position = mvp_draw * world'
```

`mvp_draw` is the unchanged per-draw `PrimParams.mvp`. When `use_instance == 0` the
shader skips the instance multiply entirely and computes `gl_Position = mvp_draw *
vec4(world_pos,1)` — **bit-identical to the v1.2 path** (§8 row C proves it pixel-exact).

### 4.2 The non-instanced additive default (byte-identity)

`instance_count == 0` **OR** `instance_alloc == 0` → today's non-instanced path,
byte-identical. Mechanically: the host sets `PrimParams.use_instance = 0` and issues
`vkCmdDraw(consumed, 1, 0, 0)` / `drawPrimitives:…vertexCount:consumed` (no
`instanceCount`), taking the exact same shader code path and draw-call arity as v1.2.
A v1.3 record with a zero instance tail is therefore indistinguishable at the pixel
level from the equivalent v1.2 record — the additive-default contract the parent
requires.

### 4.3 Per-instance tint (COLORMAP semantics)

`instance_attr_alloc != 0` (and the draw is instanced) overrides the vertex color
source: each instance's color is `LUT[ idx(attr_i) ]` where `attr_i` is that instance's
scalar, `idx` is the **same** index rule the shader and CPU reference already share
(`geom.vert:79-84`):
`t = (v==v && vmax>vmin) ? clamp((v-vmin)/(vmax-vmin),0,1) : 0; idx = floor(t*255 +
0.5)`. The value is looked up **once per instance in the vertex shader** (uniform across
the instance's vertices) and reaches the fragment stage through `v_color`. `LAMBERT`, if
enabled, multiplies this tint by the lit term exactly as it multiplies any other color
today (`geom.vert:93-101`). When `instance_attr_alloc == 0`, coloring is whatever the
base `color_mode` specifies (vertex COLORMAP / VERTEX_RGBA / FLAT / TEXTURE), unchanged.
The instance-tint path reuses the existing per-frame LUT ring (`vulkan_renderer.cpp:
1130-1151`, binding 4) — the LUT is a per-draw resource already, and one tint LUT per
instanced draw fits the same ring.

### 4.4 LAMBERT under instancing — the byte-exact CPU reference

**Decision (c), restated:** instanced LAMBERT draws require the instance matrix to be
rigid + uniform-scale (rotation `R`, uniform scale `s>0`, translation `t`). The normal
under the combined transform is then, up to the positive scalar `1/s` that `normalize`
removes, the composition of the instance rotation with the existing per-draw normal
matrix. The vertex shader computes, in this exact float order:

```
n_model  = normalize(world_normal)                 // as today
n_inst.x = im0.x*n_model.x + im1.x*n_model.y + im2.x*n_model.z   // instance upper-3x3 * n
n_inst.y = im0.y*n_model.x + im1.y*n_model.y + im2.y*n_model.z   //   (im0/im1/im2 = instance
n_inst.z = im0.z*n_model.x + im1.z*n_model.y + im2.z*n_model.z   //    matrix columns 0/1/2, xyz)
nvs      = normalize(n_inst.x*nmat0.xyz + n_inst.y*nmat1.xyz + n_inst.z*nmat2.xyz)
lit      = 0.30 + 0.70 * max(dot(nvs, vec3(0,0,1)), 0)
```

`nmat0/1/2` are the **unchanged** per-draw double-precision normal-matrix columns
(host-computed from `view·draw_model` — `vulkan_renderer.cpp:1096`,
`metal_renderer.mm:920`). When `use_instance == 0`, the `n_inst` step is skipped and
`nvs` reduces to today's `normalize(n.x*nmat0 + n.y*nmat1 + n.z*nmat2)` — byte-identical.

**The gfx CPU reference** (§8) computes the whole chain in the same float op order on
the host and compares within the v1.2 Lambert tolerance (±2 RGB LSB, alpha exact — the
bar `geometry_v1_2.md` verification row 6 already uses). Positions, tint, and depth
rows are **exact** (0 LSB); only the Lambert-lit rows carry the ±2-LSB tolerance, and
only because a `normalize` sits in the path — the same reason v1.2's Lambert row is a
tolerance row, not a regression from R3. An out-of-class instance matrix on a LAMBERT
draw **would be silently mis-lit**: for shear or non-uniform scale, raw-upper-3×3 +
`normalize` does NOT equal the inverse-transpose even up to a positive scalar, so the
image would be wrong under a clean status line — a direct violation of "never a wrong
image." The restriction is therefore **enforced, not documented**: gate G14 (§5.1)
refuses the whole frame when a LAMBERT-instanced draw carries an out-of-class instance
matrix, pixels untouched. The fleet demo uses grid translations + uniform scale, well
inside the class. UNLIT instanced draws (the fleet's default tint mode) never touch
normals and are unrestricted.

---

## 5. Caps bit 3 + the gate battery (d)

`CALIPER_GEOM_CAP_INSTANCED (1u<<3)`, granted only when
`supports_geometry_primitives() && supports_geometry_instanced()`
(`tensor_bridge.cpp:459-468`). Absent → the whole instanced path is inert and the
applet ladders down (TwinScope: "fleet needs instanced geometry (cap absent)" —
twin-exemplar §1.7). Validation stays **atomic**: one bad draw refuses the whole frame
and leaves the target pixels untouched (`tensor_bridge.cpp:49-50` contract). All gates
run in the host validator (§3) **and** are re-gated in each backend before any encode
(defense in depth — `vulkan_renderer.cpp:906-1077`, `metal_renderer.mm` mirror). Each
gate below is listed with its refusal log line; the host uses the existing
`geom_prims: draw %u refused: <reason>` sink (`tensor_bridge.cpp:647-651`), Vulkan uses
`dev_bail("primitives: <reason>")` (`:919+`), Metal uses `metal_geom_fail("primitives:
<reason>")` (`:358-361`). The instance block runs only under `GeomRev::V1_3` and only
when `color_mode`/topology otherwise validates.

| # | gate | condition | refusal reason string |
|---|---|---|---|
| G1 | instanced-cap applicability | `instance_alloc!=0 || instance_attr_alloc!=0` requires the caps bit live (backend `supports_geometry_instanced()`) | `instanced geometry unsupported` |
| G2 | N>0 with alloc | `instance_alloc!=0` requires `instance_count>0` (mirrors the "zero vertices" gate `:678`) | `instanced draw needs N>0` |
| G3 | N bound | `instance_count <= UINT32_MAX` (Vulkan `instanceCount`/`gl_InstanceIndex` are u32; `metal_renderer.mm` instanceCount is `NSUInteger` but re-bound to u32 for parity) | `too many instances` |
| G4 | matrix alignment | `instance_offset % 4 == 0` (the uniform 4-byte offset rule, `range_ok` `:618`) | `instance offset misaligned` |
| G5 | matrix overflow + bounds | `instance_count * 64` overflow-safe AND `instance_offset + N*64 <= imported size` (16 f32 = 64 B/instance; `range_ok(size, off, N, 64, "instances")`) | `instances out of imported bounds` |
| G6 | matrix base fits u32 | `instance_offset / 4 <= UINT32_MAX` (rides a `PrimParams` u32 base). Fires in **BOTH** the host validator **and** each backend re-gate, like the existing base checks (`vulkan_renderer.cpp:978,993`; Metal's uv twin `metal_renderer.mm:933-934`) — the Metal parity ledger stays honest | `instance base exceeds 32 bits` |
| G7 | matrix alloc live | `imported_.find(instance_alloc)` resolves (host) / renderer `imported_` resolves (backend) | `unknown instance alloc` |
| G8 | attr requires instances | `instance_attr_alloc!=0` requires `instance_alloc!=0 && instance_count>0` (a tint with nothing to tint is refused, not ignored) | `instance attr without instances` |
| G9 | attr alignment | `instance_attr_offset % 4 == 0` | `instance attr offset misaligned` |
| G10 | attr bound | `instance_count * 4` overflow-safe AND `instance_attr_offset + N*4 <= imported size` (`range_ok(…, N, 4, "instance attr")`) | `instance attr out of imported bounds` |
| G11 | attr alloc live | `imported_.find(instance_attr_alloc)` resolves | `unknown instance attr alloc` |
| G12 | tint needs colormap | `instance_attr_alloc!=0` requires `colormap_lut(colormap) != nullptr` (reuses the COLORMAP LUT gate `:743-745`) | `instance tint needs colormap` |
| G13 | LAMBERT+instanced needs normals | already covered by the existing "lambert needs normals" gate `:719`; no new gate (the normal stream is per-vertex, shared across instances) | *(existing)* `lambert needs normals` |
| G14 | LAMBERT rigidity (the §4.4 enforcement) | when `shade_mode==LAMBERT && instance_count>0`: every instance upper-3×3 is orthogonal-up-to-uniform-scale within the §5.1 tolerance | `instanced lambert needs rigid+uniform-scale` |

### 5.1 The G14 rigidity check (tolerance is part of the contract)

For each instance matrix, with `c0,c1,c2` the columns of the upper-3×3 (f32, read as
imported) and `s̄² = (‖c0‖² + ‖c1‖² + ‖c2‖²) / 3`, the draw is refused unless, for a
**contract constant `kGeomRigidTol = 1e-4`** (relative, dimensionless):

- `|c_i · c_j| <= kGeomRigidTol * s̄²` for all `i != j` (columns orthogonal), AND
- `| ‖c_i‖² − s̄² | <= kGeomRigidTol * s̄²` for each `i` (columns equal-length), AND
- `s̄² > 0` (a zero/degenerate upper-3×3 is refused).

*Why `1e-4`:* f32 rotation/scale matrices composed from a handful of f32 ops carry
relative error ~`1e-6`, so `1e-4` gives two orders of headroom over legitimate
composition noise, while any intentional shear or non-uniform scale registers at
`1e-2`+ — two-plus orders above the gate and four-plus above composition noise — and
the residual non-orthogonality it
admits perturbs the lit term well inside the ±2-LSB Lambert row tolerance (§4.4). The
number is part of the byte-exact contract: it is pinned in the header next to the caps
bit and asserted in the G14 gate test, never a tunable.

*Cost:* an O(N) scan of `N*64` bytes, run **only** on LAMBERT-instanced draws
(comparable to the existing per-draw bounds scans; a pose-only or UNLIT fleet never
pays it). *Placement honesty:* the instance matrices live in an **imported device
allocation** — the host's `imported_` table resolves ids and sizes, never a mapped
host pointer (`tensor_bridge.cpp:688-691`), so while the gate's *contract* (atomicity,
reason string, refusal-before-any-encode) belongs to the validator battery above, its
*execution* lands in each backend's re-gate stage, where the imported bytes are
addressable via a bounded `N*64`-byte readback of the instance range — before any
clear or encode, refusal surfacing through the same whole-frame-false path with pixels
untouched. Both backends run the identical comparison in the identical float order
(Metal transcribed, §7).

*Readback mechanics (binding, not guidance):* **(a)** all LAMBERT-instanced draws in
the array share **one** staged-copy submit (one fence wait) placed after the metadata
gate loop and before the render submit — never one round-trip per draw. **(b)** the
staging buffer is a grow-only host-visible buffer reusing the `ensure_buffer` pattern
of `geom_prim_params_` (`vulkan_renderer.cpp:1120-1128`) — never allocated per frame.
**(c) Ledger note:** G14 is the hot path's **first device-contents read** (every
existing re-gate reads metadata only). Cost: one extra fenced round-trip + `N*64`
bytes over PCIe per frame containing LAMBERT-instanced draws (fleet: N=50 → 3.2 KB,
microseconds against a path that already fence-waits its render — `submit_once`,
`vulkan_renderer.cpp:1735-1753`); it scales with the same `N*64` bytes the draw itself
reads. Coherence rides the drain-before-publish temporal half (`geometry_v1.h:66-90`):
the producer drained before flipping the slot, so the bytes are complete and stable
when read. On Metal, shared-storage imported buffers may be read via `contents()`
directly (no blit) — same comparison, same float order. The refusal-purity claim is
unaffected: the copy touches only staging, never the view.

**Whole-frame refusal & pixel purity:** every `return false` above leaves the view's
pixels exactly as they were, because (as on every existing path) nothing is cleared or
encoded until the entire draw array passes gating — the host builds the resolved vector
first (`tensor_bridge.cpp:640-791`) and the backend builds its `Enc` vector before
`vkCmdBeginRenderPass` (`vulkan_renderer.cpp:932-1112`). **Released-alloc refusal:** a
freed `instance_alloc`/`instance_attr_alloc` fails G7/G11 (`imported_.find` miss) on
both the host and the backend re-gate, so a use-after-release draws nothing. **Backend
re-gate for parity:** the Vulkan and Metal binding models differ (SSBO descriptor vs
`setVertexBuffer` index), so each backend re-runs G1-G12 against its own `imported_`
table before binding — plus G14, which executes *only* there (§5.1) — the same
discipline the v1.2 texture gate uses (`vulkan_renderer.cpp:1070-1076` re-checks the
sampled texture; Metal mirrors).

---

## 6. Vulkan backend (run-proven on this box)

### 6.1 Draw call

Change the two instanced-count arguments at `vulkan_renderer.cpp:1270` from `1` to the
per-draw instance count:
`vkCmdDraw(cb, encs[i].consumed, encs[i].use_instance ? encs[i].n_inst : 1u, 0, 0);`
`gl_InstanceIndex` then ranges `[0, N)` in the vertex shader. Indexed draws remain
NON-indexed (the shader pulls + clamps indices, `:1268-1269`) — instancing composes
with that unchanged.

### 6.2 Instance buffers — descriptor growth

Two new readonly SSBO bindings on the geom-prim set layout (`vulkan_renderer.cpp:
2405-2422`), extending bindings 0-7:

- **binding 8** — `Inst { float im[]; }` — the `(N,16)` column-major matrices,
  vertex stage.
- **binding 9** — `InstAttr { uint iattr[]; }` — the `(N,)` per-instance scalars, bound
  as `uint[]` so NaN bit patterns survive to `uintBitsToFloat` (the same trick binding 3
  uses, `geom.vert:18`), vertex stage.

Grow `bindings[8]` → `bindings[10]`, `li.bindingCount = 10`; the descriptor pool's
`STORAGE_BUFFER` count grows from `6u*cap` to `8u*cap` (`vulkan_renderer.cpp:2596`).
Absent-source binding follows the existing placeholder trick (bind `pos` when a stream
is unused so the shader's guarded read is harmless — `:1172-1190`): when
`use_instance==0` bind `pos` at 8; when `use_instance_attr==0` bind `pos` at 9. The
descriptor writes extend the `bi[7]`/`wr[8]` arrays at `:1174-1217` to `bi[9]`/`wr[10]`.

**Tint LUT at binding 4 (explicit — do not follow the current predicate literally):**
today the LUT ring slot is assigned only for COLORMAP draws (`e.lut_slot >= 0` when
`d.color_mode == COLORMAP`, slot assignment `:1048-1052`, ring write `:1148-1151`,
binding `:1179-1185`); every other draw placeholder-binds `pos` at binding 4
(`:1184`). R3 changes the slot-assignment predicate to **`d.lut256 != nullptr`** — set
by the host for COLORMAP **or** instanced-tint draws (§3.3 resolution rule) — so an
instanced-tint draw over a FLAT/VERTEX_RGBA/TEXTURE base still receives the real LUT
at binding 4. An executor who leaves the COLORMAP-only predicate in place ships a tint
path that reads placeholder garbage; §8 row B (tint over a FLAT base) exists to catch
exactly that.

### 6.3 PrimParams growth — the three hand-synced copies

`PrimParams` is 176 bytes today with a `uv_base` + three trailing `pad0/pad1/pad2`
uints (`geom.vert:47-50`, `vulkan_renderer.cpp:136-147`, `metal_renderer.mm:298-309`,
all `static_assert(… == 176)`). The comment naming the three copies to move together is
`geom.vert:26-29` ("THREE hand-synced copies … grep `PrimParams` when growing"). Grow to
**192 bytes** by consuming the three pads and adding one 16-byte tail block:

```
uint uv_base;            // 160  (unchanged)
uint use_instance;       // 164  (was pad0) — 0/1
uint inst_base;          // 168  (was pad1) — instance_offset / 4
uint use_instance_attr;  // 172  (was pad2) — 0/1
uint inst_attr_base;     // 176  — instance_attr_offset / 4
uint pad0, pad1, pad2;   // 180/184/188  (re-pad to 192)
```

Update all three `static_assert`s to `== 192` and the GLSL `std140` block, the Vulkan
`sizeof(PrimParams)` dynamic-UBO write (`:1146-1147`), and the MSL struct in lockstep.

### 6.4 `geom.vert` diff sketch

```glsl
layout(std430, set = 0, binding = 8) readonly buffer Inst     { float im[];    };
layout(std430, set = 0, binding = 9) readonly buffer InstAttr { uint  iattr[]; };
// … in main():
mat4 M;   // instance model (identity when not instanced)
if (p.use_instance != 0u) {
    uint b = p.inst_base + 16u * uint(gl_InstanceIndex);
    M = mat4(vec4(im[b+0],im[b+1],im[b+2],im[b+3]),   // column 0
             vec4(im[b+4],im[b+5],im[b+6],im[b+7]),   // column 1
             vec4(im[b+8],im[b+9],im[b+10],im[b+11]), // column 2
             vec4(im[b+12],im[b+13],im[b+14],im[b+15]));
    gl_Position = p.mvp * (M * vec4(wp, 1.0));
} else {
    gl_Position = p.mvp * vec4(wp, 1.0);              // byte-identical to v1.2
}
// per-instance tint overrides the color source (before the color_mode switch):
if (p.use_instance != 0u && p.use_instance_attr != 0u) {
    float v = uintBitsToFloat(iattr[p.inst_base==0u ? uint(gl_InstanceIndex)   // see note
                                     : uint(gl_InstanceIndex)]);               // iattr base below
    float t = (v==v && p.vmax>p.vmin) ? clamp((v-p.vmin)/(p.vmax-p.vmin),0.0,1.0) : 0.0;
    c = unpack_rgba(lut[uint(t*255.0+0.5)]);
} else { /* existing color_mode switch, geom.vert:78-91 */ }
// LAMBERT: apply M's upper-3x3 to n before nmat (§4.4) when use_instance.
```

*(Note: the instance-attr index is `p.inst_attr_base + gl_InstanceIndex` — the sketch
above is schematic; the real diff uses `inst_attr_base` for the attr SSBO and
`inst_base` for the matrix SSBO. The LUT for the tint is bound at binding 4 as for
COLORMAP.)* The `geom_prim_pipeline` cache key (`:1084-1086`,
`geom_prim_pipeline(topology, blend, depth_flags, has_depth, textured)`) gains no new
axis — instancing is a uniform-driven branch, not a new pipeline (the `size_px`
unconditional-write note at `geom.vert:104-111` shows the project prefers a
uniform branch over a pipeline explosion; the same rationale applies here).

---

## 7. Metal backend (transcription contract — NEVER claim verified here)

Transcribed and reviewed on this Windows box against the Vulkan reference and the shared
CPU references; **no Apple GPU executes any of it in this pass** — the hardware claims
fold into the next macOS session (the v1.2 hardware-pass doc `2026-07-10-metal-macos-
v1_2-hardware-pass.md` is the precedent format, and §11 below reserves the §8-style
addendum). Risk class: MSL compile, stage-in signature, pipeline-creation, and
instance-index-semantics surprises a review cannot catch.

- **Draw call:** `metal_renderer.mm:973`
  `[re drawPrimitives:e.prim vertexStart:0 vertexCount:e.consumed]` →
  `…instanceCount:(e.use_instance ? e.n_inst : 1)]`. MSL pulls the matrix by
  `uint iid [[instance_id]]`.
- **Buffers:** vertex buffer indices 0-6 and PrimParams@5 are taken
  (`metal_renderer.mm:964-971`); bind the instance matrices at **index 7** and the
  instance attr at **index 8** (`setVertexBuffer:…atIndex:7/8`), placeholder-bound to
  `e.pos` when a stream is unused, mirroring the Vulkan trick. The fragment texture stays
  at texture index 0 (`:972`) — no collision.
- **MSL:** grow the in-shader `PrimParams` (`metal_renderer.mm:161-182`) and the host
  `PrimParams` (`:298-309`) to 192 bytes in lockstep with §6.3; add `geom_compute`
  parameters `device const float* im [[buffer(7)]]` and `device const uint* iattr
  [[buffer(8)]]`, threaded through both `geom_vs` (`:257-267`) and `geom_vs_point`
  (`:269-285`) with `uint iid [[instance_id]]`. Apply the instance matrix and the
  §4.4 normal composition with the **identical float op order** as GLSL.
- **Tint LUT (buffer index 4):** Metal binds the LUT by value —
  `setVertexBytes:(e.d->lut256 ? e.d->lut256 : kZeroLut) … atIndex:4`
  (`metal_renderer.mm:968-969`) — so the §3.3 host resolution rule is sufficient: with
  `lut256` populated for instanced-tint draws, the correct LUT lands at index 4
  regardless of base color_mode. The `kZeroLut` fallback must never be reachable for a
  tint draw (G12 refuses upstream); the Metal twin of §8 row B pins it.
- **Day-one gate parity:** G1-G12 plus G14 (§5/§5.1) fire in the Metal re-gate before
  encode, byte-for-byte the same reasons and the same G14 float order, so a Metal build
  refuses exactly what Vulkan refuses even before hardware runs. Fix the T3-ledgered cosmetic while here if it recurs: the Metal refusal
  log double-prefix (`geom_prims: primitives:`) noted in remaining-work §1.2 — do not add
  a new instance of it.
- **`supports_geometry_instanced()`** on `MetalRenderer` returns
  `supports_external_import()` (mirroring `supports_geometry_textured()`,
  `vulkan_renderer.cpp:777-779`).

---

## 8. gfx byte-exact rows (both backends, §9.2 register)

Rows follow the `gfx_main.cpp` pattern: CPU-computed reference image,
`debug_readback_rgba8`, byte compare; geometry chosen pixel-center-unambiguous. Each
row exists in the **Vulkan block** (`#ifdef CALIPER_HAVE_VULKAN`, ~`gfx_main.cpp:
1000-2350`, **runs on this box**) and the mirrored **Metal block**
(`#ifdef CALIPER_HAVE_METAL`, ~`:3170-4750`, **transcribed, runs in the mac pass**),
gated on `metal_env().ok`. Add a small CPU helper `instanced_geom_reference(...)` that
composes the transform and the §4.4 normal chain in host float, reused by every row.
**Matrix association is pinned (byte-load-bearing):** the reference applies `M_i` to
the vertex **vector first**, then `mvp_draw` to the result —
`p' = mvp_draw * (M_i * vec4(v, 1))` — matching the shader's grouping (§6.4
`p.mvp * (M * vec4(wp, 1.0))`). It must **never** premultiply `(mvp_draw * M_i)` into
one matrix: matrix-matrix-then-vector groups the float sums differently and breaks
Row A's 0-LSB claim.

- **Row A — pose-only fleet, N distinct transforms (exact).** One triangle/quad, N=4
  instances at four axis-separated translations (pixel-center-unambiguous), FLAT color,
  no depth. Reference: four quads at the four `M_i`-transformed screen rects.
  `last_device_path == "primitives-imported"`. **Exact (0 LSB).**
- **Row B — per-instance tint through the LUT (exact).** Same N=4, `instance_attr` = 4
  distinct scalars spanning `[vmin,vmax]` at exact quantized LUT indices, MAGMA,
  UNLIT — and the base record's `color_mode` set to **FLAT**, so the row also proves
  the binding-4 tint-LUT rule (§6.2: a COLORMAP-only LUT predicate would read
  placeholder garbage here). Reference: each quad = `unpack_rgba(LUT[idx(attr_i)])`.
  **Exact (0 LSB)** — the tint path has no `normalize`.
- **Row C — `instance_count==0` == non-instanced, byte-identical (exact).** Draw the
  same geometry twice into two views: once as a v1.2 record, once as a v1.3 record with
  `instance_alloc==0`. Byte-compare the two readbacks directly (not just against a CPU
  ref) — the additive-default compat proof. **Exact (0 LSB).**
- **Row D — instanced LAMBERT, rigid instance rotations (±2 LSB).** N=2, two instances
  each rotated (rigid) + translated, LAMBERT + normals. Reference: §4.4 chain in host
  float. **±2 RGB LSB, alpha exact** (the sole tolerance row, matching v1.2 row 6).
- **Row E — gate refusals leave pixels untouched.** Draw a known good frame, then
  attempt each of: (i) misaligned `instance_offset` (G4); (ii) `N*64` overflow past the
  imported allocation (G5); (iii) `instance_alloc!=0` with `instance_count==0` (G2);
  (iv) a **released** instance alloc (G7); (v) `instance_attr` present with
  `instance_alloc==0` (G8); (vi) `instance_attr` without a valid colormap (G12);
  (vii) a **sheared** instance matrix on a LAMBERT-instanced draw (G14, §5.1 — shear
  well above `kGeomRigidTol`, e.g. `m[4]=0.1`). Each returns `false` and the readback
  equals the last-good frame byte-for-byte (mirror the refusal-purity structure
  `gfx_main.cpp:1785-1912` / `4030-4159`).

---

## 9. TwinScope fleet — the run-proof (§2.6)

Replace the split-view stand-in with the real 50-variant fleet. The applet **already
batches all 50 variants** — `ThermalSim` holds `sim.T` shape `(B=50, V_sim)` and
`sim.active` `(B,K)` (`twin_scope.cpp:297,466-468`; `kVariants=50` at `:67`, variant 0
is the hero) — so this is a **draw path, not new physics**. The split stand-in to remove
is the `MODE_SPLIT` two-half `specs` construction at `twin_scope.cpp:778-784` and its
two-half draw loops at `:871-916`.

- **Per-variant scalars (new reduction, worker-side):** the worker currently reduces the
  peak only for the hero variant (`cpu3.select(0,0).max()`, `:483`). Add a batched
  reduction `peakT = sim.T.amax(/*dim=*/1)` → `(50,)` and, when the error field is the
  selected tint, `peakErr = (sim_field - net_field).abs()` reduced per variant → `(50,)`.
  A UI toggle (peak-T vs peak-|error|) selects which `(50,)` tensor is published as the
  instance attr — the "peak T or peak |error|" toggle the exemplar §1.4 names.
- **Pose tensor:** a pool-born `(50,16)` f32 column-major model-matrix tensor, grid
  layout with the hero at the front (exemplar §1.4). Static after init unless
  click-to-promote reorders it. Built once, imported via the ExportablePool →
  bridge-v1.2 `import_allocation` like the mesh streams (`twin_scope.cpp:311-321,806-816`).
- **One instanced draw:** the mesh (`positions`/`normals`/`uvs`/`indices`) drawn once as
  a `CaliperGeomDrawV1_3` with `instance_alloc`=the `(50,16)` pose, `instance_count`=50,
  `instance_attr_alloc`=the selected `(50,)` scalar, `colormap=MAGMA`, LAMBERT + OPAQUE +
  depth. `draw_primitives` through the new v1.3 entry.
- **Click-to-promote:** applet-side ray-cast against the fleet's pose tensor (flow_scope
  pattern, already used for the hero source pick at `twin_scope.cpp:787+`); the picked
  unit's variant index swaps into the hero slot (variant 0's pose ↔ picked pose, or a
  view-state promotion) — zero ABI, pure applet state.
- **Publish path inherits drain-before-publish — anchor it explicitly.** The fleet's pose
  and attr tensors are a NEW worker→frame publish path, and the memory-stability contract
  in `sdk/include/caliper/services/geometry_v1.h:66-90` (both halves: SPATIAL
  triple-buffer + TEMPORAL drain, the geometry ABI is **permanently the drain rung** — it
  carries no STREAM_ORDERED channel) **binds every** such publish (restated at
  `geometry_v1_1.h:96-103` for `draw_primitives`, and `GEOMETRY.md:200-203` /
  `ZEROCOPY.md:213-259` name R3's `(N,16)` pose + `(N,)` attr as inheriting it). The
  worker MUST `torch::cuda::synchronize()` / MPS-synchronize **before** flipping
  `ready_slot` for the pose/attr slots, exactly as it already does for the field textures
  (`twin_scope.cpp:493-507`). Static poses are drained once at publish; live per-variant
  attrs drain every publish. The instance attr `(50,)` tensor is triple-buffered in the
  slot pool like `vert_slot`/`tex_slot` (`:334-348`).
- **Honest ladder:** cap bit 3 absent → hero-only, fleet panel says "fleet needs
  instanced geometry (cap absent)" (exemplar §1.7). Zero-copy claimed only when the
  instanced draw actually drew (the per-draw provenance discipline `:920-927`).
- **Run-proof:** Vulkan/CUDA on this box — fleet renders, tints track live peak-T /
  peak-|error|, click-to-promote works, status line proves the imported path. Metal/MPS
  in the mac pass (do not claim here).

---

## 10. Task breakdown (subagent-driven execution)

Ordered; each task is one implementer dispatch (Opus), each ends on **full ctest green**
(the suite is 8/8 today) and an incremental commit, each reviewed before the next.
Verification is by artifacts only — subagent report text is data, not proof
(v1_1 execution-plan S2 discipline).

1. **T1 — ABI + SDK + host vend.** `geometry_v1_3.h` record + `static_assert`s; the
   `GeomRev` enum refactor (§3, replacing `bool v12`); `geom_caps` bit 3 +
   `supports_geometry_instanced()` default-false query; `kGeom13` vend +
   `geo_draw_primitives_v13` + `kIds`; `HostGeomDraw` instance fields + the §3.3
   tint-LUT resolution rule; `geom_draw_v1_3_defaults()`; the full `caliper::Geometry`
   wrapper surface (§3.3: `g13_`, `has_instanced()`, the v1_3 overload, the widening
   tiers). **Gate:** `test_abi` pins (§2.4, incl. the v1.3-only-host widening
   regression) green; existing suite green; no backend yet grants the cap, so no
   pixels change. Commit.
2. **T2 — host validator gate battery.** The instance block (G1-G12) in
   `geom_draw_primitives_impl` under `GeomRev::V1_3`; the `geom_draw_primitives_v1_3`
   wrapper. (G14's *execution* lands in T3/T4's backend re-gate — §5.1 placement.)
   **Gate:** new `test_tensor_bridge.cpp` cases (mirror the v1.2 textured-gate
   battery `:718-780`) — atomic refusal per gate, valid instanced draw resolves. Commit.
3. **T3 — Vulkan backend.** `PrimParams` → 192 (all three copies in lockstep, §6.3);
   bindings 8/9 + pool growth + the binding-4 tint-LUT predicate change (§6.2);
   `geom.vert` instance pull + tint + §4.4 normal; `vkCmdDraw` instance count; backend
   re-gate incl. G14 (§5.1). **Gate:** builds; gfx rows A-E present and green on this
   box (row E incl. the G14 refusal); v1/v1.1/v1.2 rows untouched;
   `primitives-imported` observed. Commit.
4. **T4 — Metal transcription.** MSL + host `PrimParams` growth, buffer indices 7/8,
   `instanceCount`, day-one gate parity (§7). **Gate:** transcription reviewed against T3
   line-by-line; the `.mm` is compiled-out on this box (`CALIPER_HAVE_METAL` unset), so
   the gate is review + "the Metal gfx rows are written and mirror the Vulkan block" —
   **no hardware claim.** Commit.
5. **T5 — TwinScope fleet.** Per-variant reduction + `(50,)` publish; `(50,16)` pose;
   the single instanced draw replacing the split stand-in; click-to-promote;
   drain-before-publish on the new path (§9). **Gate:** runs zero-copy on Vulkan/CUDA
   here, status line proves it, the fleet tints track live state; the smoke test
   (physics/logic only) stays green. Commit.
6. **T6 — docs closeout** (§11). Commit.

Between every task: `ctest` green, `git commit`, orchestrator review. macOS hardware
verification of T4/T5 is a **separate session** folded into the next mac pass.

---

## 11. Docs closeout (§2.7)

- **`GEOMETRY.md`** — flip R3 (`:618`) from DIRECTIONAL to SHIPPED (both platforms once
  the mac pass lands; Vulkan-proven at merge), and the §11 goal text
  ("R2+R3 together ARE the twin demo — complete").
- **`ROADMAP.md` §6** — tick the R3 / instanced-transforms / twin-flagship items on the
  same checkbox discipline (verified-by-artifact only; the Metal tick waits on the mac
  session).
- **`ZEROCOPY.md`** — add an instanced-geometry row per origin, mirroring the primitives
  rows at `:342-343` ("imported `(N,16)` poses + `(N,)` attr, one draw, 0 in-VRAM
  copies"); the drain-rung note at `:213-259` already anticipates R3 — confirm it reads
  correctly post-merge.
- **Whitepaper** (`docs/report/caliper-whitepaper.tex`) — figure candidate: the 50-unit
  fleet, each housing tinted by live peak-T, hero draped at texture resolution — the R2+R3
  twin in one frame.
- **Mac-pass addendum** — if any new Apple-guarded code lands (it does: the MSL instance
  pull, buffer 7/8 binds), add a §8-style hardware-pass addendum to
  `2026-07-10-metal-macos-v1_2-hardware-pass.md` (or a v1_3 sibling) listing the
  Metal-specific acceptance gates: MSL compiles, `instance_id` semantics correct, the
  gfx rows A-E byte-exact on Apple Silicon, TwinScope fleet on MPS with the drain branch
  exercised live.

---

## Tree observations (reported, not silently reconciled)

- **No contradiction** between the authority docs and the tree on the ABI shape: the
  `bool v12` single axis is exactly at `tensor_bridge.cpp:590-596` / `tensor_bridge.h:
  151-159` as remaining-work §2.1 describes; the v1.2 `base`-member record-growth
  mechanism is at `geometry_v1_2.h:20-25`; the drain-rung inheritance for R3 is
  pre-declared at `geometry_v1.h:66-90`, `geometry_v1_1.h:96-103`, `GEOMETRY.md:200-203`.
- **Gap the fleet task must fill (not a contradiction):** the exemplar §1.4 says the
  fleet tints each unit by "peak T or peak |error|," but the applet today computes the
  peak only for the **hero** variant (`twin_scope.cpp:483`, `cpu3.select(0,0).max()`) —
  the batched per-variant `(50,)` reduction does not yet exist. §9/T5 add it; the sim
  already produces the batched `(50, V_sim)` state that makes it a reduction, not new
  physics.
- **Minor register drift (report-only):** `GEOMETRY.md:618` still labels R3
  "DIRECTIONAL" while remaining-work §0 shows R3 as "▢ Not started" — same status, two
  words; §11 reconciles at merge.

---

## Invariants (hold forever, never relitigate — restated from ROADMAP.md / GEOMETRY.md)

- Data flows **tensors → pixels → ImGui**, one way. No render-to-tensor, ever.
- No applet-supplied shaders; appearance is the fixed menu. Instancing is a
  uniform-driven branch on the one geom pipeline pair, not an applet shader.
- Honest ladders: a missing cap degrades to a working slow path and says so; never a
  wrong image, never a false status line. Zero-copy claimed only when the path drew.
- Increments ship against a demonstrated applet need (TwinScope's fleet), byte-exact-
  verified on both ecosystems, as prefix-identical additive revisions. `reserved0` stays
  NULL; no new entry points; the frozen 192-byte v1.1 prefix and 216-byte v1.2 record are
  never touched.

---

## Hardware addendum (Mac pass, 2026-07-11)

This section records what execution flipped versus the body above; the body is left
intact as written (Vulkan-first framing and all). It does **not** rewrite the spec —
it reconciles it with what hardware proved.

**(a) Platform order inverted (user-directed).** The body assumes a Vulkan-first,
Windows/NVIDIA run-proof with Metal transcribed (§7). This session ran the opposite,
by user direction: **Metal-first on Apple Silicon, run-proven**, with **Vulkan the
transcription**. The Vulkan hardware pass is deferred to a Windows session and has its
own execution spec — `docs/superpowers/specs/2026-07-11-geometry-v1_3-vulkan-windows-hardware-pass.md`
(committed `c50fa2c`; do not edit it). Every §7-style "NEVER claim verified" caution
therefore now applies to **Vulkan**, not Metal. Run-proven on Metal: gfx **49/49** live
incl. rows A–E plus a dedicated private-storage G14 row; TwinScope's 50-housing fleet
live zero-copy on MPS with per-draw provenance; the fleet tint window fixed at
`e71fce3`.

**(b) §5.1's `contents()`-direct read was REVERSED by hardware.** The body (§5.1, end)
states: "On Metal, shared-storage imported buffers may be read via `contents()`
directly (no blit)." Hardware refuted this. torch MPS tensors import as
`MTLStorageModePrivate`, so `contents()` is not host-addressable for the G14 rigidity
read. Metal G14 was therefore reworked to the **same collect → one-staged-copy →
compare** shape the Vulkan re-gate uses: a single staging blit of the LAMBERT-instanced
instance ranges, one fence, then the identical float-order comparison (`5403d92`,
`5aa2f73`). The correct §5.1 reading for Metal is now "private-storage imported buffers
are read via a staged blit, same as Vulkan," and this is **pinned by a dedicated
private-storage gfx row** so a future `contents()`-direct regression is caught byte-exact
rather than silently re-broken. The refusal-purity and atomicity claims are unchanged —
the copy still touches only staging, never the view.

**(c) Carry-forwards (honest, future-pass).**
- `inst_attr_base` lacks a G6-equivalent u32 base guard. The instance-**matrix** base is
  guarded (G6, both backends); the instance-**attr** base rides the same `PrimParams`
  u32 slot without its own explicit `offset/4 <= UINT32_MAX` gate. It is **host-level and
  bounded by G10** (the attr bounds gate) in practice, so no shipped path can overflow it
  today; a dedicated `inst_attr_base` u32 guard mirroring G6 is a clean future-pass
  addition on **both** backends. Noted, not a blocker.
- G5/G10 refusal lines ride `range_ok`'s own sink without the `geom_prims: draw %u
  refused:` draw-index prefix that the other instance gates carry. Adjudicated **correct**
  (intentional — `range_ok` is the shared bounds sink and its message names the stream),
  recorded here so a reviewer does not re-flag it as an inconsistency.

**(d) §9 fleet tint needed a peak-window correction on hardware.** A max-statistic on the
field window saturated every variant to `MAGMA[255]` (all housings pinned to the LUT
top), because the peak reduction fed the LUT `vmax` directly and the window's max sat at
the ceiling. Fixed with **spread headroom** on the tint window so the 50 variants span
the LUT instead of collapsing to its top entry (`e71fce3`). The §4.3 index rule and the
byte-exact tint row (§8 row B) are unchanged; only the applet-side `vmin`/`vmax` window
feeding the reduction moved.

**(e) Exemplar revision — the §9 fleet was reverted; `instance_scope` is R3's shipped
exemplar.** The §9 TwinScope fleet was implemented and run-proven zero-copy on MPS
(50 housings, ONE instanced draw, live per-variant tint — `40697f1`), then **reverted**
on user adjudication (`85698f6`). The §9 fleet UX buried the rung's value: a single
per-housing scalar tint replaced the hero's per-vertex detail in the old split view, and
even with the (d) spread-headroom fix the unbounded max-statistic saturates any fixed
window — instancing's point (per-vertex detail across N objects, one draw) was lost, not
shown. R3's shipped exemplar is instead **`instance_scope`** (`bfe6da9`) — a dedicated
applet drawing N gems (slider 1–5000, default 1000) in ONE instanced draw from live
device-tensor poses + tints, run-proven on MPS with the log line "first zero-copy
instanced frame drawn — 1000 objects, 1 draw call, 0 mesh copies". TwinScope reverts to
the R2 surface twin (hero-only). What remains in-tree from §9 is applet-independent: the
per-variant reduction machinery (`predict_batch` + its tests) stays, unused by any
shipped applet, available to a future population-twin. The byte-exact gfx rows A–E (49/49)
were never part of the fleet and remain R3's correctness proof, untouched by the swap;
the v1_3 ABI/backends are likewise untouched.
