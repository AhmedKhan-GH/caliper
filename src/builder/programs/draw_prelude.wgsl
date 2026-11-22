// Caliper draw prelude, version 1 (design/programmable-graphics.md, Prelude).
// The host puts this text first in every pass of a program's draw half, in
// the app and in the browser module, and binds group 0. Every name that
// starts with caliper_, Caliper or CALIPER_ belongs to the host.

// The view of one frame. Matrices map world positions to clip space as the
// panel's camera does; with the camera mode none they are identities.
struct CaliperView {
    clip_from_world: mat4x4<f32>,
    world_from_clip: mat4x4<f32>,
    view_from_world: mat4x4<f32>,
    // The eye in world space; w is 1 for a perspective camera, else 0.
    eye: vec4<f32>,
    // The host target: x and y of its origin (0), width and height in pixels.
    viewport: vec4<f32>,
    // The pointer in pixels from the top left, its buttons as bits, and 1
    // when it is over the panel, else 0.
    pointer: vec4<f32>,
    // Seconds since the install, seconds since the previous frame, the
    // pixel ratio of the display, and 0.
    time: vec4<f32>,
    // The cursor's step, the newest step, the step of the resources that
    // this frame shows, and 1 when the cursor is live, else 0.
    step: vec4<u32>,
    // Frames since the draw buffers' last reset, 1 on the first frame after
    // a reset, the program's pick id, and the count of caliper_selection.
    frame: vec4<u32>,
}

// One entry of the linked selection: the pick id of the program that wrote
// it, the item, flags (bit 0: the newest pick), and the tensor indices of
// the item map, -1 where the map has fewer.
struct CaliperSelected {
    pick: u32,
    item: u32,
    flags: u32,
    pad: u32,
    index: vec4<i32>,
}

// The output of caliper_fullscreen: uv is (0, 0) at the top left and (1, 1)
// at the bottom right of the target.
struct CaliperFullscreen {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// The fragment output of a pass with pick = true.
struct CaliperPicked {
    @location(0) color: vec4<f32>,
    @location(1) id: vec2<u32>,
}

@group(0) @binding(0) var<uniform> caliper_view: CaliperView;
@group(0) @binding(1) var<storage, read> caliper_selection: array<CaliperSelected>;
@group(0) @binding(2) var caliper_linear: sampler;
@group(0) @binding(3) var caliper_nearest: sampler;
@group(0) @binding(4) var caliper_colormaps: texture_2d<f32>;

// The rows of caliper_colormaps.
const CALIPER_VIRIDIS: u32 = 0u;
const CALIPER_MAGMA: u32 = 1u;
const CALIPER_RDBU: u32 = 2u;
const CALIPER_GREYS: u32 = 3u;
const CALIPER_MAX_SELECTED: u32 = 256u;

// The ID that a pass with pick = true writes at location 1.
fn caliper_id(item: u32) -> vec2<u32> {
    return vec2<u32>(caliper_view.frame.z, item);
}

// The clip position of a world position.
fn caliper_clip(world: vec3<f32>) -> vec4<f32> {
    return caliper_view.clip_from_world * vec4<f32>(world, 1.0);
}

// The color of t, from 0 to 1, on a host colormap, linear between its texels.
fn caliper_colormap(map: u32, t: f32) -> vec4<f32> {
    let size = textureDimensions(caliper_colormaps);
    let row = i32(min(map, size.y - 1u));
    let x = clamp(t, 0.0, 1.0) * f32(size.x - 1u);
    let x0 = i32(floor(x));
    let x1 = min(x0 + 1, i32(size.x) - 1);
    let a = textureLoad(caliper_colormaps, vec2<i32>(x0, row), 0);
    let b = textureLoad(caliper_colormaps, vec2<i32>(x1, row), 0);
    return mix(a, b, x - f32(x0));
}

// The count of valid entries of caliper_selection.
fn caliper_selection_count() -> u32 {
    return min(caliper_view.frame.w, min(arrayLength(&caliper_selection), CALIPER_MAX_SELECTED));
}

// Whether an item of this program is in the linked selection.
fn caliper_selected(item: u32) -> bool {
    let n = caliper_selection_count();
    for (var i = 0u; i < n; i = i + 1u) {
        let s = caliper_selection[i];
        if (s.pick == caliper_view.frame.z && s.item == item) {
            return true;
        }
    }
    return false;
}

// Whether a selected item of any program has these tensor indices; an index
// below 0 matches every index.
fn caliper_selected_index(index: vec4<i32>) -> bool {
    let n = caliper_selection_count();
    for (var i = 0u; i < n; i = i + 1u) {
        let s = caliper_selection[i].index;
        if (all((index == s) | (index < vec4<i32>(0)))) {
            return true;
        }
    }
    return false;
}

// The vertex entry of a render pass without `vertex`: one triangle that
// covers the target, drawn with 3 vertices.
@vertex
fn caliper_fullscreen(@builtin(vertex_index) index: u32) -> CaliperFullscreen {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: CaliperFullscreen;
    out.position = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    out.uv = uv;
    return out;
}
