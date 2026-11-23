// The draw half of segmentation@1, as the built-in segmentation image draws: one quad over the
// pixels, flat at the pose with one world unit a pixel, or on a camera's image plane, where a pixel
// (u, v) lies at ((u - cx) d / fx, (v - cy) d / fy, d) in the camera's axes before the pose. Each
// fragment takes the class of its nearest pixel in the color of that class, from `palette` below
// 65,536, with the alpha times the opacity in 1/255 steps,
// blended over the panel without a depth write. The ID is the index of the pixel,
// row * width + column.

struct MaskOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// A packed color, r | g << 8 | b << 16 | a << 24, as four channels from 0 to 1.
fn unpack(c: u32) -> vec4<f32> {
    return vec4<f32>(f32(c & 0xffu), f32((c >> 8u) & 0xffu), f32((c >> 16u) & 0xffu), f32(c >> 24u)) / 255.0;
}

// Whether a float is finite, decided on its bits.
fn is_finite(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}

// The world position of a point of the image in pixels, x to the right and y down from its top
// left corner.
fn place(px: vec2<f32>) -> vec4<f32> {
    let i = info[0];
    let pose = mat4x4<f32>(i.c0, i.c1, i.c2, i.c3);
    if (params.placement == 1u) {
        let d = params.plane;
        var q = vec3<f32>((px.x - params.cx) * d / params.fx, (px.y - params.cy) * d / params.fy, d);
        if (params.convention == 1u) {
            q = vec3<f32>(q.x, -q.y, -q.z);
        }
        return pose * vec4<f32>(q, 1.0);
    }
    return pose * vec4<f32>(px, 0.0, 1.0);
}

// The color of a class.
fn class_color(id: u32) -> vec4<f32> {
    if (id < counts.palette) {
        return unpack(palette[id].rgba);
    }
    return unpack(palette_color(id));
}

// The opacity as the built-in's tint holds it, in steps of 1/255.
fn opacity() -> f32 {
    return floor(clamp(params.opacity, 0.0, 1.0) * 255.0 + 0.5) / 255.0;
}

@vertex
fn mask_vs(@builtin(vertex_index) v: u32) -> MaskOut {
    var corners = array<vec2<f32>, 6>(vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
                                      vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0));
    var out: MaskOut;
    out.uv = corners[min(v, 5u)];
    out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    let size = info[0].size;
    if (size.x == 0u || size.y == 0u || counts.pixels < size.x * size.y) {
        return out;
    }
    let world = place(out.uv * vec2<f32>(size.xy));
    if (!is_finite(world.x) || !is_finite(world.y) || !is_finite(world.z) || !is_finite(world.w)) {
        return out;
    }
    out.position = caliper_view.clip_from_world * world;
    return out;
}

@fragment
fn mask_fs(in: MaskOut) -> CaliperPicked {
    let size = info[0].size.xy;
    let p = min(vec2<u32>(max(in.uv, vec2<f32>(0.0)) * vec2<f32>(size)), size - vec2<u32>(1u));
    let item = p.y * size.x + p.x;
    let c = class_color(textureLoad(pixels, p, 0).r);
    var out: CaliperPicked;
    out.color = vec4<f32>(c.rgb, c.a * opacity());
    out.id = pick::id(item);
    return out;
}
