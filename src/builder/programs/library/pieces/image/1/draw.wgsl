// The draw half of image@1, as the built-in image draws: one quad over the image's pixels, flat
// at the pose with one world unit a pixel and row 0 at y 0, or on a camera's image plane, where a
// pixel (u, v) lies at ((u - cx) d / fx, (v - cy) d / fy, d) in the camera's axes before the pose.
// The color is the texture `pixels` sampled with the hardware's bilinear filter and clamp to the
// edge, as the built-in samples its texture, times a tint of the opacity in 1/255 steps, blended
// over the panel. The ID of a fragment is the index of its nearest pixel, row * width + column,
// which the built-in pick gives as the event's index.

struct ImageOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Whether a float is finite, decided on its bits.
fn is_finite(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}

// The world position of a point of the image in pixels, x to the right and y down from the top
// left corner of the image.
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

// The opacity as the built-in's tint holds it, in steps of 1/255.
fn opacity() -> f32 {
    return floor(clamp(params.opacity, 0.0, 1.0) * 255.0 + 0.5) / 255.0;
}

@vertex
fn image_vs(@builtin(vertex_index) v: u32) -> ImageOut {
    var corners = array<vec2<f32>, 6>(vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
                                      vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0));
    var out: ImageOut;
    out.uv = corners[min(v, 5u)];
    let size = info[0].size;
    out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    let tex = textureDimensions(pixels);
    if (size.x == 0u || size.y == 0u || tex.x != size.x || tex.y != size.y || counts.pixels < size.x * size.y) {
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
fn image_fs(in: ImageOut) -> CaliperPicked {
    let size = textureDimensions(pixels);
    let s = textureSample(pixels, caliper_linear, in.uv);
    let p = min(vec2<u32>(max(in.uv, vec2<f32>(0.0)) * vec2<f32>(size)), size - vec2<u32>(1u));
    var out: CaliperPicked;
    out.color = s * vec4<f32>(1.0, 1.0, 1.0, opacity());
    out.id = pick::id(p.y * size.x + p.x);
    return out;
}
