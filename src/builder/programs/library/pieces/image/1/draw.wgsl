// The draw half of image@1, as the built-in image draws: one quad over the image's pixels, flat
// at the pose with one world unit a pixel and row 0 at y 0, or on a camera's image plane, where a
// pixel (u, v) lies at ((u - cx) d / fx, (v - cy) d / fy, d) in the camera's axes before the pose.
// The color is the bilinear sample of the pixels with clamp to the edge, its alpha times the
// opacity in 1/255 steps, blended over the panel. The ID of a fragment is the index of its
// nearest pixel, row * width + column, which the built-in pick gives as the event's index.

struct ImageOut {
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

fn texel(x: i32, y: i32, w: i32, h: i32) -> vec4<f32> {
    let c = vec2<u32>(vec2<i32>(clamp(x, 0, w - 1), clamp(y, 0, h - 1)));
    return unpack(pixels[c.y * u32(w) + c.x].rgba);
}

@vertex
fn image_vs(@builtin(vertex_index) v: u32) -> ImageOut {
    var corners = array<vec2<f32>, 6>(vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
                                      vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0));
    var out: ImageOut;
    out.uv = corners[min(v, 5u)];
    let size = info[0].size;
    out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
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
fn image_fs(in: ImageOut) -> CaliperPicked {
    let size = info[0].size.xy;
    let w = i32(size.x);
    let h = i32(size.y);
    let t = in.uv * vec2<f32>(size) - 0.5;
    let f = floor(t);
    let k = t - f;
    let x = i32(f.x);
    let y = i32(f.y);
    let top = mix(texel(x, y, w, h), texel(x + 1, y, w, h), k.x);
    let bottom = mix(texel(x, y + 1, w, h), texel(x + 1, y + 1, w, h), k.x);
    let s = mix(top, bottom, k.y);
    let p = min(vec2<u32>(max(in.uv, vec2<f32>(0.0)) * vec2<f32>(size)), size - vec2<u32>(1u));
    var out: CaliperPicked;
    out.color = vec4<f32>(s.rgb, s.a * opacity());
    out.id = pick::id(p.y * size.x + p.x);
    return out;
}
