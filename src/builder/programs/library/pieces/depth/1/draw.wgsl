// The draw half of depth@1, as the built-in depth image draws. The first pass draws one quad over
// the pixels, flat at the pose with one world unit a pixel, or on a camera's image plane, where a
// pixel (u, v) lies at ((u - cx) d / fx, (v - cy) d / fy, d) in the camera's axes before the pose.
// Each fragment takes its nearest pixel through the colormap, with the alpha times the opacity in
// 1/255 steps. The colormap level of a depth is the host reference's,
// round((v - vmin) / (vmax - vmin) * 255) clamped, and NaN at level 0. A range end with its bit of
// `auto_range` comes from the data, and a range from the data with no depth is 0 to 1, and a flat one
// widens by 0.5 on each side. The second pass, on a camera with `cloud` 1, draws each depth that
// is finite and above 0 as a disc at its back-projected point, ((u + 0.5 - cx) z / fx,
// (v + 0.5 - cy) z / fy, z), of the built-in's size for the count of pixels. The ID of both passes
// is the index of the pixel, row * width + column.

struct DepthOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct CloudOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) corner: vec2<f32>,
    @location(2) @interpolate(flat) item: u32,
}

fn hidden() -> vec4<f32> {
    return vec4<f32>(0.0, 0.0, 2.0, 1.0);
}

// Whether a float is finite, decided on its bits.
fn is_finite(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}

fn all_finite(v: vec4<f32>) -> bool {
    return is_finite(v.x) && is_finite(v.y) && is_finite(v.z) && is_finite(v.w);
}

fn pose() -> mat4x4<f32> {
    let i = info[0];
    return mat4x4<f32>(i.c0, i.c1, i.c2, i.c3);
}

// A point in the camera's axes from its pixel and its depth.
fn camera_point(px: vec2<f32>, d: f32) -> vec4<f32> {
    var q = vec3<f32>((px.x - params.cx) * d / params.fx, (px.y - params.cy) * d / params.fy, d);
    if (params.convention == 1u) {
        q = vec3<f32>(q.x, -q.y, -q.z);
    }
    return vec4<f32>(q, 1.0);
}

// The world position of a point of the image in pixels, x to the right and y down from its top
// left corner.
fn place(px: vec2<f32>) -> vec4<f32> {
    if (params.placement == 1u) {
        return pose() * camera_point(px, params.plane);
    }
    return pose() * vec4<f32>(px, 0.0, 1.0);
}

// The colormap's range.
fn depth_range() -> vec2<f32> {
    var r = vec2<f32>(params.vmin, params.vmax);
    let data = info[0].range;
    let lo = (params.auto_range & 1u) != 0u;
    let hi = (params.auto_range & 2u) != 0u;
    if (lo) {
        r.x = data.x;
    }
    if (hi) {
        r.y = data.y;
    }
    if ((lo || hi) && data.z == 0.0) {
        return vec2<f32>(0.0, 1.0);
    }
    if ((lo || hi) && r.x == r.y) {
        return vec2<f32>(r.x - 0.5, r.y + 0.5);
    }
    return r;
}

// The color of a depth: the host colormap's texel at the level of the host's reference.
fn depth_color(v: f32) -> vec4<f32> {
    var level = 0u;
    if ((bitcast<u32>(v) & 0x7fffffffu) <= 0x7f800000u) {
        let r = depth_range();
        let span = r.y - r.x;
        var t = 0.0;
        if (span != 0.0) {
            t = (v - r.x) / span;
        }
        if (!(t >= 0.0)) {
            t = 0.0;
        } else if (t > 1.0) {
            t = 1.0;
        }
        level = min(u32(t * 255.0 + 0.5), 255u);
    }
    let size = textureDimensions(caliper_colormaps);
    let x = min(level * (size.x - 1u) / 255u, size.x - 1u);
    return textureLoad(caliper_colormaps, vec2<u32>(x, min(params.colormap, size.y - 1u)), 0);
}

// The opacity as the built-in's tint holds it, in steps of 1/255.
fn opacity() -> f32 {
    return floor(clamp(params.opacity, 0.0, 1.0) * 255.0 + 0.5) / 255.0;
}

@vertex
fn depth_vs(@builtin(vertex_index) v: u32) -> DepthOut {
    var corners = array<vec2<f32>, 6>(vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
                                      vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0));
    var out: DepthOut;
    out.uv = corners[min(v, 5u)];
    out.position = hidden();
    let size = info[0].size;
    if (size.x == 0u || size.y == 0u || counts.pixels < size.x * size.y) {
        return out;
    }
    let world = place(out.uv * vec2<f32>(size.xy));
    if (all_finite(world)) {
        out.position = caliper_view.clip_from_world * world;
    }
    return out;
}

@fragment
fn depth_fs(in: DepthOut) -> CaliperPicked {
    let size = info[0].size.xy;
    let p = min(vec2<u32>(max(in.uv, vec2<f32>(0.0)) * vec2<f32>(size)), size - vec2<u32>(1u));
    let item = p.y * size.x + p.x;
    let c = depth_color(pixels[item].depth);
    var out: CaliperPicked;
    out.color = vec4<f32>(c.rgb, c.a * opacity());
    out.id = pick::id(item);
    return out;
}

// The built-in's size of a point for a count of points, in pixels at a pixel ratio of 1.
fn point_size(n: u32) -> f32 {
    if (n > 50000u) {
        return 3.0;
    }
    if (n > 5000u) {
        return 4.0;
    }
    if (n > 500u) {
        return 5.5;
    }
    return 7.0;
}

@vertex
fn cloud_vs(@builtin(instance_index) i: u32, @builtin(vertex_index) v: u32) -> CloudOut {
    var corners = array<vec2<f32>, 6>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
                                      vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    var out: CloudOut;
    out.corner = corners[min(v, 5u)];
    out.item = i;
    out.position = hidden();
    if (params.cloud == 0u || params.placement != 1u) {
        return out;
    }
    let size = info[0].size;
    let n = size.x * size.y;
    if (size.x == 0u || i >= n || counts.pixels < n) {
        return out;
    }
    let bits = bitcast<u32>(pixels[i].depth);
    // Zero, negative, infinite and NaN depths stay out, decided on the bits.
    if ((bits >> 31u) != 0u || bits == 0u || (bits & 0x7fffffffu) >= 0x7f800000u) {
        return out;
    }
    let d = bitcast<f32>(bits);
    let px = vec2<f32>(f32(i % size.x), f32(i / size.x)) + 0.5;
    let world = pose() * camera_point(px, d);
    if (!all_finite(world)) {
        return out;
    }
    let c = caliper_view.clip_from_world * world;
    let k = (point_size(n) * caliper_view.time.z / caliper_view.viewport.zw) * c.w;
    out.position = vec4<f32>(c.xy + out.corner * k, c.z, c.w);
    out.color = depth_color(d);
    return out;
}

@fragment
fn cloud_fs(in: CloudOut) -> CaliperPicked {
    if (dot(in.corner, in.corner) > 1.0) {
        discard;
    }
    var out: CaliperPicked;
    out.color = in.color;
    out.id = pick::id(in.item);
    return out;
}
