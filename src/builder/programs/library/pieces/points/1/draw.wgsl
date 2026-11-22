// The draw half of points@1, as the built-in points primitive draws: each point is a quad of two
// triangles around its projected center, `size` pixels wide at the display's pixel ratio, a disc
// when `shape` is 1, with depth and the point's ID. A colormap color is the host colormap's
// texel of the level that the host's reference gives a value, (v - vmin) / (vmax - vmin) * 255
// rounded, with NaN at level 0. A point with a NaN coordinate is not drawn.

struct PointOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) item: u32,
}

fn is_nan(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

// A packed color, r | g << 8 | b << 16 | a << 24, as four channels from 0 to 1.
fn unpack(c: u32) -> vec4<f32> {
    return vec4<f32>(f32(c & 0xffu), f32((c >> 8u) & 0xffu), f32((c >> 16u) & 0xffu), f32(c >> 24u)) / 255.0;
}

// The colormap level of a value, 0 to 255, as the host's reference computes it.
fn level(v: f32) -> u32 {
    if (is_nan(v)) {
        return 0u;
    }
    let span = params.vmax - params.vmin;
    var t = 0.0;
    if (span != 0.0) {
        t = (v - params.vmin) / span;
    }
    if (!(t >= 0.0)) {
        t = 0.0;
    } else if (t > 1.0) {
        t = 1.0;
    }
    return min(u32(t * 255.0 + 0.5), 255u);
}

fn point_color(bits: u32) -> vec4<f32> {
    if (params.color_mode == 1u) {
        let size = textureDimensions(caliper_colormaps);
        let x = min(level(bitcast<f32>(bits)) * (size.x - 1u) / 255u, size.x - 1u);
        return textureLoad(caliper_colormaps, vec2<u32>(x, min(params.colormap, size.y - 1u)), 0);
    }
    if (params.color_mode == 2u) {
        return unpack(bits);
    }
    return unpack(params.color);
}

@vertex
fn point_vs(@builtin(instance_index) i: u32, @builtin(vertex_index) v: u32) -> PointOut {
    var corners = array<vec2<f32>, 6>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
                                      vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    let corner = corners[min(v, 5u)];
    let p = points[i];
    var out: PointOut;
    out.uv = corner;
    out.item = i;
    out.color = point_color(p.bits);
    if (is_nan(p.position.x) || is_nan(p.position.y) || is_nan(p.position.z)) {
        out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return out;
    }
    let c = camera::clip(p.position.xyz);
    let size = params.size * caliper_view.time.z;
    let k = (size / caliper_view.viewport.zw) * c.w;
    out.position = vec4<f32>(c.xy + corner * k, c.z, c.w);
    return out;
}

@fragment
fn point_fs(in: PointOut) -> CaliperPicked {
    if (params.shape == 1u && dot(in.uv, in.uv) > 1.0) {
        discard;
    }
    var out: CaliperPicked;
    out.color = vec4<f32>(in.color.rgb * in.color.a, in.color.a);
    out.id = pick::id(in.item);
    return out;
}
