// The draw half of path@1, as a lines entity in strip mode draws the path operator's output: one
// strip through the kept rows in order, each join of vertex i and i + 1 a quad of two triangles,
// `width` pixels wide at the display's pixel ratio, whose ends extend by half the width, after a
// clip of the join at the near plane. The color of each end is its row's value through the host
// colormap, the texel of the level that the host's reference gives it, (v - vmin) / (vmax - vmin)
// * 255 rounded, and it blends along the join; an end of the range from the rows is the first or
// the last kept row, and an empty range widens by 0.5 on each side, as the entity's range does. A
// join with a NaN coordinate is not drawn. One color with an alpha below 255 draws in the second
// pass, which tests depth without writing it.

struct PathOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) item: u32,
}

fn is_nan(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

fn any_nan3(v: vec3<f32>) -> bool {
    return is_nan(v.x) || is_nan(v.y) || is_nan(v.z);
}

// A position that the clipper removes.
fn hidden() -> vec4<f32> {
    return vec4<f32>(0.0, 0.0, 2.0, 1.0);
}

// A packed color, r | g << 8 | b << 16 | a << 24, as four channels from 0 to 1.
fn unpack(c: u32) -> vec4<f32> {
    return vec4<f32>(f32(c & 0xffu), f32((c >> 8u) & 0xffu), f32((c >> 16u) & 0xffu), f32(c >> 24u)) / 255.0;
}

// The colormap's range: vmin and vmax, with the ends that `range` names from the values, which
// rise with the rows: the first vertex's and the last's. An empty range widens by 0.5 on each side.
fn value_range() -> vec2<f32> {
    var r = vec2<f32>(params.vmin, params.vmax);
    if (params.range == 0u || counts.vertices == 0u) {
        return r;
    }
    var lo = vertices[0].value;
    var hi = vertices[counts.vertices - 1u].value;
    if (lo == hi) {
        lo = lo - 0.5;
        hi = hi + 0.5;
    }
    if ((params.range & 1u) != 0u) {
        r.x = lo;
    }
    if ((params.range & 2u) != 0u) {
        r.y = hi;
    }
    return r;
}

// The colormap level of a value, 0 to 255, as the host's reference computes it.
fn level(v: f32, r: vec2<f32>) -> u32 {
    if (is_nan(v)) {
        return 0u;
    }
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
    return min(u32(t * 255.0 + 0.5), 255u);
}

fn end_color(value: f32, r: vec2<f32>) -> vec4<f32> {
    if (params.color_mode == 1u) {
        let size = textureDimensions(caliper_colormaps);
        let x = min(level(value, r) * (size.x - 1u) / 255u, size.x - 1u);
        return textureLoad(caliper_colormaps, vec2<u32>(x, min(params.colormap, size.y - 1u)), 0);
    }
    return unpack(params.color);
}

fn translucent() -> bool {
    return params.color_mode != 1u && (params.color >> 24u) < 255u;
}

// The clip position of a world position, moved toward the camera by the depth offset.
fn clip_of(p: vec3<f32>) -> vec4<f32> {
    var c = camera::clip(p);
    c.z = c.z - f32(params.depth_offset) * (1.0 / 8192.0) * c.w;
    return c;
}

// Corner c of a quad of two triangles, as (end 0 or 1, side -1 or 1).
fn quad_corner(c: u32) -> vec2<f32> {
    var t = array<vec2<f32>, 6>(vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
                                vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    return t[min(c, 5u)];
}

// Clips a segment to the near plane (clip z >= 0); false when all of it is behind.
fn clip_near(a: ptr<function, vec4<f32>>, b: ptr<function, vec4<f32>>) -> bool {
    let za = (*a).z;
    let zb = (*b).z;
    if (za < 0.0 && zb < 0.0) {
        return false;
    }
    if (za < 0.0) {
        *a = mix(*a, *b, za / (za - zb));
    }
    if (zb < 0.0) {
        *b = mix(*b, *a, zb / (zb - za));
    }
    return true;
}

// Corner c of segment (a, b) in clip space, as a quad width_px wide whose ends extend by half
// the width.
fn segment_corner(a_in: vec4<f32>, b_in: vec4<f32>, c: u32, width_px: f32) -> vec4<f32> {
    var a = a_in;
    var b = b_in;
    if (!clip_near(&a, &b)) {
        return hidden();
    }
    let half_vp = 0.5 * caliper_view.viewport.zw;
    let sa = a.xy / a.w * half_vp;
    let sb = b.xy / b.w * half_vp;
    let len = length(sb - sa);
    let dir = select(vec2<f32>(1.0, 0.0), (sb - sa) / len, len > 1e-6);
    let q = quad_corner(c);
    let w = max(width_px, 1.0) * 0.5;
    let off = vec2<f32>(-dir.y, dir.x) * (q.y * w) + dir * select(-w, w, q.x > 0.5);
    let e = select(a, b, q.x > 0.5);
    return vec4<f32>(e.xy + off / half_vp * e.w, e.z, e.w);
}

fn path_vertex(i: u32, v: u32, want_translucent: bool) -> PathOut {
    let c = v % 6u;
    var out: PathOut;
    out.item = i;
    if (i + 1u >= counts.vertices || translucent() != want_translucent) {
        out.position = hidden();
        return out;
    }
    let s = vertices[i];
    let t = vertices[i + 1u];
    let at_b = quad_corner(c).x > 0.5;
    out.color = end_color(select(s.value, t.value, at_b), value_range());
    let a = vec3<f32>(s.x, s.y, s.z);
    let b = vec3<f32>(t.x, t.y, t.z);
    if (any_nan3(a) || any_nan3(b)) {
        out.position = hidden();
        return out;
    }
    out.position = segment_corner(clip_of(a), clip_of(b), c, params.width * caliper_view.time.z);
    return out;
}

@vertex
fn opaque_vs(@builtin(instance_index) i: u32, @builtin(vertex_index) v: u32) -> PathOut {
    return path_vertex(i, v, false);
}

@vertex
fn translucent_vs(@builtin(instance_index) i: u32, @builtin(vertex_index) v: u32) -> PathOut {
    return path_vertex(i, v, true);
}

@fragment
fn path_fs(in: PathOut) -> CaliperPicked {
    var out: CaliperPicked;
    out.color = in.color;
    out.id = pick::id(in.item);
    return out;
}
