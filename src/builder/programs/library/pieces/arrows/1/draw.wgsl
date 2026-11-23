// The draw half of arrows@1, as the built-in arrows primitive draws: each arrow is a shaft and a
// flat head in screen space, nine vertices. After a clip of the arrow at the near plane, the head
// is a triangle a fifth of the arrow's length on the screen, at least three widths wide, and the
// shaft is a quad `width` pixels wide at the display's pixel ratio from the origin to the head's
// base, whose ends extend by half the width. The arrow has the color of its source. A colormap
// color is the host colormap's texel of the level that the host's reference gives a value,
// (v - vmin) / (vmax - vmin) * 255 rounded, with NaN at level 0; an end of the range from the
// values is their minimum or maximum, a range from the values that is not finite is 0 to 1, and
// an empty one widens by 0.5 on each side. An arrow with a NaN coordinate at its origin or its tip
// is not drawn. An arrow of one color with an alpha below 255, or of packed RGBA when any has such
// an alpha, draws in the second pass, which tests depth without writing it.

struct ArrowOut {
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

fn is_finite(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}

// A position that the clipper removes.
fn hidden() -> vec4<f32> {
    return vec4<f32>(0.0, 0.0, 2.0, 1.0);
}

// A packed color, r | g << 8 | b << 16 | a << 24, as four channels from 0 to 1.
fn unpack(c: u32) -> vec4<f32> {
    return vec4<f32>(f32(c & 0xffu), f32((c >> 8u) & 0xffu), f32((c >> 16u) & 0xffu), f32(c >> 24u)) / 255.0;
}

// The colormap's range: vmin and vmax, with the ends that `range` names from the values.
fn value_range() -> vec2<f32> {
    var r = vec2<f32>(params.vmin, params.vmax);
    if (params.range == 0u) {
        return r;
    }
    let known = counts.info > 0u && (info[0].flags & 1u) == 0u;
    var finite = known;
    if ((params.range & 1u) != 0u) {
        r.x = info[0].lo;
    }
    if ((params.range & 2u) != 0u) {
        r.y = info[0].hi;
    }
    finite = finite && is_finite(r.x) && is_finite(r.y);
    if (!finite) {
        return vec2<f32>(0.0, 1.0);
    }
    if (r.x == r.y) {
        return vec2<f32>(r.x - 0.5, r.y + 0.5);
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

fn end_color(bits: u32, r: vec2<f32>) -> vec4<f32> {
    if (params.color_mode == 1u) {
        let size = textureDimensions(caliper_colormaps);
        let x = min(level(bitcast<f32>(bits), r) * (size.x - 1u) / 255u, size.x - 1u);
        return textureLoad(caliper_colormaps, vec2<u32>(x, min(params.colormap, size.y - 1u)), 0);
    }
    if (params.color_mode == 2u) {
        return unpack(bits);
    }
    return unpack(params.color);
}

fn translucent() -> bool {
    if (params.color_mode == 0u) {
        return (params.color >> 24u) < 255u;
    }
    if (params.color_mode == 2u) {
        return counts.info > 0u && (info[0].flags & 2u) != 0u;
    }
    return false;
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

fn arrow_vertex(i: u32, vid: u32, want_translucent: bool) -> ArrowOut {
    let s = arrows[i];
    var out: ArrowOut;
    out.item = i;
    out.color = end_color(s.color, value_range());
    let origin = vec3<f32>(s.ox, s.oy, s.oz);
    let tip = origin + vec3<f32>(s.vx, s.vy, s.vz);
    if (translucent() != want_translucent || s.index == 0xffffffffu || any_nan3(origin) || any_nan3(tip)) {
        out.position = hidden();
        return out;
    }
    var a = clip_of(origin);
    var b = clip_of(tip);
    if (!clip_near(&a, &b)) {
        out.position = hidden();
        return out;
    }
    let half_vp = 0.5 * caliper_view.viewport.zw;
    let sa = a.xy / a.w * half_vp;
    let sb = b.xy / b.w * half_vp;
    let len = length(sb - sa);
    let dir = select(vec2<f32>(1.0, 0.0), (sb - sa) / len, len > 1e-6);
    let w = max(params.width * caliper_view.time.z, 1.0);
    let head = 0.2 * len;
    let base = sb - dir * head;
    let v = min(vid, 8u);
    if (v < 6u) {
        out.position = segment_corner(a, vec4<f32>(base / half_vp * b.w, b.z, b.w), v, w);
        return out;
    }
    let side = vec2<f32>(-dir.y, dir.x) * max(1.5 * w, 0.4 * head);
    var p = sb;
    if (v == 7u) {
        p = base + side;
    }
    if (v == 8u) {
        p = base - side;
    }
    out.position = vec4<f32>(p / half_vp * b.w, b.z, b.w);
    return out;
}

@vertex
fn opaque_vs(@builtin(instance_index) i: u32, @builtin(vertex_index) v: u32) -> ArrowOut {
    return arrow_vertex(i, v, false);
}

@vertex
fn translucent_vs(@builtin(instance_index) i: u32, @builtin(vertex_index) v: u32) -> ArrowOut {
    return arrow_vertex(i, v, true);
}

@fragment
fn arrow_fs(in: ArrowOut) -> CaliperPicked {
    var out: CaliperPicked;
    out.color = in.color;
    out.id = pick::id(in.item);
    return out;
}
