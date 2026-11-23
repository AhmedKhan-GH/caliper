// The draw half of frustum@1, as the built-in draws a camera entity's frustum with its lines
// program: eight segments of the unit frustum, from its centre (0, 0, 0) to the corners (s, t, 1)
// of the unit square and then the square's four sides, mapped by the matrix of `frustum`, each a
// quad `width` pixels wide at the display's pixel ratio whose ends extend by half the width, after
// a clip of the segment at the near plane, in one color, with the depth offset. Every segment
// gives the ID of element 0, the camera.

struct FrustumOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

fn hidden() -> vec4<f32> {
    return vec4<f32>(0.0, 0.0, 2.0, 1.0);
}

fn unpack(c: u32) -> vec4<f32> {
    return vec4<f32>(f32(c & 0xffu), f32((c >> 8u) & 0xffu), f32((c >> 16u) & 0xffu), f32(c >> 24u)) / 255.0;
}

// End e (0 or 1) of segment k of the unit frustum.
fn unit_end(k: u32, e: u32) -> vec3<f32> {
    var corner = array<vec3<f32>, 4>(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 1.0),
                                     vec3<f32>(0.0, 1.0, 1.0));
    if (k < 4u) {
        return select(vec3<f32>(0.0), corner[k], e == 1u);
    }
    return corner[(k - 4u + e) % 4u];
}

fn clip_of(p: vec3<f32>) -> vec4<f32> {
    let f = frustum[0];
    let world = mat4x4<f32>(f.c0, f.c1, f.c2, f.c3) * vec4<f32>(p, 1.0);
    var c = caliper_view.clip_from_world * world;
    c.z = c.z - f32(params.depth_offset) * (1.0 / 8192.0) * c.w;
    return c;
}

fn quad_corner(c: u32) -> vec2<f32> {
    var t = array<vec2<f32>, 6>(vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
                                vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    return t[min(c, 5u)];
}

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

@vertex
fn frustum_vs(@builtin(instance_index) k: u32, @builtin(vertex_index) v: u32) -> FrustumOut {
    var out: FrustumOut;
    out.color = unpack(params.color);
    let c = v % 6u;
    if (counts.frustum == 0u) {
        out.position = hidden();
        return out;
    }
    out.position = segment_corner(clip_of(unit_end(k, 0u)), clip_of(unit_end(k, 1u)), c, params.width * caliper_view.time.z);
    return out;
}

@fragment
fn frustum_fs(in: FrustumOut) -> CaliperPicked {
    var out: CaliperPicked;
    out.color = in.color;
    out.id = pick::id(0u);
    return out;
}
