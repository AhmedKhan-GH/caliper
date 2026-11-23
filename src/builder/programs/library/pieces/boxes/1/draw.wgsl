// The draw half of boxes@1, as the built-in boxes primitive draws: each box is the unit cube, or
// the unit square in 2-D, posed by its center, half sizes and rotation. Its 12 edges are quads
// `width` pixels wide at the display's pixel ratio with square caps, and its 6 faces, with
// `fill`, are triangles at a quarter of the box's alpha. In 2-D a box draws the 4 edges of the
// square at z = -1 and the face at z = +1, both flat at z = 0.
//
// The built-in draws the edges of all boxes as one opaque draw that writes depth, or, when an
// alpha is below 255, as one translucent draw that writes none, and the faces after them as a
// translucent draw. Its pick pass draws edges and faces again with depth, thus the nearest wins.
// The passes here follow that: the edges in pass 0 or 1 by the same rule, the faces in pass 2,
// and the IDs of translucent edges and of faces in passes 3 and 4, which write depth and leave
// the color as it is. A pass that does not apply to the boxes moves each vertex out of the clip
// volume. The depth offset moves the edges only, in 3-D, as it moves the built-in's batch.

struct BoxOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) item: u32,
}

const BOX_DEPTH_UNIT: f32 = 1.0 / 8192.0;

// A position that the clipper removes (z > w).
fn box_hidden() -> vec4<f32> {
    return vec4<f32>(0.0, 0.0, 2.0, 1.0);
}

// A packed color, r | g << 8 | b << 16 | a << 24, as four channels from 0 to 1.
fn box_unpack(c: u32) -> vec4<f32> {
    return vec4<f32>(f32(c & 0xffu), f32((c >> 8u) & 0xffu), f32((c >> 16u) & 0xffu), f32((c >> 24u) & 0xffu)) / 255.0;
}

fn box_color(b: Box) -> vec4<f32> {
    if (params.color_mode == 1u) {
        return box_unpack(b.rgba);
    }
    if (params.color_mode == 2u) {
        return box_unpack(b.class_rgba);
    }
    return box_unpack(params.color);
}

// Whether the built-in draws these boxes translucent: the one color's alpha below 255, or in the
// mode of RGBA, the alpha of a box below 255.
fn box_translucent() -> bool {
    if ((params.color >> 24u) < 255u) {
        return true;
    }
    return params.color_mode == 1u && (u32(boxes[0].half_size.w) & 1u) != 0u;
}

// The column-major pose of a box: translation, rotation, then the half sizes as scale.
fn box_pose(b: Box) -> mat4x4<f32> {
    var q = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    let n2 = dot(b.rotation, b.rotation);
    if (n2 > 0.0) {
        q = b.rotation * inverseSqrt(n2);
    }
    let x = q.x;
    let y = q.y;
    let z = q.z;
    let w = q.w;
    let c0 = vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y));
    let c1 = vec3<f32>(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x));
    let c2 = vec3<f32>(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y));
    let h = b.half_size.xyz;
    return mat4x4<f32>(vec4<f32>(c0 * h.x, 0.0), vec4<f32>(c1 * h.y, 0.0), vec4<f32>(c2 * h.z, 0.0),
                       vec4<f32>(b.center.xyz, 1.0));
}

fn on_axis(axis: u32, along: f32, s: f32, t: f32) -> vec3<f32> {
    if (axis == 0u) {
        return vec3<f32>(along, s, t);
    }
    if (axis == 1u) {
        return vec3<f32>(s, along, t);
    }
    return vec3<f32>(s, t, along);
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

// Corner c of segment (a, b) in clip space, as a quad width_px wide whose ends are extended by
// half the width (square caps).
fn segment_corner(a_in: vec4<f32>, b_in: vec4<f32>, c: u32, width_px: f32) -> vec4<f32> {
    var a = a_in;
    var b = b_in;
    if (!clip_near(&a, &b)) {
        return box_hidden();
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

// The clip position of a point of an edge, with the depth offset in 3-D.
fn edge_clip(pose: mat4x4<f32>, local: vec3<f32>, dims: u32) -> vec4<f32> {
    var c = caliper_view.clip_from_world * (pose * vec4<f32>(local, 1.0));
    if (dims == 3u && params.depth_offset != 0) {
        c.z = c.z - f32(params.depth_offset) * BOX_DEPTH_UNIT * c.w;
    }
    return c;
}

// Vertex v of the 12 edges of box i, 72 vertices; the first 24 are the square at z = -1, which
// is all a box in 2-D draws.
fn edge_vertex(i: u32, v: u32, shown: bool) -> BoxOut {
    let b = boxes[i];
    var out: BoxOut;
    out.item = i;
    out.color = box_color(b);
    let dims = u32(b.center.w);
    if (!shown || (dims == 2u && v >= 24u)) {
        out.position = box_hidden();
        return out;
    }
    let e = v / 6u;
    var axis = (e % 4u) / 2u;
    let s = select(-1.0, 1.0, (e & 1u) != 0u);
    var t = select(-1.0, 1.0, e >= 4u);
    if (e >= 8u) {
        axis = 2u;
        t = select(-1.0, 1.0, (e & 2u) != 0u);
    }
    let pose = box_pose(b);
    let a = edge_clip(pose, on_axis(axis, -1.0, s, t), dims);
    let z = edge_clip(pose, on_axis(axis, 1.0, s, t), dims);
    out.position = segment_corner(a, z, v % 6u, params.width * caliper_view.time.z);
    return out;
}

// Vertex v of the 6 faces of box i, 36 vertices; the first 6 are the face at z = +1, which is all
// a box in 2-D draws.
fn face_vertex(i: u32, v: u32) -> BoxOut {
    let b = boxes[i];
    var out: BoxOut;
    out.item = i;
    let c = box_color(b);
    out.color = vec4<f32>(c.rgb, c.a * 0.25);
    let dims = u32(b.center.w);
    if (params.fill == 0u || (dims == 2u && v >= 6u)) {
        out.position = box_hidden();
        return out;
    }
    let face = v / 6u;
    var order = array<u32, 3>(2u, 0u, 1u);
    let axis = order[min(face / 2u, 2u)];
    let sgn = select(1.0, -1.0, (face & 1u) != 0u);
    let q = quad_corner(v % 6u);
    let pose = box_pose(b);
    out.position = caliper_view.clip_from_world * (pose * vec4<f32>(on_axis(axis, sgn, q.x * 2.0 - 1.0, q.y), 1.0));
    return out;
}

@vertex
fn wire_vs(@builtin(instance_index) i: u32, @builtin(vertex_index) v: u32) -> BoxOut {
    return edge_vertex(i, v, !box_translucent());
}

@vertex
fn wire_see_vs(@builtin(instance_index) i: u32, @builtin(vertex_index) v: u32) -> BoxOut {
    return edge_vertex(i, v, box_translucent());
}

@vertex
fn wire_id_vs(@builtin(instance_index) i: u32, @builtin(vertex_index) v: u32) -> BoxOut {
    return edge_vertex(i, v, box_translucent());
}

@vertex
fn face_vs(@builtin(instance_index) i: u32, @builtin(vertex_index) v: u32) -> BoxOut {
    return face_vertex(i, v);
}

@vertex
fn face_id_vs(@builtin(instance_index) i: u32, @builtin(vertex_index) v: u32) -> BoxOut {
    return face_vertex(i, v);
}

@fragment
fn wire_fs(in: BoxOut) -> CaliperPicked {
    var out: CaliperPicked;
    out.color = in.color;
    out.id = pick::id(in.item);
    return out;
}

@fragment
fn see_fs(in: BoxOut) -> @location(0) vec4<f32> {
    return in.color;
}

@fragment
fn id_fs(in: BoxOut) -> CaliperPicked {
    var out: CaliperPicked;
    out.color = vec4<f32>(0.0);
    out.id = pick::id(in.item);
    return out;
}
