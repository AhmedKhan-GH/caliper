// The draw half of mesh@1, as the built-in mesh primitive draws: each triangle at its clip
// positions, a vertex with a NaN coordinate out of the picture. With shading 1, each vertex color
// is lit by the headlight from its normal, ambient 0.30; with shading 0, each face is lit from the
// normal that the derivatives of its position give, on both of its sides. A colormap color is the
// host colormap's texel of the level that the host's reference gives a value, (v - vmin) / (vmax -
// vmin) * 255 rounded, with NaN at level 0, as points@1 computes it.
//
// The built-in draws an opaque mesh with depth, and a translucent one, with an alpha below 255 in
// its one color or a vertex, without depth; its pick pass draws it with depth, thus the nearest
// face wins. Pass 0 or 1 draws by the same rule, and pass 2 writes the IDs of a
// translucent mesh with depth and leaves the color as it is. A pass that does not apply moves each
// vertex out of the clip volume. Each pass draws the index buffer `faces`, thus a vertex runs once
// for each index and the faces share their vertices, as in the built-in's indexed draw. The item of
// a face is its first vertex, the flat value of the face's first index.

struct MeshOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) item: u32,
    @location(2) obj: vec3<f32>,
    @location(3) uv: vec2<f32>,
}

const MESH_DEPTH_UNIT: f32 = 1.0 / 8192.0;

fn mesh_hidden() -> vec4<f32> {
    return vec4<f32>(0.0, 0.0, 2.0, 1.0);
}

fn is_nan(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

// A packed color, r | g << 8 | b << 16 | a << 24, as four channels from 0 to 1.
fn mesh_unpack(c: u32) -> vec4<f32> {
    return vec4<f32>(f32(c & 0xffu), f32((c >> 8u) & 0xffu), f32((c >> 16u) & 0xffu), f32((c >> 24u) & 0xffu)) / 255.0;
}

// The colormap level of a value, 0 to 255, as the host's reference computes it.
fn mesh_level(v: f32) -> u32 {
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

fn mesh_color(v: Vertex) -> vec4<f32> {
    if (params.color_mode == 1u) {
        let size = textureDimensions(caliper_colormaps);
        let x = min(mesh_level(v.position.w) * (size.x - 1u) / 255u, size.x - 1u);
        return textureLoad(caliper_colormaps, vec2<u32>(x, min(params.colormap, size.y - 1u)), 0);
    }
    if (params.color_mode == 2u) {
        return mesh_unpack(v.rgba);
    }
    return mesh_unpack(params.color);
}

// Whether the built-in draws the mesh translucent.
fn mesh_translucent() -> bool {
    if ((params.color >> 24u) < 255u) {
        return true;
    }
    // The flag of vertex 0: a vertex RGBA, or a texel of a textured mesh, with an alpha below 255.
    return (params.color_mode == 2u || counts.tex > 0u) && vertices[0].normal.w != 0.0;
}

// A normal in view space through the rotation of the view, normalized.
fn view_normal(n: vec3<f32>) -> vec3<f32> {
    let m = caliper_view.view_from_world;
    return normalize(n.x * m[0].xyz + n.y * m[1].xyz + n.z * m[2].xyz);
}

// The headlight: ambient 0.30, the alpha untouched.
fn lambert(c: vec4<f32>, n: vec3<f32>) -> vec4<f32> {
    let lit = 0.30 + 0.70 * max(dot(view_normal(n), vec3<f32>(0.0, 0.0, 1.0)), 0.0);
    return vec4<f32>(c.rgb * lit, c.a);
}

// Vertex `index` of the mesh, an element of the index buffer.
fn face_vertex(index: u32, shown: bool) -> MeshOut {
    let v = vertices[index];
    var out: MeshOut;
    out.item = index;
    out.obj = v.position.xyz;
    out.uv = v.uv;
    var c = mesh_color(v);
    if (params.shading == 1u) {
        c = lambert(c, normalize(v.normal.xyz));
    }
    out.color = c;
    if (!shown || is_nan(v.position.x) || is_nan(v.position.y) || is_nan(v.position.z)) {
        out.position = mesh_hidden();
        return out;
    }
    var p = caliper_view.clip_from_world * vec4<f32>(v.position.xyz, 1.0);
    if (params.depth_offset != 0) {
        p.z = p.z - f32(params.depth_offset) * MESH_DEPTH_UNIT * p.w;
    }
    out.position = p;
    return out;
}

// The color of a fragment: the texture times the vertex color's RGB with the texture's alpha, as the
// built-in's textured fragment; else a flat face, or the vertex color.
fn shade(in: MeshOut) -> vec4<f32> {
    // `counts` is uniform, thus the sample stays in uniform control flow.
    if (counts.tex > 0u) {
        let s = textureSample(tex, caliper_linear, in.uv);
        return vec4<f32>(s.rgb * in.color.rgb, s.a);
    }
    let n = cross(dpdx(in.obj), dpdy(in.obj));
    if (params.shading == 0u) {
        let lit = 0.30 + 0.70 * abs(view_normal(n).z);
        return vec4<f32>(in.color.rgb * lit, in.color.a);
    }
    return in.color;
}

@vertex
fn face_vs(@builtin(vertex_index) i: u32) -> MeshOut {
    return face_vertex(i, !mesh_translucent());
}

@vertex
fn face_see_vs(@builtin(vertex_index) i: u32) -> MeshOut {
    return face_vertex(i, mesh_translucent());
}

@vertex
fn face_id_vs(@builtin(vertex_index) i: u32) -> MeshOut {
    return face_vertex(i, mesh_translucent());
}

@fragment
fn face_fs(in: MeshOut) -> CaliperPicked {
    var out: CaliperPicked;
    out.color = shade(in);
    out.id = pick::id(in.item);
    return out;
}

@fragment
fn see_fs(in: MeshOut) -> @location(0) vec4<f32> {
    return shade(in);
}

@fragment
fn id_fs(in: MeshOut) -> CaliperPicked {
    var out: CaliperPicked;
    out.color = vec4<f32>(0.0);
    out.id = pick::id(in.item);
    return out;
}
