// The derive half of mesh@1: one element of `vertices` for each row of `positions`, with its value
// in position.w, its normal, its RGBA and its index; and three elements of `faces`, the index
// buffer of the draw passes, for each row of `indices`, or for every three vertices when indexed is
// 0 or `indices` is unbound, with each index clamped to the last vertex. Unbound `normals` and
// `colors` have the shape (0, 0, 0, 0): no normal, and the one color without a value. Vertex 0
// also holds the flags of all vertices in normal.w: 1 when the RGBA of a vertex has an alpha below
// 255. A tensor with one row or one value gives it to every vertex, as a stream of stride 0 does in
// the built-in. With `uvs` and `texture` bound, each vertex takes its uv, the texture resource `tex` takes
// the pixels of `texture` as packed RGBA8 the way the producer converts them, and a texel with an
// alpha below 255 sets the flag too, as it makes the built-in's textured mesh translucent; without
// them the texture has no texels.

const TEXTURE_THREADS: u32 = 16384u;   // 64 workgroups of 256

fn mesh_vertex_count() -> u32 {
    return min(shapes.positions.z, arrayLength(&vertices));
}

fn mesh_triangle_count() -> u32 {
    let v = mesh_vertex_count();
    if (v == 0u) {
        return 0u;
    }
    let most = arrayLength(&faces) / 3u;
    if (params.indexed == 0u || shapes.indices.w == 0u) {
        return min(v / 3u, most);
    }
    if (shapes.indices.w < 3u) {
        return 0u;
    }
    return min(shapes.indices.z, most);
}

// A channel from 0 to 1 as a byte, as the scene's Python side converts one: clamped, times 255,
// plus 0.5, truncated. NaN gives 0.
fn mesh_byte(v: f32) -> u32 {
    if (!(v > 0.0)) {
        return 0u;
    }
    return u32(min(v, 1.0) * 255.0 + 0.5);
}

// Channel k of `colors` from 0 to 255: the byte of u8 colors, or an f32 color from 0 to 1.
fn mesh_channel(k: u32) -> u32 {
    let v = colors_at(k);
    if (colors_type == CALIPER_U8) {
        return u32(v);
    }
    return mesh_byte(v);
}

// The RGBA of vertex i from the rows of `colors` (3 or 4 channels), packed r | g << 8 | b << 16 |
// a << 24. A vertex without a row takes the one color.
fn mesh_rgba(i: u32) -> u32 {
    let rows = shapes.colors.z;
    let w = shapes.colors.w;
    if (w < 3u || (rows != 1u && i >= rows)) {
        return params.color;
    }
    let base = select(i, 0u, rows == 1u) * w;
    var a = 255u;
    if (w > 3u) {
        a = mesh_channel(base + 3u);
    }
    return mesh_channel(base) | (mesh_channel(base + 1u) << 8u) | (mesh_channel(base + 2u) << 16u) | (a << 24u);
}

// The value of vertex i, element i of `colors`, or NaN without one.
fn mesh_value(i: u32) -> f32 {
    if (shapes.colors.w == 0u) {
        return bitcast<f32>(0x7fc00000u);
    }
    let n = shapes.colors.z * shapes.colors.w;   // the elements of a tensor of rank 1 or 2
    if (n == 1u) {
        return colors_at(0u);
    }
    if (i < n) {
        return colors_at(i);
    }
    return bitcast<f32>(0x7fc00000u);
}

@compute @workgroup_size(256)
fn copy_vertices(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = mesh_vertex_count();
    if (id.x == 0u) {
        atomicStore(&vertices_count, n);
    }
    if (id.x >= n) {
        return;
    }
    let i = id.x;
    let d = shapes.positions.w;
    let base = i * d;
    var p = vec4<f32>(positions[base], 0.0, 0.0, mesh_value(i));
    if (d > 1u) {
        p.y = positions[base + 1u];
    }
    if (d > 2u) {
        p.z = positions[base + 2u];
    }
    var normal = vec4<f32>(0.0);
    let nw = shapes.normals.w;
    if (nw >= 3u && i < shapes.normals.z) {
        normal = vec4<f32>(normals[i * nw], normals[i * nw + 1u], normals[i * nw + 2u], 0.0);
    }
    var uv = vec2<f32>(0.0);
    let uw = shapes.uvs.w;
    if (uw >= 2u && i < shapes.uvs.z) {
        uv = vec2<f32>(uvs[i * uw], uvs[i * uw + 1u]);
    }
    vertices[i] = Vertex(p, normal, mesh_rgba(i), i, uv);
}

// The width, the height and the channels of `texture`, with a size of 0 when there is no texture:
// uvs or the texture unbound, another shape, or more pixels than the resource holds.
fn texture_dims() -> vec3<u32> {
    let s = shapes.texture;
    if (s.w == 0u || shapes.uvs.w < 2u) {
        return vec3<u32>(0u, 0u, 1u);
    }
    var d = vec3<u32>(s.w, s.z, 1u);
    if (s.w == 3u || s.w == 4u) {
        d = vec3<u32>(s.z, s.y, s.w);
    } else if (s.y != 1u) {
        return vec3<u32>(0u, 0u, 1u);
    }
    if (d.x > 8192u || d.y > 8192u || d.y > arrayLength(&tex) / d.x) {
        return vec3<u32>(0u, 0u, 1u);
    }
    return d;
}

// Channel i of the texture from 0 to 255: the byte of a u8 texture, or an f32 value from 0 to 1.
fn texture_channel(i: u32) -> u32 {
    let v = texture_at(i);
    if (texture_type == CALIPER_U8) {
        return u32(v);
    }
    return mesh_byte(v);
}

@compute @workgroup_size(256)
fn copy_texture(@builtin(global_invocation_id) id: vec3<u32>) {
    let d = texture_dims();
    let n = d.x * d.y;
    if (id.x == 0u) {
        tex_size = d.xy;   // 0 by 0: no texels
    }
    var see = false;
    for (var i = id.x; i < n; i = i + TEXTURE_THREADS) {
        var rgba = 0xff000000u;
        if (d.z == 1u) {
            let g = texture_channel(i);
            rgba = rgba | g | (g << 8u) | (g << 16u);
        } else {
            let b = i * d.z;
            rgba = texture_channel(b) | (texture_channel(b + 1u) << 8u) | (texture_channel(b + 2u) << 16u);
            if (d.z == 4u) {
                rgba = rgba | (texture_channel(b + 3u) << 24u);
            } else {
                rgba = rgba | 0xff000000u;
            }
        }
        tex[i] = rgba;
        see = see || (rgba >> 24u) < 255u;
    }
    if (see) {
        vertices[0].normal.w = 1.0;
    }
}

@compute @workgroup_size(256)
fn copy_triangles(@builtin(global_invocation_id) id: vec3<u32>) {
    let t = mesh_triangle_count();
    if (id.x == 0u) {
        atomicStore(&faces_count, t * 3u);
    }
    if (id.x >= t) {
        return;
    }
    let last = mesh_vertex_count() - 1u;
    let f = id.x * 3u;
    if (params.indexed == 0u || shapes.indices.w == 0u) {
        faces[f] = f;
        faces[f + 1u] = f + 1u;
        faces[f + 2u] = f + 2u;
        return;
    }
    // An index is read as the bits of an i32, thus a negative index is above the last vertex too.
    let base = id.x * shapes.indices.w;
    let a = min(bitcast<u32>(indices[base]), last);
    let b = min(bitcast<u32>(indices[base + 1u]), last);
    let c = min(bitcast<u32>(indices[base + 2u]), last);
    faces[f] = a;
    faces[f + 1u] = b;
    faces[f + 2u] = c;
}

// The flags into vertex 0, after copy_vertices: every vertex with an alpha below 255 writes the
// same 1, thus the order of the writes does not matter.
@compute @workgroup_size(256)
fn mark_alpha(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < mesh_vertex_count() && (vertices[id.x].rgba >> 24u) < 255u) {
        vertices[0].normal.w = 1.0;
    }
}
