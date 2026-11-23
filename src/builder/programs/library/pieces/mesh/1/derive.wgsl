// The derive half of mesh@1: one element of `vertices` for each row of `positions`, with its value
// in position.w, its normal, its RGBA and its index; and three elements of `faces`, the index
// buffer of the draw passes, for each row of `indices`, or for every three vertices when indexed is
// 0 or `indices` is unbound, with each index clamped to the last vertex. Unbound `normals` and
// `colors` have the shape (0, 0, 0, 0): no normal, and the one color without a value. Vertex 0
// also holds the flags of all vertices in normal.w: 1 when the RGBA of a vertex has an alpha below
// 255. A tensor with one row or one value gives it to every vertex, as a stream of stride 0 does in
// the built-in.

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
        a = mesh_byte(colors[base + 3u]);
    }
    return mesh_byte(colors[base]) | (mesh_byte(colors[base + 1u]) << 8u) | (mesh_byte(colors[base + 2u]) << 16u) |
           (a << 24u);
}

// The value of vertex i, element i of `colors`, or NaN without one.
fn mesh_value(i: u32) -> f32 {
    if (shapes.colors.w == 0u) {
        return bitcast<f32>(0x7fc00000u);
    }
    let n = arrayLength(&colors);
    if (n == 1u) {
        return colors[0];
    }
    if (i < n) {
        return colors[i];
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
    vertices[i] = Vertex(p, normal, mesh_rgba(i), i);
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
