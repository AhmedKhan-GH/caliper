// The derive half of path@1, the path operator's polyline: every s-th row of `rows` (R, D), with
// the smallest s for which at most max_points rows remain, and never more than `vertices` holds.
// Each kept row is a vertex at its columns x, y and z (z 0 when its column is below 0 or past the
// row), and its value is the row's index, as the operator's reference gives them. A column of x
// or y past the row keeps no vertex.

// a / b rounded up, for a above 0 and b above 0, without a sum that can wrap.
fn ceil_div(a: u32, b: u32) -> u32 {
    return (a - 1u) / b + 1u;
}

@compute @workgroup_size(256)
fn keep_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    let r = shapes.rows.z;
    let d = shapes.rows.w;
    let limit = min(max(params.max_points, 1u), arrayLength(&vertices));
    var stride = 1u;
    if (r > limit) {
        stride = ceil_div(r, limit);
    }
    var count = 0u;
    if (r > 0u && params.x >= 0 && params.y >= 0 && u32(params.x) < d && u32(params.y) < d) {
        count = ceil_div(r, stride);
    }
    if (id.x == 0u) {
        atomicStore(&vertices_count, count);
    }
    let p = id.x;
    if (p >= count) {
        return;
    }
    let row = p * stride;
    let base = row * d;
    var z = 0.0;
    if (params.z >= 0 && u32(params.z) < d) {
        z = rows[base + u32(params.z)];
    }
    vertices[p] = Vertex(rows[base + u32(params.x)], rows[base + u32(params.y)], z, f32(row), row);
}
