// The derive half of points@1: one element of `points` for each row of `positions`, with the
// bits of the row's value (NaN where `values` is shorter), and the count of rows.

@compute @workgroup_size(256)
fn copy_points(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = min(shapes.positions.z, arrayLength(&points));
    if (id.x == 0u) {
        atomicStore(&points_count, n);
    }
    if (id.x >= n) {
        return;
    }
    let d = shapes.positions.w;
    let base = id.x * d;
    var p = vec4<f32>(positions[base], 0.0, 0.0, 1.0);
    if (d > 1u) {
        p.y = positions[base + 1u];
    }
    if (d > 2u) {
        p.z = positions[base + 2u];
    }
    var bits = 0x7fc00000u;
    if (id.x < shapes.values.w) {
        bits = bitcast<u32>(values[id.x]);
    }
    points[id.x] = Point(p, bits, id.x);
}
