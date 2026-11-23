// The derive half of grid_material: one point for each of the first 512 values of the weight, at the place
// where the geometry program grid puts it, with the value beside it.
@compute @workgroup_size(64)
fn place(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= 512u) {
        return;
    }
    var v = 0.0;
    if (i < shapes.w.z * shapes.w.w) {
        v = w[i];
    }
    let p = vec4<f32>(f32(i % 8u) * 0.25 - 0.875, f32(i / 8u) * (2.0 / 64.0) - 1.0, v, 1.0);
    points[i] = Point(p, v);
}
