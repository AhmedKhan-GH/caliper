// The draw half of grid_material: each point a square of `size` pixels over the grid program's point at the
// same place, in orange for a low value and in yellow for a high one.
struct MaterialOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn material_vs(@builtin(instance_index) i: u32, @builtin(vertex_index) k: u32) -> MaterialOut {
    var corners = array<vec2<f32>, 6>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
                                      vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    let p = points[i];
    let c = caliper_view.clip_from_world * vec4<f32>(p.position.xyz, 1.0);
    let half = params.size * caliper_view.time.z / caliper_view.viewport.zw;
    var out: MaterialOut;
    out.position = vec4<f32>(c.xy + corners[min(k, 5u)] * half * c.w, c.z, c.w);
    out.color = vec4<f32>(1.0, 0.55 + 0.45 * clamp(p.value * 4.0, -1.0, 1.0), 0.05, 1.0);
    return out;
}

@fragment
fn material_fs(in: MaterialOut) -> @location(0) vec4<f32> {
    return in.color;
}
