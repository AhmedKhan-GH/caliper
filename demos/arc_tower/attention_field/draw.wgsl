// Rows are the query token, columns the key token.
fn value(layer: u32, head: u32, i: u32, j: u32) -> f32 {
    return pattern[((layer * H + head) * T + i) * T + j];
}

// The pattern of the head that the params name, sampled with bilinear weights between the tokens,
// in magma, with a light contour line at each eighth of the range.
@fragment
fn field(in: CaliperFullscreen) -> @location(0) vec4<f32> {
    let layer = u32(clamp(params.layer, 0, 1));
    let head = u32(clamp(params.head, 0, i32(H) - 1));
    let p = in.uv * f32(T) - 0.5;
    let a = vec2<u32>(clamp(floor(p), vec2<f32>(0.0), vec2<f32>(f32(T - 1u))));
    let b = min(a + 1u, vec2<u32>(T - 1u));
    let f = clamp(p - floor(p), vec2<f32>(0.0), vec2<f32>(1.0));
    let v = mix(mix(value(layer, head, a.y, a.x), value(layer, head, a.y, b.x), f.x),
                mix(value(layer, head, b.y, a.x), value(layer, head, b.y, b.x), f.x), f.y);
    let t = clamp(v * params.gain, 0.0, 1.0);
    var color = colormap::magma(t);
    if (abs(fract(t * 8.0) - 0.5) < 0.04) {
        color = mix(color, vec4<f32>(1.0), 0.35);
    }
    return color;
}
