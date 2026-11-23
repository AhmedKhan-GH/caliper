// Rows are the query token, from the top; columns the key token, from the left.
fn value(layer: u32, head: u32, i: u32, j: u32) -> f32 {
    return pattern[((layer * H + head) * T + i) * T + j];
}

// The attention grid of the head that the params name: one flat cell for each pair of tokens, in
// magma, a thin gap between cells, the empty half where a token would look ahead, an outline on
// the cells that an induction head fills, and the row and column under the pointer lit.
@fragment
fn grid(in: CaliperFullscreen) -> @location(0) vec4<f32> {
    let bg = vec4<f32>(0.094, 0.090, 0.086, 1.0);
    let span = vec2<f32>(GX1 - GX0, GY1 - GY0);
    let c = (in.uv - vec2<f32>(GX0, GY0)) / span * f32(T);
    let px = fwidth(c);   // cells per pixel, before any branch
    if (any(c < vec2<f32>(0.0)) || any(c >= vec2<f32>(f32(T)))) {
        return bg;
    }
    let layer = u32(clamp(params.layer, 0, 1));
    let head = u32(clamp(params.head, 0, i32(H) - 1));
    let j = u32(c.x);
    let i = u32(c.y);
    let f = fract(c);
    if (j > i) {
        return bg;
    }
    let w = value(layer, head, i, j);
    var color = colormap::magma(clamp(w * params.gain, 0.0, 1.0));

    // The row and the column under the pointer.
    let pv = caliper_view.pointer;
    if (pv.w > 0.5) {
        let pc = (pv.xy / caliper_view.viewport.zw - vec2<f32>(GX0, GY0)) / span * f32(T);
        if (all(pc >= vec2<f32>(0.0)) && all(pc < vec2<f32>(f32(T)))) {
            let hover = vec2<u32>(pc);
            if (hover.x == j || hover.y == i) {
                color = mix(color, vec4<f32>(1.0), 0.12);
            }
        }
    }

    // The induction cells: from token t of the second copy to the token after its first occurrence.
    let induction = i >= S + P && i < S + 2u * P && j + P == i + 1u;
    if (induction && (any(f < 2.0 * px) || any(f > vec2<f32>(1.0) - 2.0 * px))) {
        color = vec4<f32>(0.40, 0.82, 0.80, 1.0);
    } else if (any(f < px)) {
        color = mix(color, bg, 0.8);
    }
    return color;
}
