// Draw the 10x10 row-normalized confusion matrix, highlighting the selected digit's row and column.
@fragment
fn grid(in: CaliperFullscreen) -> @location(0) vec4<f32> {
    let bg = vec4<f32>(0.094, 0.090, 0.086, 1.0);
    let span = vec2<f32>(GX1 - GX0, GY1 - GY0);
    let c = (in.uv - vec2<f32>(GX0, GY0)) / span * f32(CLASSES);
    let px = fwidth(c);
    if (any(c < vec2<f32>(0.0)) || any(c >= vec2<f32>(f32(CLASSES)))) {
        return bg;
    }
    let col = u32(c.x);
    let row = u32(c.y);
    let f = fract(c);
    let val = matrix[row * CLASSES + col];
    var color = colormap::magma(clamp(val * params.gain, 0.0, 1.0));

    let sel_digit = u32(clamp(params.digit, 0, i32(CLASSES) - 1));
    let sel_pick = u32(clamp(params.picked, 0, i32(CLASSES) - 1));
    if (row == sel_digit) {
        color = mix(color, vec4<f32>(0.30, 0.86, 0.84, 1.0), 0.32);
    }
    if (row == sel_pick || col == sel_pick) {
        if (any(f < 2.0 * px) || any(f > vec2<f32>(1.0) - 2.0 * px)) {
            color = vec4<f32>(0.96, 0.80, 0.30, 1.0);
        }
    } else if (any(f < px)) {
        color = mix(color, bg, 0.8);
    }
    return color;
}
