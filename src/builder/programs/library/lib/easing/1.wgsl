// lib module easing@1 (design/programmable-graphics.md, Lib Modules): curves
// from 0 to 1 for motion that draw buffers keep between frames. A piece that
// imports it calls easing::smooth and easing::out_cubic. Released; never
// changes.

fn easing__smooth(t: f32) -> f32 {
    let x = clamp(t, 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
}

fn easing__out_cubic(t: f32) -> f32 {
    let x = 1.0 - clamp(t, 0.0, 1.0);
    return 1.0 - x * x * x;
}
